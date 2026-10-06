"""Predeclared 2x2 input/background ablation; never promotes a model."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .background_abstention import WAFERS, checked_patch, digest, load, save
from .independent_background import (
    IndependentBackgroundCNN,
    metrics,
    reviewed_candidates,
    triage,
)

CONTRACT = 'center_background_v1_rgb128_uint8_mad_box9_31_center48_gaussian_context'
ARMS = ('raw_base', 'focused_base', 'raw_added', 'focused_added')


class FocusedBackgroundCNN(IndependentBackgroundCNN):
    """Keep trainable architecture fixed; change its two image views only."""
    def __init__(self):
        super().__init__()
        yy, xx = torch.meshgrid(torch.arange(128)-63.5, torch.arange(128)-63.5, indexing='ij')
        radius = xx.square()+yy.square()
        self.register_buffer('annulus', (radius >= 32**2) & (radius <= 60**2))
        self.register_buffer('context_weight', torch.exp(-radius/(2*32**2))[None, None])

    def views(self, image):
        gray = image.mean(1, keepdim=True)
        channels = []
        for size in (9, 31):
            pad = size//2
            residual = gray-F.avg_pool2d(F.pad(gray, (pad,pad,pad,pad), mode='reflect'), size, 1)
            ring = residual[:, :, self.annulus]
            median = ring.median(-1, keepdim=True).values
            scale = (ring-median).abs().median(-1).values[:, :, None, None]*1.4826
            scale = scale.clamp_min(1/255)
            channels.append((residual/scale).clamp(-4,4)/8)
        features = torch.cat([.5+channels[0], .5+channels[1], 2*channels[0].abs()], 1)
        # Both branches still receive 64x64x3; model parameter count is unchanged.
        center = F.interpolate(features[:, :, 40:88, 40:88], size=(64,64), mode='bilinear', align_corners=False)
        neutral = image.new_tensor([.5,.5,0])[None,:,None,None]
        context = F.avg_pool2d(neutral+(features-neutral)*self.context_weight, 2)
        return center, context

    def forward(self, image):
        center, context = self.views(image)
        return self.head(torch.cat([self.center(center), self.context(context)],1)).squeeze(1)


def validate_added(data, info, known_ids, known_hashes):
    if (data.get('schema') != 'xrt_weak_background_v1' or data.get('test_evaluated') is not False
            or data.get('cohort_manifest_sha256') != info['cohort_manifest_sha256']
            or data.get('coordinate_repair_sha256') != info['coordinate_repair_sha256']):
        raise ValueError('Additional weak-background contract mismatch')
    rows = data['candidates']
    ids, hashes = set(known_ids), set(known_hashes)
    for row in rows:
        if (row['wafer'] not in WAFERS or row.get('review_actor') != 'Codex_AI'
                or row.get('human_verified') is not False or row.get('expert_ground_truth') is not False
                or row.get('decision') not in ('background_candidate','uncertain','artifact_candidate')
                or row.get('weak_training_eligible') is not (row['decision']=='background_candidate')):
            raise ValueError('Invalid additional provenance or holdout wafer')
        if row['id'] in ids or row['sha256'] in hashes:
            raise ValueError('Duplicate additional identity or patch content')
        ids.add(row['id'])
        hashes.add(row['sha256'])
    if {r['wafer'] for r in rows if r['weak_training_eligible']} != set(WAFERS):
        raise ValueError('Additional weak negatives required in all development wafers')
    return rows


def fold_ids(rows, wafer, added):
    if wafer not in WAFERS or any(r['wafer'] not in WAFERS for r in rows):
        raise ValueError('Development wafers only')
    train = [i for i,r in enumerate(rows) if r['wafer'] != wafer and r['train_eligible']
             and (added or not r['additional'])]
    held = [i for i,r in enumerate(rows) if r['wafer'] == wafer]
    if not train or not held:
        raise ValueError('Empty training or held-out split')
    return train, held


def predict(model, tensors):
    model.eval()
    values = []
    with torch.inference_mode():
        for batch, in DataLoader(TensorDataset(tensors), batch_size=64):
            values.extend(model(batch.cuda().float()/255).sigmoid().cpu().tolist())
    return values


def train_model(model, rows, images, train_ids, draws, epochs, seed):
    # Additional backgrounds share the old random-background stratum's sampling
    # mass, so augmentation does not silently increase the negative class prior.
    groups = [rows[i]['sample_group'] for i in train_ids]
    counts = Counter(groups)
    labels = torch.tensor([0. if rows[i]['stratum'].startswith('background_') else 1. for i in train_ids])
    generator = torch.Generator().manual_seed(seed+1)
    sampler = WeightedRandomSampler([1/counts[g] for g in groups], draws, replacement=True, generator=generator)
    loader = DataLoader(TensorDataset(images[train_ids], labels), batch_size=32, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    history = []
    for epoch in range(epochs):
        model.train()
        total, loss_sum = 0, 0.
        for inputs, target in loader:
            inputs, target = inputs.cuda().float()/255, target.cuda()
            inputs = torch.rot90(inputs,random.randrange(4),(2,3))
            if random.random()<.5:
                inputs = inputs.flip(3)
            inputs = (inputs*random.uniform(.8,1.2)+random.uniform(-.04,.04)).clamp(0,1)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(inputs),target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(target)
            total += len(target)
        history.append({'epoch':epoch+1,'loss':loss_sum/total})
        print(f'epoch {epoch+1}/{epochs}: {loss_sum/total:.4f}',flush=True)
    return history


def run(cohort, background, baseline, observations, added, frozen, output, epochs=16, seed=42):
    if epochs < 1 or not torch.cuda.is_available():
        raise ValueError('Positive epochs and CUDA required')
    torch.set_num_threads(4)
    output, observations, added, frozen = map(Path,(output,observations,added,frozen))
    output.mkdir(parents=True,exist_ok=False)
    rows, images, _, info = load(cohort,background,baseline)
    prior = json.loads((frozen/'plan.json').read_text(encoding='utf-8'))
    for key in ('cohort_manifest_sha256','background_sha256','baseline_oof_sha256','coordinate_repair_sha256'):
        if info[key] != prior[key]:
            raise ValueError('Frozen comparison input changed')
    if digest(observations) != prior['observations_sha256'] or prior['test_evaluated'] is not False:
        raise ValueError('Frozen observation or test contract mismatch')
    for row in rows:
        row.update(stratum='background_random' if row['label']=='BACKGROUND' else row['label'],train_eligible=True,additional=False)
    extra = []
    review = reviewed_candidates(observations)
    for row in review:
        eligible = row['decision']=='weak_background'
        rows.append({'patch_id':row['id'],'wafer':row['wafer'],'stratum':'background_hard' if eligible else row['decision'],
                     'train_eligible':eligible,'additional':False})
        extra.append(checked_patch(observations.parent,row))
    import csv
    with (Path(cohort)/'samples.csv').open(encoding='utf-8-sig') as stream:
        known_hashes = {r['sha256'] for r in csv.DictReader(stream)}
    old_bg = json.loads(Path(background).read_text(encoding='utf-8'))
    known_hashes.update(r['sha256'] for r in old_bg['candidates'])
    known_hashes.update(r['sha256'] for r in review)
    additions = validate_added(json.loads(added.read_text(encoding='utf-8')), info,
                               [r['patch_id'] for r in rows],known_hashes)
    for row in additions:
        eligible = row['weak_training_eligible']
        rows.append({'patch_id':row['id'],'wafer':row['wafer'],'source_id':row['source_id'],
                     'stratum':'background_added' if eligible else 'added_uncertain', 'train_eligible':eligible,'additional':True})
        extra.append(checked_patch(added.parent,row))
    for row in rows:
        row['sample_group'] = 'background_random' if row['stratum']=='background_added' else row['stratum']
    images = torch.cat([images,torch.from_numpy(np.stack(extra).transpose(0,3,1,2).copy())])
    plan = {'schema':'center_background_ablation_v1','contract':CONTRACT,'arms':ARMS,'epochs_fixed':epochs,'seed':seed,
            'folds':WAFERS,'test_evaluated':False,'expert_ground_truth':False,'existing_type_model_changed':False,
            'background_threshold_fixed':.95,'defect_threshold_fixed':.8,'learning_rate':.0003,'weight_decay':.01,
            'sampling':'equal original stratum mass; additions share random background mass; same draws per fold in all arms',
            'control':'raw_base retrained with same seeded sampler as other arms; frozen old predictions also reported separately',
            'evaluation':'all arms use exactly identical held-out positives, old/new weak backgrounds and uncertain challenges',
            'gate':'same old/new weak strata >=70%; provided retention overall>=99%, each type>=98%, possible_defect>=98%; additionally >=2 hard-background wafers and whole-image validation',
            'observations_sha256':digest(observations),'added_sha256':digest(added),'frozen_plan_sha256':digest(frozen/'plan.json'),
            'limitations':['AI weak negatives are not expert truth','Development wafers reused across experiments; not final generalization estimate'],**info}
    save(output/'plan.json',plan)
    old_rows = json.loads((frozen/'oof.json').read_text(encoding='utf-8'))
    old_scores = {(r['wafer'],r['patch_id']):r['defect_probability'] for r in old_rows}
    results = {}
    for wafer in WAFERS:
        base_ids, held_ids = fold_ids(rows,wafer,False)
        held_rows = [rows[i] for i in held_ids]
        draws = min(6000,len(base_ids))
        reference = IndependentBackgroundCNN().cuda()
        checkpoint = frozen/f'wafer{wafer}_binary.pt'
        reference.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
        frozen_values = predict(reference,images[held_ids])
        discrepancy = max(abs(p-old_scores[wafer,r['patch_id']]) for r,p in zip(held_rows,frozen_values,strict=True) if not r['additional'])
        if discrepancy > 1e-5:
            raise ValueError('Frozen predictions failed reproduction')
        results[wafer] = {'frozen':metrics(held_rows,frozen_values),'frozen_checkpoint_sha256':digest(checkpoint),
                          'reproduction_max_abs_difference':discrepancy,'draws_per_epoch':draws,'arms':{}}
        save(output/f'wafer{wafer}_frozen_oof.json',[{**r,'defect_probability':p,'status':triage(p)} for r,p in zip(held_rows,frozen_values,strict=True)])
        del reference
        for arm in ARMS:
            print(f'BEGIN wafer={wafer} arm={arm}',flush=True)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            train_ids, _ = fold_ids(rows,wafer,arm.endswith('_added'))
            model = (FocusedBackgroundCNN() if arm.startswith('focused') else IndependentBackgroundCNN()).cuda()
            history = train_model(model,rows,images,train_ids,draws,epochs,seed)
            values = predict(model,images[held_ids])
            result = metrics(held_rows,values)
            results[wafer]['arms'][arm] = result
            prefix = output/f'wafer{wafer}_{arm}'
            torch.save(model.state_dict(),str(prefix)+'.pt')
            save(str(prefix)+'_history.json',history)
            save(str(prefix)+'_oof.json',[{**r,'defect_probability':p,'status':triage(p)} for r,p in zip(held_rows,values,strict=True)])
            save(str(prefix)+'_metrics.json',result)
            save(output/'progress.json',results)
            print(f'END wafer={wafer} arm={arm} passed={result["passed"]}',flush=True)
            del model
    hard_wafers = {r['wafer'] for r in rows if r['stratum']=='background_hard'}
    decision = {'schema':'center_background_ablation_result_v1','results':results,
                'all_wafer_numeric_pass':{a:all(results[w]['arms'][a]['passed'] for w in WAFERS) for a in ARMS},
                'cross_wafer_hard_background_support':len(hard_wafers)>=2,'deployment_status':'not_promoted',
                'whole_image_validated':False,'test_evaluated':False,'existing_model_changed':False,'plan_sha256':digest(output/'plan.json')}
    save(output/'decision.json',decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','background','baseline','observations','added','frozen','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args = parser.parse_args()
    run(**vars(args))


if __name__=='__main__':
    main()
