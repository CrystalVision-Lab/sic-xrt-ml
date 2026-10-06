"""Offline, wafer-separated review training. Never updates a desktop model.

Presence, type and coordinate correction have separate targets and losses.
These are candidate-patch metrics, not whole-wafer detection accuracy.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

CLASSES = ('BPD','TED','TSD')
WAFERS = ('1','2','3','4','5','6','7','9')
CONTRACT = 'review_multitask_v1_rgb128_raw_offset32'


def digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def checked_path(root,relative,sha):
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or digest(path) != sha:
        raise ValueError('Package path/hash mismatch')
    return path


def load_feedback(root):
    root=Path(root).resolve()
    info=json.loads((root/'dataset.json').read_text(encoding='utf-8'))
    if (info.get('schema') != 'xrt_feedback_dataset_v1' or info.get('input_schema') != 'xrt_feedback_v1'
            or info.get('patch_contract') != 'rgb128_uint8_raw_pixel_xy' or info.get('max_offset_px') != 32
            or info.get('classes') != list(CLASSES) or info.get('test_evaluated') is not False
            or info.get('expert_ground_truth') is not False or info.get('exhaustive_annotations') is not False
            or info.get('automatic_model_promotion') is not False):
        raise ValueError('Unsupported feedback dataset contract')
    samples=json.loads(checked_path(root,info['samples_path'],info['samples_sha256']).read_text(encoding='utf-8'))
    records=json.loads(checked_path(root,info['records_path'],info['records_sha256']).read_text(encoding='utf-8'))
    by_record={r['id']:r for r in records}
    if len(by_record) != len(records):
        raise ValueError('Duplicate records')
    sources={s['id']:s for s in info['sources']}
    if len(sources) != len(info['sources']) or any(s['wafer'] not in WAFERS for s in sources.values()):
        raise ValueError('Invalid source/holdout wafer')
    for source in sources.values():
        metadata=source['source']
        if (metadata.get('coordinate_space') != 'raw_pixel_xy' or metadata.get('orientation') != 'encoded_no_exif_rotation'
                or source['id'] != metadata['sha256']+':'+str(metadata['page_index'])
                or type(metadata.get('page_index')) is not int or metadata['page_index']<0
                or any(type(metadata.get(k)) is not int or metadata[k]<=0 for k in ('width','height'))):
            raise ValueError('Invalid encoded source coordinates')
        feedback=json.loads(checked_path(root,source['review_path'],source['review_sha256']).read_text(encoding='utf-8'))
        if (feedback['schema'] != 'xrt_feedback_v1' or feedback['source'] != source['source']
                or feedback['wafer_id'] != source['wafer'] or feedback['expert_ground_truth'] is not False):
            raise ValueError('Review provenance mismatch')
        latest={e['record_id']:e for e in feedback['events']}
        for record in records:
            if record['source_id'] == source['id']:
                event=record['event']
                if (event != latest.get(event['record_id']) or event['actor'] != 'human' or not event['reviewer'].strip()
                        or record['id'] != feedback['session_id']+':'+event['record_id']
                        or record['wafer'] != source['wafer'] or record['review_sha256'] != source['review_sha256']):
                    raise ValueError('Latest human event required')
    if any(r['source_id'] not in sources or r['wafer'] != sources[r['source_id']]['wafer'] for r in records):
        raise ValueError('Orphan record or holdout wafer')
    ids, images=[],[]
    for row in samples:
        if row['id'] in ids:
            raise ValueError('Duplicate sample IDs')
        ids.append(row['id'])
        record=by_record[row['record_id']]
        event=record['event']; target=event['target']; source=sources[row['source_id']]
        if (row['wafer'] not in WAFERS or row['wafer'] != record['wafer'] or row['wafer'] != source['wafer']
                or row['source_id'] != record['source_id'] or row['review_sha256'] != record['review_sha256']
                or row.get('label_basis') != 'human_review' or row.get('expert_ground_truth') is not False
                or target['objectness'] is None or target['duplicate_of']):
            raise ValueError('Invalid sample provenance or excluded decision')
        if (row['objectness'] != int(target['objectness']) or row['objectness_mask'] is not True
                or row['type'] != target['label'] or row['type_mask'] is not (target['objectness'] is True and target['label'] in CLASSES)
                or row['location_mask'] is not (target['objectness'] is True and target['location_confirmed'])):
            raise ValueError('Mask/target mismatch')
        if row['role'] not in ('current','original'):
            raise ValueError('Unknown crop role')
        xy=target if row['role']=='current' else event['original']
        crop=[round(xy['x'])-64,round(xy['y'])-64]
        anchor=[crop[0]+64.,crop[1]+64.]
        offset=[target['x']-anchor[0],target['y']-anchor[1]] if row['location_mask'] else [0.,0.]
        if row['crop_xy'] != crop or row['anchor_xy'] != anchor or row['offset_xy'] != offset or not np.isfinite(offset).all() or max(map(abs,offset))>32:
            raise ValueError('Coordinate correction mismatch')
        if crop[0]<0 or crop[1]<0 or crop[0]+128>source['source']['width'] or crop[1]+128>source['source']['height']:
            raise ValueError('Incomplete source patch')
        pixels=tifffile.imread(checked_path(root,row['path'],row['sha256']))
        if pixels.shape != (128,128,3) or pixels.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB128')
        images.append(pixels)
        row.update(origin='human',event=event,patch_sha256=row['sha256'],input_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    if len(samples) != info['sample_count'] or info['masks'] != {k:sum(r[k] for r in samples) for k in ('objectness_mask','type_mask','location_mask')}:
        raise ValueError('Dataset counts mismatch')
    if not samples:
        raise ValueError('No usable feedback samples; review some defects/background first')
    return samples,images,{'dataset_sha256':digest(root/'dataset.json'),'samples_sha256':info['samples_sha256']}


def load_retained(cohort):
    """Retain original provider/derived labels without claiming human review."""
    root=Path(cohort).resolve()
    info=json.loads((root/'cohort.json').read_text(encoding='utf-8'))
    if info.get('schema') != 'wafer_grouped_development_cohort' or info.get('schema_version') != 1:
        raise ValueError('Existing development cohort contract required')
    if info.get('expert_ground_truth') is not False or info.get('test_evaluated') is not False:
        raise ValueError('Development research cohort only')
    manifest=checked_path(root,'samples.csv',info['manifest_sha256'])
    hashes=json.loads((root/'output_hashes.json').read_text(encoding='utf-8'))
    sources_path=checked_path(root,'sources.json',hashes['sources.json'])
    source_rows=json.loads(sources_path.read_text(encoding='utf-8'))
    sources={s['source_id']:s for s in source_rows}
    if len(sources) != len(source_rows) or any(s['wafer'] not in WAFERS for s in source_rows):
        raise ValueError('Invalid retained source identities')
    with manifest.open(encoding='utf-8-sig',newline='') as stream:
        old=list(csv.DictReader(stream))
    rows,images=[],[]
    ids=set()
    for sample in old:
        if sample['wafer'] not in WAFERS or sample['label'] not in CLASSES or sample['patch_id'] in ids:
            raise ValueError('Invalid retained labels or holdout wafer')
        ids.add(sample['patch_id'])
        image=tifffile.imread(checked_path(root,sample['path'],sample['sha256']))
        if image.shape != (128,128,3) or image.dtype != np.uint8:
            raise ValueError('Retained patch must be uint8 RGB128')
        source=sources[sample['source_id']]
        if source['wafer'] != sample['wafer']:
            raise ValueError('Retained source wafer mismatch')
        rows.append({'id':'retained:'+sample['patch_id'],'record_id':'retained:'+sample['patch_id'],
            'wafer':sample['wafer'],'source_id':source['sha256']+':0','patch_sha256':sample['sha256'],
            'source_xy':[float(sample['x']),float(sample['y'])],'input_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
            'type':sample['label'],'type_mask':True,'objectness':1,'objectness_mask':True,
            'offset_xy':[0.,0.],'location_mask':False,'origin':'retained','role':'current',
            'label_basis':'existing_cohort_reference','expert_ground_truth':False})
        images.append(image)
    if not rows:
        raise ValueError('Retained cohort must not be empty')
    return rows,images,{'retained_manifest_sha256':info['manifest_sha256'],'retained_cohort_sha256':digest(root/'cohort.json'),
                       'retained_sources_sha256':digest(sources_path)}


def split_rows(rows,validation_wafer):
    if validation_wafer not in WAFERS or any(r['wafer'] not in WAFERS for r in rows):
        raise ValueError('Wafer 8 is reserved; valid development wafer required')
    seen={}
    for row in rows:
        for key in ('source_id','patch_sha256','input_sha256'):
            value=(key,row[key])
            if value in seen and seen[value] != row['wafer']:
                raise ValueError('Same physical source/patch assigned to different wafers')
            seen[value]=row['wafer']
    train=[i for i,r in enumerate(rows) if r['wafer'] != validation_wafer]
    validation=[i for i,r in enumerate(rows) if r['wafer']==validation_wafer]
    if not train or not validation:
        raise ValueError('Nonempty wafer-separated train/validation required')
    return train,validation


def primary(rows,ids):
    # One reviewed record counts once. Prefer the original candidate input so
    # coordinate correction is compared with the current candidate location.
    chosen={}
    for i in ids:
        r=rows[i]
        if r['record_id'] not in chosen or r['role']=='original':
            chosen[r['record_id']]=i
    return list(chosen.values())


def readiness(rows,train,validation):
    human_train=primary(rows,[i for i in train if rows[i]['origin']=='human'])
    human_val=primary(rows,[i for i in validation if rows[i]['origin']=='human'])
    def counts(ids):
        return dict(Counter(str(rows[i]['objectness']) for i in ids))
    tr,va=counts(human_train),counts(human_val)
    reasons=[]
    if min(tr.get('0',0),tr.get('1',0))<10:
        reasons.append('학습 웨이퍼에서 결함 10개·배경 10개 이상을 직접 확인하세요.')
    if min(va.get('0',0),va.get('1',0))<5:
        reasons.append('검증 웨이퍼에서 결함 5개·배경 5개 이상을 직접 확인하세요.')
    if not any(rows[i]['origin']=='retained' for i in train):
        reasons.append('기존 개발 자료를 지정해야 이전 종류 자료도 함께 학습합니다.')
    type_counts=Counter(rows[i]['type'] for i in primary(rows,train) if rows[i]['type_mask'])
    location_train=sum(rows[i]['location_mask'] for i in human_train)
    location_val=sum(rows[i]['location_mask'] for i in human_val)
    return {'can_train':not reasons,'reasons':reasons,'human_train_presence':tr,'human_validation_presence':va,
            'type_train_counts':dict(type_counts),'train_type_head':all(type_counts[k]>=5 for k in CLASSES),
            'train_location_head':location_train>=10,'location_train_count':location_train,'location_validation_count':location_val,
            'human_train_records':len(human_train),'human_validation_records':len(human_val)}


class ReviewCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def branch():
            return nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                 nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                 nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4),nn.Flatten())
        self.center,self.context=branch(),branch()
        self.shared=nn.Sequential(nn.Linear(2048,128),nn.ReLU(),nn.Dropout(.2))
        self.presence,self.kind,self.location=nn.Linear(128,1),nn.Linear(128,3),nn.Linear(128,2)

    def forward(self,images):
        gray=images.mean(1,keepdim=True)
        residual=gray-F.avg_pool2d(F.pad(gray,(7,7,7,7),mode='reflect'),15,1)
        scale=residual.square().mean((2,3),keepdim=True).sqrt().clamp_min(1/255)
        view=torch.cat([gray,(.5+.15*residual/scale).clamp(0,1),(.5+4*residual).clamp(0,1)],1)
        features=self.shared(torch.cat([self.center(view[:,:,32:96,32:96]),self.context(F.avg_pool2d(view,2))],1))
        return self.presence(features).squeeze(1),self.kind(features),self.location(features).tanh()


def masked_loss(outputs,truth,type_enabled=True,location_enabled=True):
    presence,kind,location=outputs
    # truth = objectness, type_id (placeholder 0 when unknown), dx/32,dy/32,
    #         objectness_mask,type_mask,location_mask
    zero=presence.sum()*0
    def masked(values,mask):
        return values[mask].mean() if mask.any() else zero
    a=masked(F.binary_cross_entropy_with_logits(presence,truth[:,0],reduction='none'),truth[:,4].bool())
    b=masked(F.cross_entropy(kind,truth[:,1].long(),reduction='none'),truth[:,5].bool()) if type_enabled else zero
    c=masked(F.smooth_l1_loss(location,truth[:,2:4],reduction='none').mean(1),truth[:,6].bool()) if location_enabled else zero
    return a+b+c,{'presence':a,'type':b,'location':c}


def sampling_weights(rows,train):
    # Half old data, half reviewed data; each stratum receives equal mass.
    groups=[(rows[i]['origin'],str(rows[i]['objectness']) if rows[i]['origin']=='human' else rows[i]['type']) for i in train]
    counts=Counter(groups)
    strata=Counter(g[0] for g in counts)
    return torch.tensor([1/(2*strata[g[0]]*counts[g]) for g in groups],dtype=torch.double)


def evaluate(rows,ids,outputs,type_enabled,location_enabled):
    presence,kind,location=outputs
    probs=presence.sigmoid().numpy(); labels=kind.argmax(1).numpy(); offsets=location.numpy()*32
    # Inputs here are aligned to ids and already one record each.
    target=np.array([rows[i]['objectness'] for i in ids],dtype=bool)
    prediction=probs>=.5
    tp=int((target&prediction).sum()); fn=int((target&~prediction).sum()); fp=int((~target&prediction).sum()); tn=int((~target&~prediction).sum())
    presence_metrics={'count':len(ids),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'defect_recall':tp/(tp+fn) if tp+fn else None,
                      'background_rejection':tn/(tn+fp) if tn+fp else None,'baseline_always_defect_recall':1. if target.any() else None,
                      'baseline_always_defect_background_rejection':0. if (~target).any() else None}
    matrix=np.zeros((3,3),dtype=int); old_matrix=np.zeros((3,3),dtype=int); paired=np.zeros((3,3),dtype=int)
    corrected=[]; original=[]
    for j,i in enumerate(ids):
        r=rows[i]
        if type_enabled and r['type_mask']:
            y=CLASSES.index(r['type']); matrix[y,labels[j]]+=1
            old=r.get('event',{}).get('original',{}).get('predicted_type')
            if old in CLASSES:
                old_matrix[y,CLASSES.index(old)]+=1; paired[y,labels[j]]+=1
        if location_enabled and r['location_mask']:
            expected=np.array(r['offset_xy']); corrected.append(float(np.linalg.norm(offsets[j]-expected)))
            original.append(float(np.linalg.norm(expected)))
    def type_metrics(m):
        f1=[2*m[k,k]/(m[k].sum()+m[:,k].sum()) for k in range(3) if m[k].sum()]
        return {'confusion':m.tolist(),'count':int(m.sum()),'support':dict(zip(CLASSES,map(int,m.sum(1)),strict=True)),
                'macro_f1_supported_classes':float(np.mean(f1)) if f1 else None}
    return {'presence':presence_metrics,'type':type_metrics(matrix) if type_enabled else None,
            'paired_type_baseline':type_metrics(old_matrix) if type_enabled else None,'paired_type_new':type_metrics(paired) if type_enabled else None,
            'location':{'count':len(corrected),'mean_error_px':float(np.mean(corrected)) if corrected else None,
                        'median_error_px':float(np.median(corrected)) if corrected else None,
                        'original_mean_error_px':float(np.mean(original)) if original else None} if location_enabled else None,
            'scope':'reviewed_candidate_patches_not_whole_image_recall','expert_ground_truth':False}


def inspect(dataset,retained,validation_wafer):
    rows,images,info=load_feedback(dataset)
    old,old_images,old_info=load_retained(retained)
    rows+=old; images+=old_images; info.update(old_info)
    train,validation=split_rows(rows,validation_wafer)
    # Original byte hash and encoded frame must match before source coordinates
    # can override old labels. Never infer JPEG/TIFF reflection correspondence.
    human=[rows[i] for i in train if rows[i]['origin']=='human']
    def conflicts(r):
        if r['origin'] != 'retained':
            return False
        for reviewed in human:
            if r['input_sha256'] == reviewed['input_sha256']:
                return True
            if r['source_id'] == reviewed['source_id']:
                for xy in (reviewed['event']['original'],reviewed['event']['target']):
                    if max(abs(r['source_xy'][0]-xy['x']),abs(r['source_xy'][1]-xy['y']))<=32:
                        return True
        return False
    excluded=[i for i in train if conflicts(rows[i])]
    train=[i for i in train if i not in set(excluded)]
    info['retained_training_overlap_excluded']=len(excluded)
    return rows,images,info,train,validation,readiness(rows,train,validation)


def run(dataset,retained,validation_wafer,output,epochs=15,seed=42,device='cpu'):
    if type(epochs) is not int or epochs<1 or device not in ('cpu','cuda'):
        raise ValueError('Positive fixed epochs and cpu/cuda required')
    rows,images,info,train,validation,ready=inspect(dataset,retained,validation_wafer)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    plan={'schema':'xrt_feedback_training_v1','contract':CONTRACT,**info,'validation_wafer':validation_wafer,
          'epochs_fixed':epochs,'seed':seed,'readiness':ready,'test_evaluated':False,'automatic_model_promotion':False,
          'expert_ground_truth':False,'full_image_detector':False,'threshold_fixed':.5,'old_data_sampling_mass':.5}
    save(output/'plan.json',plan)
    if not ready['can_train']:
        report={**plan,'status':'needs_review_data','trained':False,'reasons':ready['reasons']}
        save(output/'report.json',report)
        return report
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model=ReviewCNN().to(device)
    tensors=torch.from_numpy(np.stack(images).transpose(0,3,1,2).copy())
    truth=torch.tensor([[r['objectness'],CLASSES.index(r['type']) if r['type_mask'] else 0,
                         r['offset_xy'][0]/32,r['offset_xy'][1]/32,r['objectness_mask'],r['type_mask'],r['location_mask']] for r in rows],dtype=torch.float32)
    weights=sampling_weights(rows,train)
    draws=min(4096,max(256,len(train)))
    loader=DataLoader(TensorDataset(tensors[train],truth[train]),batch_size=32,
                      sampler=WeightedRandomSampler(weights,draws,replacement=True,generator=torch.Generator().manual_seed(seed)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    history=[]
    for epoch in range(epochs):
        model.train(); losses=[]
        for pixels,targets in loader:
            pixels=pixels.to(device).float()/255; targets=targets.to(device)
            # No geometric augmentation: subtype orientation and delta targets
            # must remain in the encoded image coordinate frame.
            pixels=(pixels*random.uniform(.9,1.1)+random.uniform(-.02,.02)).clamp(0,1)
            optimizer.zero_grad(set_to_none=True)
            loss,_=masked_loss(model(pixels),targets,ready['train_type_head'],ready['train_location_head'])
            loss.backward(); optimizer.step(); losses.append(loss.item())
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses))})
        print(f"검수 학습 {epoch+1}/{epochs} · loss {history[-1]['train_loss']:.4f}",flush=True)
    def predict(ids):
        model.eval(); chunks=[[],[],[]]
        with torch.inference_mode():
            for pixels, in DataLoader(TensorDataset(tensors[ids]),batch_size=64):
                for index,values in enumerate(model(pixels.to(device).float()/255)):
                    chunks[index].append(values.cpu())
        return tuple(torch.cat(values) for values in chunks)
    human_val=primary(rows,[i for i in validation if rows[i]['origin']=='human'])
    reference_val=primary(rows,[i for i in validation if rows[i]['origin']=='retained'])
    metrics=evaluate(rows,human_val,predict(human_val),ready['train_type_head'],ready['train_location_head'])
    reference_metrics=evaluate(rows,reference_val,predict(reference_val),ready['train_type_head'],False) if reference_val else None
    # Conservative readiness is not automatic promotion approval. Support and
    # whole-image detection still require additional independent evaluation.
    support=metrics['type']['support'] if metrics['type'] else {}
    checks={'human_presence_support':min(ready['human_validation_presence'].values())>=20,
            'defect_recall':metrics['presence']['defect_recall'] is not None and metrics['presence']['defect_recall']>=.95,
            'background_rejection':metrics['presence']['background_rejection'] is not None and metrics['presence']['background_rejection']>=.7,
            'type_support':all(support.get(k,0)>=10 for k in CLASSES),
            'type_non_decrease':bool(metrics['paired_type_baseline'] and metrics['paired_type_baseline']['count']>=30 and
                metrics['paired_type_new']['macro_f1_supported_classes']>=metrics['paired_type_baseline']['macro_f1_supported_classes']),
            'coordinate_support':bool(metrics['location'] and metrics['location']['count']>=20),
            'coordinate_improvement':bool(metrics['location'] and metrics['location']['mean_error_px'] is not None and
                metrics['location']['mean_error_px']<metrics['location']['original_mean_error_px']),
            'retained_reference_evaluated':bool(reference_metrics and reference_metrics['type']),
            'retained_reference_non_decrease':False,
            'whole_image_recall_verified':False}
    checkpoint=output/'research_model.pt'
    torch.save({'contract':CONTRACT,'state_dict':model.cpu().state_dict(),'classes':CLASSES,'max_offset_px':32,
                'trained_heads':{'presence':True,'type':ready['train_type_head'],'location':ready['train_location_head']},'plan':plan},checkpoint)
    report={**plan,'status':'completed_research','trained':True,'history':history,'human_validation':metrics,
            'retained_validation_reference_agreement':reference_metrics,'comparison_checks':checks,
            'candidate_patch_checks_passed':all(checks.values()),'model_sha256':digest(checkpoint),
            'deployed':False,'deployment_ready':False,'train_samples_used':len(train),'draws_per_epoch':draws,
            'model_versions_under_review':sorted({r['event']['original'].get('model_sha256') for r in rows if r['origin']=='human' and r['event']['original'].get('model_sha256')})}
    save(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True); parser.add_argument('--retained',required=True)
    parser.add_argument('--validation-wafer',required=True,choices=WAFERS)
    parser.add_argument('--output'); parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu'); parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    if args.inspect:
        result=inspect(args.dataset,args.retained,args.validation_wafer)[-1]
    elif not args.output:
        parser.error('--output required for training')
    else:
        result=run(args.dataset,args.retained,args.validation_wafer,args.output,args.epochs,device=args.device)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
