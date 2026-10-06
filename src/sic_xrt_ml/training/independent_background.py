"""Independent binary gate; existing three-class predictions are never changed."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .background_abstention import WAFERS, checked_patch, digest, load, save

CONTRACT = 'independent_background_v1_rgb128_uint8_defect_probability'
BACKGROUND_THRESHOLD = .95
DEFECT_THRESHOLD = .8


def reviewed_candidates(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 'xrt_detection_observations_v1' or data.get('test_evaluated') is not False:
        raise ValueError('Development AI observation contract required')
    rows = data['candidates']
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty unique observation IDs required')
    for row in rows:
        if (row['wafer'] not in WAFERS or row.get('human_verified') is not False
                or row.get('expert_ground_truth') is not False or row.get('review_actor') != 'Codex_AI'
                or row.get('decision') not in ('weak_background', 'possible_defect', 'uncertain')
                or row.get('training_eligible') is not (row['decision'] == 'weak_background')):
            raise ValueError('Invalid wafer or weak observation provenance')
    return rows


class IndependentBackgroundCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def branch():
            return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten())
        self.center, self.context = branch(), branch()
        self.head = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 1))

    def forward(self, image):
        gray = image.mean(1, keepdim=True)
        residual = gray-F.avg_pool2d(F.pad(gray, (7, 7, 7, 7), mode='reflect'), 15, 1)
        rms = residual.square().mean((2, 3), keepdim=True).sqrt().clamp_min(1/255)
        # Preserve raw contrast amplitude alongside the normalized view.
        features = torch.cat([gray, (.5+.15*residual/rms).clamp(0, 1), (.5+4*residual).clamp(0, 1)], 1)
        return self.head(torch.cat([self.center(features[:, :, 32:96, 32:96]),
                                    self.context(F.avg_pool2d(features, 2))], 1)).squeeze(1)


def triage(defect_probability):
    value = float(defect_probability)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Finite probability in [0,1] required')
    if 1-value >= BACKGROUND_THRESHOLD:
        return 'background_suspect'
    if value >= DEFECT_THRESHOLD:
        return 'defect_candidate'
    return 'uncertain'


def metrics(rows, scores):
    if len(rows) != len(scores):
        raise ValueError('Row/score length mismatch')
    statuses = [triage(s) for s in scores]
    result = {}
    for group in sorted({r['stratum'] for r in rows}):
        selected = [statuses[i] for i, r in enumerate(rows) if r['stratum'] == group]
        count = len(selected)
        result[group] = {'count': count, 'background_suspect': selected.count('background_suspect'),
                         'uncertain': selected.count('uncertain'), 'defect_candidate': selected.count('defect_candidate'),
                         'background_rate': selected.count('background_suspect')/count}
    positives = [r for k, r in result.items() if k in ('BPD', 'TED', 'TSD')]
    total = sum(r['count'] for r in positives)
    overall = sum(r['background_suspect'] for r in positives)/total if total else None
    supported_weak = [r for k, r in result.items() if k.startswith('background_')]
    checks = {'positive_retention_overall': overall is not None and overall <= .01,
              'positive_retention_each_type': bool(positives) and all(r['background_rate'] <= .02 for r in positives),
              'weak_background_rejection': bool(supported_weak) and all(r['background_rate'] >= .7 for r in supported_weak),
              'possible_defect_retention': result.get('possible_defect', {}).get('background_rate', 0) <= .02}
    return {'by_stratum': result, 'positive_rejection_rate': overall, 'checks': checks, 'passed': all(checks.values())}


def run(cohort, background, baseline, observations, output, epochs=16, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for this fixed workstation experiment')
    torch.set_num_threads(4)
    rows, images, _, info = load(cohort, background, baseline)
    for row in rows:
        row['stratum'] = 'background_random' if row['label'] == 'BACKGROUND' else row['label']
    observations = Path(observations)
    review = reviewed_candidates(observations)
    additions, challenge_rows, challenge_images = [], [], []
    for row in review:
        item = {'patch_id': row['id'], 'wafer': row['wafer'], 'label': 'BACKGROUND',
                'candidate_id': row['candidate_id'], 'stratum': 'background_hard'}
        patch = checked_patch(observations.parent, row)
        if row['decision'] == 'weak_background':
            rows.append(item)
            additions.append(patch)
        else:
            challenge_rows.append({**item, 'label': row['decision'], 'stratum': row['decision']})
            challenge_images.append(patch)
    if not additions:
        raise ValueError('No reviewed hard-background candidates; do not invent negatives')
    images = torch.cat([images, torch.from_numpy(np.stack(additions).transpose(0, 3, 1, 2).copy())])
    challenge_images = torch.from_numpy(np.stack(challenge_images).transpose(0, 3, 1, 2).copy())
    hard_wafers = sorted({r['wafer'] for r in rows if r['stratum'] == 'background_hard'})
    plan = {'schema': 'independent_background_plan_v1', 'contract': CONTRACT, 'epochs_fixed': epochs,
            'seed': seed, 'folds': WAFERS, 'background_threshold_fixed': BACKGROUND_THRESHOLD,
            'defect_threshold_fixed': DEFECT_THRESHOLD, 'existing_type_predictions': 'unchanged',
            'learning_rate': .0003, 'weight_decay': .01, 'sampler': 'equal stratum probability; cap6000 per epoch',
            'hard_background_wafers': hard_wafers, 'observations_sha256': digest(observations),
            'test_evaluated': False, 'expert_truth': False, 'training_counts': dict(Counter(r['stratum'] for r in rows)),
            'gate': 'each wafer: retain >=99% provided points overall and >=98% each type; reject >=70% each supported weak background stratum; retain >=98% possible defects; deployment additionally needs reviewed hard backgrounds on >=2 wafers and whole-image preservation', **info}
    save(output/'plan.json', plan)
    targets = torch.tensor([0 if r['label'] == 'BACKGROUND' else 1 for r in rows], dtype=torch.float32)
    results, predictions = {}, []
    for wafer in WAFERS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        train_ids = [i for i, r in enumerate(rows) if r['wafer'] != wafer]
        valid_ids = [i for i, r in enumerate(rows) if r['wafer'] == wafer]
        groups = Counter(rows[i]['stratum'] for i in train_ids)
        weights = [1/groups[rows[i]['stratum']] for i in train_ids]
        loader = DataLoader(TensorDataset(images[train_ids], targets[train_ids]), batch_size=32,
                            sampler=WeightedRandomSampler(weights, min(6000, len(train_ids)), replacement=True))
        model = IndependentBackgroundCNN().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
        history = []
        for epoch in range(epochs):
            model.train()
            total, loss_sum = 0, 0.
            for inputs, labels in loader:
                inputs, labels = inputs.cuda().float()/255, labels.cuda()
                inputs = torch.rot90(inputs, random.randrange(4), (2, 3))
                if random.random() < .5:
                    inputs = inputs.flip(3)
                inputs = (inputs*random.uniform(.8, 1.2)+random.uniform(-.04, .04)).clamp(0, 1)
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(inputs), labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()*len(labels)
                total += len(labels)
            history.append({'epoch': epoch+1, 'loss': loss_sum/total})
            print(f'held-out {wafer} epoch {epoch+1}/{epochs}: {loss_sum/total:.4f}', flush=True)
        torch.save(model.state_dict(), output/f'wafer{wafer}_binary.pt')
        model.eval()
        def predict(tensors, model=model):
            values = []
            with torch.inference_mode():
                for batch, in DataLoader(TensorDataset(tensors), batch_size=64):
                    values.extend(model(batch.cuda().float()/255).sigmoid().cpu().tolist())
            return values
        held_rows = [rows[i] for i in valid_ids]
        scores = predict(images[valid_ids])
        challenge_ids = [i for i, r in enumerate(challenge_rows) if r['wafer'] == wafer]
        held_rows += [challenge_rows[i] for i in challenge_ids]
        scores += predict(challenge_images[challenge_ids])
        results[wafer] = metrics(held_rows, scores)
        for row, probability in zip(held_rows, scores, strict=True):
            predictions.append({**row, 'defect_probability': probability, 'status': triage(probability),
                                'held_out_wafer': wafer})
        save(output/f'wafer{wafer}_history.json', history)
        save(output/f'wafer{wafer}_metrics.json', results[wafer])
        print(f'wafer {wafer}: {results[wafer]}', flush=True)
    result = {'schema': 'independent_background_result_v1', 'folds': results,
              'numeric_screen_passed': all(r['passed'] for r in results.values()),
              'hard_background_cross_wafer_support': len(hard_wafers) >= 2,
              'deployment_status': 'not_promoted', 'existing_type_model_changed': False, 'test_evaluated': False,
              'reason': 'Research screening only; requires all metrics, cross-wafer hard background support, and whole-image checks',
              'plan_sha256': digest(output/'plan.json')}
    save(output/'oof.json', predictions)
    save(output/'decision.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'background', 'baseline', 'observations', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.cohort, args.background, args.baseline, args.observations, args.output)


if __name__ == '__main__':
    main()
