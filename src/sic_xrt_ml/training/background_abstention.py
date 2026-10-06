"""Development-only four-class experiment; weak background is not ground truth.

No model is exported or promoted by this runner. Wafer 8 is forbidden.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

CLASSES = ('BPD', 'TED', 'TSD', 'BACKGROUND')
WAFERS = ('1', '2', '9')
CONTRACT = 'background_context_cnn_v1_rgb128_uint8_gray_local_box15_center64_context128'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def checked_patch(root, row):
    root = Path(root).resolve()
    path = (root/row['path']).resolve()
    if not path.is_relative_to(root) or digest(path) != row['sha256']:
        raise ValueError('Patch path/hash mismatch')
    image = tifffile.imread(path)
    if image.shape != (128, 128, 3) or image.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB128')
    return image


def validate_rows(positive, weak, baseline):
    if not positive or {r['wafer'] for r in positive} != set(WAFERS):
        raise ValueError('Only complete development wafers 1/2/9 allowed')
    ids = [r['patch_id'] for r in positive]
    if len(set(ids)) != len(ids) or any(r['label'] not in CLASSES[:3] for r in positive):
        raise ValueError('Invalid positive IDs/classes')
    if any(r['wafer'] not in WAFERS for r in weak):
        raise ValueError('Holdout wafer forbidden even among excluded candidates')
    if len({r['id'] for r in weak}) != len(weak):
        raise ValueError('Duplicate background IDs')
    selected = []
    for row in weak:
        if (row.get('review_actor') != 'Codex_AI' or row.get('human_verified') is not False
                or row.get('expert_ground_truth') is not False):
            raise ValueError('Explicit weak AI provenance required')
        eligible = row.get('weak_training_eligible')
        if eligible is not (row['decision'] == 'background_candidate'):
            raise ValueError('Contradictory weak eligibility')
        if eligible:
            selected.append(row)
    if {r['wafer'] for r in selected} != set(WAFERS):
        raise ValueError('Weak background required in each development wafer')
    by_id = {r['patch_id']: r for r in baseline}
    if len(by_id) != len(baseline) or set(by_id) != set(ids):
        raise ValueError('Baseline must match exact positive cohort')
    for row in positive:
        old = by_id[row['patch_id']]
        if (old['wafer'] != row['wafer'] or old['label'] != row['label']
                or old['prediction'] not in CLASSES[:3]):
            raise ValueError('Baseline wafer/label mismatch')
    return selected, by_id


def load(cohort, background, baseline_path):
    cohort, background = Path(cohort), Path(background)
    info = json.loads((cohort/'cohort.json').read_text(encoding='utf-8'))
    weak = json.loads(background.read_text(encoding='utf-8'))
    actual_hash = digest(cohort/'samples.csv')
    if (actual_hash != info['manifest_sha256'] or actual_hash != weak['cohort_manifest_sha256']
            or weak['schema'] != 'xrt_weak_background_v1' or weak.get('test_evaluated') is not False):
        raise ValueError('Cohort/background contract mismatch')
    with (cohort/'samples.csv').open(encoding='utf-8-sig', newline='') as stream:
        positive = list(csv.DictReader(stream))
    baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    selected, by_id = validate_rows(positive, weak['candidates'], baseline)
    rows, images = [], []
    for row in positive:
        images.append(checked_patch(cohort, row))
        rows.append({k: row[k] for k in ('patch_id', 'point_id', 'wafer', 'label', 'source_id')})
    for row in selected:
        images.append(checked_patch(background.parent, row))
        rows.append({'patch_id': row['id'], 'wafer': row['wafer'], 'label': 'BACKGROUND',
                     'source_id': row['source_id'], 'label_basis': 'weak_ai_background'})
    tensors = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy())
    metadata = {'cohort_manifest_sha256': actual_hash, 'background_sha256': digest(background),
                'baseline_oof_sha256': digest(baseline_path), 'coordinate_repair_sha256': weak['coordinate_repair_sha256'],
                'counts': {w: dict(Counter(r['label'] for r in rows if r['wafer'] == w)) for w in WAFERS}}
    return rows, tensors, by_id, metadata


class BackgroundContextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def branch():
            return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten())
        self.center, self.context = branch(), branch()
        self.head = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 4))

    def forward(self, images):
        gray = images.mean(dim=1, keepdim=True)
        residual = gray - F.avg_pool2d(F.pad(gray, (7, 7, 7, 7), mode='reflect'), 15, 1)
        rms = residual.square().mean((2, 3), keepdim=True).sqrt().clamp_min(1/255)
        normalized = (.5 + .15*residual/rms).clamp(0, 1)
        return self.head(torch.cat([self.center(normalized[:, :, 32:96, 32:96]),
                                    self.context(F.avg_pool2d(normalized, 2))], dim=1))


def type_metrics(target, prediction):
    matrix = np.zeros((3, 3), dtype=np.int64)
    for y, p in zip(target, prediction, strict=True):
        matrix[y, p] += 1
    f1 = []
    per_class = {}
    for i, name in enumerate(CLASSES[:3]):
        tp, support, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        score = 2*tp/(support+predicted) if support+predicted else 0.0
        per_class[name] = {'support': support, 'tp': tp, 'fp': predicted-tp,
                           'recall': tp/support if support else None, 'f1': score if support else None}
        if support:
            f1.append(score)
    return {'macro_f1_supported_classes': float(np.mean(f1)), 'confusion': matrix.tolist(),
            'per_class': per_class, 'agreement': float(np.trace(matrix)/matrix.sum())}


def evaluate(rows, probabilities, baseline, background_threshold=.95, type_threshold=.8):
    probabilities = np.asarray(probabilities)
    if (probabilities.shape != (len(rows), 4) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
        raise ValueError('Invalid four-class probabilities')
    positive = np.array([r['label'] != 'BACKGROUND' for r in rows])
    true = np.array([CLASSES.index(r['label']) for r in rows])
    reject = probabilities[:, 3] >= background_threshold
    # Conditional type score is deliberately reported separately from objectness.
    conditional = probabilities[:, :3]/probabilities[:, :3].sum(1, keepdims=True).clip(1e-12)
    prediction = conditional.argmax(1)
    candidate = type_metrics(true[positive], prediction[positive])
    old = type_metrics(true[positive], [CLASSES.index(baseline[r['patch_id']]['prediction']) for r in rows if r['label'] != 'BACKGROUND'])
    rejected_by_type = {}
    for i, name in enumerate(CLASSES[:3]):
        mask = true == i
        rejected_by_type[name] = float(reject[mask].mean()) if mask.any() else None
    # A confident conditional type can still be background. Require low background probability too.
    accepted = positive & (probabilities[:, 3] <= .2) & (conditional.max(1) >= type_threshold)
    abstention = {'type_coverage': float(accepted.sum()/positive.sum()),
                  'accepted_count': int(accepted.sum()),
                  'accepted_provider_agreement': float((prediction[accepted] == true[accepted]).mean()) if accepted.any() else None}
    weak_recall = float(reject[~positive].mean())
    positive_reject = float(reject[positive].mean())
    checks = {'positive_retention_overall': positive_reject <= .01,
              'positive_retention_each_type': all(v is None or v <= .02 for v in rejected_by_type.values()),
              'weak_background_rejection': weak_recall >= .70,
              'type_macro_f1_non_decrease': candidate['macro_f1_supported_classes'] >= old['macro_f1_supported_classes'],
              'bpd_tp_non_decrease': candidate['per_class']['BPD']['tp'] >= old['per_class']['BPD']['tp'],
              'bpd_fp_non_increase': candidate['per_class']['BPD']['fp'] <= old['per_class']['BPD']['fp']}
    return {'positive_count': int(positive.sum()), 'weak_background_count': int((~positive).sum()),
            'positive_rejection_rate': positive_reject, 'positive_rejection_by_type': rejected_by_type,
            'weak_background_rejection_rate': weak_recall, 'candidate_types': candidate, 'baseline_types': old,
            'abstention': abstention, 'checks': checks, 'passed': all(checks.values())}


def run(cohort, background, baseline_path, output, epochs=12, seed=42, device='cuda'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if epochs < 1 or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Invalid epochs or CUDA unavailable')
    torch.set_num_threads(4)
    rows, images, baseline, metadata = load(cohort, background, baseline_path)
    plan = {'schema': 'background_experiment_v1', 'contract': CONTRACT, 'classes': CLASSES,
            'epochs_fixed': epochs, 'seed': seed, 'folds': WAFERS, 'final_test_wafer': '8', 'test_evaluated': False,
            'expert_verified': False, 'sampler_power': .5, 'training_samples_per_epoch_cap': 6000,
            'background_threshold_fixed': .95, 'type_threshold_fixed': .8,
            'gate': 'all wafers: retain >=99% positive overall and >=98% each type; reject >=70% weak background; macroF1/ BPD TP not lower; BPD FP not higher',
            'limitations': ['Weak AI background is not expert truth', 'Baseline is a previous out-of-wafer research experiment; architecture differs',
                            'Passing this screen alone never authorizes whole-image deployment'],
            'device': torch.cuda.get_device_name() if device == 'cuda' else 'cpu', **metadata}
    save(output/'plan.json', plan)  # Saved before any optimization/evaluation.
    target = torch.tensor([CLASSES.index(r['label']) for r in rows])
    results, predictions = {}, []
    for wafer in WAFERS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        train_ids = torch.tensor([i for i, r in enumerate(rows) if r['wafer'] != wafer])
        valid_ids = torch.tensor([i for i, r in enumerate(rows) if r['wafer'] == wafer])
        train_y = target[train_ids]
        counts = torch.bincount(train_y, minlength=4).float()
        if (counts == 0).any():
            raise ValueError('All training classes required')
        sampler = WeightedRandomSampler(counts[train_y].pow(-.5), min(6000, len(train_ids)), replacement=True)
        loader = DataLoader(TensorDataset(images[train_ids], train_y), batch_size=32, sampler=sampler)
        model = BackgroundContextCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        history = []
        for epoch in range(epochs):
            start = time.monotonic()
            model.train()
            loss_sum, seen = 0.0, 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(device).float()/255, labels.to(device)
                inputs = torch.rot90(inputs, random.randrange(4), (2, 3))
                if random.random() < .5:
                    inputs = inputs.flip(3)
                inputs = (inputs*random.uniform(.8, 1.2) + random.uniform(-.04, .04)).clamp(0, 1)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(inputs), labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()*len(labels)
                seen += len(labels)
            history.append({'epoch': epoch+1, 'loss': loss_sum/seen, 'seconds': time.monotonic()-start})
            print(f'held-out wafer {wafer} epoch {epoch+1}/{epochs} loss {loss_sum/seen:.4f}', flush=True)
        torch.save(model.state_dict(), output/f'wafer{wafer}_model.pt')
        model.eval()
        probs = []
        with torch.inference_mode():
            for inputs, in DataLoader(TensorDataset(images[valid_ids]), batch_size=64):
                probs.extend(model(inputs.to(device).float()/255).softmax(1).cpu().tolist())
        held_rows = [rows[i] for i in valid_ids.tolist()]
        results[wafer] = evaluate(held_rows, probs, baseline)
        save(output/f'wafer{wafer}_history.json', history)
        save(output/f'wafer{wafer}_metrics.json', results[wafer])
        for row, probability in zip(held_rows, probs, strict=True):
            predictions.append({**row, 'scores': probability, 'held_out_wafer': wafer})
        print(f'wafer {wafer} checks: {results[wafer]["checks"]}', flush=True)
    decision = {'schema': 'background_experiment_result_v1', 'folds': results,
                'development_screen_passed': all(r['passed'] for r in results.values()),
                'deployment_status': 'not_promoted', 'test_evaluated': False,
                'reason': 'Development screening only; whole-image and expert validation still required',
                'plan_sha256': digest(output/'plan.json')}
    if not decision['development_screen_passed']:
        decision['reason'] = 'One or more predeclared development gates failed; existing model preserved'
    save(output/'oof.json', predictions)
    save(output/'decision.json', decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'background', 'baseline', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    run(args.cohort, args.background, args.baseline, args.output)


if __name__ == '__main__':
    main()
