"""Fixed-checkpoint diagnostics, never a held-out threshold selection policy."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sic_xrt_ml.training.background_abstention import (
    WAFERS,
    checked_patch,
    digest,
    load,
    save,
)
from sic_xrt_ml.training.independent_background import (
    IndependentBackgroundCNN,
    reviewed_candidates,
)


def retention_bound(rows, defect_probability):
    """Optimistic diagnostic bound computed on evaluation labels, NOT deployable.

    Background score >= threshold is rejected. Find the smallest threshold
    retaining >=99% positives overall and >=98% per type/possible-defect group.
    """
    probability = np.asarray(defect_probability, dtype=float)
    if probability.shape != (len(rows),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Invalid probabilities')
    background = 1-probability
    positive = np.array([r['stratum'] in ('BPD', 'TED', 'TSD') for r in rows])
    if not positive.any():
        raise ValueError('Provided positives required')
    constraints = [(positive, .01)]
    for group in ('BPD', 'TED', 'TSD', 'possible_defect'):
        mask = np.array([r['stratum'] == group for r in rows])
        if mask.any():
            constraints.append((mask, .02))
    threshold = 0.
    for mask, rate in constraints:
        values = np.sort(background[mask])[::-1]
        allowed = int(np.floor(len(values)*rate))
        threshold = max(threshold, float(np.nextafter(values[allowed], np.inf)))
    removed = background >= threshold
    strata = {}
    for group in sorted({r['stratum'] for r in rows}):
        mask = np.array([r['stratum'] == group for r in rows])
        strata[group] = {'n': int(mask.sum()), 'removed': int(removed[mask].sum()), 'removed_rate': float(removed[mask].mean())}
    weak = [v for k, v in strata.items() if k.startswith('background_')]
    return {'diagnostic_only': True, 'not_a_deployment_threshold': threshold,
            'all_weak_strata_can_reach_70_percent': bool(weak) and all(v['removed_rate'] >= .7 for v in weak),
            'strata': strata}


def distribution(rows, probabilities):
    result = {}
    for group in sorted({r['stratum'] for r in rows}):
        selected = np.array([p for r, p in zip(rows, probabilities, strict=True) if r['stratum'] == group])
        result[group] = {'n': len(selected), 'defect_probability_quantiles_0_05_50_95_100': np.quantile(selected, [0,.05,.5,.95,1]).tolist(),
                         'background_at_095': int(((1-selected) >= .95).sum())}
    return result


def run(cohort, background, baseline, observations, experiment, output):
    output, experiment, observations = Path(output), Path(experiment), Path(observations)
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((experiment/'plan.json').read_text(encoding='utf-8'))
    rows, tensors, _, info = load(cohort, background, baseline)
    for key in ('cohort_manifest_sha256', 'background_sha256', 'baseline_oof_sha256', 'coordinate_repair_sha256'):
        if info[key] != plan[key]:
            raise ValueError('Experiment input changed')
    if digest(observations) != plan['observations_sha256'] or plan['test_evaluated'] is not False:
        raise ValueError('Observation or test contract mismatch')
    for row in rows:
        row['stratum'] = 'background_random' if row['label'] == 'BACKGROUND' else row['label']
    extra = []
    for row in reviewed_candidates(observations):
        rows.append({'patch_id': row['id'], 'wafer': row['wafer'], 'stratum':
                     'background_hard' if row['decision']=='weak_background' else row['decision']})
        extra.append(checked_patch(observations.parent, row))
    tensors = torch.cat([tensors, torch.from_numpy(np.stack(extra).transpose(0,3,1,2).copy())])
    original = json.loads((experiment/'oof.json').read_text(encoding='utf-8'))
    old = {(r['held_out_wafer'], r['patch_id']): r['defect_probability'] for r in original}
    if len(old) != len(original) or len({r['patch_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate evaluation identities')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    results = {}
    save(output/'diagnostic_plan.json', {'frozen_experiment_plan_sha256': digest(experiment/'plan.json'),
        'test_evaluated': False, 'existing_model_changed': False, 'new_training': False,
        'purpose': 'Compare train/held-out distributions; optimistic held-out retention bound is diagnostic, never deployment tuning'})
    for wafer in WAFERS:
        checkpoint = experiment/f'wafer{wafer}_binary.pt'
        model = IndependentBackgroundCNN().to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        values = []
        with torch.inference_mode():
            for batch, in DataLoader(TensorDataset(tensors), batch_size=64):
                values.extend(model(batch.to(device).float()/255).sigmoid().cpu().tolist())
        data = [{**r, 'defect_probability':p} for r,p in zip(rows, values, strict=True)]
        held = [r for r in data if r['wafer'] == wafer]
        discrepancy = max(abs(r['defect_probability']-old[wafer,r['patch_id']]) for r in held)
        if discrepancy > 1e-5:
            raise ValueError(f'Cannot reproduce fixed predictions: {discrepancy}')
        trained = [r for r in data if r['wafer'] != wafer and r['stratum'] not in ('possible_defect', 'uncertain')]
        report = {'checkpoint_sha256':digest(checkpoint), 'held_out_prediction_max_abs_difference':discrepancy,
                  'train_counts':dict(Counter(r['stratum'] for r in trained)),
                  'training':distribution(trained, [r['defect_probability'] for r in trained]),
                  'held_out':distribution(held, [r['defect_probability'] for r in held]),
                  'optimistic_held_out_bound':retention_bound(held, [r['defect_probability'] for r in held])}
        save(output/f'wafer{wafer}_scores.json', data)
        results[wafer] = report
        print(wafer, json.dumps(report), flush=True)
    save(output/'diagnosis.json', results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','background','baseline','observations','experiment','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    run(args.cohort,args.background,args.baseline,args.observations,args.experiment,args.output)


if __name__ == '__main__':
    main()
