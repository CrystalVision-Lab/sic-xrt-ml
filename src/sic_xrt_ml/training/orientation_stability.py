"""Fixed C4/D4 inference comparison on development folds; no subtype relabeling."""

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .bpd_context import EqualProbabilityEnsemble
from .grouped_development import (
    LOCAL_CONTRAST_CONTRACT,
    WAFERS,
    DevelopmentImages,
    DevelopmentView,
    load_development_model,
    split_wafer,
    verify_cohort,
)
from .patch_classifier import CLASSES, file_hash, to_tensor, write_json
from .recall_budget import bpd_counts, load_candidate, read_json, score, validate_oof

MODES = {'identity': 1, 'c4': 4, 'd4': 8}


def transform(images, index):
    if index not in range(8):
        raise ValueError('Unknown orientation')
    rotated = torch.rot90(images, index % 4, (-2, -1))
    return rotated if index < 4 else rotated.flip(-1)


class OrientationMean(torch.nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown orientation averaging mode')
        self.model, self.mode = model, mode

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 128, 128):
            raise ValueError('Expected normalized RGB NCHW 128 input')
        probabilities = torch.stack([self.model(transform(images, i)).softmax(1)
                                     for i in range(MODES[self.mode])]).mean(0)
        return probabilities.clamp_min(1e-12).log()


def choose(candidates):
    base = next(c for c in candidates if c['mode'] == 'identity')
    eligible = []
    for item in candidates:
        if item['mode'] == 'identity' or item['metrics']['macro_f1'] < base['metrics']['macro_f1']:
            continue
        allowed = True
        for w in WAFERS:
            original, new = base['by_wafer'][w], item['by_wafer'][w]
            b, n = bpd_counts(original), bpd_counts(new)
            if (np.asarray(original['confusion_matrix']).sum(1).tolist()
                    != np.asarray(new['confusion_matrix']).sum(1).tolist()
                    or n['true_positives'] < b['true_positives']
                    or n['false_positives'] > b['false_positives']):
                allowed = False
        if (allowed and bpd_counts(item['by_wafer']['1'])['false_positives']
                < bpd_counts(base['by_wafer']['1'])['false_positives']):
            eligible.append(item)
    if not eligible:
        return base
    return min(eligible, key=lambda c: (bpd_counts(c['by_wafer']['1'])['false_positives'],
                                       bpd_counts(c['metrics'])['false_positives'],
                                       -c['metrics']['macro_f1'], MODES[c['mode']]))


def load_orientation_candidate(pointer, device='cpu'):
    plan = read_json(pointer)
    if (plan.get('schema_version') != 1 or plan.get('task') != 'orientation_mean_research_candidate'
            or plan.get('mode') not in MODES or plan.get('classes') != list(CLASSES)
            or plan.get('preprocessing') != LOCAL_CONTRAST_CONTRACT
            or plan.get('target') != 'base_type_only'
            or plan.get('transform_order') != 'rot90_k_then_optional_horizontal_flip'
            or plan.get('weights') != [1/MODES[plan['mode']]]*MODES[plan['mode']]):
        raise ValueError('Invalid orientation contract')
    parent = Path(plan['parent_pointer'])
    if file_hash(parent) != plan['parent_sha256']:
        raise ValueError('Parent pointer integrity failure')
    if read_json(parent)['manifest_sha256'] != plan['manifest_sha256']:
        raise ValueError('Parent dataset mismatch')
    return OrientationMean(load_candidate(parent, device), plan['mode']).to(device).eval()


def load_fold(previous, prior, heldout, manifest, device):
    if heldout not in WAFERS:
        raise ValueError('Reserved or unknown wafer forbidden')
    models = []
    for member in prior['selected']['members']:
        folds = read_json(previous/f'{member}_folds.json')
        if sorted(f['wafer'] for f in folds) != sorted(WAFERS):
            raise ValueError('Invalid fold coverage')
        expected = next(f for f in folds if f['wafer'] == heldout)
        run = previous/f'{member}_holdout{heldout}'
        if file_hash(run/'best_model.pt') != expected['checkpoint_sha256']:
            raise ValueError('Fold checkpoint integrity failure')
        model, checkpoint = load_development_model(run, device)
        config = checkpoint['config']
        if (config['train_wafers'] != sorted(set(WAFERS)-{heldout})
                or config['heldout_wafer'] != heldout or config['manifest_sha256'] != manifest
                or config['preprocessing'] != LOCAL_CONTRAST_CONTRACT
                or config['epochs'] != 12 or checkpoint['epoch'] != 12
                or config['recipe']['name'] != member):
            raise ValueError('Fold training provenance mismatch')
        models.append(model)
    if len(models) != 2:
        raise ValueError('Expected two fixed seed models')
    return EqualProbabilityEnsemble(models).to(device).eval()


def run_comparison(root, previous, output, device='cuda'):
    root, previous, output = (Path(p).resolve() for p in (root, previous, output))
    if output.exists() or output.is_relative_to(root):
        raise ValueError('New output outside development dataset required')
    contract, rows = verify_cohort(root)
    prior = read_json(previous/'result.json')
    parent = Path(prior['candidate_pointer'])
    if (prior['status'] != 'completed' or prior['selected']['name'] != 'power075_two_seeds'
            or file_hash(parent) != prior['candidate_sha256']
            or read_json(parent)['manifest_sha256'] != contract['manifest_sha256']):
        raise ValueError('Incompatible previous candidate')
    baseline_path = previous/'power075_two_seeds_oof.json'
    baseline = validate_oof(read_json(baseline_path), rows)
    if score(baseline) != prior['selected']['pooled_OOF']:
        raise ValueError('Baseline predictions disagree with recorded metrics')
    output.mkdir(parents=True)
    plan = {'schema_version': 1, 'created_at': datetime.now(UTC).isoformat(),
            'task': 'fixed_orientation_development_comparison', 'modes': MODES,
            'baseline_result_sha256': file_hash(previous/'result.json'),
            'baseline_oof_sha256': file_hash(baseline_path), 'parent_sha256': file_hash(parent),
            'manifest_sha256': contract['manifest_sha256'], 'wafer_groups': list(WAFERS),
            'forbidden_wafers': ['8'], 'new_training_runs': 0,
            'selection': 'all_wafer_TP_nondecrease;all_wafer_FP_nonincrease;W1_FP_strict_decrease;macro_nondecrease;min_W1_FP_then_total_FP_then_max_macro_then_min_views',
            'test_evaluated': False, 'independent_final_test': False, 'adaptive_development': True}
    write_json(output/'plan.json', plan)
    cache = DevelopmentImages(root, rows)
    outputs, diagnostics, timings = {m: [] for m in MODES}, [], {}
    for heldout in WAFERS:
        _, indices = split_wafer(rows, heldout)
        model = load_fold(previous, prior, heldout, contract['manifest_sha256'], device)
        loader = DataLoader(DevelopmentView(cache, indices, False), batch_size=32, shuffle=False)
        seen, elapsed = 0, np.zeros(8)
        with torch.inference_mode():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                views = []
                for orientation in range(8):
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    views.append(model(transform(inputs, orientation)).softmax(1).cpu().numpy())
                    elapsed[orientation] += time.perf_counter()-started
                stacked = np.stack(views, axis=1)
                for j, probabilities in enumerate(stacked):
                    row = cache.rows[indices[seen+j]]
                    identity = {k: row[k] for k in ('point_id', 'patch_id', 'wafer', 'label')}
                    for mode, count in MODES.items():
                        mean = probabilities[:count].astype(np.float64).mean(0)
                        outputs[mode].append(identity | {'scores': mean.tolist(),
                                                        'prediction': CLASSES[int(mean.argmax())]})
                    diagnostics.append(identity | {'phase': row['phase'],
                        'context_other_type_count': int(row['context_other_type_count']),
                        'orientation_predictions': [CLASSES[int(p.argmax())] for p in probabilities],
                        'orientation_probabilities': probabilities.tolist(),
                        'BPD_score_range': float(np.ptp(probabilities[:, 0])),
                        'disagrees': len(set(probabilities.argmax(1).tolist())) > 1})
                seen += len(inputs)
        timings[heldout] = {'samples': seen, 'seconds_by_orientation': elapsed.tolist()}
        write_json(output/'progress.json', {'completed_wafers': list(timings), 'test_evaluated': False})
        print(f'Completed W{heldout}: {seen} samples x 8 orientations x 2 models', flush=True)
        del model
    comparison = []
    for mode in MODES:
        outputs[mode] = validate_oof(outputs[mode], rows)
        if mode == 'identity' and (
                not np.allclose([p['scores'] for p in baseline], [p['scores'] for p in outputs[mode]], atol=1e-6)
                or [p['prediction'] for p in baseline] != [p['prediction'] for p in outputs[mode]]):
            raise ValueError('Recomputed baseline changed')
        metrics = score(outputs[mode])
        item = {'mode': mode, 'metrics': metrics, 'BPD': bpd_counts(metrics),
                'by_wafer': {w: score([p for p in outputs[mode] if p['wafer'] == w]) for w in WAFERS}}
        comparison.append(item)
        write_json(output/f'{mode}_oof.json', outputs[mode])
    write_json(output/'orientation_diagnostics.json', diagnostics)
    write_json(output/'timings.json', timings)
    write_json(output/'comparison.json', comparison)
    selected = choose(comparison)
    pointer = output/'candidate.json'
    write_json(pointer, {'schema_version': 1, 'task': 'orientation_mean_research_candidate',
        'target': 'base_type_only', 'mode': selected['mode'], 'classes': list(CLASSES),
        'preprocessing': LOCAL_CONTRAST_CONTRACT, 'transform_order': 'rot90_k_then_optional_horizontal_flip',
        'weights': [1/MODES[selected['mode']]]*MODES[selected['mode']],
        'parent_pointer': str(parent), 'parent_sha256': file_hash(parent),
        'manifest_sha256': contract['manifest_sha256'], 'research_only': True, 'test_evaluated': False})
    loaded = load_orientation_candidate(pointer, device)
    direct = OrientationMean(load_candidate(parent, device), selected['mode']).to(device).eval()
    probe = torch.stack([to_tensor(cache.images[i]) for i in range(3)]).to(device)
    with torch.inference_mode():
        if not torch.allclose(loaded(probe).softmax(1), direct(probe).softmax(1), atol=1e-6):
            raise ValueError('Candidate serialization changed predictions')
    verify_cohort(root)
    if file_hash(baseline_path) != plan['baseline_oof_sha256'] or file_hash(parent) != plan['parent_sha256']:
        raise ValueError('Prior experiment changed')
    result = {'status': 'completed', 'comparison': comparison, 'selected': selected,
              'baseline_retained': selected['mode'] == 'identity', 'serialization_verified': True,
              'baseline_reproduced': True, 'candidate_pointer': str(pointer),
              'candidate_sha256': file_hash(pointer), 'new_training_runs': 0,
              'test_evaluated': False, 'independent_final_test': False, 'research_only': True,
              'server_model_replaced': False, 'expert_ground_truth': False,
              'limitations': ['adaptive_development_selection', 'BPD_in_two_wafers_only',
                              'orientation_averaging_not_valid_for_direction_subtypes',
                              'extra_inference_cost', 'provider_labels_not_expert_verified']}
    write_json(output/'result.json', result)
    return result
