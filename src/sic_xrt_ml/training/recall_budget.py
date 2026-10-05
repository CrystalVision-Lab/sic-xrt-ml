"""Predeclared development comparison with an explicit BPD false-positive budget."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .bpd_context import EqualProbabilityEnsemble, average_predictions
from .grouped_development import (
    LOCAL_CONTRAST_CONTRACT,
    WAFERS,
    DevelopmentImages,
    fit,
    load_development_model,
    predictions,
    split_wafer,
    verify_cohort,
)
from .patch_classifier import (
    CLASSES,
    classification_metrics,
    file_hash,
    to_tensor,
    write_json,
)

BASELINE = 'power050_seed42'
SPECS = {
    BASELINE: (.5, 42), 'power050_seed43': (.5, 43),
    'power075_seed42': (.75, 42), 'power075_seed43': (.75, 43),
}
ENSEMBLES = {
    'power050_two_seeds': [BASELINE, 'power050_seed43'],
    'power075_two_seeds': ['power075_seed42', 'power075_seed43'],
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_oof(records, rows):
    """Require exactly one valid prediction per development identity, in canonical order."""
    keys = ('patch_id', 'point_id', 'wafer', 'label')
    expected = sorted(tuple(r[k] for k in keys) for r in rows)
    actual = sorted(tuple(r[k] for k in keys) for r in records)
    if (not expected or len({r[0] for r in expected}) != len(expected) or actual != expected
            or any(r['wafer'] not in WAFERS for r in rows)):
        raise ValueError('OOF identity/wafer coverage mismatch')
    for r in records:
        scores = np.asarray(r['scores'], dtype=float)
        if (scores.shape != (3,) or not np.isfinite(scores).all()
                or (scores < 0).any() or (scores > 1).any()
                or not np.isclose(scores.sum(), 1, atol=1e-6)
                or r['prediction'] != CLASSES[int(scores.argmax())]):
            raise ValueError('Invalid OOF probabilities or prediction')
    return sorted(records, key=lambda r: r['patch_id'])


def score(records):
    matrix = np.zeros((3, 3), dtype=np.int64)
    for r in records:
        matrix[CLASSES.index(r['label']), CLASSES.index(r['prediction'])] += 1
    return classification_metrics(matrix)


def bpd_counts(metrics):
    matrix = np.asarray(metrics['confusion_matrix'])
    return {'true_positives': int(matrix[0, 0]),
            'false_negatives': int(matrix[0, 1:].sum()),
            'false_positives': int(matrix[1:, 0].sum())}


def select_candidate(candidates, baseline_name=BASELINE):
    baseline = next(c for c in candidates if c['name'] == baseline_name)
    budget = bpd_counts(baseline['pooled_OOF'])['false_positives']
    floor = baseline['pooled_OOF']['macro_f1']
    eligible = [c for c in candidates
                if bpd_counts(c['pooled_OOF'])['false_positives'] <= budget
                and c['pooled_OOF']['macro_f1'] >= floor
                and np.asarray(c['pooled_OOF']['confusion_matrix']).sum(axis=1).tolist()
                == np.asarray(baseline['pooled_OOF']['confusion_matrix']).sum(axis=1).tolist()]
    # A change requires strictly higher BPD recall; ties retain the established baseline.
    improving = [c for c in eligible if bpd_counts(c['pooled_OOF'])['true_positives']
                 > bpd_counts(baseline['pooled_OOF'])['true_positives']]
    if not improving:
        return baseline
    return max(improving, key=lambda c: (bpd_counts(c['pooled_OOF'])['true_positives'],
                                        c['pooled_OOF']['macro_f1'], -len(c['members'])))


def load_candidate(pointer, device='cpu'):
    """Load hash-pinned normalized full-development members, never held-out-fold models."""
    plan = read_json(pointer)
    members = plan.get('members', [])
    if (plan.get('schema_version') != 1 or plan.get('task') != 'normalized_recall_budget_candidate'
            or plan.get('classes') != list(CLASSES) or plan.get('preprocessing') != LOCAL_CONTRAST_CONTRACT
            or not members or plan.get('weights') != [1/len(members)]*len(members)
            or len({m['checkpoint_sha256'] for m in members}) != len(members)):
        raise ValueError('Invalid candidate contract')
    models = []
    for member in members:
        if file_hash(Path(member['run_dir'])/'best_model.pt') != member['checkpoint_sha256']:
            raise ValueError('Candidate checkpoint integrity failure')
        model, checkpoint = load_development_model(member['run_dir'], device)
        config = checkpoint['config']
        if (config['model'] != 'NormalizedBPDContextCNN_v1'
                or config['manifest_sha256'] != plan['manifest_sha256']
                or config['train_wafers'] != list(WAFERS) or config['heldout_wafer'] is not None
                or config['preprocessing'] != plan['preprocessing']
                or config['seed'] != member['seed'] or config['recipe'] != member['recipe']
                or config['epochs'] != plan['epochs'] or checkpoint['epoch'] != plan['epochs']):
            raise ValueError('Candidate training/input contract mismatch')
        models.append(model)
    return (models[0] if len(models) == 1 else EqualProbabilityEnsemble(models)).to(device).eval()


def run_comparison(root, previous, output, device='cuda'):
    root, previous, output = (Path(p).resolve() for p in (root, previous, output))
    if output.exists() or output.is_relative_to(root):
        raise ValueError('New output outside development cohort required')
    contract, rows = verify_cohort(root)
    prior, prior_plan = read_json(previous/'result.json'), read_json(previous/'plan.json')
    if (prior['status'] != 'completed' or prior_plan['manifest_sha256'] != contract['manifest_sha256']
            or prior_plan['epochs'] != 12 or prior_plan['seed'] != 42
            or prior['selected_recipe']['name'] != 'normalized_context_sqrt'):
        raise ValueError('Incompatible baseline development experiment')
    baseline_path = previous/'normalized_context_sqrt_oof.json'
    baseline = validate_oof(read_json(baseline_path), rows)
    if score(baseline) != prior['selected_OOF']:
        raise ValueError('Baseline OOF metrics disagree with recorded result')
    prior_item = next(r for r in prior['comparison'] if r['recipe'] == prior['selected_recipe'])
    for fold in prior_item['folds']:
        run = previous/f'normalized_context_sqrt_holdout{fold["wafer"]}'
        checkpoint = torch.load(run/'best_model.pt', map_location='cpu', weights_only=True)
        config = checkpoint['config']
        expected_train = sorted(set(WAFERS)-{fold['wafer']})
        recorded = read_json(run/'heldout_predictions.json')
        if (file_hash(run/'best_model.pt') != fold['checkpoint_sha256']
                or config['train_wafers'] != expected_train or config['heldout_wafer'] != fold['wafer']
                or config['manifest_sha256'] != contract['manifest_sha256']
                or config['seed'] != 42 or config['epochs'] != 12
                or config['recipe'] != prior['selected_recipe']
                or sorted(recorded, key=lambda r: r['patch_id'])
                != [r for r in baseline if r['wafer'] == fold['wafer']]):
            raise ValueError('Baseline fold provenance mismatch')
    if file_hash(Path(prior['model_run'])/'best_model.pt') != prior['checkpoint_sha256']:
        raise ValueError('Baseline final checkpoint changed')
    output.mkdir(parents=True)
    plan = {'schema_version': 1, 'task': 'development_BPD_recall_with_FP_budget',
            'created_at': datetime.now(UTC).isoformat(), 'manifest_sha256': contract['manifest_sha256'],
            'baseline_result_sha256': file_hash(previous/'result.json'),
            'baseline_oof_sha256': file_hash(baseline_path), 'specs': SPECS, 'ensembles': ENSEMBLES,
            'epochs': 12, 'wafer_groups': list(WAFERS), 'forbidden_wafers': ['8'],
            'maximum_BPD_false_positives': bpd_counts(prior['selected_OOF'])['false_positives'],
            'minimum_macro_f1': prior['selected_OOF']['macro_f1'],
            'selection': 'strict_BPD_recall_gain_within_FP_and_macro_budget;then_macro;then_fewer_members',
            'ensemble_weights': 'equal;fixed_before_training', 'test_evaluated': False,
            'adaptive_development_experiment': True, 'independent_final_test': False}
    write_json(output/'plan.json', plan)
    cache = DevelopmentImages(root, rows)
    all_oof, results = {BASELINE: baseline}, []
    for name, (power, seed) in SPECS.items():
        if name == BASELINE:
            continue
        recipe = {'name': name, 'variant': 'center32_context128', 'sampling_power': power,
                  'normalization': 'local_contrast_v1'}
        oof, folds = [], []
        for heldout in WAFERS:
            train, valid = split_wafer(rows, heldout)
            run = output/f'{name}_holdout{heldout}'
            model = fit(cache, train, recipe, contract, run, 12, seed, device, heldout)
            pred, metrics = predictions(model, cache, valid, device)
            write_json(run/'heldout_predictions.json', pred)
            folds.append({'wafer': heldout, 'metrics': metrics,
                          'checkpoint_sha256': file_hash(run/'best_model.pt')})
            oof.extend(pred)
            write_json(output/'progress.json', {'name': name, 'completed_folds': folds})
            del model
        all_oof[name] = validate_oof(oof, rows)
        write_json(output/f'{name}_folds.json', folds)
    for name, members in ENSEMBLES.items():
        combined, _ = average_predictions([all_oof[m] for m in members])
        for r, original in zip(combined, baseline):
            r['wafer'] = original['wafer']
        all_oof[name] = validate_oof(combined, rows)
    for name, records in all_oof.items():
        write_json(output/f'{name}_oof.json', records)
        results.append({'name': name, 'members': ENSEMBLES.get(name, [name]),
                        'pooled_OOF': score(records), 'BPD': bpd_counts(score(records)),
                        'by_wafer': {w: score([r for r in records if r['wafer'] == w]) for w in WAFERS}})
    write_json(output/'comparison.json', results)
    selected = select_candidate(results)
    member_specs, models = [], []
    for name in selected['members']:
        power, seed = SPECS[name]
        if name == BASELINE:
            run, recipe = Path(prior['model_run']), prior['selected_recipe']
            model, _ = load_development_model(run, device)
        else:
            recipe = {'name': name, 'variant': 'center32_context128', 'sampling_power': power,
                      'normalization': 'local_contrast_v1'}
            run = output/f'development_{name}'
            model = fit(cache, list(range(len(rows))), recipe, contract, run, 12, seed, device, None)
        models.append(model.eval())
        member_specs.append({'run_dir': str(run), 'checkpoint_sha256': file_hash(run/'best_model.pt'),
                             'recipe': recipe, 'seed': seed})
    pointer = output/'candidate.json'
    write_json(pointer, {'schema_version': 1, 'task': 'normalized_recall_budget_candidate',
        'classes': list(CLASSES), 'preprocessing': LOCAL_CONTRAST_CONTRACT,
        'manifest_sha256': contract['manifest_sha256'], 'epochs': 12, 'selected': selected['name'],
        'members': member_specs, 'weights': [1/len(models)]*len(models),
        'research_only': True, 'expert_ground_truth': False, 'test_evaluated': False})
    reloaded = load_candidate(pointer, device)
    assembled = models[0] if len(models) == 1 else EqualProbabilityEnsemble(models).eval()
    probe = torch.stack([to_tensor(cache.images[i]) for i in range(3)]).to(device)
    with torch.inference_mode():
        if not torch.allclose(assembled(probe).softmax(1), reloaded(probe).softmax(1), atol=1e-6):
            raise ValueError('Candidate reload differs')
    verify_cohort(root)
    if file_hash(baseline_path) != plan['baseline_oof_sha256']:
        raise ValueError('Baseline changed during comparison')
    result = {'status': 'completed', 'comparison': results, 'selected': selected,
              'baseline_retained': selected['name'] == BASELINE, 'candidate_pointer': str(pointer),
              'candidate_sha256': file_hash(pointer), 'serialization_verified': True,
              'new_grouped_training_runs': 9, 'test_evaluated': False, 'independent_final_test': False,
              'research_only': True, 'expert_ground_truth': False, 'server_model_replaced': False,
              'limitations': ['adaptive_OOF_selection', 'BPD_in_two_wafers_only', 'W9_has_no_BPD',
                              'only_two_seeds', 'provider_labels_not_expert_verified']}
    write_json(output/'result.json', result)
    return result
