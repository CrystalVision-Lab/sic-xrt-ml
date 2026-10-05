"""Fixed spatial ablations with whole-wafer development folds and strict adoption."""

import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .bpd_context import (
    ContextClassifier,
    EqualProbabilityEnsemble,
    average_predictions,
)
from .grouped_development import (
    WAFERS,
    DevelopmentImages,
    DevelopmentView,
    normalize_local_contrast,
    predictions,
    sampling_weights,
    split_wafer,
    verify_cohort,
)
from .orientation_stability import OrientationMean
from .patch_classifier import CLASSES, file_hash, run_epoch, to_tensor, write_json
from .recall_budget import bpd_counts, load_candidate, read_json, score, validate_oof

RECIPES = (
    {'name': 'center32_jitter4', 'center_size': 32, 'jitter_px': 4},
    {'name': 'center64_nojitter', 'center_size': 64, 'jitter_px': 0},
)
SEEDS = (42, 43)
EPOCHS = 12
CONTRACT = 'uint8_RGB128_to_float32_NCHW_div255;gray_mean;reflect_box15_highpass;rms_floor1over255;0.5plus0.15z_clip01;center_native_resolution;context128_avgpool2;no_marks'


def translate_reflect(image, dy, dx):
    if dy not in range(-4, 5) or dx not in range(-4, 5):
        raise ValueError('Training displacement must be within four pixels')
    padded = torch.nn.functional.pad(image, (4, 4, 4, 4), mode='reflect')
    return padded[:, 4+dy:132+dy, 4+dx:132+dx]


class SpatialView(DevelopmentView):
    def __init__(self, cache, indices, training, jitter):
        super().__init__(cache, indices, training)
        if jitter not in (0, 4):
            raise ValueError('Unsupported jitter')
        self.jitter = jitter

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        if self.training and self.jitter:
            dy, dx = torch.randint(-4, 5, (2,)).tolist()
            image = translate_reflect(image, dy, dx)
        return image, label


class SpatialClassifier(torch.nn.Module):
    def __init__(self, center_size):
        super().__init__()
        if center_size not in (32, 64):
            raise ValueError('Unsupported center size')
        self.center_size = center_size
        self.classifier = ContextClassifier('center32_context128')

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 128, 128):
            raise ValueError('Expected RGB128 NCHW input')
        images = normalize_local_contrast(images)
        start = (128-self.center_size)//2
        inner = self.classifier.center.features(images[:, :, start:128-start, start:128-start]).flatten(1)
        outer = self.classifier.context(torch.nn.functional.avg_pool2d(images, 2)).flatten(1)
        return self.classifier.fusion(torch.cat((inner, outer), 1))


def fit_spatial(cache, indices, recipe, contract, run, seed, device, heldout):
    if recipe not in RECIPES or seed not in SEEDS or heldout not in (*WAFERS, None):
        raise ValueError('Unsupported fixed experiment')
    expected = (list(range(len(cache.rows))) if heldout is None else split_wafer(cache.rows, heldout)[0])
    if sorted(indices) != expected or any(r['wafer'] not in WAFERS for r in cache.rows):
        raise ValueError('Training fold contains heldout or unknown data')
    run.mkdir()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    training_rows = [cache.rows[i] for i in indices]
    sampler = WeightedRandomSampler(sampling_weights(training_rows, .75), len(indices), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(SpatialView(cache, indices, True, recipe['jitter_px']), batch_size=32, sampler=sampler)
    model = SpatialClassifier(recipe['center_size']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    config = {'model': 'NormalizedSpatialCNN_v1', 'classes': list(CLASSES), 'preprocessing': CONTRACT,
              'recipe': recipe, 'seed': seed, 'epochs': EPOCHS, 'sampling_power': .75,
              'manifest_sha256': contract['manifest_sha256'],
              'train_wafers': sorted({r['wafer'] for r in training_rows}), 'heldout_wafer': heldout,
              'training_counts': dict(Counter(r['label'] for r in training_rows)),
              'batch_size': 32, 'learning_rate': .001, 'epoch_selection': 'fixed_last_epoch',
              'augmentation': 'train_only_D4_contrast0.8to1.2_brightness0.04_then_optional_reflect_integer_jitter4',
              'research_only': True, 'test_evaluated': False}
    write_json(run/'config.json', config)
    history = []
    for epoch in range(1, EPOCHS+1):
        started = time.perf_counter()
        metrics = run_epoch(model, loader, torch.device(device), optimizer)
        history.append({'epoch': epoch, 'training': metrics, 'seconds': time.perf_counter()-started})
        write_json(run/'history.json', history)
        print(f'{recipe["name"]} seed{seed} W{heldout} epoch{epoch}/{EPOCHS} '
              f'trainF1={metrics["macro_f1"]:.3f} {history[-1]["seconds"]:.1f}s', flush=True)
    torch.save({'config': config, 'epoch': EPOCHS,
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}, run/'best_model.pt')
    return model.eval()


def load_spatial(run, device='cpu'):
    checkpoint = torch.load(Path(run)/'best_model.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if (config.get('model') != 'NormalizedSpatialCNN_v1' or config.get('classes') != list(CLASSES)
            or config.get('preprocessing') != CONTRACT or config.get('recipe') not in RECIPES
            or config.get('seed') not in SEEDS or config.get('epochs') != EPOCHS
            or config.get('sampling_power') != .75 or checkpoint['epoch'] != EPOCHS):
        raise ValueError('Invalid spatial checkpoint contract')
    model = SpatialClassifier(config['recipe']['center_size']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), checkpoint


def selection_reasons(item, baseline):
    reasons = []
    if item['metrics']['macro_f1'] < baseline['metrics']['macro_f1']:
        reasons.append('macro_f1_decreased')
    for wafer in WAFERS:
        original, current = baseline['by_wafer'][wafer], item['by_wafer'][wafer]
        b, n = bpd_counts(original), bpd_counts(current)
        if np.asarray(original['confusion_matrix']).sum(1).tolist() != np.asarray(current['confusion_matrix']).sum(1).tolist():
            reasons.append(f'W{wafer}_support_changed')
        if n['true_positives'] < b['true_positives']:
            reasons.append(f'W{wafer}_BPD_TP_decreased')
        if n['false_positives'] > b['false_positives']:
            reasons.append(f'W{wafer}_BPD_FP_increased')
    if bpd_counts(item['by_wafer']['1'])['false_positives'] >= bpd_counts(baseline['by_wafer']['1'])['false_positives']:
        reasons.append('W1_FP_not_strictly_reduced')
    return reasons


def select_spatial(comparison):
    baseline = next(c for c in comparison if c['name'] == 'baseline_identity')
    eligible = [c for c in comparison if c['name'] != baseline['name'] and not selection_reasons(c, baseline)]
    if not eligible:
        return baseline
    return min(eligible, key=lambda c: (bpd_counts(c['by_wafer']['1'])['false_positives'],
        bpd_counts(c['metrics'])['false_positives'], -c['metrics']['macro_f1'], c['views']))


def load_spatial_candidate(pointer, device='cpu'):
    plan = read_json(pointer)
    if (plan.get('schema_version') != 1 or plan.get('task') != 'spatial_research_candidate'
            or plan.get('mode') not in ('identity', 'c4') or plan.get('classes') != list(CLASSES)
            or plan.get('preprocessing') != CONTRACT or plan.get('recipe') not in RECIPES
            or plan.get('weights') != [.5, .5] or len(plan.get('members', [])) != 2
            or [m['seed'] for m in plan['members']] != list(SEEDS)
            or len({m['checkpoint_sha256'] for m in plan['members']}) != 2):
        raise ValueError('Invalid spatial candidate contract')
    models = []
    for member in plan['members']:
        if file_hash(Path(member['run_dir'])/'best_model.pt') != member['checkpoint_sha256']:
            raise ValueError('Spatial checkpoint integrity failure')
        model, checkpoint = load_spatial(member['run_dir'], device)
        config = checkpoint['config']
        if (config['train_wafers'] != list(WAFERS) or config['heldout_wafer'] is not None
                or config['manifest_sha256'] != plan['manifest_sha256']
                or config['recipe'] != plan['recipe'] or config['seed'] != member['seed']):
            raise ValueError('Spatial training provenance mismatch')
        models.append(model)
    return OrientationMean(EqualProbabilityEnsemble(models), plan['mode']).to(device).eval()


def run_comparison(root, previous, output, diagnostic_summary, device='cuda'):
    root, previous, output = (Path(p).resolve() for p in (root, previous, output))
    if output.exists() or output.is_relative_to(root):
        raise ValueError('New output outside dataset required')
    contract, rows = verify_cohort(root)
    prior = read_json(previous/'result.json')
    prior_plan = read_json(previous/'plan.json')
    orientation_pointer = read_json(Path(prior['candidate_pointer']))
    parent = Path(orientation_pointer['parent_pointer'])
    if (prior['status'] != 'completed' or not prior['baseline_reproduced']
            or not prior['baseline_retained'] or prior_plan['manifest_sha256'] != contract['manifest_sha256']
            or file_hash(Path(prior['candidate_pointer'])) != prior['candidate_sha256']
            or file_hash(parent) != orientation_pointer['parent_sha256']):
        raise ValueError('Incompatible or changed prior experiment')
    comparison, baseline_records = [], {}
    for mode in ('identity', 'c4'):
        records = validate_oof(read_json(previous/f'{mode}_oof.json'), rows)
        entry = next(c for c in prior['comparison'] if c['mode'] == mode)
        if score(records) != entry['metrics']:
            raise ValueError('Prior OOF metrics mismatch')
        baseline_records[mode] = records
        comparison.append(entry | {'name': f'baseline_{mode}', 'views': 1 if mode == 'identity' else 4,
                                   'recipe': None})
    output.mkdir(parents=True)
    plan = {'schema_version': 1, 'created_at': datetime.now(UTC).isoformat(),
        'task': 'spatial_development_ablation', 'manifest_sha256': contract['manifest_sha256'],
        'previous_result_sha256': file_hash(previous/'result.json'),
        'prior_oof_hashes': {m: file_hash(previous/f'{m}_oof.json') for m in ('identity', 'c4')},
        'diagnostic_summary_sha256': file_hash(diagnostic_summary),
        'recipes': list(RECIPES), 'seeds': list(SEEDS), 'epochs': EPOCHS, 'sampling_power': .75,
        'inference_modes': ['identity', 'c4'], 'seed_ensemble': 'equal_probability',
        'wafer_groups': list(WAFERS), 'forbidden_wafers': ['8'],
        'selection': 'per_wafer_BPD_TP_nondecrease_FP_nonincrease;W1_FP_strict_decrease;macro_nondecrease;min_W1FP_then_totalFP_then_max_macro_then_min_views',
        'adaptive_development': True, 'independent_final_test': False, 'test_evaluated': False}
    write_json(output/'plan.json', plan)
    write_json(output/'diagnostic_summary.json', read_json(diagnostic_summary))
    for mode, records in baseline_records.items():
        write_json(output/f'baseline_{mode}_oof.json', records)
    cache = DevelopmentImages(root, rows)
    completed = []
    for recipe in RECIPES:
        by_seed = {}
        for seed in SEEDS:
            oof = {'identity': [], 'c4': []}
            for heldout in WAFERS:
                train, valid = split_wafer(rows, heldout)
                run = output/f'{recipe["name"]}_seed{seed}_holdout{heldout}'
                model = fit_spatial(cache, train, recipe, contract, run, seed, device, heldout)
                fold = {'recipe': recipe, 'seed': seed, 'wafer': heldout,
                        'checkpoint_sha256': file_hash(run/'best_model.pt'), 'evaluation': {}}
                for mode in ('identity', 'c4'):
                    started = time.perf_counter()
                    records, metrics = predictions(OrientationMean(model, mode).eval(), cache, valid, device)
                    fold['evaluation'][mode] = {'metrics': metrics, 'seconds': time.perf_counter()-started}
                    oof[mode].extend(records)
                    write_json(run/f'{mode}_predictions.json', records)
                completed.append(fold)
                write_json(output/'progress.json', {'completed_folds': completed, 'test_evaluated': False})
                del model
            by_seed[seed] = {m: validate_oof(records, rows) for m, records in oof.items()}
        for mode in ('identity', 'c4'):
            averaged, metrics = average_predictions([by_seed[s][mode] for s in SEEDS])
            for record, source in zip(averaged, by_seed[42][mode]):
                record['wafer'] = source['wafer']
            averaged = validate_oof(averaged, rows)
            name = recipe['name']+'_'+mode
            comparison.append({'name': name, 'recipe': recipe, 'mode': mode,
                               'views': 1 if mode == 'identity' else 4, 'metrics': metrics,
                               'BPD': bpd_counts(metrics),
                               'by_wafer': {w: score([r for r in averaged if r['wafer'] == w]) for w in WAFERS}})
            write_json(output/f'{name}_oof.json', averaged)
        write_json(output/'comparison.json', comparison)
    selected = select_spatial(comparison)
    for item in comparison:
        item['rejection_reasons'] = selection_reasons(item, comparison[0]) if item['name'] != 'baseline_identity' else []
    write_json(output/'comparison.json', comparison)
    if selected['recipe'] is None:
        pointer, loader = parent, 'sic_xrt_ml.training.recall_budget.load_candidate'
        # Baseline C4 cannot meet the predeclared per-wafer gate in this experiment.
        if selected['mode'] != 'identity':
            raise ValueError('Unexpected prior candidate selection')
        loaded = load_candidate(pointer, device)
        direct = load_candidate(pointer, device)
    else:
        models, members = [], []
        for seed in SEEDS:
            run = output/f'development_seed{seed}'
            model = fit_spatial(cache, list(range(len(rows))), selected['recipe'], contract, run, seed, device, None)
            models.append(model)
            members.append({'seed': seed, 'run_dir': str(run), 'checkpoint_sha256': file_hash(run/'best_model.pt')})
        pointer = output/'candidate.json'
        write_json(pointer, {'schema_version': 1, 'task': 'spatial_research_candidate',
            'classes': list(CLASSES), 'preprocessing': CONTRACT, 'recipe': selected['recipe'],
            'mode': selected['mode'], 'manifest_sha256': contract['manifest_sha256'],
            'members': members, 'weights': [.5, .5], 'research_only': True, 'test_evaluated': False})
        loader = 'sic_xrt_ml.training.spatial_robustness.load_spatial_candidate'
        loaded = load_spatial_candidate(pointer, device)
        direct = OrientationMean(EqualProbabilityEnsemble(models), selected['mode']).to(device).eval()
    probe = torch.stack([to_tensor(cache.images[i]) for i in range(3)]).to(device)
    with torch.inference_mode():
        if not torch.allclose(loaded(probe).softmax(1), direct(probe).softmax(1), atol=1e-6):
            raise ValueError('Serialization changed predictions')
    verify_cohort(root)
    if any(file_hash(previous/f'{m}_oof.json') != h for m, h in plan['prior_oof_hashes'].items()):
        raise ValueError('Prior OOF changed')
    result = {'status': 'completed', 'comparison': comparison, 'selected': selected,
        'baseline_retained': selected['name'] == 'baseline_identity', 'serialization_verified': True,
        'candidate_pointer': str(pointer), 'candidate_sha256': file_hash(pointer), 'loader': loader,
        'completed_folds': len(completed), 'full_development_refits': 0 if selected['recipe'] is None else 2,
        'test_evaluated': False, 'independent_final_test': False, 'research_only': True,
        'expert_ground_truth': False, 'labels_changed': False, 'server_model_replaced': False}
    write_json(output/'result.json', result)
    return result
