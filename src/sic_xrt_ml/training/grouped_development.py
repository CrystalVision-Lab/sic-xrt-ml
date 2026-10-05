"""Fixed-epoch, leave-one-wafer-out development experiments; no reserved test use."""

import json
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .bpd_context import CONTRACT, ContextClassifier, augment_training, load_model
from .patch_classifier import (
    CLASSES,
    classification_metrics,
    file_hash,
    read_csv,
    run_epoch,
    to_tensor,
    write_json,
)

WAFERS = ('1', '2', '9')
RECIPES = (
    {'name': 'center_equal', 'variant': 'center32', 'sampling_power': 1.0},
    {'name': 'center_sqrt', 'variant': 'center32', 'sampling_power': .5},
    {'name': 'context_sqrt', 'variant': 'center32_context128', 'sampling_power': .5},
)
LOCAL_CONTRAST_CONTRACT = CONTRACT + ';gray_mean;reflect_box15_highpass;rms_floor1over255;0.5plus0.15z_clip01'


def normalize_local_contrast(images):
    gray = images.mean(dim=1, keepdim=True)
    background = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(gray, (7, 7, 7, 7), mode='reflect'), 15, stride=1)
    residual = gray-background
    scale = residual.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1/255)
    return (.5+.15*residual/scale).clamp(0, 1).repeat(1, 3, 1, 1)


class NormalizedContextClassifier(torch.nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.classifier = ContextClassifier(variant)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 128, 128):
            raise ValueError('Expected RGB 128-pixel input')
        return self.classifier(normalize_local_contrast(images))


def load_development_model(run, device='cpu'):
    checkpoint = torch.load(Path(run)/'best_model.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if config.get('model') == 'BPDContextCNN_v1':
        return load_model(run, device)
    if (config.get('model') != 'NormalizedBPDContextCNN_v1'
            or config.get('preprocessing') != LOCAL_CONTRAST_CONTRACT
            or config.get('classes') != list(CLASSES)):
        raise ValueError('Unknown development model/input contract')
    model = NormalizedContextClassifier(config['variant']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), checkpoint


def verify_cohort(root):
    root = Path(root).resolve()
    contract = json.loads((root/'cohort.json').read_text(encoding='utf-8'))
    if (contract.get('schema') != 'wafer_grouped_development_cohort' or contract.get('schema_version') != 1
            or contract.get('status') != 'completed' or (root/'BUILD_FAILED.json').exists()
            or contract.get('classes') != list(CLASSES) or contract.get('allowed_wafers') != list(WAFERS)
            or contract.get('forbidden_wafers') != ['8']
            or contract.get('manifest_sha256') != file_hash(root/'samples.csv')):
        raise ValueError('Invalid development cohort contract')
    rows = read_csv(root/'samples.csv')
    if (len(rows) != contract['samples'] or len({r['patch_id'] for r in rows}) != len(rows)
            or {r['wafer'] for r in rows} != set(WAFERS)
            or any(r['split'] != 'development' or r['label'] not in CLASSES for r in rows)):
        raise ValueError('Invalid development rows or forbidden wafer')
    for key in ('source_id', 'point_id', 'sha256'):
        groups = {}
        for r in rows:
            if r[key] in groups and groups[r[key]] != r['wafer']:
                raise ValueError('Same source/point/patch occurs across wafers')
            groups[r[key]] = r['wafer']
    protected = json.loads((root/'output_hashes.json').read_text(encoding='utf-8'))
    for relative, expected in protected.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected:
            raise ValueError('Cohort file integrity failure')
    return contract, rows


def split_wafer(rows, heldout):
    if heldout not in WAFERS or any(r['wafer'] not in WAFERS or r['split'] != 'development' for r in rows):
        raise ValueError('Reserved or unknown wafer is forbidden')
    train = [i for i, r in enumerate(rows) if r['wafer'] != heldout]
    valid = [i for i, r in enumerate(rows) if r['wafer'] == heldout]
    if not valid or {rows[i]['label'] for i in train} != set(CLASSES):
        raise ValueError('Each training fold requires all three classes')
    return train, valid


def sampling_weights(rows, power):
    if power not in (.5, 1.) or not rows or any(r['wafer'] not in WAFERS for r in rows):
        raise ValueError('Invalid development sampler')
    counts = Counter(r['label'] for r in rows)
    if set(counts) != set(CLASSES):
        raise ValueError('Missing training class')
    return torch.tensor([counts[r['label']]**(-power) for r in rows], dtype=torch.double)


class DevelopmentImages:
    def __init__(self, root, rows):
        self.rows, self.images = rows, []
        root = Path(root).resolve()
        for row in rows:
            if row['wafer'] not in WAFERS or row['split'] != 'development':
                raise ValueError('Only development wafers may be loaded')
            path = (root/row['path']).resolve()
            if not path.is_relative_to(root) or file_hash(path) != row['sha256']:
                raise ValueError('Patch hash/path mismatch')
            image = tifffile.imread(path)
            if image.shape != (128, 128, 3) or image.dtype != np.uint8:
                raise ValueError('Expected uint8 RGB 128-pixel patches')
            image.flags.writeable = False
            self.images.append(image)


class DevelopmentView(Dataset):
    def __init__(self, cache, indices, training):
        self.cache, self.indices, self.training = cache, indices, training

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        tensor = to_tensor(self.cache.images[i])
        return (augment_training(tensor) if self.training else tensor,
                CLASSES.index(self.cache.rows[i]['label']))


def predictions(model, cache, indices, device):
    result, matrix = [], np.zeros((3, 3), dtype=np.int64)
    loader = DataLoader(DevelopmentView(cache, indices, False), batch_size=32, shuffle=False)
    model.eval()
    with torch.inference_mode():
        for inputs, targets in loader:
            scores = model(inputs.to(device)).softmax(1).cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError('Invalid predictions')
            start = len(result)
            for j, (probability, actual) in enumerate(zip(scores, targets.tolist())):
                row = cache.rows[indices[start+j]]
                predicted = int(probability.argmax())
                matrix[actual, predicted] += 1
                result.append({k: row[k] for k in ('point_id', 'patch_id', 'wafer', 'label')} |
                              {'prediction': CLASSES[predicted], 'scores': probability.tolist()})
    return result, classification_metrics(matrix)


def fit(cache, indices, recipe, contract, run, epochs, seed, device, heldout):
    run.mkdir()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_rows = [cache.rows[i] for i in indices]
    sampler = WeightedRandomSampler(sampling_weights(train_rows, recipe['sampling_power']),
                                    len(indices), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(DevelopmentView(cache, indices, True), sampler=sampler, batch_size=32)
    normalized = recipe.get('normalization') == 'local_contrast_v1'
    if recipe.get('normalization') not in (None, 'local_contrast_v1'):
        raise ValueError('Unknown normalization')
    model = (NormalizedContextClassifier(recipe['variant']) if normalized
             else ContextClassifier(recipe['variant'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    config = {'model': 'NormalizedBPDContextCNN_v1' if normalized else 'BPDContextCNN_v1',
              'classes': list(CLASSES), 'preprocessing': LOCAL_CONTRAST_CONTRACT if normalized else CONTRACT,
              'variant': recipe['variant'], 'manifest_sha256': contract['manifest_sha256'],
              'task': 'grouped_development_fixed_epoch', 'recipe': recipe, 'seed': seed,
              'epochs': epochs, 'epoch_selection': 'fixed_last_epoch_no_heldout_selection',
              'train_wafers': sorted({r['wafer'] for r in train_rows}), 'heldout_wafer': heldout,
              'training_counts': dict(Counter(r['label'] for r in train_rows)),
              'augmentation': 'train_only_D4_contrast0.8to1.2_brightness_minus0.04to0.04',
              'batch_size': 32, 'learning_rate': .001, 'research_only': True, 'test_evaluated': False}
    write_json(run/'config.json', config)
    history = []
    for epoch in range(1, epochs+1):
        started = time.perf_counter()
        metrics = run_epoch(model, loader, torch.device(device), optimizer)
        history.append({'epoch': epoch, 'training': metrics, 'seconds': time.perf_counter()-started})
        write_json(run/'history.json', history)
        print(f'{recipe["name"]} heldout={heldout} epoch {epoch}/{epochs} '
              f'train F1={metrics["macro_f1"]:.3f} {history[-1]["seconds"]:.1f}s', flush=True)
    torch.save({'config': config, 'epoch': epochs,
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}, run/'best_model.pt')
    # Filename is retained for shared loader compatibility; epoch was not selected on validation.
    return model


def run_comparison(root, output, epochs=12, seed=42, device='cuda'):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(root) or epochs < 1:
        raise ValueError('New output outside dataset and positive epochs required')
    contract, rows = verify_cohort(root)
    cache = DevelopmentImages(root, rows)
    output.mkdir(parents=True)
    plan = {'schema_version': 1, 'task': 'leave_one_wafer_out_development',
            'created_at': datetime.now(UTC).isoformat(), 'dataset_root': str(root),
            'manifest_sha256': contract['manifest_sha256'], 'recipes': list(RECIPES),
            'epochs': epochs, 'seed': seed, 'wafer_groups': list(WAFERS), 'forbidden_wafers': ['8'],
            'checkpoint_selection': 'fixed_last_epoch', 'recipe_selection': 'maximum_pooled_OOF_macro_F1',
            'independent_final_test': False, 'test_evaluated': False, 'research_only': True}
    write_json(output/'plan.json', plan)
    results = []
    for recipe in RECIPES:
        folds, oof = [], []
        for heldout in WAFERS:
            train, valid = split_wafer(rows, heldout)
            run = output/f'{recipe["name"]}_holdout{heldout}'
            model = fit(cache, train, recipe, contract, run, epochs, seed, device, heldout)
            pred, metrics = predictions(model, cache, valid, device)
            write_json(run/'heldout_predictions.json', pred)
            folds.append({'wafer': heldout, 'metrics': metrics, 'checkpoint_sha256': file_hash(run/'best_model.pt')})
            oof.extend(pred)
            write_json(output/'progress.json', {'recipe': recipe, 'completed_folds': folds, 'test_evaluated': False})
            print(f'FOLD DONE {recipe["name"]} W{heldout}: {json.dumps(metrics)}', flush=True)
            del model
        if len(oof) != len(rows) or {p['patch_id'] for p in oof} != {r['patch_id'] for r in rows}:
            raise ValueError('OOF coverage failure')
        matrix = np.sum([f['metrics']['confusion_matrix'] for f in folds], axis=0)
        item = {'recipe': recipe, 'folds': folds, 'pooled_OOF': classification_metrics(matrix)}
        results.append(item)
        write_json(output/f'{recipe["name"]}_oof.json', oof)
        write_json(output/'comparison.json', results)
    selected = max(results, key=lambda r: r['pooled_OOF']['macro_f1'])
    final_run = output/'development_model'
    model = fit(cache, list(range(len(rows))), selected['recipe'], contract, final_run,
                epochs, seed, device, heldout=None)
    reloaded, _ = load_model(final_run, device)
    probe = torch.stack([to_tensor(cache.images[i]) for i in range(3)]).to(device)
    model.eval()
    with torch.inference_mode():
        if not torch.allclose(model(probe), reloaded(probe), atol=1e-6):
            raise ValueError('Reloaded checkpoint differs')
    verify_cohort(root)
    result = {'status': 'completed', 'output_dir': str(output), 'comparison': results,
              'selected_recipe': selected['recipe'], 'selected_OOF': selected['pooled_OOF'],
              'model_run': str(final_run), 'checkpoint_sha256': file_hash(final_run/'best_model.pt'),
              'serialization_verified': True, 'test_evaluated': False, 'independent_final_test': False,
              'research_only': True, 'expert_ground_truth': False,
              'limitations': ['W9_has_no_BPD', 'BPD_in_two_wafers_only', 'one_seed',
                              'OOF_used_for_recipe_selection', 'provider_AI_research_labels']}
    write_json(output/'result.json', result)
    return result
