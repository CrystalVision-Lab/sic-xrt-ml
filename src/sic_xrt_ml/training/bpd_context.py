"""Fixed-cohort BPD experiments with train-only balanced sampling and augmentation."""

import json
import random
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .patch_classifier import (
    CLASSES,
    SmallPatchCNN,
    classification_metrics,
    dataset_info,
    file_hash,
    run_epoch,
    to_tensor,
    write_json,
)

CONTRACT = 'uint_RGB_to_float32_CHW;center32;context128_avgpool2_to64;no_display_marks'
VARIANTS = ('center32', 'center32_context128')


def average_predictions(members):
    """Equal weights fixed before evaluation; reject unaligned or invalid predictions."""
    if len(members) < 2 or not members[0]:
        raise ValueError('At least two nonempty prediction sets are required')
    identity = [(r['patch_id'], r['point_id'], r['label']) for r in members[0]]
    if len({r[0] for r in identity}) != len(identity):
        raise ValueError('Duplicate predictions')
    arrays = []
    for rows in members:
        if [(r['patch_id'], r['point_id'], r['label']) for r in rows] != identity:
            raise ValueError('Ensemble sample identities or labels differ')
        scores = np.asarray([r['scores'] for r in rows], dtype=np.float64)
        if (scores.shape != (len(identity), 3) or not np.isfinite(scores).all()
                or (scores < 0).any() or (scores > 1).any()
                or not np.allclose(scores.sum(1), 1, atol=1e-6)):
            raise ValueError('Invalid class probabilities')
        arrays.append(scores)
    means = np.mean(arrays, axis=0)
    result, matrix = [], np.zeros((3, 3), dtype=np.int64)
    for row, scores in zip(members[0], means):
        prediction = int(scores.argmax())
        matrix[CLASSES.index(row['label']), prediction] += 1
        result.append({k: row[k] for k in ('patch_id', 'point_id', 'label')} |
                      {'prediction': CLASSES[prediction], 'scores': scores.tolist()})
    return result, classification_metrics(matrix)


def sampling_weights(rows):
    counts = Counter(r['label'] for r in rows)
    if set(counts) != set(CLASSES) or any(r['split'] != 'train' for r in rows):
        raise ValueError('Balanced sampling needs all classes and training rows only')
    return torch.tensor([1/counts[r['label']] for r in rows], dtype=torch.double)


def augment_training(image):
    """Base-type classification only: subtype directions are NOT training targets."""
    image = torch.rot90(image, int(torch.randint(4, (1,))), (-2, -1))
    if bool(torch.rand(()) < .5):
        image = image.flip(-1)
    mean = image.mean(dim=(-2, -1), keepdim=True)
    contrast = .8 + .4*torch.rand(())
    brightness = -.04 + .08*torch.rand(())
    return ((image-mean)*contrast+mean+brightness).clamp(0, 1)


class CachedPatches(Dataset):
    def __init__(self, root, rows, training=False):
        if not rows or {r['split'] for r in rows} != ({'train'} if training else {'val'}):
            raise ValueError('Only explicit train/validation data is permitted')
        self.rows, self.training = rows, training
        self.images = []
        root = Path(root).resolve()
        for row in rows:
            path = (root/row['path']).resolve()
            if not path.is_relative_to(root) or file_hash(path) != row['sha256']:
                raise ValueError('Patch integrity check failed')
            image = tifffile.imread(path)
            if image.shape != (128, 128, 3) or str(image.dtype) != row['dtype']:
                raise ValueError('Expected raw RGB 128-pixel patches')
            # Immutable, hash-verified input snapshot; original files are never edited.
            image.flags.writeable = False
            self.images.append(image)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        tensor = to_tensor(self.images[index])
        if self.training:
            tensor = augment_training(tensor)
        return tensor, CLASSES.index(self.rows[index]['label'])


class ContextClassifier(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unsupported BPD experiment variant')
        self.variant = variant
        self.center = SmallPatchCNN()
        if variant == 'center32_context128':
            self.context = SmallPatchCNN().features
            self.fusion = nn.Sequential(nn.Flatten(), nn.Linear(2048, 128), nn.ReLU(),
                                        nn.Dropout(.2), nn.Linear(128, 3))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 128, 128):
            raise ValueError('Model expects unmarked RGB 128-pixel source patches')
        center = images[:, :, 48:80, 48:80]
        if self.variant == 'center32':
            return self.center(center)
        inner = self.center.features(center).flatten(1)
        outer = self.context(nn.functional.avg_pool2d(images, 2)).flatten(1)
        return self.fusion(torch.cat((inner, outer), dim=1))


def load_model(run, device='cpu'):
    checkpoint = torch.load(Path(run)/'best_model.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if (config.get('model') != 'BPDContextCNN_v1' or config.get('preprocessing') != CONTRACT
            or config.get('classes') != list(CLASSES) or config.get('variant') not in VARIANTS):
        raise ValueError('Unknown model/input contract')
    model = ContextClassifier(config['variant']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


class EqualProbabilityEnsemble(nn.Module):
    """Return log probabilities so softmax/argmax inference remains conventional."""
    def __init__(self, models):
        super().__init__()
        if len(models) < 2:
            raise ValueError('At least two models are required')
        self.members = nn.ModuleList(models)

    def forward(self, images):
        probability = torch.stack([m(images).softmax(1) for m in self.members]).mean(0)
        return probability.clamp_min(1e-12).log()


def load_ensemble(plan_path, device='cpu'):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    members = plan.get('members', [])
    if (plan.get('schema_version') != 1 or plan.get('task') != 'equal_probability_seed_ensemble'
            or len(members) < 2 or plan.get('weights') != [1/len(members)]*len(members)
            or len({m['checkpoint_sha256'] for m in members}) != len(members)):
        raise ValueError('Invalid equal-weight ensemble contract')
    models = []
    for member in members:
        if file_hash(Path(member['run_dir'])/'best_model.pt') != member['checkpoint_sha256']:
            raise ValueError('Ensemble checkpoint integrity failure')
        model, checkpoint = load_model(member['run_dir'], device)
        if checkpoint['config']['manifest_sha256'] != plan['manifest_sha256']:
            raise ValueError('Ensemble members were trained on different datasets')
        models.append(model)
    return EqualProbabilityEnsemble(models).to(device).eval()


def train_candidate(info, train_data, val_data, output, variant, seed, epochs=15, device='cuda'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    run = output/f'{variant}_seed{seed}_{uuid.uuid4().hex[:8]}'
    run.mkdir()
    config = {'model': 'BPDContextCNN_v1', 'variant': variant, 'classes': list(CLASSES),
              'preprocessing': CONTRACT, 'manifest_sha256': info['manifest_sha256'],
              'dataset_root': str(info['root']), 'seed': seed, 'epochs': epochs,
              'batch_size': 32, 'learning_rate': .001, 'device': device, 'torch': str(torch.__version__),
              'sampling': 'train_only_inverse_count_replacement_N_draws', 'loss_weighting': 'none',
              'augmentation': 'train_only_D4_contrast0.8to1.2_brightness_minus0.04to0.04',
              'target': 'base_type_only_not_direction_subtype', 'counts': info['counts'],
              'wafer_groups': info['wafer_groups'], 'research_only': True, 'test_evaluated': False}
    write_json(run/'config.json', config)
    sampler = WeightedRandomSampler(sampling_weights(info['rows']['train']), len(train_data),
                                    replacement=True, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_data, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, num_workers=0)
    model = ContextClassifier(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    history, best = [], -1.
    try:
        for epoch in range(1, epochs+1):
            started = time.perf_counter()
            training = run_epoch(model, train_loader, torch.device(device), optimizer)
            validation = run_epoch(model, val_loader, torch.device(device))
            history.append({'epoch': epoch, 'train': training, 'val': validation,
                            'seconds': time.perf_counter()-started})
            write_json(run/'history.json', history)
            if validation['macro_f1'] > best:
                best = validation['macro_f1']
                torch.save({'config': config, 'epoch': epoch, 'validation': validation,
                            'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                           run/'best_model.pt')
            print(f'{variant} seed={seed} {epoch}/{epochs} F1={validation["macro_f1"]:.4f} '
                  f'BPD={validation["confusion_matrix"][0][0]}/{validation["per_class"]["BPD"]["support"]} '
                  f'FP_BPD={sum(r[0] for r in validation["confusion_matrix"][1:])} '
                  f'{time.perf_counter()-started:.1f}s', flush=True)
        chosen = max(history, key=lambda r: r['val']['macro_f1'])
        result = {'status': 'completed', 'run_dir': str(run), 'variant': variant, 'seed': seed,
                  'selected_epoch': chosen['epoch'], 'metrics': chosen['val'],
                  'checkpoint_sha256': file_hash(run/'best_model.pt'), 'test_evaluated': False}
        write_json(run/'result.json', result)
        return result
    except BaseException as error:
        write_json(run/'FAILED.json', {'error': type(error).__name__, 'epochs_completed': len(history)})
        raise


def predict_validation(record, info, data, device='cuda'):
    model, checkpoint = load_model(record['run_dir'], device)
    if checkpoint['config']['manifest_sha256'] != info['manifest_sha256']:
        raise ValueError('Validation dataset changed')
    rows, matrix, offset = [], np.zeros((3, 3), dtype=np.int64), 0
    with torch.no_grad():
        for x, y in DataLoader(data, batch_size=32, num_workers=0):
            probabilities = model(x.to(device)).softmax(1).cpu().numpy()
            prediction = probabilities.argmax(1)
            np.add.at(matrix, (y.numpy(), prediction), 1)
            for p, probabilities_row in zip(prediction, probabilities):
                row = info['rows']['val'][offset]
                rows.append({'patch_id': row['patch_id'], 'point_id': row['point_id'], 'label': row['label'],
                             'prediction': CLASSES[int(p)], 'scores': probabilities_row.tolist()})
                offset += 1
    metrics = classification_metrics(matrix)
    if metrics['confusion_matrix'] != record['metrics']['confusion_matrix']:
        raise ValueError('Reloaded checkpoint predictions differ from saved validation')
    return rows


def compare_bpd(dataset_root, baseline_run, output_root, device='cuda'):
    info = dataset_info(Path(dataset_root), 128, False)
    baseline_run = Path(baseline_run)
    baseline = torch.load(baseline_run/'best_model.pt', map_location='cpu', weights_only=True)
    if baseline['config']['manifest_sha256'] != info['manifest_sha256']:
        raise ValueError('Baseline must use the unchanged corrected dataset')
    output = Path(output_root).resolve()/('bpd_comparison_'+uuid.uuid4().hex[:8])
    if output == info['root'] or info['root'] in output.parents:
        raise ValueError('Output must be outside the dataset')
    output.mkdir(parents=True)
    plan = {'schema_version': 1, 'task': 'fixed_validation_bpd_context_comparison',
            'manifest_sha256': info['manifest_sha256'], 'baseline_run': str(baseline_run),
            'baseline_checkpoint_sha256': file_hash(baseline_run/'best_model.pt'),
            'baseline_metrics': baseline['validation'], 'variants': list(VARIANTS),
            'epochs': 15, 'initial_seed': 42, 'confirmation_seed': 43,
            'selection': 'highest macro F1; promotion also requires better BPD recall than baseline',
            'labels_changed': False, 'test_evaluated': False, 'research_only': True,
            'limitation': 'Reused validation wafer; not independent generalization or confirmed physical labels'}
    write_json(output/'plan.json', plan)
    train_data = CachedPatches(info['root'], info['rows']['train'], training=True)
    val_data = CachedPatches(info['root'], info['rows']['val'])
    records = []
    try:
        for variant in VARIANTS:
            records.append(train_candidate(info, train_data, val_data, output, variant, 42, device=device))
            write_json(output/'progress.json', records)
        best = max(records, key=lambda r: r['metrics']['macro_f1'])
        b = baseline['validation']
        improved = best['metrics']['macro_f1'] > b['macro_f1'] and best['metrics']['per_class']['BPD']['recall'] > b['per_class']['BPD']['recall']
        if improved:
            records.append(train_candidate(info, train_data, val_data, output, best['variant'], 43, device=device))
            write_json(output/'progress.json', records)
        for r in records:
            write_json(Path(r['run_dir'])/'validation_predictions.json', predict_validation(r, info, val_data, device))
        if file_hash(info['root']/'기록/samples.csv') != plan['manifest_sha256']:
            raise ValueError('Manifest changed during experiment')
        if file_hash(baseline_run/'best_model.pt') != plan['baseline_checkpoint_sha256']:
            raise ValueError('Baseline changed during experiment')
        result = dict(plan, status='completed', output_dir=str(output), records=records,
                      improvement_observed=improved,
                      confirmation_improved=(len(records) == 3 and records[-1]['metrics']['macro_f1'] > b['macro_f1']
                          and records[-1]['metrics']['per_class']['BPD']['recall'] > b['per_class']['BPD']['recall']))
        write_json(output/'comparison.json', result)
        return result
    except BaseException as error:
        write_json(output/'FAILED.json', {'error': type(error).__name__, 'completed_runs': len(records)})
        raise
