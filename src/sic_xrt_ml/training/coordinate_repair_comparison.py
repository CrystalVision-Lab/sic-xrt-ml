"""Separate data-coordinate corrections from improvements due to model training."""

import json
import uuid
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .patch_classifier import (
    PatchDataset,
    SmallPatchCNN,
    checkpoint_crop,
    dataset_info,
    file_hash,
    plot_history,
    run_epoch,
    train,
    write_json,
)


def validate_repair_pair(original, repaired, repair):
    if (repair.get('task') != 'explicit_horizontal_mirror_coordinate_repair'
            or repair.get('status') != 'completed'
            or repair.get('original_type_labels_changed') is not False
            or repair.get('cohort_changed') is not False
            or repair.get('reserved_test_modified') is not False):
        raise ValueError('Expected a completed coordinate-only repair contract')
    changes = {c['patch_id']: c for c in repair['selected_patch_changes']}
    if len(changes) != repair['selected_patch_count']:
        raise ValueError('Duplicate/missing repair changes')
    observed = set()
    for split in ('train', 'val', 'test'):
        before = {r['patch_id']: r for r in original['rows'][split]}
        after = {r['patch_id']: r for r in repaired['rows'][split]}
        if before.keys() != after.keys():
            raise ValueError('Evaluation cohort changed')
        for key, old in before.items():
            new = after[key]
            if key not in changes:
                if old != new:
                    raise ValueError('Undeclared dataset change')
                continue
            if split != 'val':
                raise ValueError('This comparison requires unchanged train and test rows')
            change = changes[key]
            if old != change['old'] or new != change['new']:
                raise ValueError('Repair contract differs from manifests')
            for field in ('point_id', 'source_id', 'wafer', 'split', 'label', 'provider_fine_label', 'size', 'dtype'):
                if old[field] != new[field]:
                    raise ValueError('Coordinates-only comparison cannot change labels or identities')
            observed.add(key)
    if observed != changes.keys():
        raise ValueError('Unmatched repair record')


def evaluate_validation(run, info, device):
    checkpoint = torch.load(run/'best_model.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    crop = checkpoint_crop(config)
    model = SmallPatchCNN().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    loader = DataLoader(PatchDataset(info['root'], info['rows']['val'], config['size'], crop),
                        batch_size=config['batch_size'], num_workers=0)
    metrics = run_epoch(model, loader, torch.device(device))
    return {'run_dir': str(run), 'checkpoint_sha256': file_hash(run/'best_model.pt'),
            'selected_epoch': checkpoint['epoch'], 'input_size': crop or config['size'],
            'metrics': metrics, 'evaluation_manifest_sha256': info['manifest_sha256'],
            'split': 'val', 'test_evaluated': False}


def compare_repair(original_root, repaired_root, baseline_runs, output_root, *, retrain_crops=(None, 32), device='cuda'):
    old = dataset_info(Path(original_root), 128, False)
    new = dataset_info(Path(repaired_root), 128, False)
    repair_file = new['root']/'coordinate_repair.json'
    repair = json.loads(repair_file.read_text(encoding='utf-8'))
    validate_repair_pair(old, new, repair)
    output = Path(output_root).resolve()/('coordinate_comparison_'+uuid.uuid4().hex[:8])
    if any(root == output or root in output.parents for root in (old['root'], new['root'])):
        raise ValueError('Outputs must be outside both datasets')
    output.mkdir(parents=True, exist_ok=False)
    plan = {'schema_version': 1, 'task': 'coordinate_repair_model_comparison', 'research_only': True,
            'original_manifest_sha256': old['manifest_sha256'], 'repaired_manifest_sha256': new['manifest_sha256'],
            'repair_contract_sha256': file_hash(repair_file), 'corrected_validation_patches': repair['selected_patch_count'],
            'retrain_input_sizes': [c or 128 for c in retrain_crops], 'test_evaluated': False,
            'interpretation': 'Input repair changes the validation observations; not a claim of model learning improvement.',
            'generalization_limit': 'Reused validation wafer; original quality selection predates coordinate repair.'}
    write_json(output/'plan.json', plan)
    records = []
    reference = None
    try:
        for path in baseline_runs:
            run = Path(path).resolve()
            config = json.loads((run/'config.json').read_text(encoding='utf-8'))
            if (config['manifest_sha256'] != old['manifest_sha256'] or config['balanced_train']
                    or config['counts'] != old['counts'] or config['wafer_groups'] != old['wafer_groups']):
                raise ValueError('Baseline training data differs from original dataset')
            if reference is None:
                reference = config
            record = {'kind': 'same_checkpoint_input_repair',
                      'original': evaluate_validation(run, old, device),
                      'repaired': evaluate_validation(run, new, device)}
            if record['original']['checkpoint_sha256'] != record['repaired']['checkpoint_sha256']:
                raise ValueError('Checkpoint changed during paired evaluation')
            records.append(record)
            write_json(output/'progress.json', records)
            print(json.dumps({'kind': record['kind'], 'input_size': record['repaired']['input_size'],
                  'original_macro_f1': record['original']['metrics']['macro_f1'],
                  'repaired_macro_f1': record['repaired']['metrics']['macro_f1'],
                  'repaired_TSD_recall': record['repaired']['metrics']['per_class']['TSD']['recall']}), flush=True)
        if reference is None:
            raise ValueError('At least one baseline is required')
        for crop in retrain_crops:
            result = train(new['root'], output/'runs', epochs=reference['epochs'],
                           batch_size=reference['batch_size'], learning_rate=reference['learning_rate'],
                           size=128, balanced=False, seed=reference['seed'], device=device,
                           center_crop=crop, class_weighting=reference.get('class_weighting', 'none'))
            run = Path(result['run_dir'])
            plot_history(run)
            records.append({'kind': 'retrained_with_repaired_validation', 'repaired': evaluate_validation(run, new, device)})
            write_json(output/'progress.json', records)
        for info in (old, new):
            if file_hash(info['root']/'기록/samples.csv') != info['manifest_sha256']:
                raise ValueError('Dataset manifest changed during comparison')
        if file_hash(repair_file) != plan['repair_contract_sha256']:
            raise ValueError('Repair contract changed during comparison')
        result = dict(plan, status='completed', output_dir=str(output), records=records)
        write_json(output/'comparison.json', result)
        return result
    except BaseException as error:
        write_json(output/'FAILED.json', {'error_type': type(error).__name__, 'records_completed': len(records)})
        raise
