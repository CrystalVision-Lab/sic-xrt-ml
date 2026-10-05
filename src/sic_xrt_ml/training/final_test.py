"""One-shot held-out evaluation of a frozen research ensemble; no model selection."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .bpd_context import CONTRACT, load_ensemble
from .patch_classifier import (
    CLASSES,
    PatchDataset,
    classification_metrics,
    dataset_info,
    file_hash,
    write_json,
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def verify_audit(path, manifest_sha256, train_count):
    path = Path(path).resolve()
    audit = json.loads(path.read_text(encoding='utf-8'))
    if (audit.get('schema') != 'pretest_source_audit' or audit.get('schema_version') != 1
            or audit.get('status') not in ('passed', 'passed_with_limitations')
            or audit.get('dataset_manifest_sha256') != manifest_sha256
            or audit.get('training_patches_checked') != train_count
            or audit.get('source_pixel_and_roi_checks_passed') is not True
            or audit.get('research_test_permitted') is not True
            or audit.get('test_evaluated') is not False
            or audit.get('labels_changed') is not False
            or audit.get('coordinates_changed') is not False
            or not audit.get('artifacts')):
        raise ValueError('Pretest audit contract mismatch')
    for relative, expected in audit['artifacts'].items():
        artifact = (path.parent/relative).resolve()
        if not artifact.is_relative_to(path.parent) or file_hash(artifact) != expected:
            raise ValueError('Pretest audit artifact hash/path mismatch')
    return audit


def test_cohort(rows, expected_wafer):
    if (not rows or {r['split'] for r in rows} != {'test'}
            or {str(r['wafer']) for r in rows} != {str(expected_wafer)}
            or {r['label'] for r in rows} != set(CLASSES)
            or len({r['patch_id'] for r in rows}) != len(rows)):
        raise ValueError('Expected unique held-out rows from the reserved wafer')
    keys = ('patch_id', 'point_id', 'source_id', 'wafer', 'split', 'label', 'sha256')
    return sorted(({k: r[k] for k in keys} for r in rows), key=lambda r: r['patch_id'])


test_cohort.__test__ = False


def claim_once(ledger_root, cohort_sha256, record):
    """Exclusive receipt is intentionally retained even after a failed attempt."""
    ledger_root = Path(ledger_root)
    ledger_root.mkdir(parents=True, exist_ok=True)
    if len(cohort_sha256) != 64 or any(c not in '0123456789abcdef' for c in cohort_sha256):
        raise ValueError('Invalid cohort hash')
    receipt = ledger_root/f'{cohort_sha256}.json'
    try:
        with receipt.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
    except FileExistsError as error:
        raise ValueError('This test cohort already has an evaluation receipt; do not retune/retest') from error
    return receipt


def evaluate_frozen_ensemble(dataset_root, ensemble_path, audit_path, output,
                             expected_wafer='8', batch_size=32, device='cuda'):
    dataset_root, ensemble_path, audit_path, output = (
        Path(p).resolve() for p in (dataset_root, ensemble_path, audit_path, output))
    if batch_size < 1 or output.exists() or output.is_relative_to(dataset_root):
        raise ValueError('Positive batch size and new output outside dataset required')
    info = dataset_info(dataset_root, 128, balanced=False)
    verify_audit(audit_path, info['manifest_sha256'], len(info['rows']['train']))
    ensemble = json.loads(ensemble_path.read_text(encoding='utf-8'))
    if ensemble.get('manifest_sha256') != info['manifest_sha256']:
        raise ValueError('Frozen ensemble dataset mismatch')
    cohort = test_cohort(info['rows']['test'], expected_wafer)
    model = load_ensemble(ensemble_path, device).eval()
    protected = json.loads((dataset_root/'output_hashes.json').read_text(encoding='utf-8'))

    def verify_dataset():
        for relative, expected in protected.items():
            path = (dataset_root/relative).resolve()
            if not path.is_relative_to(dataset_root) or file_hash(path) != expected:
                raise ValueError('Dataset integrity changed')

    verify_dataset()
    data = PatchDataset(dataset_root, info['rows']['test'], 128)
    # Preflight before claiming the one-shot receipt: no forward passes or scoring.
    for i in range(len(data)):
        inputs, _ = data[i]
        if inputs.shape != (3, 128, 128) or not torch.isfinite(inputs).all():
            raise ValueError('Invalid test input contract')
    plan = {'schema_version': 1, 'task': 'frozen_ensemble_heldout_wafer_test',
            'frozen_at': datetime.now(UTC).isoformat(), 'manifest_sha256': info['manifest_sha256'],
            'audit_sha256': file_hash(audit_path), 'ensemble_plan_sha256': file_hash(ensemble_path),
            'ensemble_plan': ensemble, 'test_cohort_sha256': canonical_hash(cohort),
            'test_cohort': cohort, 'counts': info['counts'], 'wafer_groups': info['wafer_groups'],
            'model_contract': CONTRACT, 'classes': list(CLASSES), 'augmentation': 'none',
            'decision': 'argmax_of_equal_mean_probabilities', 'batch_size': batch_size,
            'device': device, 'torch': str(torch.__version__), 'research_only': True,
            'test_used_for_model_selection': False, 'output_dir': str(output)}
    # A shared sibling ledger prevents changing output directories to repeat this cohort.
    receipt = claim_once(dataset_root.parent/'final_test_ledger', plan['test_cohort_sha256'],
                         {'status': 'claimed_before_inference', 'frozen_at': plan['frozen_at'],
                          'plan_canonical_sha256': canonical_hash(plan), 'output_dir': str(output)})
    try:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'frozen_plan.json', plan)
        matrix, predictions = np.zeros((3, 3), dtype=np.int64), []
        with torch.inference_mode():
            for inputs, targets in DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0):
                probabilities = model(inputs.to(device)).softmax(1).cpu().numpy()
                if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-6):
                    raise ValueError('Non-finite or invalid model probabilities')
                start = len(predictions)
                for offset, (scores, actual) in enumerate(zip(probabilities, targets.tolist())):
                    row = info['rows']['test'][start+offset]
                    predicted = int(scores.argmax())
                    matrix[actual, predicted] += 1
                    predictions.append({k: row[k] for k in ('patch_id', 'point_id', 'wafer', 'label')} |
                                       {'prediction': CLASSES[predicted], 'scores': scores.tolist()})
        write_json(output/'test_predictions.json', predictions)
        verify_dataset()
        for member in ensemble['members']:
            if file_hash(Path(member['run_dir'])/'best_model.pt') != member['checkpoint_sha256']:
                raise ValueError('Checkpoint changed during evaluation')
        result = {'schema_version': 1, 'status': 'completed', 'test_evaluated': True,
                  'completed_at': datetime.now(UTC).isoformat(), 'metrics': classification_metrics(matrix),
                  'samples': len(predictions), 'wafer': str(expected_wafer),
                  'frozen_plan_sha256': file_hash(output/'frozen_plan.json'),
                  'predictions_sha256': file_hash(output/'test_predictions.json'),
                  'dataset_files_verified_unchanged': len(protected), 'research_only': True,
                  'test_used_for_model_selection': False, 'expert_semantic_confirmation': False,
                  'limitations': ['provider_AI_research_labels', 'single_reserved_wafer',
                                  'selected_clean_cohort_only', 'not_detection_or_subtype_evaluation'],
                  'output_dir': str(output)}
        write_json(output/'test_result.json', result)
        write_json(receipt, {'status': 'completed', 'output_dir': str(output),
                             'result_sha256': file_hash(output/'test_result.json'),
                             'frozen_plan_sha256': result['frozen_plan_sha256']})
        return result
    except BaseException as error:
        write_json(receipt, {'status': 'failed_do_not_automatically_repeat',
                             'output_dir': str(output), 'error': type(error).__name__})
        raise
