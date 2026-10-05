import json

import numpy as np
import pytest
import tifffile
import torch

from sic_xrt_ml.training import final_test
from sic_xrt_ml.training.final_test import (
    canonical_hash,
    claim_once,
    test_cohort,
    verify_audit,
)
from sic_xrt_ml.training.patch_classifier import file_hash


def rows():
    return [{'patch_id': c, 'point_id': c, 'source_id': 's', 'wafer': '8',
             'split': 'test', 'label': c, 'sha256': c} for c in ('BPD', 'TED', 'TSD')]


def test_reserved_cohort_rejects_mixed_splits_and_wafer():
    sample = rows()
    assert canonical_hash(test_cohort(sample, '8')) == canonical_hash(test_cohort(sample[::-1], '8'))
    with pytest.raises(ValueError):
        test_cohort(sample+[sample[0]], '8')
    sample[0]['split'] = 'val'
    with pytest.raises(ValueError):
        test_cohort(sample, '8')
    sample[0]['split'], sample[0]['wafer'] = 'test', '2'
    with pytest.raises(ValueError):
        test_cohort(sample, '8')


def test_receipt_blocks_repeat_even_with_new_output(tmp_path):
    key = canonical_hash(test_cohort(rows(), '8'))
    claim_once(tmp_path, key, {'output': 'first', 'status': 'started'})
    with pytest.raises(ValueError, match='already'):
        claim_once(tmp_path, key, {'output': 'second'})
    assert json.loads((tmp_path/f'{key}.json').read_text())['output'] == 'first'


def test_audit_requires_matching_manifest_and_intact_evidence(tmp_path):
    evidence = tmp_path/'evidence.json'
    evidence.write_text('{}')
    audit = {'schema': 'pretest_source_audit', 'schema_version': 1, 'status': 'passed_with_limitations',
             'dataset_manifest_sha256': 'sha', 'training_patches_checked': 3,
             'source_pixel_and_roi_checks_passed': True, 'research_test_permitted': True,
             'test_evaluated': False, 'labels_changed': False, 'coordinates_changed': False,
             'artifacts': {'evidence.json': file_hash(evidence)}}
    path = tmp_path/'audit.json'
    path.write_text(json.dumps(audit))
    assert verify_audit(path, 'sha', 3)['status'] == 'passed_with_limitations'
    with pytest.raises(ValueError, match='contract'):
        verify_audit(path, 'wrong', 3)
    evidence.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='artifact'):
        verify_audit(path, 'sha', 3)


def test_end_to_end_freezes_before_forward_and_preserves_inputs(tmp_path, monkeypatch):
    root = tmp_path/'dataset'
    root.mkdir()
    samples = rows()
    hashes = {}
    for i, row in enumerate(samples):
        path = root/f'{i}.tif'
        tifffile.imwrite(path, np.full((128, 128, 3), i*100, np.uint8), photometric='rgb')
        row.update(path=path.name, dtype='uint8', sha256=file_hash(path))
        hashes[path.name] = row['sha256']
    (root/'output_hashes.json').write_text(json.dumps(hashes))
    checkpoint = tmp_path/'best_model.pt'
    checkpoint.write_bytes(b'synthetic checkpoint')
    ensemble = tmp_path/'ensemble.json'
    ensemble.write_text(json.dumps({'manifest_sha256': 'sha',
        'members': [{'run_dir': str(tmp_path), 'checkpoint_sha256': file_hash(checkpoint)}]}))
    audit = tmp_path/'audit.json'
    audit.write_text('{}')
    info = {'rows': {'train': [1, 2, 3], 'test': samples}, 'manifest_sha256': 'sha',
            'counts': {'test': {c: 1 for c in ('BPD', 'TED', 'TSD')}},
            'wafer_groups': {'train': ['1'], 'val': ['2'], 'test': ['8']}}
    monkeypatch.setattr(final_test, 'dataset_info', lambda *a, **kw: info)
    monkeypatch.setattr(final_test, 'verify_audit', lambda *a: {})
    output = tmp_path/'result'
    calls = []

    class Fake(torch.nn.Module):
        def forward(self, inputs):
            assert (output/'frozen_plan.json').exists()
            assert len(list((tmp_path/'final_test_ledger').glob('*.json'))) == 1
            calls.append(len(inputs))
            indices = (inputs[:, 0, 0, 0]*255/100).round().long()
            return torch.nn.functional.one_hot(indices, 3).float()*10

    monkeypatch.setattr(final_test, 'load_ensemble', lambda *a: Fake())
    result = final_test.evaluate_frozen_ensemble(root, ensemble, audit, output, device='cpu')
    assert calls == [3] and result['metrics']['accuracy'] == 1
    assert result['samples'] == 3 and result['dataset_files_verified_unchanged'] == 3
    assert all(file_hash(root/p) == h for p, h in hashes.items())
    with pytest.raises(ValueError, match='already'):
        final_test.evaluate_frozen_ensemble(root, ensemble, audit, tmp_path/'second', device='cpu')
    assert calls == [3]
