import copy

import numpy as np
import pytest
import tifffile
import torch

from sic_xrt_ml.training.background_abstention import (
    BackgroundContextCNN,
    checked_patch,
    digest,
    evaluate,
    validate_rows,
)


def fixture_rows():
    positive = [{'patch_id': w, 'wafer': w, 'label': 'BPD'} for w in ('1', '2', '9')]
    weak = [{'id': 'bg'+w, 'wafer': w, 'decision': 'background_candidate',
             'weak_training_eligible': True, 'review_actor': 'Codex_AI',
             'human_verified': False, 'expert_ground_truth': False} for w in ('1', '2', '9')]
    baseline = [{**r, 'prediction': 'BPD'} for r in positive]
    return positive, weak, baseline


def test_cohort_guard_rejects_holdout_and_truth_laundering():
    positive, weak, baseline = fixture_rows()
    assert len(validate_rows(positive, weak, baseline)[0]) == 3
    for collection, key, value in ((positive, 'wafer', '8'), (weak, 'wafer', '8'),
                                   (weak, 'human_verified', True), (weak, 'expert_ground_truth', True),
                                   (weak, 'weak_training_eligible', False), (baseline, 'label', 'TED')):
        old = copy.deepcopy(collection[0])
        collection[0][key] = value
        with pytest.raises(ValueError):
            validate_rows(positive, weak, baseline)
        collection[0] = old


def test_patch_integrity_and_path_containment(tmp_path):
    root = tmp_path/'data'
    root.mkdir()
    path = root/'patch.tif'
    tifffile.imwrite(path, np.zeros((128, 128, 3), np.uint8), photometric='rgb')
    row = {'path': 'patch.tif', 'sha256': digest(path)}
    assert checked_patch(root, row).shape == (128, 128, 3)
    with pytest.raises(ValueError):
        checked_patch(root, {**row, 'sha256': 'invalid'})
    with pytest.raises(ValueError):
        checked_patch(root, {**row, 'path': '../outside.tif'})


def test_background_rejection_cannot_mask_lost_defects():
    rows = [{'patch_id': 'bpd', 'label': 'BPD'}, {'patch_id': 'bg', 'label': 'BACKGROUND'}]
    baseline = {'bpd': {'prediction': 'BPD'}}
    result = evaluate(rows, [[.009, .001, 0, .99], [.001, .001, .001, .997]], baseline)
    assert result['weak_background_rejection_rate'] == 1
    assert not result['passed']
    assert not result['checks']['positive_retention_each_type']
    assert result['abstention']['accepted_count'] == 0
    assert result['positive_rejection_by_type']['TED'] is None


def test_bpd_type_regression_fails_gate_even_when_background_good():
    rows = [{'patch_id': 'bpd', 'label': 'BPD'}, {'patch_id': 'bg', 'label': 'BACKGROUND'}]
    result = evaluate(rows, [[.01, .98, .01, 0], [.001, .001, .001, .997]], {'bpd': {'prediction': 'BPD'}})
    assert not result['checks']['bpd_tp_non_decrease']
    assert result['abstention']['accepted_provider_agreement'] == 0
    assert not result['passed']


def test_explicit_probabilities_and_small_cpu_training_step():
    with pytest.raises(ValueError):
        evaluate([{'label': 'BPD'}], [[float('nan'), 0, 0, 0]], {})
    torch.set_num_threads(2)
    model = BackgroundContextCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    inputs = torch.rand(2, 3, 128, 128)
    logits = model(inputs)
    assert logits.shape == (2, 4) and torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3])).backward()
    optimizer.step()
