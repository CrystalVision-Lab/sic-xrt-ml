import json

import pytest
import torch

from sic_xrt_ml.training.independent_background import (
    IndependentBackgroundCNN,
    metrics,
    reviewed_candidates,
    triage,
)


def test_triage_is_separate_and_preserves_ambiguous_points():
    assert triage(.01) == 'background_suspect'
    assert triage(.5) == 'uncertain'
    assert triage(.95) == 'defect_candidate'
    for value in (-1, 2, float('nan')):
        with pytest.raises(ValueError):
            triage(value)


def test_weak_background_success_cannot_hide_bpd_loss():
    rows = [{'stratum': k} for k in ('BPD', 'TED', 'TSD', 'background_hard')]
    result = metrics(rows, [.01, .9, .9, .01])
    assert result['checks']['weak_background_rejection']
    assert not result['checks']['positive_retention_each_type']
    assert not result['passed']


def test_possible_defect_not_a_negative_training_label(tmp_path):
    row = {'id': 'one', 'wafer': '1', 'human_verified': False, 'expert_ground_truth': False,
           'review_actor': 'Codex_AI', 'decision': 'possible_defect', 'training_eligible': False}
    path = tmp_path/'observations.json'
    def write():
        path.write_text(json.dumps({'schema': 'xrt_detection_observations_v1', 'test_evaluated': False,
                                    'candidates': [row]}), encoding='utf-8')
    write()
    assert reviewed_candidates(path)[0]['decision'] == 'possible_defect'
    row['training_eligible'] = True
    write()
    with pytest.raises(ValueError):
        reviewed_candidates(path)
    row['training_eligible'], row['wafer'] = False, '8'
    write()
    with pytest.raises(ValueError):
        reviewed_candidates(path)


def test_possible_defect_preservation_is_a_separate_gate():
    rows = [{'stratum': k} for k in ('BPD', 'background_hard', 'possible_defect')]
    result = metrics(rows, [.9, .01, .01])
    assert not result['checks']['possible_defect_retention']
    assert not result['passed']


def test_binary_model_shape_and_training():
    torch.set_num_threads(2)
    model = IndependentBackgroundCNN()
    logits = model(torch.rand(2, 3, 128, 128))
    assert logits.shape == (2,) and torch.isfinite(logits).all()
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
    assert any(p.grad is not None for p in model.parameters())
