import copy

import numpy as np
import pytest
import torch

from sic_xrt_ml.training.grouped_development import LOCAL_CONTRAST_CONTRACT
from sic_xrt_ml.training.orientation_stability import (
    OrientationMean,
    choose,
    load_fold,
    load_orientation_candidate,
    transform,
)
from sic_xrt_ml.training.patch_classifier import (
    classification_metrics,
    file_hash,
    write_json,
)


class DirectionalModel(torch.nn.Module):
    def forward(self, x):
        return torch.stack([x[:, 0, :40, :20].mean((1, 2))*7,
                            x[:, 1, 70:, :].mean((1, 2))*3,
                            -x[:, 2, :, 60:].mean((1, 2))], 1)


@pytest.mark.parametrize('mode,indices', [('c4', range(4)), ('d4', range(8))])
def test_averaging_is_group_invariant_and_preserves_input(mode, indices):
    torch.manual_seed(123)
    probe = torch.rand(2, 3, 128, 128)
    original = probe.clone()
    model = OrientationMean(DirectionalModel(), mode)
    expected = model(probe).softmax(1)
    for i in indices:
        assert torch.allclose(expected, model(transform(probe, i)).softmax(1), atol=1e-6)
    assert torch.equal(probe, original)
    manual = torch.stack([model.model(transform(probe, i)).softmax(1) for i in indices]).mean(0)
    assert torch.allclose(expected, manual, atol=1e-6)


def test_orientation_validation_and_identity_equivalence():
    probe = torch.rand(2, 3, 128, 128)
    model = DirectionalModel()
    assert torch.allclose(model(probe).softmax(1), OrientationMean(model, 'identity')(probe).softmax(1))
    with pytest.raises(ValueError):
        OrientationMean(model, 'unknown')
    with pytest.raises(ValueError):
        transform(probe, 8)
    with pytest.raises(ValueError):
        OrientationMean(model, 'd4')(probe[:, :, :64])
    with pytest.raises(ValueError, match='forbidden'):
        load_fold(None, None, '8', None, 'cpu')


def item(mode, matrices):
    return {'mode': mode, 'metrics': classification_metrics(np.sum(matrices, axis=0)),
            'by_wafer': {w: classification_metrics(np.array(m)) for w, m in zip(('1', '2', '9'), matrices)}}


def test_selection_prevents_aggregate_improvement_hiding_wafer_regression():
    matrices = [[[5, 5, 0], [5, 95, 0], [0, 0, 100]],
                [[5, 5, 0], [2, 98, 0], [0, 0, 100]],
                [[0, 0, 0], [2, 98, 0], [0, 0, 100]]]
    base = item('identity', matrices)
    improved = copy.deepcopy(matrices)
    improved[0][1] = [1, 99, 0]
    good = item('d4', improved)
    assert choose([base, good]) == good
    lost = copy.deepcopy(improved)
    lost[1][0] = [4, 6, 0]
    assert choose([base, item('c4', lost)]) == base
    extra_fp = copy.deepcopy(improved)
    extra_fp[2][1] = [3, 97, 0]
    assert choose([base, item('c4', extra_fp)]) == base
    assert choose([base, good, item('c4', improved)])['mode'] == 'c4'


def test_loader_checks_parent_hash_contract_and_dataset(tmp_path, monkeypatch):
    parent = tmp_path/'parent.json'
    write_json(parent, {'manifest_sha256': 'dataset'})
    plan = {'schema_version': 1, 'task': 'orientation_mean_research_candidate', 'mode': 'd4',
            'classes': ['BPD', 'TED', 'TSD'], 'preprocessing': LOCAL_CONTRAST_CONTRACT,
            'target': 'base_type_only', 'transform_order': 'rot90_k_then_optional_horizontal_flip',
            'weights': [.125]*8, 'parent_pointer': str(parent), 'parent_sha256': file_hash(parent),
            'manifest_sha256': 'dataset'}
    pointer = tmp_path/'candidate.json'
    write_json(pointer, plan)
    monkeypatch.setattr('sic_xrt_ml.training.orientation_stability.load_candidate',
                        lambda path, device: DirectionalModel())
    assert load_orientation_candidate(pointer).mode == 'd4'
    for key, value in [('weights', [.25]*4), ('manifest_sha256', 'other'),
                       ('target', 'TED_subtypes'), ('parent_sha256', 'wrong')]:
        write_json(pointer, plan | {key: value})
        with pytest.raises(ValueError):
            load_orientation_candidate(pointer)
