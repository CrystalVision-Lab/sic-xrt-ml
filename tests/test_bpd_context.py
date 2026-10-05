import numpy as np
import pytest
import tifffile
import torch

from sic_xrt_ml.training.bpd_context import (
    CachedPatches,
    ContextClassifier,
    EqualProbabilityEnsemble,
    augment_training,
    average_predictions,
    load_ensemble,
    sampling_weights,
)
from sic_xrt_ml.training.patch_classifier import file_hash


def test_sampler_equalizes_class_mass_and_rejects_validation():
    rows = [{'label': c, 'split': 'train'} for c, n in [('BPD', 2), ('TED', 5), ('TSD', 11)] for _ in range(n)]
    weights = sampling_weights(rows)
    for c in ('BPD', 'TED', 'TSD'):
        assert sum(float(w) for r, w in zip(rows, weights) if r['label'] == c) == pytest.approx(1)
    rows[0]['split'] = 'val'
    with pytest.raises(ValueError):
        sampling_weights(rows)


def test_validation_never_augments_and_rejects_test_split(tmp_path):
    image = np.arange(128*128*3, dtype=np.uint8).reshape(128, 128, 3)
    path = tmp_path/'p.tif'
    tifffile.imwrite(path, image, photometric='rgb')
    row = {'path': 'p.tif', 'sha256': file_hash(path), 'dtype': 'uint8', 'label': 'BPD', 'split': 'val'}
    data = CachedPatches(tmp_path, [row])
    a, y = data[0]
    assert y == 0 and torch.equal(a, data[0][0])
    assert np.allclose(a.permute(1, 2, 0).numpy(), image/255)
    with pytest.raises(ValueError):
        CachedPatches(tmp_path, [dict(row, split='test')])


def test_augmentation_preserves_input_and_stays_finite():
    x = torch.rand(3, 128, 128)
    old = x.clone()
    y = augment_training(x)
    assert torch.equal(x, old)
    assert y.shape == x.shape and torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1


def test_context_observes_outside_center_but_control_does_not():
    x = torch.rand(2, 3, 128, 128)
    changed = x.clone()
    changed[:, :, :40] = 0
    torch.manual_seed(4)
    control = ContextClassifier('center32').eval()
    model = ContextClassifier('center32_context128').eval()
    with torch.no_grad():
        assert torch.equal(control(x), control(changed))
        assert not torch.equal(model(x), model(changed))
        assert model(x).shape == (2, 3)
    with pytest.raises(ValueError):
        model(x[:, :, :32, :32])


def test_ensemble_uses_equal_probabilities_and_checks_identity():
    a = [{'patch_id': 'p', 'point_id': 'q', 'label': 'BPD', 'scores': [.8, .1, .1]}]
    b = [{'patch_id': 'p', 'point_id': 'q', 'label': 'BPD', 'scores': [.4, .5, .1]}]
    rows, metrics = average_predictions([a, b])
    assert rows[0]['scores'] == pytest.approx([.6, .3, .1])
    assert metrics['confusion_matrix'][0][0] == 1
    with pytest.raises(ValueError, match='identities'):
        average_predictions([a, [dict(b[0], patch_id='other')]])
    with pytest.raises(ValueError, match='probabilities'):
        average_predictions([a, [dict(b[0], scores=[float('nan'), .5, .5])]])


def test_ensemble_inference_averages_probabilities_and_rejects_bad_weight(tmp_path):
    import json

    class Fixed(torch.nn.Module):
        def __init__(self, scores):
            super().__init__()
            self.register_buffer('scores', torch.tensor(scores))

        def forward(self, x):
            return self.scores.log()[None].expand(len(x), -1)

    model = EqualProbabilityEnsemble([Fixed([.8, .1, .1]), Fixed([.4, .5, .1])])
    p = model(torch.zeros(2, 3, 128, 128)).softmax(1)
    assert torch.allclose(p, torch.tensor([[.6, .3, .1], [.6, .3, .1]]))
    path = tmp_path/'ensemble.json'
    path.write_text(json.dumps({'schema_version': 1, 'task': 'equal_probability_seed_ensemble',
                               'members': [{'checkpoint_sha256': 'a'}, {'checkpoint_sha256': 'b'}],
                               'weights': [.6, .4]}), encoding='utf-8')
    with pytest.raises(ValueError, match='contract'):
        load_ensemble(path)
