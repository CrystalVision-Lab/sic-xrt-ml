import copy

import numpy as np
import pytest
import torch

from sic_xrt_ml.training.grouped_development import NormalizedContextClassifier
from sic_xrt_ml.training.patch_classifier import (
    classification_metrics,
    file_hash,
    write_json,
)
from sic_xrt_ml.training.spatial_robustness import (
    CONTRACT,
    RECIPES,
    SpatialClassifier,
    SpatialView,
    fit_spatial,
    load_spatial_candidate,
    select_spatial,
    translate_reflect,
)


def test_shift_keeps_shape_moves_pixels_without_wrap_or_input_mutation():
    image = torch.arange(3*128*128).reshape(3, 128, 128).float()
    original = image.clone()
    shifted = translate_reflect(image, 4, -3)
    assert shifted.shape == image.shape
    assert torch.equal(shifted[:, 60, 60], image[:, 64, 57])
    assert torch.equal(shifted[:, 127, 0], image[:, 123, 3])
    assert torch.equal(image, original)
    with pytest.raises(ValueError):
        translate_reflect(image, 5, 0)


def test_eval_ignores_jitter_and_center32_reproduces_existing_model():
    class Cache:
        def __init__(self):
            self.rows = [{'label': 'TED'}]
            self.images = [np.arange(128*128*3, dtype=np.uint8).reshape(128, 128, 3)]
    assert torch.equal(SpatialView(Cache(), [0], False, 4)[0][0],
                       SpatialView(Cache(), [0], False, 0)[0][0])
    old = NormalizedContextClassifier('center32_context128').eval()
    new = SpatialClassifier(32).eval()
    new.load_state_dict(old.state_dict())
    inputs = torch.rand(2, 3, 128, 128)
    with torch.inference_mode():
        assert torch.equal(old(inputs), new(inputs))


def test_center64_uses_native64_and_same_parameter_count():
    narrow, wide = SpatialClassifier(32), SpatialClassifier(64)
    assert sum(p.numel() for p in narrow.parameters()) == sum(p.numel() for p in wide.parameters())
    shapes = []
    hook = wide.classifier.center.features.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
    assert wide(torch.rand(2, 3, 128, 128)).shape == (2, 3)
    hook.remove()
    assert shapes == [torch.Size([2, 3, 64, 64])]


def test_heldout_training_contamination_rejected_before_writing(tmp_path):
    class Cache:
        def __init__(self):
            self.rows = [{'wafer': w, 'label': c, 'split': 'development'}
                         for w in ('1', '2', '9') for c in ('BPD', 'TED', 'TSD')]
    with pytest.raises(ValueError, match='heldout'):
        fit_spatial(Cache(), list(range(9)), RECIPES[0], {}, tmp_path/'run', 42, 'cpu', '1')
    assert not (tmp_path/'run').exists()


def test_selector_retains_baseline_when_one_wafer_regresses():
    matrices = [[[5, 5, 0], [5, 95, 0], [0, 0, 100]],
                [[5, 5, 0], [2, 98, 0], [0, 0, 100]],
                [[0, 0, 0], [2, 98, 0], [0, 0, 100]]]
    def item(name, ms):
        return {'name': name, 'views': 1, 'metrics': classification_metrics(np.sum(ms, 0)),
                'by_wafer': {w: classification_metrics(np.array(m)) for w, m in zip(('1', '2', '9'), ms)}}
    base = item('baseline_identity', matrices)
    improved = copy.deepcopy(matrices)
    improved[0][1] = [1, 99, 0]
    good = item('new', improved)
    assert select_spatial([base, good]) == good
    improved[2][1] = [3, 97, 0]
    assert select_spatial([base, item('new', improved)]) == base


def test_spatial_candidate_roundtrip_and_training_provenance(tmp_path):
    members, models = [], []
    for seed in (42, 43):
        torch.manual_seed(seed)
        model = SpatialClassifier(64).eval()
        models.append(model)
        run = tmp_path/str(seed)
        run.mkdir()
        config = {'model': 'NormalizedSpatialCNN_v1', 'classes': ['BPD', 'TED', 'TSD'],
                  'preprocessing': CONTRACT, 'recipe': RECIPES[1], 'seed': seed, 'epochs': 12,
                  'sampling_power': .75, 'manifest_sha256': 'cohort',
                  'train_wafers': ['1', '2', '9'], 'heldout_wafer': None}
        torch.save({'config': config, 'epoch': 12, 'state_dict': model.state_dict()}, run/'best_model.pt')
        members.append({'seed': seed, 'run_dir': str(run), 'checkpoint_sha256': file_hash(run/'best_model.pt')})
    pointer = tmp_path/'candidate.json'
    plan = {'schema_version': 1, 'task': 'spatial_research_candidate', 'mode': 'identity',
            'classes': ['BPD', 'TED', 'TSD'], 'preprocessing': CONTRACT, 'recipe': RECIPES[1],
            'weights': [.5, .5], 'members': members, 'manifest_sha256': 'cohort'}
    write_json(pointer, plan)
    loaded = load_spatial_candidate(pointer)
    probe = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        expected = torch.stack([m(probe).softmax(1) for m in models]).mean(0)
        assert torch.allclose(loaded(probe).softmax(1), expected, atol=1e-6)
    for key, value in [('preprocessing', 'wrong'), ('manifest_sha256', 'other'), ('weights', [.8, .2])]:
        write_json(pointer, plan | {key: value})
        with pytest.raises(ValueError):
            load_spatial_candidate(pointer)
    write_json(pointer, plan)
    checkpoint_path = tmp_path/'42'/'best_model.pt'
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint['config']['heldout_wafer'] = '1'
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match='integrity'):
        load_spatial_candidate(pointer)
    plan['members'][0]['checkpoint_sha256'] = file_hash(checkpoint_path)
    write_json(pointer, plan)
    with pytest.raises(ValueError, match='provenance'):
        load_spatial_candidate(pointer)
