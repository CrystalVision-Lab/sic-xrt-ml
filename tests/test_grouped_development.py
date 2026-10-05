import numpy as np
import pytest
import torch

from sic_xrt_ml.training.grouped_development import (
    DevelopmentImages,
    DevelopmentView,
    NormalizedContextClassifier,
    load_development_model,
    normalize_local_contrast,
    sampling_weights,
    split_wafer,
)


def rows():
    return [{'wafer': w, 'label': c, 'split': 'development'}
            for w in ('1', '2', '9') for c in ('BPD', 'TED', 'TSD') if w != '9' or c != 'BPD']


def test_whole_wafer_folds_and_reserved_rejection():
    records = rows()
    for w in ('1', '2', '9'):
        train, valid = split_wafer(records, w)
        assert {records[i]['wafer'] for i in valid} == {w}
        assert w not in {records[i]['wafer'] for i in train}
        assert set(train).isdisjoint(valid) and len(train)+len(valid) == len(records)
    with pytest.raises(ValueError):
        split_wafer(records+[{'wafer': '8', 'split': 'development'}], '1')


def test_sqrt_sampling_reduces_rare_class_repetition():
    records = [{'wafer': '1', 'label': c} for c, n in [('BPD', 1), ('TED', 4), ('TSD', 9)] for _ in range(n)]
    equal, sqrt = sampling_weights(records, 1.), sampling_weights(records, .5)
    assert float(equal[0]/equal.sum()) == pytest.approx(1/3)
    assert float(sqrt[0]/sqrt.sum()) == pytest.approx(1/6)


def test_reserved_images_not_opened_and_eval_never_augments(tmp_path):
    with pytest.raises(ValueError, match='development'):
        DevelopmentImages(tmp_path, [{'wafer': '8', 'split': 'test'}])
    class Cache:
        def __init__(self):
            self.images = [np.full((128, 128, 3), 128, np.uint8)]
            self.rows = [{'label': 'TED'}]
    view = DevelopmentView(Cache(), [0], False)
    assert torch.equal(view[0][0], view[0][0]) and view[0][1] == 1


def test_local_contrast_removes_affine_intensity_and_handles_flat_input():
    torch.manual_seed(4)
    image = .2+.2*torch.rand(2, 3, 128, 128)
    before, after = normalize_local_contrast(image), normalize_local_contrast(.7*image+.1)
    assert torch.allclose(before, after, atol=1e-4)
    flat = normalize_local_contrast(torch.zeros_like(image))
    assert torch.all(flat == .5) and torch.isfinite(flat).all()


def test_normalized_checkpoint_contract_and_reload(tmp_path):
    from sic_xrt_ml.training.grouped_development import LOCAL_CONTRAST_CONTRACT
    model = NormalizedContextClassifier('center32').eval()
    config = {'model': 'NormalizedBPDContextCNN_v1', 'classes': ['BPD', 'TED', 'TSD'],
              'variant': 'center32', 'preprocessing': LOCAL_CONTRAST_CONTRACT}
    torch.save({'config': config, 'state_dict': model.state_dict()}, tmp_path/'best_model.pt')
    loaded, _ = load_development_model(tmp_path)
    probe = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        assert torch.allclose(model(probe), loaded(probe))
    config['preprocessing'] = 'wrong'
    torch.save({'config': config, 'state_dict': model.state_dict()}, tmp_path/'best_model.pt')
    with pytest.raises(ValueError, match='contract'):
        load_development_model(tmp_path)
