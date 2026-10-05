import copy

import numpy as np
import pytest
import torch

from sic_xrt_ml.training.grouped_development import (
    LOCAL_CONTRAST_CONTRACT,
    NormalizedContextClassifier,
    sampling_weights,
)
from sic_xrt_ml.training.patch_classifier import (
    classification_metrics,
    file_hash,
    write_json,
)
from sic_xrt_ml.training.recall_budget import (
    BASELINE,
    load_candidate,
    select_candidate,
    validate_oof,
)


def candidate(name, matrix):
    return {'name': name, 'members': [name], 'pooled_OOF': classification_metrics(np.array(matrix))}


def test_selection_requires_recall_gain_without_more_false_positives_or_lower_macro():
    base = candidate(BASELINE, [[5, 5, 0], [2, 98, 0], [1, 0, 99]])
    too_many_fp = candidate('fp', [[9, 1, 0], [4, 96, 0], [0, 0, 100]])
    bad_macro = candidate('macro', [[9, 1, 0], [0, 0, 100], [0, 100, 0]])
    no_gain = candidate('tie', [[5, 5, 0], [0, 100, 0], [0, 0, 100]])
    wrong_support = candidate('support', [[6, 5, 0], [0, 100, 0], [0, 0, 100]])
    assert select_candidate([base, too_many_fp, bad_macro, no_gain, wrong_support]) == base
    better = candidate('better', [[7, 3, 0], [1, 99, 0], [1, 0, 99]])
    assert select_candidate([base, too_many_fp, bad_macro, better]) == better


def test_oof_alignment_rejects_wrong_wafer_duplicate_and_inconsistent_scores():
    rows = [{'patch_id': 'a', 'point_id': 'p', 'wafer': '1', 'label': 'BPD'}]
    valid = [rows[0] | {'scores': [.6, .3, .1], 'prediction': 'BPD'}]
    assert validate_oof(valid, rows) == valid
    for invalid in ([valid[0] | {'wafer': '2'}], valid*2,
                    [valid[0] | {'scores': [.1, .8, .1]}],
                    [valid[0] | {'scores': [float('nan'), .3, .1]}]):
        with pytest.raises(ValueError):
            validate_oof(invalid, rows)


def test_intermediate_sampling_mass_between_sqrt_and_equal():
    rows = [{'wafer': '1', 'label': c} for c, n in [('BPD', 1), ('TED', 4), ('TSD', 16)] for _ in range(n)]
    probabilities = []
    for power in (.5, .75, 1.):
        weights = sampling_weights(rows, power)
        probabilities.append(float(weights[0]/weights.sum()))
    assert probabilities[0] < probabilities[1] < probabilities[2]
    assert probabilities[1] == pytest.approx(1/(1+4**.25+16**.25))


def make_pointer(tmp_path):
    members, models = [], []
    for seed in (42, 43):
        torch.manual_seed(seed)
        model = NormalizedContextClassifier('center32').eval()
        run = tmp_path/str(seed)
        run.mkdir()
        recipe = {'name': str(seed), 'variant': 'center32', 'normalization': 'local_contrast_v1'}
        config = {'model': 'NormalizedBPDContextCNN_v1', 'classes': ['BPD', 'TED', 'TSD'],
                  'variant': 'center32', 'preprocessing': LOCAL_CONTRAST_CONTRACT,
                  'manifest_sha256': 'cohort', 'train_wafers': ['1', '2', '9'],
                  'heldout_wafer': None, 'seed': seed, 'recipe': recipe, 'epochs': 12}
        torch.save({'config': config, 'epoch': 12, 'state_dict': model.state_dict()}, run/'best_model.pt')
        members.append({'run_dir': str(run), 'checkpoint_sha256': file_hash(run/'best_model.pt'),
                        'seed': seed, 'recipe': recipe})
        models.append(model)
    plan = {'schema_version': 1, 'task': 'normalized_recall_budget_candidate',
            'classes': ['BPD', 'TED', 'TSD'], 'preprocessing': LOCAL_CONTRAST_CONTRACT,
            'manifest_sha256': 'cohort', 'epochs': 12, 'members': members, 'weights': [.5, .5]}
    pointer = tmp_path/'candidate.json'
    write_json(pointer, plan)
    return pointer, plan, models


def test_candidate_reload_averages_probabilities_and_detects_corruption(tmp_path):
    pointer, plan, models = make_pointer(tmp_path)
    loaded = load_candidate(pointer)
    probe = torch.rand(2, 3, 128, 128)
    with torch.inference_mode():
        expected = torch.stack([m(probe).softmax(1) for m in models]).mean(0)
        assert torch.allclose(loaded(probe).softmax(1), expected, atol=1e-6)
    corrupt = copy.deepcopy(plan)
    corrupt['members'][0]['checkpoint_sha256'] = 'changed'
    write_json(pointer, corrupt)
    with pytest.raises(ValueError, match='integrity'):
        load_candidate(pointer)


@pytest.mark.parametrize('key,value', [('manifest_sha256', 'other'), ('epochs', 11),
                                     ('preprocessing', 'wrong'), ('weights', [.7, .3])])
def test_candidate_rejects_incompatible_metadata(tmp_path, key, value):
    pointer, plan, _ = make_pointer(tmp_path)
    plan[key] = value
    write_json(pointer, plan)
    with pytest.raises(ValueError, match='contract'):
        load_candidate(pointer)


def test_candidate_rejects_heldout_fold_even_with_matching_hash(tmp_path):
    pointer, plan, _ = make_pointer(tmp_path)
    path = tmp_path/'42'/'best_model.pt'
    checkpoint = torch.load(path, weights_only=True)
    checkpoint['config']['train_wafers'] = ['2', '9']
    checkpoint['config']['heldout_wafer'] = '1'
    torch.save(checkpoint, path)
    plan['members'][0]['checkpoint_sha256'] = file_hash(path)
    write_json(pointer, plan)
    with pytest.raises(ValueError, match='contract'):
        load_candidate(pointer)
