from copy import deepcopy

import pytest

from sic_xrt_ml.training.coordinate_repair_comparison import validate_repair_pair


def test_repair_rejects_label_changes_and_test_changes():
    row = {'patch_id': 'p', 'point_id': 'q', 'source_id': 's', 'wafer': '2', 'split': 'val',
           'label': 'TSD', 'provider_fine_label': 'TSD_a', 'size': '128', 'dtype': 'uint8', 'x': '10'}
    updated = dict(row, x='90')
    old = {'rows': {'train': [], 'test': [], 'val': [row]}}
    new = {'rows': {'train': [], 'test': [], 'val': [updated]}}
    repair = {'task': 'explicit_horizontal_mirror_coordinate_repair', 'status': 'completed',
              'original_type_labels_changed': False, 'cohort_changed': False, 'reserved_test_modified': False,
              'selected_patch_count': 1, 'selected_patch_changes': [{'patch_id': 'p', 'old': row, 'new': updated}]}
    validate_repair_pair(old, new, repair)
    bad = deepcopy(new)
    bad['rows']['val'][0]['label'] = 'TED'
    with pytest.raises(ValueError):
        validate_repair_pair(old, bad, repair)
    bad = deepcopy(new)
    bad['rows']['test'] = [dict(updated, split='test')]
    with pytest.raises(ValueError):
        validate_repair_pair(old, bad, repair)
