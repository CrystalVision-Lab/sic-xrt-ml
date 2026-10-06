import numpy as np
import pytest

from sic_xrt_ml.evaluation.background_diagnostics import distribution, retention_bound


def test_equal_scores_cannot_separate_background_from_defects():
    rows = [{'stratum':'TED'}]*100 + [{'stratum':'background_random'}]*20
    bound = retention_bound(rows, [.2]*120)
    assert bound['diagnostic_only']
    assert not bound['all_weak_strata_can_reach_70_percent']
    assert bound['strata']['background_random']['removed'] == 0


def test_boundary_ties_do_not_violate_retention():
    rows = [{'stratum':'TED'}]*100 + [{'stratum':'background_random'}]*10
    scores = [.1,.1]+[.9]*98+[.01]*10
    bound = retention_bound(rows,scores)
    assert bound['strata']['TED']['removed'] == 0
    assert bound['strata']['background_random']['removed'] == 10
    assert bound['all_weak_strata_can_reach_70_percent']


def test_small_bpd_support_and_possible_defects_protected():
    rows = [{'stratum':'TED'}]*1000 + [{'stratum':'BPD'}]*10 + [{'stratum':'possible_defect'}]*10 + [{'stratum':'background_hard'}]*10
    scores = [.9]*1000+[.03]+[.9]*9+[.02]+[.9]*9+[.025]*10
    bound = retention_bound(rows,scores)
    assert bound['strata']['BPD']['removed'] == 0
    assert bound['strata']['possible_defect']['removed'] == 0
    assert bound['strata']['background_hard']['removed'] == 0


def test_invalid_inputs_rejected():
    for scores in ([float('nan')], [-.1], [1.1], []):
        with pytest.raises(ValueError):
            retention_bound([{'stratum':'TED'}],scores)
    with pytest.raises(ValueError, match='positives'):
        retention_bound([{'stratum':'background_random'}],[.5])


def test_score_quantiles_and_fixed_threshold_counts():
    result = distribution([{'stratum':'background_random'}]*3,[.01,.02,.5])['background_random']
    assert result['background_at_095'] == 2
    assert np.isclose(result['defect_probability_quantiles_0_05_50_95_100'][2],.02)
