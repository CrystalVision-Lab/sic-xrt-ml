import math
import random

import pytest

from sic_xrt_ml.evaluation.localization_audit import evaluate_bundle, evaluate_points


def point(name, x, y=0, kind="TED"):
    return {"id": name, "x": x, "y": y, "type": kind}


def test_augmenting_assignment_not_greedy_or_class_driven():
    refs = [point("r0", 0, kind="BPD"), point("r1", 3)]
    preds = [point("p0", 1, kind="BPD"), point("p1", -1)]
    report = evaluate_points(refs, preds, 2.1)
    assert report["matched_pairs"] == 2
    assert report["matched_type_agreement"] == 0
    assert report["ambiguous_pairs"] == 2
    assert report == evaluate_points(refs[::-1], preds[::-1], 2.1)


def test_extra_candidates_are_not_background_or_extra_matches():
    report = evaluate_points([point("r", 0)], [point("a", 0), point("b", 1), point("c", 50)], 2)
    assert report["matched_pairs"] == 1
    assert report["unmatched_competing_for_reference"] == 1
    assert report["unmatched_without_reference_neighbor"] == 1
    assert len(report["unmatched_prediction_ids"]) == 2
    assert "precision" not in report and "false_positives" not in report
    assert report["evaluation_kind"].endswith("not_accuracy")


def test_empty_and_unclassified_inputs():
    assert evaluate_points([], [point("a", 0)])["reference_point_coverage"] is None
    assert evaluate_points([point("r", 0)], [point("p", 0, kind="unclassified")])["matched_type_agreement"] is None
    assert evaluate_points([point("r", 0)], [])["unmatched_reference_ids"] == ["r"]


@pytest.mark.parametrize("radius", [0, -1, math.inf, math.nan])
def test_invalid_radius(radius):
    with pytest.raises(ValueError):
        evaluate_points([], [], radius)


def test_duplicate_ids_and_invalid_coordinates():
    for rows in ([point("a", 1), point("a", 2)], [point("a", math.nan)], [point(None, 1)]):
        with pytest.raises(ValueError):
            evaluate_points(rows, [])


def test_manifest_rejects_coordinate_mismatch_and_test_split():
    ref = {"image_sha256": "a" * 64, "coordinate_space": "raw_pixel_xy", "rows": []}
    bundle = {"schema": "xrt_point_audit_inputs_v1", "split": "development", "reference": ref, "prediction": dict(ref)}
    assert evaluate_bundle(bundle)["image_sha256"] == "a" * 64
    for updates in ({"image_sha256": "b" * 64}, {"coordinate_space": "display_pixels"}):
        with pytest.raises(ValueError):
            evaluate_bundle({**bundle, "prediction": {**ref, **updates}})
    with pytest.raises(ValueError):
        evaluate_bundle({**bundle, "split": "final_test"})


def test_maximum_cardinality_against_small_exhaustive_search():
    rng = random.Random(33)
    for _ in range(50):
        refs = [point(f"r{i}", rng.randrange(10), rng.randrange(10)) for i in range(5)]
        preds = [point(f"p{i}", rng.randrange(10), rng.randrange(10)) for i in range(6)]
        def brute(i, used, refs=refs, preds=preds):
            if i == len(preds):
                return 0
            best = brute(i + 1, used)
            for j, ref in enumerate(refs):
                if not used & (1 << j) and math.hypot(ref["x"] - preds[i]["x"], ref["y"] - preds[i]["y"]) <= 3:
                    best = max(best, 1 + brute(i + 1, used | (1 << j)))
            return best
        assert evaluate_points(refs, preds, 3)["matched_pairs"] == brute(0, 0)
