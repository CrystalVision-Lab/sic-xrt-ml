"""Development-point correspondence, not certified detection accuracy.

Consumes exported original-pixel points; no Analyzer or Data Tools imports.
Unmatched predictions remain unreviewed, never automatic background labels.
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

CLASSES = ("BPD", "TED", "TSD")


def _points(rows):
    result, seen = [], set()
    for row in rows:
        ident = row["id"]
        x, y = float(row["x"]), float(row["y"])
        if not isinstance(ident, str) or not ident or ident in seen or not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Point IDs must be unique/nonempty and coordinates finite")
        seen.add(ident)
        result.append({**row, "id": ident, "x": x, "y": y})
    return sorted(result, key=lambda row: row["id"])


def evaluate_points(references, predictions, radius=20):
    """Maximum-cardinality one-to-one spatial matching, independent of class.

    Deterministic ID ordering with distance-ordered neighbors; not a minimum-
    total-distance assignment. Ambiguous neighborhoods are reported explicitly.
    Labels may be unverified and annotations incomplete, so no precision/F1.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    refs, preds = _points(references), _points(predictions)
    grid = defaultdict(list)
    for j, ref in enumerate(refs):
        grid[(math.floor(ref["x"] / radius), math.floor(ref["y"] / radius))].append(j)
    edges, reverse_degree = [], Counter()
    for pred in preds:
        gx, gy = math.floor(pred["x"] / radius), math.floor(pred["y"] / radius)
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    distance = math.hypot(pred["x"] - refs[j]["x"], pred["y"] - refs[j]["y"])
                    if distance <= radius:
                        near.append((distance, j))
        near.sort()
        edges.append([j for _, j in near])
        reverse_degree.update(j for _, j in near)
    # Iterative augmenting paths avoid recursion limits in crowded fields.
    ref_match, pred_match = {}, {}
    for start in range(len(preds)):
        queue, visited_pred, parent_ref, free_ref = deque([start]), {start}, {}, None
        while queue and free_ref is None:
            p = queue.popleft()
            for r in edges[p]:
                if r in parent_ref:
                    continue
                parent_ref[r] = p
                if r not in ref_match:
                    free_ref = r
                    break
                other = ref_match[r]
                if other not in visited_pred:
                    visited_pred.add(other)
                    queue.append(other)
        while free_ref is not None:
            p = parent_ref[free_ref]
            previous = pred_match.get(p)
            ref_match[free_ref], pred_match[p] = p, free_ref
            free_ref = previous
    matches, confusion, class_total, class_matched = [], Counter(), Counter(), Counter()
    for r in refs:
        class_total[r.get("type", "unlabeled")] += 1
    for p, r in sorted(pred_match.items()):
        pred, ref = preds[p], refs[r]
        rt, pt = ref.get("type", "unlabeled"), pred.get("type", "unclassified")
        class_matched[rt] += 1
        confusion[(rt, pt)] += 1
        matches.append({"prediction_id": pred["id"], "reference_id": ref["id"],
                        "distance_px": math.hypot(pred["x"] - ref["x"], pred["y"] - ref["y"]),
                        "reference_type": rt, "predicted_type": pt,
                        "ambiguous": len(edges[p]) > 1 or reverse_degree[r] > 1})
    types = sorted({t for pair in confusion for t in pair})
    typed = sum(n for (r, p), n in confusion.items() if r in CLASSES and p in CLASSES)
    agreed = sum(confusion[(kind, kind)] for kind in CLASSES)
    return {
        "schema": "xrt_point_audit_v1", "radius_px": radius,
        "evaluation_kind": "unverified_reference_correspondence_not_accuracy",
        "assignment": "maximum_cardinality_distance_ordered_augmenting_paths",
        "reference_count": len(refs), "prediction_count": len(preds), "matched_pairs": len(matches),
        "reference_point_coverage": len(matches) / len(refs) if refs else None,
        "unmatched_reference_ids": [r["id"] for j, r in enumerate(refs) if j not in ref_match],
        "unmatched_prediction_ids": [p["id"] for i, p in enumerate(preds) if i not in pred_match],
        "unmatched_without_reference_neighbor": sum(not near for near in edges),
        "unmatched_competing_for_reference": sum(bool(edges[i]) for i in range(len(preds)) if i not in pred_match),
        "ambiguous_pairs": sum(m["ambiguous"] for m in matches),
        "coverage_by_reference_type": {t: {"matched": class_matched[t], "total": n,
                                             "coverage": class_matched[t] / n} for t, n in sorted(class_total.items())},
        "matched_type_comparison_count": typed,
        "matched_type_agreement": agreed / typed if typed else None,
        "confusion_rows_reference_columns_prediction": {r: {p: confusion[(r, p)] for p in types} for r in types},
        "matches": matches,
        "limitations": ["Unmatched predictions are unreviewed, not proven false positives or background.",
                        "Reference point coverage is not recall of all physical defects.",
                        "Provider type agreement is not expert-verified or independent test accuracy.",
                        "Ambiguous assignments can change type agreement; matching does not use labels."],
    }


def evaluate_bundle(bundle, radius=20):
    if bundle.get("schema") != "xrt_point_audit_inputs_v1" or bundle.get("split") != "development":
        raise ValueError("Only explicitly declared development bundles are accepted")
    ref, pred = bundle["reference"], bundle["prediction"]
    digest = ref.get("image_sha256", "")
    if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            or digest != pred.get("image_sha256")
            or ref.get("coordinate_space") != "raw_pixel_xy"
            or pred.get("coordinate_space") != "raw_pixel_xy"):
        raise ValueError("Image hash and original-pixel coordinate space must match")
    result = evaluate_points(ref["rows"], pred["rows"], radius)
    result["image_sha256"] = digest
    result["reference_provenance"] = ref.get("provenance", "unspecified")
    result["prediction_provenance"] = pred.get("provenance", "unspecified")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--radius", type=float, default=20)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = evaluate_bundle(json.loads(raw), args.radius)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    # Never overwrite previous audit artifacts.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
