"""Coarse (BPD/TED/TSD) and fine (class + direction subtype) metrics from fine scores."""

import numpy as np

from ..data.provider_subtype import CLASSES, FINE_LABELS, FINE_TO_CLASS, RARE_SUPPORT
from .metrics import metrics

COARSE_ONLY_KEYS = ("BPD_as_TED", "BPD_as_TSD", "false_BPD")


def coarse_scores(fine):
    """[N,10] fine probabilities -> [N,3] class probabilities (group sums)."""
    fine = np.asarray(fine, dtype=np.float64)
    return np.stack(
        [fine[:, [i for i, c in enumerate(FINE_TO_CLASS) if c == k]].sum(1) for k in range(3)],
        axis=1,
    )


def present_macro(result):
    """Macro averages over labels with support > 0 (absent labels make the plain ones None)."""
    present = [r for r in result["per_class"] if r["support"]]
    return {
        "labels": [r["class"] for r in present],
        "balanced_accuracy": sum(r["recall"] for r in present) / len(present)
        if present
        else None,
        "macro_f1": sum(r["f1"] for r in present) / len(present) if present else None,
    }


def fine_metrics(labels, predictions):
    result = metrics(labels, predictions, FINE_LABELS)
    for key in COARSE_ONLY_KEYS:
        result.pop(key)
    result["present_label_macro"] = present_macro(result)
    result["rare_labels_support_under_30"] = [
        r["class"] for r in result["per_class"] if 0 < r["support"] < RARE_SUPPORT
    ]
    result["absent_labels"] = [r["class"] for r in result["per_class"] if not r["support"]]
    return result


def within_class_subtype(labels, fine):
    """Subtype accuracy among samples whose coarse class is predicted correctly.

    The subtype is the best fine label inside the true (= predicted) class group, so
    this isolates direction-subtype skill from coarse-class errors. BPD has no subtype.
    """
    fine = np.asarray(fine)
    coarse_pred = coarse_scores(fine).argmax(1)
    result = {}
    for k, name in enumerate(CLASSES):
        group = [i for i, c in enumerate(FINE_TO_CLASS) if c == k]
        if len(group) < 2:
            continue
        rows = [
            n
            for n, y in enumerate(labels)
            if FINE_TO_CLASS[y] == k and coarse_pred[n] == k
        ]
        matrix = [[0] * len(group) for _ in group]
        for n in rows:
            p = group[int(fine[n, group].argmax())]
            matrix[group.index(labels[n])][group.index(p)] += 1
        correct = sum(matrix[i][i] for i in range(len(group)))
        per = []
        for i, g in enumerate(group):
            support = sum(matrix[i])
            per.append(
                {
                    "label": FINE_LABELS[g],
                    "support": support,
                    "recall": matrix[i][i] / support if support else None,
                }
            )
        result[name] = {
            "samples_with_correct_class": len(rows),
            "class_samples": sum(1 for y in labels if FINE_TO_CLASS[y] == k),
            "subtype_accuracy": correct / len(rows) if rows else None,
            "label_order": [FINE_LABELS[g] for g in group],
            "confusion_matrix": matrix,
            "per_subtype": per,
        }
    total = sum(v["samples_with_correct_class"] for v in result.values())
    hits = sum(
        sum(v["confusion_matrix"][i][i] for i in range(len(v["label_order"])))
        for v in result.values()
    )
    return {
        "subtype_accuracy_given_correct_class": hits / total if total else None,
        "samples": total,
        "by_class": result,
    }


def summarize(labels, fine):
    """All candidate metrics for fine integer labels and [N,10] fine probabilities."""
    fine = np.asarray(fine)
    coarse_true = [FINE_TO_CLASS[y] for y in labels]
    coarse_pred = coarse_scores(fine).argmax(1).tolist() if len(labels) else []
    fine_pred = fine.argmax(1).tolist() if len(labels) else []
    return {
        "coarse": metrics(coarse_true, coarse_pred, CLASSES),
        "coarse_rule": "argmax of summed fine probabilities (= ONNX probabilities output)",
        "fine": fine_metrics(list(labels), fine_pred),
        "within_class": within_class_subtype(list(labels), fine),
        "fine_argmax_class_disagrees_with_coarse_argmax": sum(
            FINE_TO_CLASS[f] != c for f, c in zip(fine_pred, coarse_pred, strict=True)
        ),
    }


def _coarse_stats(y, p, k=3):
    matrix = np.zeros((k, k))
    np.add.at(matrix, (y, p), 1)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    tp = np.diag(matrix)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(k), where=denom > 0)
    return tp.sum() / len(y), recall.mean(), f1.mean()


def paired_bootstrap(labels, candidate, baseline, resamples=2000, seed=2026):
    """Percentile 95% intervals of candidate - baseline coarse metric differences.

    Resamples patches, which are not independent (neighbouring points, one Area), so
    the intervals are optimistic; they describe test-set sampling noise only.
    """
    y, a, b = (np.asarray(v) for v in (labels, candidate, baseline))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(resamples):
        i = rng.integers(0, len(y), len(y))
        diffs.append(np.subtract(_coarse_stats(y[i], a[i]), _coarse_stats(y[i], b[i])))
    diffs = np.asarray(diffs)
    point = np.subtract(_coarse_stats(y, a), _coarse_stats(y, b))
    names = ("accuracy", "balanced_accuracy", "macro_f1")
    return {
        "resamples": resamples,
        "seed": seed,
        "unit": "patch (not independent; intervals are optimistic)",
        **{
            n: {
                "difference": float(point[j]),
                "ci95": [
                    float(np.percentile(diffs[:, j], 2.5)),
                    float(np.percentile(diffs[:, j], 97.5)),
                ],
            }
            for j, n in enumerate(names)
        },
    }
