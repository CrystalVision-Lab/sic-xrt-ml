"""Classification metrics: absent classes and empty evaluations stay unavailable."""

CLASSES = ["BPD", "TED", "TSD"]


def metrics(labels, predictions):
    if len(labels) != len(predictions) or any(
        type(i) is not int or i not in range(3) for i in labels + predictions
    ):
        raise ValueError("THREE_CLASS_LABELS_REQUIRED")
    matrix = [[0] * 3 for _ in range(3)]
    for y, p in zip(labels, predictions, strict=True):
        matrix[y][p] += 1
    report = []
    for i, c in enumerate(CLASSES):
        support = sum(matrix[i])
        positive = sum(row[i] for row in matrix)
        tp = matrix[i][i]
        recall = tp / support if support else None
        precision = tp / positive if positive else (0.0 if support else None)
        f1 = (
            (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            if support
            else None
        )
        report.append(
            {
                "class": c,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    complete = all(r["support"] for r in report)
    n = len(labels)
    return {
        "accuracy": sum(matrix[i][i] for i in range(3)) / n if n else None,
        "balanced_accuracy": sum(r["recall"] for r in report) / 3 if complete else None,
        "macro_f1": sum(r["f1"] for r in report) / 3 if complete else None,
        "per_class": report,
        "confusion_matrix": matrix,
        "class_order": CLASSES,
        "BPD_as_TED": matrix[0][1],
        "BPD_as_TSD": matrix[0][2],
        "false_BPD": matrix[1][0] + matrix[2][0],
        "samples": n,
    }
