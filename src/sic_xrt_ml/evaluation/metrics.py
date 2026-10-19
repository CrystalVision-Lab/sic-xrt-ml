"""Classification metrics: absent classes and empty evaluations stay unavailable."""

CLASSES = ["BPD", "TED", "TSD"]
BACKGROUND = "NOT_DEFECT"


def metrics(labels, predictions, classes=CLASSES):
    k = len(classes)
    if len(labels) != len(predictions) or any(
        type(i) is not int or i not in range(k) for i in labels + predictions
    ):
        raise ValueError("CLASS_LABELS_REQUIRED")
    matrix = [[0] * k for _ in range(k)]
    for y, p in zip(labels, predictions, strict=True):
        matrix[y][p] += 1
    report = []
    for i, c in enumerate(classes):
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
    result = {
        "accuracy": sum(matrix[i][i] for i in range(k)) / n if n else None,
        "balanced_accuracy": sum(r["recall"] for r in report) / k if complete else None,
        "macro_f1": sum(r["f1"] for r in report) / k if complete else None,
        "per_class": report,
        "confusion_matrix": matrix,
        "class_order": list(classes),
        "BPD_as_TED": matrix[0][1],
        "BPD_as_TSD": matrix[0][2],
        "false_BPD": sum(matrix[i][0] for i in range(1, k)),
        "samples": n,
    }
    if BACKGROUND in classes:
        b = classes.index(BACKGROUND)
        # Background accepted as a defect (false alarm) / defect dismissed as background.
        result["NOT_DEFECT_as_defect"] = sum(matrix[b][j] for j in range(k) if j != b)
        result["defect_as_NOT_DEFECT"] = sum(matrix[i][b] for i in range(k) if i != b)
    return result
