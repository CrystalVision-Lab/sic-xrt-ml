"""Explicit test evaluation and Area cross-validation for provider subtype runs.

The test split is read only here, once per checkpoint (``final-evaluation.json`` lock,
as in ``compare.evaluate``). Cross-validation folds hold out one Area each; the fold's
validation Area is chosen from the remaining Areas only.
"""

import hashlib
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..data import provider_subtype as ps
from ..data.verified import SPLITS, canonical, digest
from ..training.subtype import (
    device_pixels,
    load_checkpoint,
    predict,
    selected_model,
    train_subtype,
)
from .compare import frozen_session
from .metrics import metrics
from .subtype_metrics import coarse_scores, paired_bootstrap, present_macro, summarize


def baseline_scores(bundle, pixels, chosen, batch_size=64):
    """Coarse class indices of an existing 3-class frozen bundle on the same patches."""
    session, manifest = frozen_session(bundle)
    if manifest["classes"] != ps.CLASSES:
        raise ValueError("BASELINE_CLASS_ORDER")
    predictions = []
    index = torch.as_tensor(chosen, dtype=torch.long)
    for start in range(0, len(index), batch_size):
        x = pixels[index[start : start + batch_size].to(pixels.device)].cpu()
        x = x.float().div_(255).numpy()
        scores = session.run(["probabilities"], {"images": x})[0]
        if (
            scores.shape != (len(x), 3)
            or not np.isfinite(scores).all()
            or (scores < 0).any()
            or not np.allclose(scores.sum(1), 1, atol=1e-4)
        ):
            raise ValueError("BASELINE_SCORES")
        predictions.extend(scores.argmax(1).tolist())
    return predictions, manifest


def evaluate_subtype(
    root,
    checkpoint_path,
    output,
    existing_bundle=None,
    synthetic_smoke=False,
    data=None,
    device="cpu",
):
    checkpoint = load_checkpoint(checkpoint_path)
    data = ps.validate(root, load_pixels=True) if data is None else data
    assignment = ps.split_assignment(data, checkpoint["split_assignment"])
    ready = ps.readiness(
        data,
        assignment,
        synthetic_smoke,
        checkpoint["minimum_per_class"],
        checkpoint.get("readiness_mode") == "cross_validation_fold",
    )
    if not ready["ready"]:
        raise ValueError("DATA_NOT_READY")
    if (
        checkpoint["dataset_hash"] != data.dataset_hash
        or checkpoint["synthetic_smoke"] != synthetic_smoke
        or checkpoint["fine_labels"] != ps.FINE_LABELS
    ):
        raise ValueError("CHECKPOINT_DATASET")
    lock = Path(checkpoint_path).parent / "final-evaluation.json"
    if lock.exists():
        raise ValueError("FINAL_TEST_ALREADY_USED")
    output = Path(output)
    if output.exists():
        raise ValueError("FINAL_EVALUATION_ALREADY_RECORDED")
    output.mkdir(parents=True, mode=0o700)
    checkpoint_sha = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(
            canonical(
                {
                    "dataset_hash": data.dataset_hash,
                    "checkpoint_sha256": checkpoint_sha,
                    "state": "TEST_ACCESS_STARTED",
                }
            ).decode("utf-8")
        )
    torch.set_num_threads(2)
    device = torch.device(device)
    model = selected_model(checkpoint).to(device)
    pixels = device_pixels(data, device)
    test = ps.indices(data, assignment, "test")
    labels = ps.labels(data, test)
    fine = predict(model, pixels, test, device)
    candidate = summarize(labels, fine)
    coarse_true = [ps.FINE_TO_CLASS[y] for y in labels]
    coarse_pred = [int(v) for v in np.argmax(coarse_scores(fine), 1)]
    old, old_manifest = (
        baseline_scores(existing_bundle, pixels, test) if existing_bundle else (None, None)
    )
    result = {
        "status": "SYNTHETIC_FUNCTIONAL_EVALUATION"
        if synthetic_smoke
        else "INDEPENDENT_EVALUATION_COMPLETED",
        "task": "provider_subtype",
        "dataset_hash": data.dataset_hash,
        "dataset": data.provenance(),
        "classes": ps.CLASSES,
        "fine_labels": ps.FINE_LABELS,
        "split_assignment": assignment,
        "test_areas": ready["areas_by_split"]["test"],
        "test_sample_hash": digest([(data.rows[i]["patch_id"], data.rows[i]["sha256"]) for i in test]),
        "checkpoint_sha256": checkpoint_sha,
        "architecture": checkpoint["architecture"],
        "candidate": candidate,
        "baseline": metrics(coarse_true, old, ps.CLASSES) if old is not None else None,
        "baseline_model_sha256": old_manifest["model_sha256"] if old_manifest else None,
        "baseline_bundle_id": old_manifest.get("bundle_id") if old_manifest else None,
        "candidate_minus_baseline": paired_bootstrap(coarse_true, coarse_pred, old)
        if old is not None
        else None,
        "baseline_note": "baseline training data may overlap these Areas (provider labels); "
        "comparison is indicative only"
        if old is not None
        else None,
        "labels_source": "provider",
        "expert_verified": False,
        "improvement": "NOT_VERIFIED",
        "scientific_metrics": not synthetic_smoke,
        "same_inputs_and_labels": old is not None,
        "selection_policy": "validation only; explicit final test invocation",
    }
    (output / "evaluation.json").write_bytes(canonical(result))
    (output / "predictions.json").write_bytes(
        canonical(
            [
                {
                    "patch_id": data.rows[i]["patch_id"],
                    "label": ps.FINE_LABELS[y],
                    "candidate_fine": ps.FINE_LABELS[int(np.argmax(f))],
                    "candidate_class": ps.CLASSES[c],
                    "baseline_class": ps.CLASSES[old[n]] if old is not None else None,
                }
                for n, (i, y, f, c) in enumerate(zip(test, labels, fine, coarse_pred, strict=True))
            ]
        )
    )
    return result


def fold_plan(data):
    """Leave-one-Area-out folds: validation = remaining Area with fewest patches."""
    counts = {}
    for row in data.rows:
        counts[row["area"]] = counts.get(row["area"], 0) + 1
    areas = sorted(counts, key=int)
    if len(areas) < 3:
        raise ValueError("CROSS_VALIDATION_NEEDS_3_AREAS")
    plan = []
    for held in areas:
        rest = [a for a in areas if a != held]
        val = min(rest, key=lambda a: (counts[a], int(a)))
        plan.append({a: "test" if a == held else "validation" if a == val else "train" for a in areas})
    return plan


KEYS = (
    ("coarse", "accuracy"),
    ("coarse_present", "balanced_accuracy"),
    ("coarse_present", "macro_f1"),
    ("fine", "accuracy"),
    ("fine_present", "macro_f1"),
    ("fine_present", "balanced_accuracy"),
    ("within", "subtype_accuracy_given_correct_class"),
)


def _pick(result, block, key):
    c = result["candidate"]
    if block == "fine_present":
        return c["fine"]["present_label_macro"][key]
    if block == "coarse_present":
        return c["coarse"]["present_label_macro"][key]
    if block == "within":
        return c["within_class"][key]
    return c[block][key]


def _spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "folds": len(values),
    }


def cross_validate(
    root, output, config, existing_bundle=None, synthetic_smoke=False, data=None
):
    data = ps.validate(root, load_pixels=True) if data is None else data
    output = Path(output)
    if output.exists():
        raise ValueError("NEW_RUN_DIRECTORY_REQUIRED")
    output.mkdir(parents=True, mode=0o700)
    folds = []
    for split_map in fold_plan(data):
        held = next(a for a, s in split_map.items() if s == "test")
        base = output / f"fold-area-{held}"
        report = train_subtype(
            root,
            base / "run",
            config,
            synthetic_smoke=synthetic_smoke,
            split_map=split_map,
            data=data,
            fold=True,
        )
        result = evaluate_subtype(
            root,
            base / "run" / "checkpoint.pt",
            base / "evaluation",
            existing_bundle,
            synthetic_smoke,
            data=data,
            device=config.device,
        )
        fold = {
            "held_out_area": held,
            "areas_by_split": {
                s: sorted((a for a, v in split_map.items() if v == s), key=int) for s in SPLITS
            },
            "best_epoch": report["best_epoch"],
            "test_samples": result["candidate"]["coarse"]["samples"],
            "absent_fine_labels": result["candidate"]["fine"]["absent_labels"],
            "rare_fine_labels": result["candidate"]["fine"]["rare_labels_support_under_30"],
            **{f"{b}_{k}": _pick(result, b, k) for b, k in KEYS},
            "within_class_by_class": {
                c: v["subtype_accuracy"]
                for c, v in result["candidate"]["within_class"]["by_class"].items()
            },
            "absent_classes": [
                r["class"] for r in result["candidate"]["coarse"]["per_class"] if not r["support"]
            ],
            "baseline": None
            if result["baseline"] is None
            else {
                "accuracy": result["baseline"]["accuracy"],
                **{
                    k: present_macro(result["baseline"])[k]
                    for k in ("balanced_accuracy", "macro_f1")
                },
            },
            "fine_per_label": {
                r["class"]: {"support": r["support"], "f1": r["f1"], "recall": r["recall"]}
                for r in result["candidate"]["fine"]["per_class"]
            },
        }
        folds.append(fold)
    summary = {
        "status": "SYNTHETIC_FUNCTIONAL_EVALUATION"
        if synthetic_smoke
        else "CROSS_VALIDATION_COMPLETED",
        "dataset_hash": data.dataset_hash,
        "config": asdict(config),
        "fold_rule": "hold out one Area; validation = remaining Area with fewest patches; "
        "train = the rest",
        "folds": folds,
        "summary": {f"{b}_{k}": _spread([f[f"{b}_{k}"] for f in folds]) for b, k in KEYS},
        "baseline_summary": None
        if existing_bundle is None
        else {
            k: _spread([f["baseline"][k] for f in folds])
            for k in ("accuracy", "balanced_accuracy", "macro_f1")
        },
        "labels_source": "provider",
        "expert_verified": False,
        "improvement": "NOT_VERIFIED",
    }
    (output / "crossval-summary.json").write_bytes(canonical(summary))
    return summary
