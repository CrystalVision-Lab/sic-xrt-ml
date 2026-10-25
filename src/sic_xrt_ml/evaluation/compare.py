"""Explicit independent test evaluation. Never selects hyperparameters on test."""

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ..data.consensus import CLASSES_V2
from ..data.loading import prepare
from ..data.provider_subtype import FINE_LABELS, FINE_TO_CLASS
from ..data.verified import (
    CLASSES,
    canonical,
    digest,
    loader,
    read_json,
    safe_file,
)
from ..training.engine import load_checkpoint, selected_model
from .metrics import metrics

INPUT = {
    "name": "images",
    "dtype": "float32",
    "shape": ["N", 3, 128, 128],
    "conversion": "uint8_RGB_to_NCHW_div255",
    "local_contrast_inside_model": True,
}
OUTPUT = {
    "name": "probabilities",
    "shape": ["N", 3],
    "meaning": "uncalibrated_class_scores",
}
# frozen_xrt_patch_classifier schema_version 2 = v1 + NOT_DEFECT background score.
OUTPUT_V2 = {**OUTPUT, "shape": ["N", 4]}
# schema_version -> (classes, output block, background_class)
MANIFEST_VERSIONS = {1: (CLASSES, OUTPUT, False), 2: (CLASSES_V2, OUTPUT_V2, True)}
# Optional second output of a v1 bundle (provider subtype models). Consumers that only
# run ["probabilities"] are unaffected; the block must match exactly when present.
SUBTYPE_OUTPUT = {
    "name": "subtype_probabilities",
    "shape": ["N", 10],
    "labels": FINE_LABELS,
    "class_of_label": FINE_TO_CLASS,
    "meaning": "uncalibrated_subtype_scores",
    "relation": "probabilities[:, k] = sum of subtype_probabilities over labels with class_of_label == k",
    "orientation": "tiff_pixel_grid_direction_subtypes_mirror_with_the_image",
}


def manifest_version(classes):
    """Frozen manifest schema_version for a class order (3 -> v1, 4 -> v2)."""
    for version, (expected, _output, _background) in MANIFEST_VERSIONS.items():
        if list(classes) == expected:
            return version
    raise ValueError("UNSUPPORTED_CLASS_ORDER")


def frozen_session(bundle):
    bundle = Path(bundle).absolute()
    m = read_json(safe_file(bundle, "manifest.json").read_bytes())
    version = m.get("schema_version") if isinstance(m, dict) else None
    contract = MANIFEST_VERSIONS.get(version) if type(version) is int else None
    classes, output, background = contract or (None, None, None)
    if (
        contract is None
        or m.get("schema") != "frozen_xrt_patch_classifier"
        or m.get("classes") != classes
        or m.get("input") != INPUT
        or m.get("output") != output
        or m.get("model_file") != "model.onnx"
        or m.get("status") != "validated_export"
        or m.get("research_only") is not True
        or m.get("background_class") is not background
        or m.get("localizes_defects") is not False
        or ("subtype_output" in m and (version != 1 or m["subtype_output"] != SUBTYPE_OUTPUT))
    ):
        raise ValueError("WRONG_MODEL_CONTRACT")
    outputs = [(output["name"], len(classes))]
    if "subtype_output" in m:
        outputs.append((SUBTYPE_OUTPUT["name"], SUBTYPE_OUTPUT["shape"][1]))
    model = safe_file(bundle, "model.onnx", 256 * 1024**2)
    if hashlib.sha256(model.read_bytes()).hexdigest() != m["model_sha256"]:
        raise ValueError("MODEL_HASH")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if (
        len(session.get_inputs()) != 1
        or session.get_inputs()[0].name != "images"
        or session.get_inputs()[0].type != "tensor(float)"
        or len(session.get_inputs()[0].shape) != 4
        or not isinstance(session.get_inputs()[0].shape[0], str)
        or not session.get_inputs()[0].shape[0]
        or session.get_inputs()[0].shape[1:] != [3, 128, 128]
        or len(session.get_outputs()) != len(outputs)
        or any(
            actual.name != name
            or actual.type != "tensor(float)"
            or actual.shape != [session.get_inputs()[0].shape[0], width]
            for actual, (name, width) in zip(session.get_outputs(), outputs, strict=True)
        )
    ):
        raise ValueError("ACTUAL_ONNX_CONTRACT")
    return session, m


def evaluate(
    root, checkpoint_path, output, existing_bundle=None, synthetic_smoke=False
):
    checkpoint = load_checkpoint(checkpoint_path)
    # v2 readiness reuses the per-class gate recorded at training time.
    v2 = (checkpoint.get("dataset") or {}).get("schema_version") == 2
    prepared = prepare(
        root, synthetic_smoke, checkpoint["minimum_per_class"] if v2 else None
    )
    manifest = prepared.manifest
    classes = prepared.classes
    if not prepared.readiness["ready"]:
        raise ValueError("DATA_NOT_READY")
    if (
        checkpoint["dataset_hash"] != manifest["dataset_hash"]
        or checkpoint["synthetic_smoke"] != synthetic_smoke
        or checkpoint.get("classes", CLASSES) != classes
    ):
        raise ValueError("CHECKPOINT_DATASET")
    lock = Path(checkpoint_path).parent / "final-evaluation.json"
    if lock.exists():
        raise ValueError("FINAL_TEST_ALREADY_USED")
    output = Path(output)
    if output.exists():
        raise ValueError("FINAL_EVALUATION_ALREADY_RECORDED")
    output.mkdir(parents=True, mode=0o700)
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(
            canonical(
                {
                    "dataset_hash": manifest["dataset_hash"],
                    "checkpoint_sha256": hashlib.sha256(
                        Path(checkpoint_path).read_bytes()
                    ).hexdigest(),
                    "state": "TEST_ACCESS_STARTED",
                }
            ).decode("utf-8")
        )
    model = selected_model(checkpoint)
    torch.set_num_threads(2)
    data = prepared.dataset("test")
    session, old_manifest = (
        frozen_session(existing_bundle) if existing_bundle else (None, None)
    )
    # A 3-class baseline can be scored on 4-class labels (it never predicts NOT_DEFECT).
    if old_manifest and old_manifest["classes"] != classes[: len(old_manifest["classes"])]:
        raise ValueError("BASELINE_CLASS_ORDER")
    labels = []
    new = []
    old = []
    with torch.no_grad():
        for batch in loader(data, checkpoint["config"]["batch_size"]):
            x, y = batch[0], batch[1]
            labels.extend(y.tolist())
            new.extend(model(x).argmax(1).tolist())
            if session:
                scores = session.run(["probabilities"], {"images": x.numpy()})[0]
                if (
                    scores.shape != (len(y), len(old_manifest["classes"]))
                    or not np.isfinite(scores).all()
                    or (scores < 0).any()
                    or not np.allclose(scores.sum(1), 1, atol=1e-4)
                ):
                    raise ValueError("BASELINE_SCORES")
                old.extend(scores.argmax(1).tolist())
    result = {
        "status": "SYNTHETIC_FUNCTIONAL_EVALUATION"
        if synthetic_smoke
        else "INDEPENDENT_EVALUATION_COMPLETED",
        "dataset_hash": manifest["dataset_hash"],
        "test_sample_hash": digest(prepared.identity(data.rows)),
        "checkpoint_sha256": hashlib.sha256(
            Path(checkpoint_path).read_bytes()
        ).hexdigest(),
        "candidate": metrics(labels, new, classes),
        "baseline": metrics(labels, old, classes) if session else None,
        "baseline_model_sha256": old_manifest["model_sha256"] if old_manifest else None,
        "improvement": "NOT_VERIFIED",
        "scientific_metrics": not synthetic_smoke,
        "same_inputs_and_labels": bool(session),
        "selection_policy": "validation only; explicit final test invocation",
    }
    if prepared.schema_version == 2:
        # Random-queue samples are the unbiased subset for accuracy estimates.
        chosen = [i for i, r in enumerate(data.rows) if r["random_sample"]]
        result |= {
            "classes": classes,
            "dataset": prepared.provenance(),
            "coverage": prepared.readiness["coverage"],
            "random_sample_candidate": metrics(
                [labels[i] for i in chosen], [new[i] for i in chosen], classes
            ),
            "random_sample_baseline": metrics(
                [labels[i] for i in chosen], [old[i] for i in chosen], classes
            )
            if session
            else None,
        }
    (output / "evaluation.json").write_bytes(canonical(result))
    (output / "predictions.json").write_bytes(
        canonical(
            [
                {
                    "sample_id": r["sample_id"],
                    "label": y,
                    "candidate": n,
                    "baseline": o if session else None,
                }
                for r, y, n, o in zip(
                    data.rows,
                    labels,
                    new,
                    old if session else [None] * len(new),
                    strict=True,
                )
            ]
        )
    )
    return result
