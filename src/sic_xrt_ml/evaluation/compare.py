"""Explicit independent test evaluation. Never selects hyperparameters on test."""

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ..data.verified import (
    CLASSES,
    VerifiedDataset,
    canonical,
    digest,
    loader,
    read_json,
    readiness,
    safe_file,
    validate,
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


def frozen_session(bundle):
    bundle = Path(bundle).absolute()
    m = read_json(safe_file(bundle, "manifest.json").read_bytes())
    if (
        m.get("schema") != "frozen_xrt_patch_classifier"
        or m.get("schema_version") != 1
        or m.get("classes") != CLASSES
        or m.get("input") != INPUT
        or m.get("output") != OUTPUT
        or m.get("model_file") != "model.onnx"
        or m.get("status") != "validated_export"
        or m.get("research_only") is not True
        or m.get("background_class") is not False
        or m.get("localizes_defects") is not False
    ):
        raise ValueError("WRONG_MODEL_CONTRACT")
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
        or session.get_inputs()[0].shape != ["N", 3, 128, 128]
        or len(session.get_outputs()) != 1
        or session.get_outputs()[0].name != "probabilities"
        or session.get_outputs()[0].type != "tensor(float)"
        or session.get_outputs()[0].shape != ["N", 3]
    ):
        raise ValueError("ACTUAL_ONNX_CONTRACT")
    return session, m


def evaluate(
    root, checkpoint_path, output, existing_bundle=None, synthetic_smoke=False
):
    manifest = validate(root)
    ready = readiness(manifest, synthetic_smoke)
    if not ready["ready"]:
        raise ValueError("DATA_NOT_READY")
    checkpoint = load_checkpoint(checkpoint_path)
    if (
        checkpoint["dataset_hash"] != manifest["dataset_hash"]
        or checkpoint["synthetic_smoke"] != synthetic_smoke
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
    data = VerifiedDataset(root, "test", manifest)
    session, old_manifest = (
        frozen_session(existing_bundle) if existing_bundle else (None, None)
    )
    labels = []
    new = []
    old = []
    with torch.no_grad():
        for x, y in loader(data, checkpoint["config"]["batch_size"]):
            labels.extend(y.tolist())
            new.extend(model(x).argmax(1).tolist())
            if session:
                scores = session.run(["probabilities"], {"images": x.numpy()})[0]
                if (
                    scores.shape != (len(y), 3)
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
        "test_sample_hash": digest(
            [(r["sample_id"], r["class_label"], r["crop_hash"]) for r in data.rows]
        ),
        "checkpoint_sha256": hashlib.sha256(
            Path(checkpoint_path).read_bytes()
        ).hexdigest(),
        "candidate": metrics(labels, new),
        "baseline": metrics(labels, old) if session else None,
        "baseline_model_sha256": old_manifest["model_sha256"] if old_manifest else None,
        "improvement": "NOT_VERIFIED",
        "scientific_metrics": not synthetic_smoke,
        "same_inputs_and_labels": bool(session),
        "selection_policy": "validation only; explicit final test invocation",
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
