"""New, research-only candidate artifacts. No replacement or distribution."""

import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch

from ..data.verified import CLASSES, canonical, digest, read_json
from ..evaluation.compare import INPUT, OUTPUT, frozen_session
from ..training.engine import load_checkpoint, selected_model
from ..training.model import ARCHITECTURE, ScoreExport


def export(checkpoint_path, output, evaluation_path):
    checkpoint = load_checkpoint(checkpoint_path)
    evaluation = read_json(Path(evaluation_path).read_bytes())
    sha = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    if (
        evaluation["checkpoint_sha256"] != sha
        or evaluation["dataset_hash"] != checkpoint["dataset_hash"]
    ):
        raise ValueError("EVALUATION_PROVENANCE")
    if evaluation["scientific_metrics"] == checkpoint["synthetic_smoke"]:
        raise ValueError("EVALUATION_ORIGIN")
    out = Path(output)
    if out.exists():
        raise ValueError("NEW_CANDIDATE_DIRECTORY_REQUIRED")
    out.mkdir(parents=True, mode=0o700)
    torch.set_num_threads(2)
    model = ScoreExport(selected_model(checkpoint)).eval()
    rng = np.random.default_rng(2026)
    example = torch.from_numpy(rng.random((1, 3, 128, 128), dtype=np.float32))
    path = out / "model.onnx"
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=["images"],
        output_names=["probabilities"],
        dynamic_axes={"images": {0: "N"}, "probabilities": {0: "N"}},
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    manifest = {
        "schema": "frozen_xrt_patch_classifier",
        "schema_version": 1,
        "status": "validated_export",
        "classes": CLASSES,
        "input": INPUT,
        "output": OUTPUT,
        "model_file": "model.onnx",
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "research_only": True,
        "independent_test_passed": False,
        "background_class": False,
        "localizes_defects": False,
        "architecture": ARCHITECTURE,
        "dataset_hash": checkpoint["dataset_hash"],
        "config_hash": checkpoint["config_hash"],
        "code_revision": checkpoint["code_revision"],
        "checkpoint_sha256": sha,
        "evaluation_hash": digest(evaluation),
        "synthetic_smoke": checkpoint["synthetic_smoke"],
        "improvement": "NOT_VERIFIED",
        "candidate_status": "FUNCTIONAL_TEST_ONLY"
        if checkpoint["synthetic_smoke"]
        else "AWAITING_EXPERT_MODEL_ACCEPTANCE",
    }
    (out / "manifest.json").write_bytes(canonical(manifest))
    try:
        session, _ = frozen_session(out)
        parity = []
        for batch in (1, 7):
            x = rng.random((batch, 3, 128, 128), dtype=np.float32)
            with torch.no_grad():
                expected = model(torch.from_numpy(x)).numpy()
            actual = session.run(["probabilities"], {"images": x})[0]
            if (
                actual.shape != (batch, 3)
                or not np.isfinite(actual).all()
                or not np.allclose(actual, expected, atol=1e-5, rtol=1e-4)
            ):
                raise ValueError("ONNX_RUNTIME_PARITY")
            parity.append(
                {
                    "batch": batch,
                    "max_absolute_error": float(np.abs(actual - expected).max()),
                }
            )
        manifest["parity"] = {
            "provider": "CPUExecutionProvider",
            "atol": 1e-5,
            "rtol": 1e-4,
            "results": parity,
        }
        (out / "manifest.json").write_bytes(canonical(manifest))
        (out / "candidate-registry.json").write_bytes(
            canonical(
                {
                    "schema": "xrt_model_candidate/v1",
                    "candidate_id": manifest["model_sha256"],
                    "manifest_sha256": hashlib.sha256(
                        (out / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "dataset_hash": checkpoint["dataset_hash"],
                    "evaluation": evaluation,
                    "status": manifest["candidate_status"],
                    "distribution": "NOT_AUTHORIZED",
                    "replaces_existing_model": False,
                }
            )
        )
    except Exception:
        (out / "manifest.json").unlink(missing_ok=True)
        raise
    return manifest
