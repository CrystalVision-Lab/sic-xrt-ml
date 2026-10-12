"""End-to-end CPU functional gates, including optimizer, resume and ORT parity."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from conftest import save_manifest

from sic_xrt_ml.data.verified import VerifiedDataset, loader, readiness, validate
from sic_xrt_ml.evaluation.compare import evaluate, frozen_session
from sic_xrt_ml.evaluation.metrics import metrics
from sic_xrt_ml.export.onnx_bundle import export
from sic_xrt_ml.training.engine import TrainConfig, load_checkpoint, train
from sic_xrt_ml.training.model import PatchClassifier


def test_loader_input_and_readiness(dataset):
    root, m = dataset
    assert validate(root) == m
    data = VerifiedDataset(root, "train")
    x, y = next(iter(loader(data, 3)))
    assert x.dtype == torch.float32 and tuple(x.shape) == (3, 3, 128, 128)
    assert y.tolist() == [0, 1, 2] and 0 <= x.min() <= x.max() <= 1
    assert readiness(m)["status"] == "DATA_NOT_READY"
    assert readiness(m, True)["ready"]
    m["rows"][0]["data_origin"] = "real_research"
    with pytest.raises(ValueError, match="OVERRIDE_FORBIDDEN"):
        readiness(m, True)


@pytest.mark.parametrize(
    "key",
    [
        "source_hash",
        "wafer_id",
        "group_id",
        "run_id",
        "target_id",
        "provider_group",
        "before_after_group",
        "duplicate_group",
        "split_group_id",
    ],
)
def test_group_leakage(dataset, key):
    root, m = dataset
    m["rows"][3][key] = m["rows"][0][key] = (
        "8" if key == "wafer_id" else m["rows"][0][key] or "same-group"
    )
    save_manifest(root, m)
    with pytest.raises(ValueError, match="LEAKAGE"):
        validate(root)


@pytest.mark.parametrize(
    "bad",
    [
        "class",
        "input",
        "sha",
        "pixel",
        "path",
        "label",
        "rights",
        "bounds",
        "approval",
        "split",
        "report",
        "extra",
        "duplicate_json",
        "dhash",
        "holdout",
    ],
)
def test_invalid_contract(dataset, bad):
    root, m = dataset
    if bad == "class":
        m["class_mapping"] = {"TED": 0, "BPD": 1, "TSD": 2}
    elif bad == "input":
        m["preprocessing_contract"]["crop_policy"] = "resize"
    elif bad == "sha":
        (root / m["rows"][0]["image_path"]).write_bytes(b"corrupt")
    elif bad == "pixel":
        m["rows"][0]["pixel_hash"] = "0" * 64
    elif bad == "path":
        m["rows"][0]["image_path"] = "../outside.png"
    elif bad == "label":
        m["rows"][0]["class_label"] = "UNKNOWN"
    elif bad == "rights":
        m["rows"][0]["training_use_authorized"] = False
    elif bad == "bounds":
        m["rows"][0]["crop"] = [200, 200, 128, 128]
    elif bad == "approval":
        m["rows"][0]["review_ids"] = []
    elif bad == "dhash":
        m["rows"][0]["dhash"] = "0" * 16
    elif bad == "holdout":
        m["rows"][0]["evaluation_only"] = True
    save_manifest(root, m)
    if bad == "split":
        (root / "splits/train.jsonl").write_text("")
    elif bad == "report":
        (root / "eligibility-report.json").write_text("{}")
    elif bad == "extra":
        (root / "images/unreferenced.png").write_bytes(b"extra")
    elif bad == "duplicate_json":
        (root / "manifest.json").write_text('{"schema":1,"schema":2}')
    with pytest.raises((ValueError, KeyError)):
        validate(root)


def test_asset_changed_after_load(dataset):
    root, _ = dataset
    data = VerifiedDataset(root, "train")
    (root / data.rows[0]["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="CHANGED_AFTER"):
        data[0]


def test_metrics_absent_and_bpd_errors():
    m = metrics([0, 0, 1, 2], [1, 2, 0, 2])
    assert m["confusion_matrix"] == [[0, 1, 1], [1, 0, 0], [0, 0, 1]]
    assert m["BPD_as_TED"] == 1 and m["BPD_as_TSD"] == 1 and m["false_BPD"] == 1
    assert metrics([0], [0])["macro_f1"] is None
    assert metrics([], [])["accuracy"] is None
    assert metrics([0], [0])["per_class"][1]["recall"] is None


def test_forward_backprop_internal_contrast():
    model = PatchClassifier()
    x = torch.rand(2, 3, 128, 128)
    assert not torch.equal(x, model.contrast(x))
    logits = model(x)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
    assert logits.shape == (2, 3) and model.head.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        TrainConfig(augmentation="random_rotation")


def test_exact_epoch_resume(dataset, tmp_path):
    root, _ = dataset
    cfg = TrainConfig(epochs=2, patience=10, batch_size=2)
    uninterrupted = tmp_path / "uninterrupted"
    part = tmp_path / "part"
    resumed = tmp_path / "resumed"
    train(root, uninterrupted, cfg, synthetic_smoke=True)
    train(root, part, replace(cfg, epochs=1), synthetic_smoke=True)
    train(root, resumed, cfg, resume=part / "checkpoint.pt", synthetic_smoke=True)
    a = load_checkpoint(uninterrupted / "checkpoint.pt")
    b = load_checkpoint(resumed / "checkpoint.pt")
    assert a["history"] == b["history"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert all(
        torch.equal(a["best_model"][k], b["best_model"][k]) for k in a["best_model"]
    )
    with pytest.raises(ValueError, match="DATA_NOT_READY"):
        train(root, tmp_path / "real", cfg)
    with pytest.raises(ValueError, match="RESUME_CONTRACT"):
        train(
            root,
            tmp_path / "wrong-resume",
            replace(cfg, seed=10),
            resume=part / "checkpoint.pt",
            synthetic_smoke=True,
        )


def test_evaluation_onnx_parity_contract_and_candidate(dataset, tmp_path):
    root, _ = dataset
    run = tmp_path / "run"
    evaluation = tmp_path / "evaluation"
    bundle = tmp_path / "bundle"
    result = train(root, run, TrainConfig(epochs=1), synthetic_smoke=True)
    assert result["status"] == "TRAINING_PIPELINE_VERIFIED"
    evaluated = evaluate(root, run / "checkpoint.pt", evaluation, synthetic_smoke=True)
    assert (
        not evaluated["scientific_metrics"]
        and evaluated["improvement"] == "NOT_VERIFIED"
    )
    with pytest.raises(ValueError, match="FINAL_TEST_ALREADY_USED"):
        evaluate(
            root,
            run / "checkpoint.pt",
            tmp_path / "second-evaluation",
            synthetic_smoke=True,
        )
    with pytest.raises(ValueError, match="FINAL_TEST_ALREADY_USED_RESUME"):
        train(
            root,
            tmp_path / "after-test",
            TrainConfig(epochs=2),
            resume=run / "checkpoint.pt",
            synthetic_smoke=True,
        )
    manifest = export(run / "checkpoint.pt", bundle, evaluation / "evaluation.json")
    assert manifest["classes"] == ["BPD", "TED", "TSD"]
    assert manifest["input"]["local_contrast_inside_model"] is True
    assert max(r["max_absolute_error"] for r in manifest["parity"]["results"]) < 1e-5
    session, _ = frozen_session(bundle)
    assert session.run(None, {"images": np.zeros((3, 3, 128, 128), dtype=np.float32)})[
        0
    ].shape == (3, 3)
    registry = json.loads((bundle / "candidate-registry.json").read_text())
    assert (
        registry["distribution"] == "NOT_AUTHORIZED"
        and registry["status"] == "FUNCTIONAL_TEST_ONLY"
    )
    another = tmp_path / "another-run"
    train(root, another, TrainConfig(epochs=1), synthetic_smoke=True)
    comparison = evaluate(
        root, another / "checkpoint.pt", tmp_path / "comparison", bundle, True
    )
    assert (
        comparison["same_inputs_and_labels"]
        and comparison["candidate"] == comparison["baseline"]
    )
    # Existing approved bundle uses "batch", not "N"; symbolic name is not a semantic shape change.
    import hashlib

    import onnx

    graph = onnx.load(str(bundle / "model.onnx"))
    graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    graph.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    onnx.save(graph, str(bundle / "model.onnx"))
    manifest["model_sha256"] = hashlib.sha256(
        (bundle / "model.onnx").read_bytes()
    ).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    frozen_session(bundle)
    bad = deepcopy(manifest)
    bad["classes"] = ["TED", "BPD", "TSD"]
    (bundle / "manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="WRONG_MODEL_CONTRACT"):
        frozen_session(bundle)


def test_early_stop_rejects_resume(dataset, tmp_path, monkeypatch):
    from sic_xrt_ml.training import engine

    root, _ = dataset
    monkeypatch.setattr(engine, "evaluate_loader", lambda *args: {"loss": 1.0})
    run = tmp_path / "stopped"
    result = train(root, run, TrainConfig(epochs=5, patience=1), synthetic_smoke=True)
    assert result["epochs_completed"] == 2
    assert load_checkpoint(run / "checkpoint.pt")["early_stopped"]
    with pytest.raises(ValueError, match="RESUME_CONTRACT_OR_STOPPED"):
        train(
            root,
            tmp_path / "resume-stopped",
            TrainConfig(epochs=6, patience=1),
            resume=run / "checkpoint.pt",
            synthetic_smoke=True,
        )


def test_symlink_guard_and_oversize(dataset, monkeypatch):
    from pathlib import Path

    from sic_xrt_ml.data.verified import safe_file

    root, m = dataset
    path = root / m["rows"][0]["image_path"]
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == path or original(self))
    with pytest.raises(ValueError, match="SYMLINK"):
        safe_file(root, m["rows"][0]["image_path"])
    monkeypatch.setattr(Path, "is_symlink", original)
    path.write_bytes(b"x" * (4 * 1024**2 + 1))
    with pytest.raises(ValueError, match="OVERSIZED"):
        safe_file(root, m["rows"][0]["image_path"])
