"""verified_xrt_dataset/v2 -> 4-class training, evaluation and frozen v2 ONNX export.

Only generated synthetic_fixture data; no company images or expert labels.
"""

import json
import struct
import sys
import zlib
from dataclasses import replace

import numpy as np
import pytest
import torch
from conftest import V2_CLASSES, save_v2_manifest, v2_row

from sic_xrt_ml.data.consensus import ConsensusDataset
from sic_xrt_ml.data.loading import parse_minimum, prepare
from sic_xrt_ml.data.verified import digest, loader
from sic_xrt_ml.evaluation.compare import evaluate, frozen_session
from sic_xrt_ml.evaluation.metrics import metrics
from sic_xrt_ml.export.onnx_bundle import export
from sic_xrt_ml.training.engine import TrainConfig, load_checkpoint, train


def test_v2_validate_coverage_and_readiness(dataset_v2):
    root, m = dataset_v2
    data = prepare(root, synthetic_smoke=True)
    assert data.schema_version == 2 and data.classes == V2_CLASSES
    ready = data.readiness
    assert ready["ready"] and ready["status"] == "TRAINING_PIPELINE_VERIFIED_INPUT"
    cov = ready["coverage"]
    assert cov["crop_coverage"] == m["crop_coverage"] == round(26 / 27, 4)
    assert cov["usable_samples"] == 24
    assert cov["skipped"] == {"NO_CROP": 1, "CROP_GRAY": 1, "CROP_NOT_128": 1}
    by = cov["by_split_class"]
    assert by["train"]["BPD"] == {
        "samples": 3,
        "with_crop": 2,
        "usable": 2,
        "skipped": {"NO_CROP": 1, "CROP_NOT_128": 0, "CROP_GRAY": 0},
    }
    assert by["train"]["NOT_DEFECT"]["skipped"]["CROP_GRAY"] == 1
    assert by["validation"]["TSD"]["skipped"]["CROP_NOT_128"] == 1
    assert ready["class_counts"]["test"] == {c: 2 for c in V2_CLASSES}
    # Default gate 10/5/5 per class (NOT_DEFECT included) and synthetic origin block READY.
    assert prepare(root).readiness["status"] == "DATA_NOT_READY"
    assert prepare(root).readiness["minimum_per_class"] == {
        "train": 10,
        "validation": 5,
        "test": 5,
    }
    assert not prepare(root, min_per_class=parse_minimum("1,1,1")).readiness["ready"]
    assert not prepare(root, True, parse_minimum("3,1,1")).readiness["ready"]
    x, y, w = next(iter(loader(data.dataset("train"), 4)))
    assert x.dtype == torch.float32 and tuple(x.shape) == (4, 3, 128, 128)
    assert 0 <= x.min() <= x.max() <= 1 and set(y.tolist()) <= {0, 1, 2, 3}
    assert w.dtype == torch.float32 and set(w.tolist()) <= {1.0, 0.5}
    weights = data.class_weights()
    assert len(weights) == 4 and all(v > 0 for v in weights)


def test_v2_real_origin_gate(tmp_path):
    from conftest import write_v2_dataset

    root, m = write_v2_dataset(tmp_path / "real", per_class=1, extras=False)
    for r in m["rows"]:
        r["data_origin"] = ["real_research"]
    save_v2_manifest(root, m)
    with pytest.raises(ValueError, match="OVERRIDE_FORBIDDEN"):
        prepare(root, synthetic_smoke=True)
    assert prepare(root).readiness["status"] == "DATA_NOT_READY"
    assert prepare(root, min_per_class=parse_minimum("1,1,1")).readiness["status"] == (
        "READY"
    )
    with pytest.raises(ValueError, match="MIN_PER_CLASS"):
        parse_minimum("0,1,1")


def _png16():
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 128, 128, 16, 2, 0, 0, 0))
        + chunk(
            b"IDAT",
            zlib.compress(b"".join(b"\0" + b"\x80\x00" * (128 * 3) for _ in range(128))),
        )
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize(
    "bad",
    [
        "schema",
        "class_mapping",
        "crop_contract",
        "dataset_hash",
        "content_hash",
        "png_sha",
        "pixel_hash",
        "dhash",
        "label_index",
        "background_flag",
        "weight",
        "sample_id",
        "rect_bounds",
        "rect_size",
        "holdout",
        "wafer8",
        "source_leak",
        "wafer_leak",
        "duplicate_pixels",
        "near_duplicate",
        "extra_file",
        "split_file",
        "report",
        "png16",
        "path",
    ],
)
def test_v2_invalid_contract(dataset_v2, bad):
    root, m = dataset_v2
    rows = m["rows"]
    with_crop = [r for r in rows if r["crop"] and r["crop"]["rect"][2] == 128]
    a, b = with_crop[0], next(r for r in with_crop if r["split"] != with_crop[0]["split"])
    save = True
    if bad == "schema":
        m["schema_version"] = 3
    elif bad == "class_mapping":
        m["class_mapping"] = {"BPD": 0, "TED": 1, "TSD": 2, "BACKGROUND": 3}
    elif bad == "crop_contract":
        m["crop_contract"]["resize"] = "bilinear"
    elif bad == "dataset_hash":
        m["dataset_version"] = "tampered"
        (root / "manifest.json").write_text(
            json.dumps(m, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        save = False
    elif bad == "content_hash":
        m["content_hash"] = "0" * 64
        m["dataset_hash"] = digest({k: v for k, v in m.items() if k != "dataset_hash"})
        (root / "manifest.json").write_text(
            json.dumps(m, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        save = False
    elif bad == "png_sha":
        (root / a["crop"]["image_path"]).write_bytes(b"corrupt")
        save = False
    elif bad == "pixel_hash":
        a["crop"]["pixel_hash"] = "0" * 64
    elif bad == "dhash":
        a["crop"]["dhash"] = "0" * 16
    elif bad == "label_index":
        a["class_index"] = (a["class_index"] + 1) % 4
    elif bad == "background_flag":
        a["is_background"] = not a["is_background"]
    elif bad == "weight":
        a["weight"] = 0
    elif bad == "sample_id":
        a["sample_id"] = "0" * 64
    elif bad == "rect_bounds":
        a["crop"]["rect"] = [1000, 700, 128, 128]
    elif bad == "rect_size":
        a["crop"]["rect"] = [0, 0, 120, 128]
    elif bad == "holdout":
        row = next(r for r in rows if r["split"] == "train")
        row["evaluation_only"] = True
    elif bad == "wafer8":
        row = next(r for r in rows if r["split"] == "train")
        row["wafer_id"] = ["8"]
    elif bad == "source_leak":
        b["source_sha256"] = a["source_sha256"]
        b["sample_id"] = digest([b["source_sha256"], b["page"], b["candidate_id"] + "x"])
        b["candidate_id"] += "x"
    elif bad == "wafer_leak":
        a["wafer_id"] = b["wafer_id"] = ["3"]
    elif bad == "duplicate_pixels":
        b["crop"] = dict(a["crop"])
    elif bad == "near_duplicate":
        array = np.asarray(
            __import__("PIL.Image", fromlist=["Image"]).open(
                root / a["crop"]["image_path"]
            )
        ).copy()
        array[0, 0, 0] ^= 1  # different pixels, same dHash
        rows.remove(b)
        new = v2_row(root, b["split"], b["label"], 500, array=array)
        rows.append(new)
        (root / b["crop"]["image_path"]).unlink()
    elif bad == "extra_file":
        (root / "images" / "unreferenced.png").write_bytes(b"x")
    elif bad == "png16":
        raw = _png16()
        import hashlib

        (root / a["crop"]["image_path"]).unlink()
        a["crop"]["sha256"] = hashlib.sha256(raw).hexdigest()
        a["crop"]["image_path"] = "images/" + a["crop"]["sha256"] + ".png"
        (root / a["crop"]["image_path"]).write_bytes(raw)
    elif bad == "path":
        a["crop"]["image_path"] = "../outside.png"
    if save:
        save_v2_manifest(root, m)
    if bad == "split_file":
        (root / "splits" / "test.jsonl").write_text("")
    elif bad == "report":
        (root / "consensus-report.json").write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        prepare(root, synthetic_smoke=True)


def test_v2_asset_changed_after_load(dataset_v2):
    root, _ = dataset_v2
    data = prepare(root, synthetic_smoke=True)
    dataset = data.dataset("train")
    assert isinstance(dataset, ConsensusDataset)
    (root / dataset.rows[0]["crop"]["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="CHANGED_AFTER"):
        dataset[0]


def test_four_class_metrics():
    m = metrics([0, 1, 3, 3, 2], [0, 3, 3, 1, 2], V2_CLASSES)
    assert m["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    assert m["class_order"] == V2_CLASSES and m["per_class"][3]["class"] == "NOT_DEFECT"
    assert m["NOT_DEFECT_as_defect"] == 1 and m["defect_as_NOT_DEFECT"] == 1
    assert m["per_class"][3]["recall"] == 0.5
    with pytest.raises(ValueError):
        metrics([4], [0], V2_CLASSES)
    assert "NOT_DEFECT_as_defect" not in metrics([0], [0])


def test_v2_resume_exact(dataset_v2, tmp_path):
    root, _ = dataset_v2
    cfg = TrainConfig(epochs=2, patience=10, batch_size=4)
    train(root, tmp_path / "full", cfg, synthetic_smoke=True)
    train(root, tmp_path / "part", replace(cfg, epochs=1), synthetic_smoke=True)
    train(
        root,
        tmp_path / "resumed",
        cfg,
        resume=tmp_path / "part" / "checkpoint.pt",
        synthetic_smoke=True,
    )
    a = load_checkpoint(tmp_path / "full" / "checkpoint.pt")
    b = load_checkpoint(tmp_path / "resumed" / "checkpoint.pt")
    assert a["history"] == b["history"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    with pytest.raises(ValueError, match="RESUME_CONTRACT"):
        train(
            root,
            tmp_path / "other-gate",
            cfg,
            resume=tmp_path / "part" / "checkpoint.pt",
            synthetic_smoke=True,
            min_per_class=parse_minimum("2,1,1"),
        )


def test_v2_train_evaluate_export_onnx(dataset_v2, dataset, tmp_path):
    root, m = dataset_v2
    run, evaluation, bundle = tmp_path / "run", tmp_path / "evaluation", tmp_path / "bundle"
    result = train(
        root, run, TrainConfig(epochs=2, batch_size=4, patience=10), synthetic_smoke=True
    )
    assert result["status"] == "TRAINING_PIPELINE_VERIFIED"
    assert result["classes"] == V2_CLASSES and result["epochs_completed"] == 2
    assert result["dataset"]["schema_version"] == 2
    assert result["dataset"]["dataset_id"] == m["dataset_id"]
    checkpoint = load_checkpoint(run / "checkpoint.pt")
    assert checkpoint["model"]["head.weight"].shape == (4, 32)
    validation = checkpoint["history"][-1]["validation"]
    assert len(validation["confusion_matrix"]) == 4 and validation["samples"] == 8
    evaluated = evaluate(root, run / "checkpoint.pt", evaluation, synthetic_smoke=True)
    cand = evaluated["candidate"]
    assert cand["class_order"] == V2_CLASSES and cand["samples"] == 8
    assert [len(r) for r in cand["confusion_matrix"]] == [4, 4, 4, 4]
    assert [p["support"] for p in cand["per_class"]] == [2, 2, 2, 2]
    assert "NOT_DEFECT_as_defect" in cand and "defect_as_NOT_DEFECT" in cand
    assert evaluated["random_sample_candidate"]["samples"] == 4
    assert evaluated["coverage"]["usable_samples"] == 24
    assert not evaluated["scientific_metrics"]
    manifest = export(run / "checkpoint.pt", bundle, evaluation / "evaluation.json")
    assert manifest["schema"] == "frozen_xrt_patch_classifier"
    assert manifest["schema_version"] == 2
    assert manifest["classes"] == V2_CLASSES
    assert manifest["background_class"] is True
    assert manifest["localizes_defects"] is False
    assert manifest["input"] == {
        "name": "images",
        "dtype": "float32",
        "shape": ["N", 3, 128, 128],
        "conversion": "uint8_RGB_to_NCHW_div255",
        "local_contrast_inside_model": True,
    }
    assert manifest["output"] == {
        "name": "probabilities",
        "shape": ["N", 4],
        "meaning": "uncalibrated_class_scores",
    }
    assert [r["batch"] for r in manifest["parity"]["results"]] == [1, 7]
    assert max(r["max_absolute_error"] for r in manifest["parity"]["results"]) < 1e-5
    session, loaded = frozen_session(bundle)
    assert loaded["schema_version"] == 2
    for n in (1, 5):
        scores = session.run(
            None, {"images": np.random.default_rng(n).random((n, 3, 128, 128), np.float32)}
        )[0]
        assert scores.shape == (n, 4) and scores.dtype == np.float32
        assert np.allclose(scores.sum(1), 1, atol=1e-5) and (scores >= 0).all()
    registry = json.loads((bundle / "candidate-registry.json").read_text())
    assert registry["schema"] == "xrt_model_candidate/v1"
    assert registry["dataset"]["schema"] == "verified_xrt_dataset"
    assert registry["dataset"]["schema_version"] == 2
    assert registry["dataset"]["dataset_id"] == m["dataset_id"]
    assert registry["dataset"]["content_hash"] == m["content_hash"]
    assert registry["dataset_hash"] == m["dataset_hash"] == manifest["dataset_hash"]
    assert registry["evaluation"]["classes"] == V2_CLASSES
    assert {p.name for p in bundle.iterdir()} == {
        "model.onnx",
        "manifest.json",
        "candidate-registry.json",
    }
    # Same-input comparison against the new v2 bundle and against a 3-class v1 bundle.
    another = tmp_path / "another"
    train(root, another, TrainConfig(epochs=2, batch_size=4, patience=10), synthetic_smoke=True)
    comparison = evaluate(root, another / "checkpoint.pt", tmp_path / "cmp", bundle, True)
    assert comparison["candidate"] == comparison["baseline"]
    v1_root, _ = dataset
    v1_run = tmp_path / "v1-run"
    train(v1_root, v1_run, TrainConfig(epochs=1), synthetic_smoke=True)
    evaluate(v1_root, v1_run / "checkpoint.pt", tmp_path / "v1-eval", synthetic_smoke=True)
    v1_bundle = tmp_path / "v1-bundle"
    v1_manifest = export(
        v1_run / "checkpoint.pt", v1_bundle, tmp_path / "v1-eval" / "evaluation.json"
    )
    assert v1_manifest["schema_version"] == 1 and v1_manifest["background_class"] is False
    assert "dataset" not in json.loads((v1_bundle / "candidate-registry.json").read_text())
    third = tmp_path / "third"
    train(root, third, TrainConfig(epochs=1, batch_size=4), synthetic_smoke=True)
    old = evaluate(root, third / "checkpoint.pt", tmp_path / "vs-v1", v1_bundle, True)
    assert [row[3] for row in old["baseline"]["confusion_matrix"]] == [0, 0, 0, 0]
    assert old["baseline"]["per_class"][3]["recall"] == 0.0
    # v2 checkpoints cannot be evaluated on a v1 dataset and vice versa.
    with pytest.raises(ValueError):
        evaluate(v1_root, third / "checkpoint.pt", tmp_path / "wrong", None, True)
    with pytest.raises(ValueError, match="MIN_PER_CLASS_V2_ONLY"):
        prepare(v1_root, True, parse_minimum("1,1,1"))
    bad = dict(manifest, background_class=False)
    (bundle / "manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="WRONG_MODEL_CONTRACT"):
        frozen_session(bundle)


def test_cli_validate_v2(dataset_v2, monkeypatch, capsys):
    from sic_xrt_ml.__main__ import main

    root, _ = dataset_v2
    monkeypatch.setattr(
        sys,
        "argv",
        ["sic_xrt_ml", "validate", str(root), "--synthetic-smoke", "--min-per-class", "2,2,2"],
    )
    main()
    out = json.loads(capsys.readouterr().out)
    assert out["ready"] and out["minimum_per_class"] == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    assert out["dataset"]["schema_version"] == 2
    assert out["coverage"]["by_split_class"]["validation"]["TSD"]["usable"] == 2
