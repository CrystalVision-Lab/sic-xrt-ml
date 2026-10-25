"""provider_subtype_point_dataset/v1 -> fine-label training, evaluation, ONNX export.

Only generated synthetic patches in tmp_path; no company images or provider labels.
"""

import csv
import hashlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sic_xrt_ml.data import provider_subtype as ps
from sic_xrt_ml.data.loading import is_provider_subtype, prepare
from sic_xrt_ml.evaluation.compare import frozen_session
from sic_xrt_ml.evaluation.subtype_eval import (
    cross_validate,
    evaluate_subtype,
    fold_plan,
)
from sic_xrt_ml.evaluation.subtype_metrics import (
    coarse_scores,
    paired_bootstrap,
    summarize,
    within_class_subtype,
)
from sic_xrt_ml.export.subtype_bundle import export_subtype
from sic_xrt_ml.training.subtype import (
    SubtypeTrainConfig,
    load_checkpoint,
    train_subtype,
)
from sic_xrt_ml.training.subtype_model import (
    MEDIUM,
    MEDIUM_BN_CENTER,
    SMALL,
    SubtypeClassifier,
    SubtypeScoreExport,
)

AREAS = {"1": "train", "9": "train", "2": "val", "8": "test"}
FIELDS = [
    "patch_id",
    "path",
    "sha256",
    "point_id",
    "image_id",
    "image",
    "area",
    "wafer",
    "stage",
    "split",
    "label",
    "subtype",
    "fine_label",
    "x",
    "y",
    "roi_x",
    "roi_y",
    "coordinate_transform",
    "pixel_source",
    "center_x",
    "center_y",
    "left",
    "top",
    "size",
    "dtype",
    "channels",
    "labels_source",
    "expert_verified",
    "review_status",
]
SMOKE = SubtypeTrainConfig(epochs=1, batch_size=16, architecture=SMALL)


def _tiff(array, **options):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="TIFF", **options)
    return buffer.getvalue()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_patch(root, row, raw):
    path = root / row["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    row["sha256"] = _sha(raw)


def finalize(root, manifest, rows, recount=True):
    """Write samples.csv, manifest.json and output_hashes.json consistently."""
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    samples = ("﻿" + text.getvalue()).encode("utf-8")
    (root / "samples.csv").write_bytes(samples)
    manifest["samples_sha256"] = _sha(samples)
    if recount:
        counts = {}
        for r in rows:
            by = counts.setdefault(r["split"], {}).setdefault(r["label"], {})
            by[r["subtype"] or "-"] = by.get(r["subtype"] or "-", 0) + 1
        manifest["counts"] = {"patches": len(rows), "by_split_class_subtype": counts}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    hashes = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and rel not in (
            "output_hashes.json",
            "preview/mean_by_area_stage_display_only.png",
        ):
            hashes[rel] = _sha(path.read_bytes())
    (root / "output_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def write_provider_dataset(root, per_label=2, synthetic=True):
    root = Path(root)
    root.mkdir(parents=True)
    rows, images = [], []
    rng = np.random.default_rng(7)
    for area, split in AREAS.items():
        image_id = f"{int(area):012x}"
        images.append({"image_id": image_id, "area": area, "split": split, "stage": "before"})
        for f, fine in enumerate(ps.FINE_LABELS):
            label, _, subtype = fine.partition("_")
            for k in range(per_label):
                n = len(rows)
                x, y = 200.5 + 130 * k, 300.25 + 130 * f
                patch_id = f"{n:020x}"
                row = {
                    "patch_id": patch_id,
                    "path": f"patches_128/{split}/{fine}/{patch_id}.tif",
                    "sha256": "",
                    "point_id": patch_id,
                    "image_id": image_id,
                    "image": f"{area} Area/synthetic.tif",
                    "area": area,
                    "wafer": area,
                    "stage": "before",
                    "split": split,
                    "label": label,
                    "subtype": subtype,
                    "fine_label": fine,
                    "x": repr(x),
                    "y": repr(y),
                    "roi_x": repr(x),
                    "roi_y": repr(y),
                    "coordinate_transform": "identity",
                    "pixel_source": "tiff",
                    "center_x": str(int(x)),
                    "center_y": str(int(y)),
                    "left": str(int(x) - 64),
                    "top": str(int(y) - 64),
                    "size": "128",
                    "dtype": "uint8",
                    "channels": "RGB",
                    "labels_source": "provider",
                    "expert_verified": "False",
                    "review_status": "provider_label_unverified",
                }
                # A faint label-dependent blob so the tiny model has something to fit.
                array = rng.integers(90, 160, (128, 128, 3), dtype=np.uint8)
                array[50 + f : 70 + f, 40 + 4 * f : 60 + 4 * f] += 60
                write_patch(root, row, _tiff(array))
                rows.append(row)
    (root / "images.json").write_text(json.dumps(images), encoding="utf-8")
    (root / "points.csv").write_text("point_id,status\n", encoding="utf-8")
    (root / "roi_files.csv").write_text("relpath,status\n", encoding="utf-8")
    (root / "preview").mkdir()
    (root / "preview" / "contact_sheet.png").write_bytes(_tiff(np.zeros((4, 4, 3), np.uint8)))
    # Display-only file the producer does not hash: must be tolerated.
    (root / "preview" / "mean_by_area_stage_display_only.png").write_bytes(b"display")
    manifest = {
        "schema": "provider_subtype_point_dataset",
        "schema_version": 1,
        "code_revision": {"package": "synthetic", "git_commit": "0" * 40, "dirty": False},
        "labels_source": "provider",
        "expert_verified": False,
        "research_only": True,
        "classes": ps.CLASSES,
        "subtypes": ps.SUBTYPES,
        "fine_labels": ps.FINE_LABELS,
        "patch": {
            "size": 128,
            "dtype": "uint8",
            "channels": "RGB",
            "format": "TIFF uncompressed",
            "resized": False,
            "normalized": False,
        },
        "split_map": AREAS,
    }
    manifest = json.loads(json.dumps(manifest))  # no shared module constants
    if synthetic:
        manifest["synthetic_fixture"] = True
    finalize(root, manifest, rows)
    return root, manifest, rows


@pytest.fixture
def provider(tmp_path):
    return write_provider_dataset(tmp_path / "provider")


def test_validate_readiness_and_routing(provider):
    root, _, rows = provider
    data = ps.validate(root, load_pixels=True)
    assert len(data.rows) == len(rows) == 80 and data.synthetic
    assert data.pixels.shape == (80, 3, 128, 128) and data.pixels.dtype == np.uint8
    assert data.dataset_hash == _sha((root / "manifest.json").read_bytes())
    first = np.asarray(Image.open(root / rows[0]["path"]))
    assert np.array_equal(data.pixels[0], first.transpose(2, 0, 1))
    assignment = ps.split_assignment(data)
    assert assignment == {"1": "train", "2": "validation", "8": "test", "9": "train"}
    ready = ps.readiness(data, assignment, synthetic_smoke=True)
    assert ready["status"] == "TRAINING_PIPELINE_VERIFIED_INPUT"
    assert ready["fine_counts"]["train"]["TED_e"] == 4
    assert ready["class_counts"]["test"] == {"BPD": 2, "TED": 12, "TSD": 6}
    assert ready["areas_by_split"]["validation"] == ["2"]
    assert ready["rare_fine_labels"]["test"] == ps.FINE_LABELS
    # Synthetic data never passes the real-data gate.
    assert not ps.readiness(data, assignment)["ready"]
    assert is_provider_subtype(root)
    with pytest.raises(ValueError, match="DATASET_SCHEMA"):
        prepare(root)  # v1/v2 entry point is unchanged
    with pytest.raises(ValueError, match="SPLIT_OVERRIDE"):
        ps.split_assignment(data, {"1": "train", "2": "train", "8": "test", "9": "train"})


def test_real_flag_gate(tmp_path):
    root, _, _ = write_provider_dataset(tmp_path / "real", synthetic=False)
    data = ps.validate(root)
    with pytest.raises(ValueError, match="OVERRIDE_FORBIDDEN"):
        ps.readiness(data, ps.split_assignment(data), synthetic_smoke=True)
    gate = ps.minimum_gate({"train": 1, "validation": 1, "test": 1})
    assert ps.readiness(data, ps.split_assignment(data), minimum=gate)["status"] == "READY"


def _regray(root, rows):
    gray = Image.fromarray(np.zeros((128, 128), np.uint8))
    buffer = io.BytesIO()
    gray.save(buffer, format="TIFF")
    write_patch(root, rows[0], buffer.getvalue())


REJECTIONS = {
    "schema": "DATASET_SCHEMA",
    "research_only": "LABEL_PROVENANCE",
    "expert_verified": "LABEL_PROVENANCE",
    "labels_source": "LABEL_PROVENANCE",
    "fine_labels": "LABEL_SET",
    "patch_format": "MODEL_INPUT_CONTRACT",
    "patch_bytes": "PATCH_SHA_MISMATCH",
    "samples_sha": "SAMPLES_SHA_MISMATCH",
    "manifest_hash": "MANIFEST_SHA_MISMATCH",
    "missing_hash": "UNREFERENCED_FILE",
    "extra_file": "UNREFERENCED_FILE",
    "build_failed": "UNREFERENCED_FILE",
    "area_two_splits": "AREA_SPLIT_MAP",
    "split_map": "SPLIT_MAP",
    "gray_patch": "TIFF_RGB8_PATCH_REQUIRED",
    "small_patch": "TIFF_RGB8_PATCH_REQUIRED",
    "compressed_patch": "TIFF_RGB8_PATCH_REQUIRED",
    "fine_label_mismatch": "LABEL_SET",
    "unknown_subtype": "LABEL_SET",
    "coordinates": "SAMPLE_COORDINATES",
    "counts": "SPLIT_CLASS_COUNTS",
    "image_area": "IMAGE_AREA",
    "duplicate_patch": "DUPLICATE_PATCH",
    "row_expert_verified": "LABEL_PROVENANCE",
    "path": "ASSET_PATH_CONTRACT",
}


@pytest.mark.parametrize("bad", sorted(REJECTIONS))
def test_invalid_contract(provider, bad):
    root, m, rows = provider
    r = rows[0]
    if bad == "schema":
        m["schema_version"] = 2
    elif bad == "research_only":
        m["research_only"] = False
    elif bad == "expert_verified":
        m["expert_verified"] = True
    elif bad == "labels_source":
        m["labels_source"] = "expert"
    elif bad == "fine_labels":
        m["fine_labels"] = ps.FINE_LABELS[:-1]
    elif bad == "patch_format":
        m["patch"]["resized"] = True
    elif bad == "split_map":
        m["split_map"]["3"] = "holdout"
    elif bad == "area_two_splits":
        # Area 1 rows in train and validation: rejected even with a consistent folder.
        moved = (root / r["path"]).read_bytes()
        (root / r["path"]).unlink()
        r["split"] = "val"
        r["path"] = f"patches_128/val/{r['fine_label']}/{r['patch_id']}.tif"
        write_patch(root, r, moved)
    elif bad == "gray_patch":
        _regray(root, rows)
    elif bad == "small_patch":
        write_patch(root, r, _tiff(np.zeros((64, 64, 3), np.uint8)))
    elif bad == "compressed_patch":
        write_patch(root, r, _tiff(np.zeros((128, 128, 3), np.uint8), compression="tiff_lzw"))
    elif bad == "fine_label_mismatch":
        r["fine_label"] = "TED_a"
    elif bad == "unknown_subtype":
        r["label"], r["subtype"], r["fine_label"] = "TED", "g", "TED_g"
    elif bad == "coordinates":
        r["left"] = str(int(r["left"]) + 1)
    elif bad == "image_area":
        r["image_id"] = rows[-1]["image_id"]
    elif bad == "duplicate_patch":
        raw = (root / r["path"]).read_bytes()
        write_patch(root, rows[1], raw)
    elif bad == "row_expert_verified":
        r["expert_verified"] = "True"
    elif bad == "path":
        r["path"] = "../outside.tif"
    if bad not in ("patch_bytes", "samples_sha", "manifest_hash", "missing_hash", "extra_file"):
        finalize(root, m, rows, recount=bad != "counts")
    if bad == "counts":
        m["counts"]["patches"] += 1
        finalize(root, m, rows, recount=False)
    elif bad == "patch_bytes":
        (root / r["path"]).write_bytes(_tiff(np.zeros((128, 128, 3), np.uint8)))
    elif bad == "samples_sha":
        (root / "samples.csv").write_bytes((root / "samples.csv").read_bytes() + b"\r\n")
    elif bad == "manifest_hash":
        (root / "manifest.json").write_text(
            json.dumps(m) + " ", encoding="utf-8"
        )
    elif bad == "missing_hash":
        hashes = json.loads((root / "output_hashes.json").read_text())
        del hashes[r["path"]]
        (root / "output_hashes.json").write_text(json.dumps(hashes))
    elif bad == "extra_file":
        (root / "patches_128" / "train" / "BPD" / "stray.tif").write_bytes(b"x")
    elif bad == "build_failed":
        (root / "BUILD_FAILED.json").write_text("{}")
    with pytest.raises(ValueError, match=REJECTIONS[bad]):
        ps.validate(root)


def test_patch_changed_after_validation_is_caught_on_next_validate(provider):
    root, _, rows = provider
    ps.validate(root)
    (root / rows[3]["path"]).write_bytes(_tiff(np.ones((128, 128, 3), np.uint8)))
    with pytest.raises(ValueError, match="SHA_MISMATCH"):
        ps.validate(root)


def test_fold_readiness_allows_absent_held_out_labels(provider):
    root, _, _ = provider
    data = ps.validate(root)
    for row in data.rows:
        if row["area"] == "8" and row["label"] == "BPD":
            row["area"] = "8x"  # simulate a held-out Area without BPD (in memory only)
    data.rows = [r for r in data.rows if r["area"] != "8x"]
    plan = {"1": "train", "2": "validation", "8": "test", "9": "train"}
    assignment = ps.split_assignment(data, plan)
    assert not ps.readiness(data, assignment, True)["ready"]
    fold = ps.readiness(data, assignment, True, fold=True)
    assert fold["ready"] and fold["mode"] == "cross_validation_fold"
    assert "BPD" in fold["absent_fine_labels"]["test"]


def test_class_weights_and_config_guards(provider):
    root, _, _ = provider
    data = ps.validate(root)
    a = ps.split_assignment(data)
    assert ps.class_weights(data, a, "none") is None
    assert ps.class_weights(data, a, "balanced") == pytest.approx([1.0] * 10)
    counts = ps.fine_counts(data, a)["train"]
    sq = ps.class_weights(data, a, "sqrt_balanced")
    mean = sum(w * counts[f] for w, f in zip(sq, ps.FINE_LABELS, strict=True)) / sum(
        counts.values()
    )
    assert mean == pytest.approx(1.0)
    for bad in (
        {"augmentation": "hflip"},
        {"augmentation": "rotate90"},
        {"architecture": "xrt_local_contrast_cnn/v1"},
        {"class_weighting": "focal"},
        {"device": "tpu"},
        {"ema_decay": 1.0},
        {"ema_decay": True},
    ):
        with pytest.raises(ValueError, match="TRAIN_CONFIG"):
            SubtypeTrainConfig(**bad)


def test_subtype_metrics():
    fine = np.zeros((4, 10))
    fine[0, 0] = 1  # BPD -> BPD
    fine[1, [1, 2]] = [0.3, 0.7]  # TED_a -> TED_b (class right, subtype wrong)
    fine[2, [7, 8]] = [0.6, 0.4]  # TSD_a -> TSD_a
    fine[3, [3, 9]] = [0.4, 0.6]  # TED_c -> TSD (class wrong)
    labels = [0, 1, 7, 3]
    assert coarse_scores(fine).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.4, 0.6]]
    s = summarize(labels, fine)
    assert s["coarse"]["accuracy"] == 0.75
    assert s["fine"]["accuracy"] == 0.5
    assert "BPD_as_TED" not in s["fine"] and "BPD_as_TED" in s["coarse"]
    assert s["fine"]["macro_f1"] is None  # absent labels
    assert s["fine"]["present_label_macro"]["labels"] == ["BPD", "TED_a", "TED_c", "TSD_a"]
    w = within_class_subtype(labels, fine)
    assert w["samples"] == 2 and w["subtype_accuracy_given_correct_class"] == 0.5
    assert w["by_class"]["TED"]["subtype_accuracy"] == 0.0
    assert w["by_class"]["TSD"]["subtype_accuracy"] == 1.0
    assert "BPD" not in w["by_class"]
    boot = paired_bootstrap([0, 1, 2, 1], [0, 1, 2, 1], [0, 1, 1, 1], resamples=50)
    assert boot["accuracy"]["difference"] == 0.25


def test_export_module_sums_coarse():
    model = SubtypeScoreExport(SubtypeClassifier(MEDIUM)).eval()
    with torch.no_grad():
        coarse, fine = model(torch.rand(3, 3, 128, 128))
    assert coarse.shape == (3, 3) and fine.shape == (3, 10)
    assert torch.allclose(coarse[:, 1], fine[:, 1:7].sum(1))
    assert torch.allclose(coarse.sum(1), torch.ones(3))


def test_exact_resume(provider, tmp_path):
    root, _, _ = provider
    data = ps.validate(root, load_pixels=True)
    cfg = replace(SMOKE, epochs=2, patience=10, ema_decay=0.9)
    train_subtype(root, tmp_path / "full", cfg, synthetic_smoke=True, data=data)
    train_subtype(root, tmp_path / "part", replace(cfg, epochs=1), synthetic_smoke=True, data=data)
    train_subtype(
        root,
        tmp_path / "resumed",
        cfg,
        resume=tmp_path / "part" / "checkpoint.pt",
        synthetic_smoke=True,
        data=data,
    )
    a = load_checkpoint(tmp_path / "full" / "checkpoint.pt")
    b = load_checkpoint(tmp_path / "resumed" / "checkpoint.pt")
    assert a["history"] == b["history"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert all(torch.equal(a["ema_model"][k], b["ema_model"][k]) for k in a["model"])
    assert not torch.equal(a["ema_model"]["head.weight"], a["model"]["head.weight"])
    with pytest.raises(ValueError, match="RESUME_CONTRACT"):
        train_subtype(
            root,
            tmp_path / "other",
            replace(cfg, class_weighting="none"),
            resume=tmp_path / "part" / "checkpoint.pt",
            synthetic_smoke=True,
            data=data,
        )


def test_train_evaluate_export_two_outputs(provider, tmp_path):
    root, _, _ = provider
    run, evaluation, bundle = tmp_path / "run", tmp_path / "eval", tmp_path / "bundle"
    cfg = replace(SMOKE, architecture=MEDIUM_BN_CENTER, ema_decay=0.5)
    report = train_subtype(root, run, cfg, synthetic_smoke=True)
    assert report["status"] == "TRAINING_PIPELINE_VERIFIED" and report["task"] == "provider_subtype"
    assert report["split_assignment"]["8"] == "test" and report["epochs_completed"] == 1
    result = evaluate_subtype(root, run / "checkpoint.pt", evaluation, synthetic_smoke=True)
    c = result["candidate"]
    assert c["coarse"]["samples"] == 20 and c["coarse"]["class_order"] == ps.CLASSES
    assert len(c["fine"]["confusion_matrix"]) == 10
    assert [r["support"] for r in c["fine"]["per_class"]] == [2] * 10
    assert set(c["within_class"]["by_class"]) == {"TED", "TSD"}
    assert result["test_areas"] == ["8"] and not result["scientific_metrics"]
    with pytest.raises(ValueError, match="FINAL_TEST_ALREADY_USED"):
        evaluate_subtype(root, run / "checkpoint.pt", tmp_path / "again", synthetic_smoke=True)
    with pytest.raises(ValueError, match="FINAL_TEST_ALREADY_USED_RESUME"):
        train_subtype(
            root, tmp_path / "r", replace(cfg, epochs=2), run / "checkpoint.pt", True
        )
    manifest = export_subtype(run / "checkpoint.pt", bundle, evaluation / "evaluation.json")
    # Exact frozen v1 blocks the Analyzer compares, plus the new subtype block.
    assert manifest["schema"] == "frozen_xrt_patch_classifier"
    assert manifest["schema_version"] == 1 and manifest["status"] == "validated_export"
    assert manifest["classes"] == ["BPD", "TED", "TSD"]
    assert manifest["output"] == {
        "name": "probabilities",
        "shape": ["N", 3],
        "meaning": "uncalibrated_class_scores",
    }
    assert manifest["input"]["conversion"] == "uint8_RGB_to_NCHW_div255"
    assert manifest["model_file"] == "model.onnx" and manifest["research_only"] is True
    assert manifest["background_class"] is False and manifest["localizes_defects"] is False
    assert manifest["subtype_output"]["name"] == "subtype_probabilities"
    assert manifest["subtype_output"]["labels"] == ps.FINE_LABELS
    assert manifest["labels_source"] == "provider" and manifest["expert_verified"] is False
    assert manifest["bundle_id"].startswith("provider-subtype-")
    assert [r["batch"] for r in manifest["parity"]["results"]] == [1, 7]
    assert manifest["model_sha256"] == _sha((bundle / "model.onnx").read_bytes())
    session, _ = frozen_session(bundle)
    x = np.random.default_rng(1).random((5, 3, 128, 128), np.float32)
    # A v1 consumer asks for "probabilities" only.
    (only,) = session.run(["probabilities"], {"images": x})
    coarse, fine = session.run(["probabilities", "subtype_probabilities"], {"images": x})
    assert np.array_equal(only, coarse) and fine.shape == (5, 10)
    assert np.allclose(coarse[:, 2], fine[:, 7:].sum(1), atol=1e-6)
    registry = json.loads((bundle / "candidate-registry.json").read_text())
    assert registry["dataset"]["schema"] == "provider_subtype_point_dataset"
    assert {p.name for p in bundle.iterdir()} == {
        "model.onnx",
        "manifest.json",
        "candidate-registry.json",
    }
    # Same model, single-output graph for consumers that require exactly one output.
    single = tmp_path / "single"
    coarse_manifest = export_subtype(
        run / "checkpoint.pt", single, evaluation / "evaluation.json", coarse_only=True
    )
    assert "subtype_output" not in coarse_manifest
    assert coarse_manifest["bundle_id"].startswith("provider-class-")
    assert coarse_manifest["parity"]["outputs"] == ["probabilities"]
    single_session, _ = frozen_session(single)
    assert [o.name for o in single_session.get_outputs()] == ["probabilities"]
    (single_coarse,) = single_session.run(None, {"images": x})
    assert np.allclose(single_coarse, coarse, atol=1e-6)
    # Same-patch comparison against an existing 3-class bundle.
    other = tmp_path / "other"
    train_subtype(root, other, SMOKE, synthetic_smoke=True)
    compared = evaluate_subtype(root, other / "checkpoint.pt", tmp_path / "cmp", bundle, True)
    assert compared["baseline"]["samples"] == 20 and compared["same_inputs_and_labels"]
    assert set(compared["candidate_minus_baseline"]) >= {"accuracy", "macro_f1"}
    with pytest.raises(ValueError, match="NEW_CANDIDATE"):
        export_subtype(run / "checkpoint.pt", bundle, evaluation / "evaluation.json")
    with pytest.raises(ValueError, match="EVALUATION_PROVENANCE"):
        export_subtype(other / "checkpoint.pt", tmp_path / "b2", evaluation / "evaluation.json")
    bad = json.loads((bundle / "manifest.json").read_text())
    bad["subtype_output"]["labels"] = list(reversed(ps.FINE_LABELS))
    (bundle / "manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="WRONG_MODEL_CONTRACT"):
        frozen_session(bundle)
    del bad["subtype_output"]  # two ONNX outputs without the block: actual graph mismatch
    (bundle / "manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="ACTUAL_ONNX_CONTRACT"):
        frozen_session(bundle)


def test_cross_validation_folds(provider, tmp_path):
    root, _, _ = provider
    data = ps.validate(root, load_pixels=True)
    plan = fold_plan(data)
    assert [next(a for a, s in p.items() if s == "test") for p in plan] == ["1", "2", "8", "9"]
    # Equal counts: validation = lowest remaining Area number; never the held-out Area.
    assert plan[0] == {"1": "test", "2": "validation", "8": "train", "9": "train"}
    assert plan[2] == {"1": "validation", "2": "train", "8": "test", "9": "train"}
    summary = cross_validate(root, tmp_path / "cv", SMOKE, synthetic_smoke=True, data=data)
    assert [f["held_out_area"] for f in summary["folds"]] == ["1", "2", "8", "9"]
    assert summary["summary"]["coarse_accuracy"]["folds"] == 4
    assert (tmp_path / "cv" / "fold-area-9" / "evaluation" / "evaluation.json").exists()


def test_cli_validate_provider(provider, monkeypatch, capsys):
    from sic_xrt_ml.__main__ import main

    root, _, _ = provider
    monkeypatch.setattr(
        sys, "argv", ["sic_xrt_ml", "validate", str(root), "--synthetic-smoke"]
    )
    main()
    out = json.loads(capsys.readouterr().out)
    assert out["ready"] and out["dataset"]["schema"] == "provider_subtype_point_dataset"
    assert out["dataset"]["labels_source"] == "provider"


def test_repository_config_parses():
    from sic_xrt_ml.data.verified import read_json

    path = Path(__file__).resolve().parents[1] / "configs" / "provider-subtype.json"
    config = SubtypeTrainConfig(**read_json(path.read_bytes()))
    assert config.augmentation == "none" and config.architecture.endswith("_mbc/v1")
