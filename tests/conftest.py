"""Entirely generated RGB fixtures; no company images or expert labels."""

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sic_xrt_ml.data.verified import CLASSES, CONTRACT, canonical, digest


def write_dataset(root, rows=None):
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(exist_ok=True)
    if rows is None:
        rows = []
        for i, split in enumerate(("train", "validation", "test")):
            for j, c in enumerate(CLASSES):
                array = np.random.default_rng(100 + i * 3 + j).integers(
                    0, 256, (128, 128, 3), dtype=np.uint8
                )
                image = Image.fromarray(array)
                path = root / "images" / "stage.png"
                image.save(path)
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                path.rename(root / "images" / (sha + ".png"))
                a = list(image.convert("L").resize((9, 8)).tobytes())
                dh = sum(
                    (a[y * 9 + x] > a[y * 9 + x + 1]) << (y * 8 + x)
                    for y in range(8)
                    for x in range(8)
                )
                ident = f"synthetic-{i}-{j}"
                rows.append(
                    {
                        "sample_id": ident,
                        "project_id": "synthetic-project",
                        "asset_id": digest(ident),
                        "source_id": f"source-{i}",
                        "source_hash": digest(f"source-{i}"),
                        "source_width": 256,
                        "source_height": 256,
                        "run_id": f"run-{i}",
                        "target_id": ident,
                        "review_ids": ["synthetic-review-" + ident],
                        "approval_id": "synthetic-approval-" + ident,
                        "approval_revision": 1,
                        "adjudicator_id": "synthetic-admin",
                        "approval_reason": "Synthetic functional test ONLY",
                        "approval_status": "APPROVED",
                        "review_head_hash": digest(["synthetic-review-" + ident]),
                        "group_evidence": "Generated independent synthetic groups ONLY",
                        "approval_time": "2026-10-12T00:00:00+00:00",
                        "group_mapping_id": f"mapping-{i}",
                        "group_mapping_revision": 1,
                        "class_label": c,
                        "crop_hash": sha,
                        "pixel_hash": hashlib.sha256(array.tobytes()).hexdigest(),
                        "dhash": f"{dh:016x}",
                        "crop": [j * 10, j * 10, 128, 128],
                        "image_path": "images/" + sha + ".png",
                        "model_provenance": {
                            "model_id": "synthetic-not-a-real-model",
                            "model_sha256": "a" * 64,
                        },
                        "group_id": f"group-{i}",
                        "wafer_id": "8" if split == "test" else str(i),
                        "provider_group": f"provider-{i}",
                        "before_after_group": None,
                        "duplicate_group": None,
                        "evaluation_only": split == "test",
                        "data_origin": "synthetic_fixture",
                        "training_use_authorized": True,
                        "split": split,
                        "split_group_id": digest(f"group-{i}"),
                        "leakage_component": digest(f"group-{i}"),
                    }
                )
    files = {r["image_path"]: r["crop_hash"] for r in rows}
    report = {
        "collected_crops": len(rows),
        "approved_usable_crops": len(rows),
        "excluded": [],
        "synthetic_fixture": True,
    }
    manifest = {
        "schema": "verified_xrt_dataset",
        "schema_version": 1,
        "dataset_id": "synthetic-functional-fixture",
        "dataset_version": "synthetic-v1",
        "created_at": "2026-10-12T00:00:00+00:00",
        "created_by": "synthetic-admin",
        "project_ids": sorted({r["project_id"] for r in rows}),
        "included_source_ids": sorted({r["source_id"] for r in rows}),
        "included_review_ids": sorted({r for row in rows for r in row["review_ids"]}),
        "approval_ids": sorted({r["approval_id"] for r in rows}),
        "split_group_ids": sorted({r["split_group_id"] for r in rows}),
        "class_mapping": {c: i for i, c in enumerate(CLASSES)},
        "preprocessing_contract": deepcopy(CONTRACT),
        "seed": 2026,
        "rows": rows,
        "files": files,
        "file_manifest_hash": digest(files),
        "split_class_counts": {
            s: {
                c: sum(r["split"] == s and r["class_label"] == c for r in rows)
                for c in CLASSES
            }
            for s in ("train", "validation", "test")
        },
        "eligibility_report": report,
        "readiness": "DATA_NOT_READY",
    }
    manifest["dataset_hash"] = digest(manifest)
    save_manifest(root, manifest)
    return root, manifest


def save_manifest(root, m):
    m["dataset_hash"] = digest({k: v for k, v in m.items() if k != "dataset_hash"})
    (root / "manifest.json").write_bytes(canonical(m))
    (root / "eligibility-report.json").write_bytes(canonical(m["eligibility_report"]))
    for split in ("train", "validation", "test"):
        (root / "splits" / (split + ".jsonl")).write_bytes(
            b"".join(canonical(r) + b"\n" for r in m["rows"] if r["split"] == split)
        )


@pytest.fixture
def dataset(tmp_path):
    return write_dataset(tmp_path / "synthetic-dataset")


# ---- verified_xrt_dataset/v2 (Analyzer review-label consensus export) ----------
V2_CLASSES = ["BPD", "TED", "TSD", "NOT_DEFECT"]


def _png(array):
    import io

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def v2_row(root, split, label, index, array=None, crop=True, origin="synthetic_fixture"):
    """One consensus row shaped like label_datasets.consensus()/apply_rights()."""
    from sic_xrt_ml.data.consensus import fingerprint

    source = digest(f"v2-source-{split}-{label}-{index}")
    candidate = f"cand-{index}"
    row = {
        "sample_id": digest([source, 0, candidate]),
        "source_sha256": source,
        "page": 0,
        "candidate_id": candidate,
        "x": 300.0,
        "y": 200.0,
        "source_width": 1024,
        "source_height": 768,
        "label": label,
        "class_index": V2_CLASSES.index(label),
        "is_background": label == "NOT_DEFECT",
        "weight": 1.0 if index % 2 == 0 else 0.5,
        "n_reviewers": 1,
        "n_unsure": 0,
        "agreement": 1.0,
        "votes": {label: 1},
        "modes": ["key"],
        "batch_implicit_only": False,
        "queue_kinds": ["random" if index % 2 == 0 else "uncertain"],
        "random_sample": index % 2 == 0,
        "random_seeds": [7] if index % 2 == 0 else [],
        "predicted": "TED",
        "scores": [0.2, 0.5, 0.3],
        "model_version": "synthetic-not-a-real-model",
        "model_sha256": "a" * 64,
        "model_agrees": label == "TED",
        "model_predictions": [
            {
                "model_version": "synthetic-not-a-real-model",
                "model_sha256": "a" * 64,
                "predicted": "TED",
                "scores": [0.2, 0.5, 0.3],
            }
        ],
        "reviewer_ids": ["synthetic-reviewer"],
        "effective_event_ids": [digest(["event", source])[:32]],
        "history_event_ids": [digest(["event", source])[:32]],
        "crop": None,
        "split": split,
        "split_group_id": digest(sorted([source])),
        "evaluation_only": False,
        "group_mapping_ids": [],
        "data_origin": [origin],
        "wafer_id": None,
    }
    if crop:
        if array is None:
            seed = int(source[:8], 16)
            array = np.random.default_rng(seed).integers(
                0, 256, (128, 128, 3), dtype=np.uint8
            )
            # Class-dependent brightness so a couple of epochs can learn something.
            array[40:88, 40:88] = 60 * V2_CLASSES.index(label)
        raw = _png(array)
        sha = hashlib.sha256(raw).hexdigest()
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "images" / (sha + ".png")).write_bytes(raw)
        with Image.open(root / "images" / (sha + ".png")) as image:
            pixel, dh = fingerprint(image)
        h, w = array.shape[:2]
        row["crop"] = {
            "sha256": sha,
            "rect": [236, 136, w, h],
            "event_id": row["effective_event_ids"][0],
            "image_path": "images/" + sha + ".png",
            "pixel_hash": pixel,
            "dhash": None if dh is None else f"{dh:016x}",
        }
    return row


def write_v2_dataset(root, per_class=2, extras=True):
    """Generated v2 export: 4 classes x 3 splits plus no-crop/gray/small samples."""
    import json

    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(exist_ok=True)
    rows = []
    for split in ("train", "validation", "test"):
        for label in V2_CLASSES:
            for i in range(per_class):
                rows.append(v2_row(root, split, label, i))
    if extras:
        rows.append(v2_row(root, "train", "BPD", 100, crop=False))
        gray = np.random.default_rng(5).integers(0, 256, (128, 128), dtype=np.uint8)
        rows.append(v2_row(root, "train", "NOT_DEFECT", 101, array=gray))
        small = np.random.default_rng(6).integers(0, 256, (64, 96, 3), dtype=np.uint8)
        rows.append(v2_row(root, "validation", "TSD", 102, array=small))
    rows.sort(key=lambda r: r["sample_id"])
    manifest = {
        "schema": "verified_xrt_dataset",
        "schema_version": 2,
        "label_schema": "review_label/v1",
        "parameters": {
            "seed": 0,
            "min_reviewers": 1,
            "consensus": "majority",
            "implicit_weight": 1.0,
            "received_before": "2026-10-19T00:00:00+00:00",
        },
        "label_events": len(rows),
        "label_event_set_hash": digest(sorted(r["effective_event_ids"] for r in rows)),
        "class_mapping": {c: i for i, c in enumerate(V2_CLASSES)},
        "background_class": "NOT_DEFECT",
        "crop_contract": {
            "format": "png",
            "dtype": "uint8",
            "modes": ["L", "RGB"],
            "max_edge": 128,
            "geometry": "original_pixels_centered_shifted_inside_source",
            "resize": "none",
        },
        "split_policy": {
            "unit": "source_sha256_component",
            "validation": 0.15,
            "test": 0.15,
            "hash": "sha256(canonical({seed, source}))",
            "dhash_max_hamming": 4,
            "dhash_min_gray_std": 2.0,
        },
        "rows": rows,
        "conflicts": [],
        "conflict_counts": {},
        "excluded": [],
        "exclusion_counts": {},
        "readiness": "DATA_NOT_READY",
        "dataset_id": "00000000-0000-4000-8000-000000000002",
        "dataset_version": "synthetic-v2",
        "created_at": "2026-10-19T00:01:00+00:00",
        "created_by": "synthetic-admin",
    }
    save_v2_manifest(root, manifest)
    return root, json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def save_v2_manifest(root, m):
    """Recompute derived fields/hashes and write files as label_datasets.export_version()."""
    import json

    rows = m["rows"]
    m["included_source_sha256"] = sorted({r["source_sha256"] for r in rows})
    m["split_group_ids"] = sorted({r["split_group_id"] for r in rows})
    m["files"] = {r["crop"]["image_path"]: r["crop"]["sha256"] for r in rows if r["crop"]}
    m["file_manifest_hash"] = digest(m["files"])
    m["split_class_counts"] = {
        s: {c: sum(r["label"] == c for r in rows if r["split"] == s) for c in V2_CLASSES}
        for s in ("train", "validation", "test")
    }
    m["crop_coverage"] = round(len(m["files"]) / len(rows), 4) if rows else 0.0
    skip = {"dataset_id", "dataset_version", "created_at", "created_by", "dataset_hash",
            "content_hash"}
    m.pop("dataset_hash", None)
    m["content_hash"] = digest({k: v for k, v in m.items() if k not in skip})
    m["dataset_hash"] = digest(m)

    def write(name, value):
        (root / name).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    for split in ("train", "validation", "test"):
        (root / "splits" / (split + ".jsonl")).write_text(
            "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                    for r in rows if r["split"] == split),
            encoding="utf-8",
        )
    write("consensus-report.json", {k: m[k] for k in (
        "conflicts", "conflict_counts", "excluded", "exclusion_counts", "parameters")})
    write("manifest.json", m)


@pytest.fixture
def dataset_v2(tmp_path):
    return write_v2_dataset(tmp_path / "synthetic-v2-export")
