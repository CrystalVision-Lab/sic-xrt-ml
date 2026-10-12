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
