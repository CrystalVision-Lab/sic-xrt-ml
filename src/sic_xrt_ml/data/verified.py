"""Strict consumer of the published verified_xrt_dataset/v1 artifact contract."""

import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASSES = ["BPD", "TED", "TSD"]
SPLITS = ("train", "validation", "test")
CONTRACT = {
    "shape": [3, 128, 128],
    "source": "uint8_RGB",
    "conversion": "uint8_RGB_to_NCHW_div255",
    "crop_policy": "exact_128_no_resize",
    "local_contrast_inside_model": True,
}


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NON_FINITE_JSON")),
    )


def safe_file(root, relative, limit=4 * 1024**2):
    p = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or ":" in relative
        or p.is_absolute()
        or ".." in p.parts
        or str(p) != relative
    ):
        raise ValueError("UNSAFE_RELATIVE_PATH")
    path = root.joinpath(*p.parts)
    if any(
        x.is_symlink() or (hasattr(x, "is_junction") and x.is_junction())
        for x in (path, *path.parents)
    ):
        raise ValueError("SYMLINK_OR_JUNCTION")
    if (
        not path.is_file()
        or not path.resolve().is_relative_to(root.resolve())
        or path.stat().st_size > limit
    ):
        raise ValueError("MISSING_OR_OVERSIZED_FILE")
    return path


def validate(root):
    root = Path(root).absolute()
    manifest = read_json(safe_file(root, "manifest.json", 32 * 1024**2).read_bytes())
    if (manifest.get("schema"), manifest.get("schema_version")) != (
        "verified_xrt_dataset",
        1,
    ):
        raise ValueError("DATASET_SCHEMA")
    if (
        manifest.get("class_mapping") != {c: i for i, c in enumerate(CLASSES)}
        or manifest.get("preprocessing_contract") != CONTRACT
    ):
        raise ValueError("MODEL_INPUT_CONTRACT")
    if digest(
        {k: v for k, v in manifest.items() if k != "dataset_hash"}
    ) != manifest.get("dataset_hash"):
        raise ValueError("DATASET_HASH_MISMATCH")
    rows, files = manifest["rows"], manifest["files"]
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= 4096
        or not isinstance(files, dict)
        or digest(files) != manifest["file_manifest_hash"]
    ):
        raise ValueError("MANIFEST_SIZE_OR_FILE_HASH")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("DUPLICATE_SAMPLE_ID")
    wanted = {r["image_path"]: r["crop_hash"] for r in rows}
    if wanted != files:
        raise ValueError("FILE_INVENTORY")
    expected = {
        "manifest.json",
        "eligibility-report.json",
        *("splits/" + s + ".jsonl" for s in SPLITS),
        *files,
    }
    for path in root.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("SYMLINK_OR_JUNCTION")
        if path.is_file() and path.relative_to(root).as_posix() not in expected:
            raise ValueError("UNREFERENCED_FILE")
    report = read_json(
        safe_file(root, "eligibility-report.json", 32 * 1024**2).read_bytes()
    )
    if report != manifest["eligibility_report"]:
        raise ValueError("REPORT_MISMATCH")
    for split in SPLITS:
        data = safe_file(root, "splits/" + split + ".jsonl", 32 * 1024**2).read_text(
            encoding="utf-8"
        )
        actual = [read_json(line) for line in data.splitlines()]
        if actual != [r for r in rows if r["split"] == split]:
            raise ValueError("SPLIT_FILE_MISMATCH")
    associations, pixels, dhashes = {}, set(), []
    for row in rows:
        if (
            row["split"] not in SPLITS
            or row["class_label"] not in CLASSES
            or row["training_use_authorized"] is not True
        ):
            raise ValueError("LABEL_SPLIT_OR_RIGHTS")
        if (
            row["data_origin"] not in {"real_research", "synthetic_fixture"}
            or type(row["evaluation_only"]) is not bool
        ):
            raise ValueError("DATA_ORIGIN_OR_HOLDOUT")
        if row["wafer_id"] is not None and (
            not isinstance(row["wafer_id"], str)
            or not re.fullmatch(r"0|[1-9][0-9]*", row["wafer_id"])
        ):
            raise ValueError("CANONICAL_EXPLICIT_WAFER_ID_REQUIRED")
        if (row["evaluation_only"] or row["wafer_id"] == "8") and row[
            "split"
        ] != "test":
            raise ValueError("HOLDOUT_LEAKAGE")
        for key in (
            "approval_id",
            "adjudicator_id",
            "approval_time",
            "approval_reason",
            "group_evidence",
            "group_mapping_id",
            "group_id",
            "provider_group",
            "source_id",
            "run_id",
            "target_id",
            "project_id",
        ):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError("LABEL_OR_SOURCE_PROVENANCE")
        if row["approval_status"] != "APPROVED" or row["review_head_hash"] != digest(
            sorted(row["review_ids"])
        ):
            raise ValueError("APPROVAL_HEAD_PROVENANCE")
        if (
            not row["review_ids"]
            or type(row["approval_revision"]) is not int
            or row["approval_revision"] < 1
        ):
            raise ValueError("APPROVAL_PROVENANCE")
        if not row.get("model_provenance"):
            raise ValueError("MODEL_PROVENANCE")
        for key in (
            "source_hash",
            "crop_hash",
            "pixel_hash",
            "split_group_id",
            "leakage_component",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("SHA_FORMAT")
        if row["image_path"] != "images/" + row["crop_hash"] + ".png":
            raise ValueError("ASSET_PATH_CONTRACT")
        x, y, w, h = row["crop"]
        if (
            any(
                type(v) is not int
                for v in (x, y, w, h, row["source_width"], row["source_height"])
            )
            or min(x, y) < 0
            or (w, h) != (128, 128)
            or x + w > row["source_width"]
            or y + h > row["source_height"]
        ):
            raise ValueError("ORIGINAL_COORDINATES")
        raw = safe_file(root, row["image_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["crop_hash"]:
            raise ValueError("PNG_SHA_MISMATCH")
        if (
            raw[:8] != b"\x89PNG\r\n\x1a\n"
            or raw[12:16] != b"IHDR"
            or raw[24:26] != b"\x08\x02"
        ):
            raise ValueError("PNG_RGB8_ENCODING_REQUIRED")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (128, 128):
                raise ValueError("RGB8_EXACT_PATCH_REQUIRED")
            if hashlib.sha256(image.tobytes()).hexdigest() != row["pixel_hash"]:
                raise ValueError("PIXEL_SHA_MISMATCH")
            a = list(image.convert("L").resize((9, 8)).tobytes())
            dh = sum(
                (a[y * 9 + x] > a[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8)
                for x in range(8)
            )
        if row["dhash"] != f"{dh:016x}":
            raise ValueError("DHASH_MISMATCH")
        for old, split in dhashes:
            if (dh ^ old).bit_count() <= 4 and split != row["split"]:
                raise ValueError("NEAR_DUPLICATE_LEAKAGE")
        dhashes.append((dh, row["split"]))
        if row["pixel_hash"] in pixels:
            raise ValueError("DUPLICATE_PIXELS")
        pixels.add(row["pixel_hash"])
        for key in (
            "source_hash",
            "wafer_id",
            "group_id",
            "run_id",
            "target_id",
            "provider_group",
            "before_after_group",
            "duplicate_group",
            "split_group_id",
            "leakage_component",
        ):
            if row[key] is None:
                continue
            token = (key, row[key])
            if token in associations and associations[token] != row["split"]:
                raise ValueError("GROUP_LEAKAGE:" + key)
            associations[token] = row["split"]
    for key, values in (
        ("project_ids", {r["project_id"] for r in rows}),
        ("included_source_ids", {r["source_id"] for r in rows}),
        ("included_review_ids", {i for r in rows for i in r["review_ids"]}),
        ("approval_ids", {r["approval_id"] for r in rows}),
        ("split_group_ids", {r["split_group_id"] for r in rows}),
    ):
        if key != "project_ids" and manifest[key] != sorted(values):
            raise ValueError("PROVENANCE_INVENTORY")
        if key == "project_ids" and not values.issubset(set(manifest[key])):
            raise ValueError("PROJECT_INVENTORY")
    expected_counts = {
        s: dict(Counter(r["class_label"] for r in rows if r["split"] == s))
        for s in SPLITS
    }
    if manifest["split_class_counts"] != expected_counts:
        raise ValueError("SPLIT_CLASS_COUNTS")
    return manifest


def readiness(manifest, synthetic_smoke=False):
    rows = manifest["rows"]
    synthetic = all(r["data_origin"] == "synthetic_fixture" for r in rows)
    counts = {
        s: dict(Counter(r["class_label"] for r in rows if r["split"] == s))
        for s in SPLITS
    }
    minimum = {"train": 10, "validation": 5, "test": 5}
    if synthetic_smoke and not synthetic:
        raise ValueError("SYNTHETIC_OVERRIDE_FORBIDDEN_FOR_REAL_DATA")
    sufficient = all(
        counts[s].get(c, 0) >= (1 if synthetic_smoke else minimum[s])
        for s in SPLITS
        for c in CLASSES
    )
    groups = {
        s: len({r["split_group_id"] for r in rows if r["split"] == s}) for s in SPLITS
    }
    ready = (
        sufficient
        and all(groups.values())
        and (
            synthetic_smoke
            or not any(r["data_origin"] != "real_research" for r in rows)
        )
    )
    return {
        "status": "TRAINING_PIPELINE_VERIFIED_INPUT"
        if ready and synthetic_smoke
        else "READY"
        if ready
        else "DATA_NOT_READY",
        "ready": bool(ready),
        "synthetic": synthetic,
        "class_counts": counts,
        "groups": groups,
        "minimum_per_class": {
            s: (1 if synthetic_smoke else minimum[s]) for s in SPLITS
        },
    }


class VerifiedDataset(Dataset):
    def __init__(self, root, split, manifest=None):
        self.root = Path(root).absolute()
        self.manifest = validate(self.root) if manifest is None else manifest
        if split not in SPLITS:
            raise ValueError("SPLIT_REQUIRED")
        self.rows = [r for r in self.manifest["rows"] if r["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = safe_file(self.root, row["image_path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["crop_hash"]:
            raise ValueError("ASSET_CHANGED_AFTER_VALIDATION")
        with Image.open(io.BytesIO(raw)) as image:
            array = np.asarray(image).copy()
        # No augmentation or spatial transform. Internal contrast is in the model.
        return torch.from_numpy(array.transpose(2, 0, 1).copy()).float().div_(
            255
        ), CLASSES.index(row["class_label"])


def loader(dataset, batch_size, shuffle=False, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )
