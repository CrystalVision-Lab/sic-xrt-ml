"""Strict consumer of the Analyzer verified_xrt_dataset/v2 (review-label consensus) export.

Producer: Analyzer ``server/xrt_server/label_datasets.py`` via
``python -m xrt_server.label_dataset_admin``. Only the published folder format is
consumed; no producer source is copied. Crops may be absent, gray (L) or smaller
than 128 px. Only exact 128x128 RGB8 crops match the frozen model input contract
(the Analyzer also never feeds gray or incomplete patches to the model), so other
samples are skipped and counted in ``crop_coverage`` instead of being converted.
"""

import hashlib
import io
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageStat
from torch.utils.data import Dataset

from .verified import SPLITS, digest, read_json, safe_file

SCHEMA = "verified_xrt_dataset"
SCHEMA_VERSION = 2
LABEL_SCHEMA = "review_label/v1"
BACKGROUND = "NOT_DEFECT"
CLASSES_V2 = ["BPD", "TED", "TSD", BACKGROUND]
CLASS_MAPPING = {c: i for i, c in enumerate(CLASSES_V2)}
CROP_CONTRACT = {
    "format": "png",
    "dtype": "uint8",
    "modes": ["L", "RGB"],
    "max_edge": 128,
    "geometry": "original_pixels_centered_shifted_inside_source",
    "resize": "none",
}
SPLIT_POLICY = {
    "unit": "source_sha256_component",
    "validation": 0.15,
    "test": 0.15,
    "hash": "sha256(canonical({seed, source}))",
    "dhash_max_hamming": 4,
    "dhash_min_gray_std": 2.0,
}
MANIFEST_KEYS = {
    "schema",
    "schema_version",
    "label_schema",
    "parameters",
    "label_events",
    "label_event_set_hash",
    "class_mapping",
    "background_class",
    "crop_contract",
    "split_policy",
    "included_source_sha256",
    "split_group_ids",
    "rows",
    "files",
    "file_manifest_hash",
    "split_class_counts",
    "conflicts",
    "conflict_counts",
    "excluded",
    "exclusion_counts",
    "crop_coverage",
    "readiness",
    "dataset_id",
    "dataset_version",
    "created_at",
    "created_by",
    "content_hash",
    "dataset_hash",
}
CONTENT_HASH_SKIP = {
    "dataset_id",
    "dataset_version",
    "created_at",
    "created_by",
    "dataset_hash",
    "content_hash",
}
REPORT_KEYS = ("conflicts", "conflict_counts", "excluded", "exclusion_counts", "parameters")
ROW_KEYS = {
    "sample_id",
    "source_sha256",
    "page",
    "candidate_id",
    "x",
    "y",
    "source_width",
    "source_height",
    "label",
    "class_index",
    "is_background",
    "weight",
    "n_reviewers",
    "reviewer_ids",
    "effective_event_ids",
    "random_sample",
    "crop",
    "split",
    "split_group_id",
    "evaluation_only",
    "group_mapping_ids",
    "data_origin",
    "wafer_id",
}
CROP_KEYS = {"sha256", "rect", "event_id", "image_path", "pixel_hash", "dhash"}
ORIGINS = {"real_research", "synthetic_fixture"}
MAX_ROWS = 20000
DEFAULT_MINIMUM = {"train": 10, "validation": 5, "test": 5}
# Reasons a consensus sample is not a model input (counted, never converted).
NO_CROP, CROP_NOT_128, CROP_GRAY = "NO_CROP", "CROP_NOT_128", "CROP_GRAY"
SKIP_REASONS = (NO_CROP, CROP_NOT_128, CROP_GRAY)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _int(value, low=0):
    return type(value) is int and value >= low


def _number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _strings(value, empty=False):
    return (
        isinstance(value, list)
        and (empty or bool(value))
        and all(isinstance(v, str) and v for v in value)
    )


def fingerprint(image):
    """(pixel_hash, dhash or None for near-flat crops), as the producer defines them."""
    pixel = hashlib.sha256(
        f"{image.mode}:{image.width}x{image.height}:".encode() + image.tobytes()
    ).hexdigest()
    grey = image.convert("L")
    if ImageStat.Stat(grey).stddev[0] < SPLIT_POLICY["dhash_min_gray_std"]:
        return pixel, None
    a = list(grey.resize((9, 8)).tobytes())
    return pixel, sum(
        (a[y * 9 + x] > a[y * 9 + x + 1]) << (y * 8 + x)
        for y in range(8)
        for x in range(8)
    )


def _check_row(row):
    if not isinstance(row, dict) or not ROW_KEYS <= set(row):
        raise ValueError("ROW_FIELDS")
    if (
        not _sha(row["source_sha256"])
        or not _int(row["page"])
        or not isinstance(row["candidate_id"], str)
        or not row["candidate_id"]
        or row["sample_id"]
        != digest([row["source_sha256"], row["page"], row["candidate_id"]])
    ):
        raise ValueError("SAMPLE_IDENTITY")
    if (
        row["label"] not in CLASSES_V2
        or row["class_index"] != CLASS_MAPPING[row["label"]]
        or type(row["class_index"]) is not int
        or row["is_background"] is not (row["label"] == BACKGROUND)
    ):
        raise ValueError("LABEL_CLASS_MAPPING")
    if not _number(row["weight"]) or not 0 < row["weight"] <= 1:
        raise ValueError("SAMPLE_WEIGHT")
    if (
        not _int(row["n_reviewers"], 1)
        or not _strings(row["reviewer_ids"])
        or not _strings(row["effective_event_ids"])
        or type(row["random_sample"]) is not bool
    ):
        raise ValueError("CONSENSUS_PROVENANCE")
    if (
        not _int(row["source_width"], 1)
        or not _int(row["source_height"], 1)
        or not _number(row["x"])
        or not _number(row["y"])
        or not 0 <= row["x"] <= row["source_width"]
        or not 0 <= row["y"] <= row["source_height"]
    ):
        raise ValueError("ORIGINAL_COORDINATES")
    if (
        row["split"] not in SPLITS
        or not _sha(row["split_group_id"])
        or type(row["evaluation_only"]) is not bool
    ):
        raise ValueError("SPLIT_FIELDS")
    if row["data_origin"] is not None and (
        not _strings(row["data_origin"]) or not set(row["data_origin"]) <= ORIGINS
    ):
        raise ValueError("DATA_ORIGIN")
    if row["wafer_id"] is not None and (
        not _strings(row["wafer_id"])
        or not all(re.fullmatch(r"0|[1-9][0-9]*", w) for w in row["wafer_id"])
    ):
        raise ValueError("CANONICAL_EXPLICIT_WAFER_ID_REQUIRED")
    if not isinstance(row["group_mapping_ids"], list):
        raise ValueError("GROUP_MAPPING_IDS")  # noqa: TRY004 -- stable contract error
    # Same holdout rule as v1: evaluation-only groups and Wafer 8 are test only.
    if (row["evaluation_only"] or "8" in (row["wafer_id"] or [])) and row[
        "split"
    ] != "test":
        raise ValueError("HOLDOUT_LEAKAGE")
    crop = row["crop"]
    if crop is None:
        return
    if not isinstance(crop, dict) or set(crop) != CROP_KEYS:
        raise ValueError("CROP_FIELDS")
    if (
        not _sha(crop["sha256"])
        or not _sha(crop["pixel_hash"])
        or crop["image_path"] != "images/" + crop["sha256"] + ".png"
        or not (
            crop["dhash"] is None
            or (isinstance(crop["dhash"], str) and re.fullmatch(r"[0-9a-f]{16}", crop["dhash"]))
        )
    ):
        raise ValueError("ASSET_PATH_CONTRACT")
    rect = crop["rect"]
    if (
        not isinstance(rect, list)
        or len(rect) != 4
        or any(type(v) is not int for v in rect)
    ):
        raise ValueError("CROP_RECT")
    x, y, w, h = rect
    if (
        min(x, y) < 0
        or not 0 < w <= CROP_CONTRACT["max_edge"]
        or not 0 < h <= CROP_CONTRACT["max_edge"]
        or x + w > row["source_width"]
        or y + h > row["source_height"]
    ):
        raise ValueError("CROP_RECT")


def _check_crop_file(root, crop):
    """Verify PNG bytes and decoded pixels; return the model-input skip reason or None."""
    raw = safe_file(root, crop["image_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != crop["sha256"]:
        raise ValueError("PNG_SHA_MISMATCH")
    # 8-bit gray (0) or RGB (2) only; reject 16-bit/palette before any implicit decode.
    if (
        raw[:8] != b"\x89PNG\r\n\x1a\n"
        or raw[12:16] != b"IHDR"
        or raw[24:26] not in (b"\x08\x00", b"\x08\x02")
    ):
        raise ValueError("PNG_UINT8_L_OR_RGB_REQUIRED")
    w, h = crop["rect"][2:]
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or image.mode not in CROP_CONTRACT["modes"]:
            raise ValueError("PNG_UINT8_L_OR_RGB_REQUIRED")
        if image.size != (w, h):
            raise ValueError("CROP_SIZE_RECT_MISMATCH")
        pixel, dh = fingerprint(image)
        mode = image.mode
    if pixel != crop["pixel_hash"]:
        raise ValueError("PIXEL_SHA_MISMATCH")
    if crop["dhash"] != (None if dh is None else f"{dh:016x}"):
        raise ValueError("DHASH_MISMATCH")
    if (w, h) != (128, 128):
        return CROP_NOT_128, dh
    if mode != "RGB":
        return CROP_GRAY, dh
    return None, dh


def validate(root):
    """Validate a v2 export folder. Return (manifest, {sample_id: skip reason or None})."""
    root = Path(root).absolute()
    manifest = read_json(safe_file(root, "manifest.json", 256 * 1024**2).read_bytes())
    if not isinstance(manifest, dict) or (
        manifest.get("schema"),
        manifest.get("schema_version"),
        manifest.get("label_schema"),
    ) != (SCHEMA, SCHEMA_VERSION, LABEL_SCHEMA):
        raise ValueError("DATASET_SCHEMA")
    if not MANIFEST_KEYS <= set(manifest):
        raise ValueError("MANIFEST_FIELDS")
    if (
        manifest["class_mapping"] != CLASS_MAPPING
        or manifest["background_class"] != BACKGROUND
        or manifest["crop_contract"] != CROP_CONTRACT
        or manifest["split_policy"] != SPLIT_POLICY
    ):
        raise ValueError("MODEL_INPUT_CONTRACT")
    if digest({k: v for k, v in manifest.items() if k != "dataset_hash"}) != manifest[
        "dataset_hash"
    ]:
        raise ValueError("DATASET_HASH_MISMATCH")
    if (
        digest({k: v for k, v in manifest.items() if k not in CONTENT_HASH_SKIP})
        != manifest["content_hash"]
    ):
        raise ValueError("CONTENT_HASH_MISMATCH")
    rows, files = manifest["rows"], manifest["files"]
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= MAX_ROWS
        or not isinstance(files, dict)
        or digest(files) != manifest["file_manifest_hash"]
    ):
        raise ValueError("MANIFEST_SIZE_OR_FILE_HASH")
    for row in rows:
        _check_row(row)
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("DUPLICATE_SAMPLE_ID")
    if files != {r["crop"]["image_path"]: r["crop"]["sha256"] for r in rows if r["crop"]}:
        raise ValueError("FILE_INVENTORY")
    expected = {
        "manifest.json",
        "consensus-report.json",
        *("splits/" + s + ".jsonl" for s in SPLITS),
        *files,
    }
    for path in root.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("SYMLINK_OR_JUNCTION")
        if path.is_file() and path.relative_to(root).as_posix() not in expected:
            raise ValueError("UNREFERENCED_FILE")
    report = read_json(
        safe_file(root, "consensus-report.json", 256 * 1024**2).read_bytes()
    )
    if report != {k: manifest[k] for k in REPORT_KEYS}:
        raise ValueError("REPORT_MISMATCH")
    for split in SPLITS:
        data = safe_file(root, "splits/" + split + ".jsonl", 256 * 1024**2).read_text(
            encoding="utf-8"
        )
        if [read_json(line) for line in data.splitlines()] != [
            r for r in rows if r["split"] == split
        ]:
            raise ValueError("SPLIT_FILE_MISMATCH")
    eligibility, associations, pixels = {}, {}, set()
    bands = defaultdict(list)
    for row in rows:
        reason = NO_CROP
        if row["crop"]:
            reason, dh = _check_crop_file(root, row["crop"])
            if row["crop"]["pixel_hash"] in pixels:
                raise ValueError("DUPLICATE_PIXELS")
            pixels.add(row["crop"]["pixel_hash"])
            if dh is not None:
                # Hamming <= 4 over 64 bits leaves at least one of five 13-bit bands equal.
                for band in range(5):
                    bucket = bands[(band, (dh >> (band * 13)) & 0x1FFF)]
                    for other, split in bucket:
                        if (
                            split != row["split"]
                            and (dh ^ other).bit_count()
                            <= SPLIT_POLICY["dhash_max_hamming"]
                        ):
                            raise ValueError("NEAR_DUPLICATE_LEAKAGE")
                    bucket.append((dh, row["split"]))
        eligibility[row["sample_id"]] = reason
        tokens = [
            ("source_sha256", row["source_sha256"]),
            ("split_group_id", row["split_group_id"]),
            *(("wafer_id", w) for w in row["wafer_id"] or []),
            *(("group_mapping_id", str(g)) for g in row["group_mapping_ids"]),
        ]
        for token in tokens:
            if token in associations and associations[token] != row["split"]:
                raise ValueError("GROUP_LEAKAGE:" + token[0])
            associations[token] = row["split"]
    if manifest["included_source_sha256"] != sorted(
        {r["source_sha256"] for r in rows}
    ) or manifest["split_group_ids"] != sorted({r["split_group_id"] for r in rows}):
        raise ValueError("PROVENANCE_INVENTORY")
    counts = {
        s: {c: sum(r["label"] == c for r in rows if r["split"] == s) for c in CLASSES_V2}
        for s in SPLITS
    }
    if manifest["split_class_counts"] != counts:
        raise ValueError("SPLIT_CLASS_COUNTS")
    if manifest["crop_coverage"] != round(len(files) / len(rows), 4):
        raise ValueError("CROP_COVERAGE")
    return manifest, eligibility


def coverage(manifest, eligibility):
    """Per split/class sample, model-input and skip counts (all four classes, zeros kept)."""
    result = {}
    for split in SPLITS:
        result[split] = {}
        for c in CLASSES_V2:
            rows = [
                r for r in manifest["rows"] if r["split"] == split and r["label"] == c
            ]
            skipped = Counter(eligibility[r["sample_id"]] for r in rows)
            result[split][c] = {
                "samples": len(rows),
                "with_crop": sum(r["crop"] is not None for r in rows),
                "usable": skipped.pop(None, 0),
                "skipped": {k: skipped.get(k, 0) for k in SKIP_REASONS},
            }
    total = len(manifest["rows"])
    usable = sum(v is None for v in eligibility.values())
    return {
        "crop_coverage": manifest["crop_coverage"],
        "model_input_coverage": round(usable / total, 4) if total else 0.0,
        "usable_samples": usable,
        "skipped": dict(Counter(v for v in eligibility.values() if v is not None)),
        "by_split_class": result,
    }


def minimum_gate(value=None, synthetic_smoke=False):
    if value is None:
        return {s: 1 for s in SPLITS} if synthetic_smoke else dict(DEFAULT_MINIMUM)
    if (
        not isinstance(value, dict)
        or set(value) != set(SPLITS)
        or any(type(v) is not int or not 1 <= v <= 100000 for v in value.values())
    ):
        raise ValueError("MIN_PER_CLASS")
    return {s: value[s] for s in SPLITS}


def readiness(manifest, eligibility, synthetic_smoke=False, minimum=None):
    rows = [r for r in manifest["rows"] if eligibility[r["sample_id"]] is None]
    synthetic = bool(manifest["rows"]) and all(
        r["data_origin"] == ["synthetic_fixture"] for r in manifest["rows"]
    )
    if synthetic_smoke and not synthetic:
        raise ValueError("SYNTHETIC_OVERRIDE_FORBIDDEN_FOR_REAL_DATA")
    gate = minimum_gate(minimum, synthetic_smoke)
    counts = {
        s: {c: sum(r["label"] == c for r in rows if r["split"] == s) for c in CLASSES_V2}
        for s in SPLITS
    }
    groups = {
        s: len({r["split_group_id"] for r in rows if r["split"] == s}) for s in SPLITS
    }
    sufficient = all(counts[s][c] >= gate[s] for s in SPLITS for c in CLASSES_V2)
    no_synthetic = not any(
        "synthetic_fixture" in (r["data_origin"] or []) for r in manifest["rows"]
    )
    ready = sufficient and all(groups.values()) and (synthetic_smoke or no_synthetic)
    return {
        "status": "TRAINING_PIPELINE_VERIFIED_INPUT"
        if ready and synthetic_smoke
        else "READY"
        if ready
        else "DATA_NOT_READY",
        "ready": bool(ready),
        "synthetic": synthetic,
        "dataset_schema_version": SCHEMA_VERSION,
        "classes": CLASSES_V2,
        "class_counts": counts,
        "groups": groups,
        "minimum_per_class": gate,
        "coverage": coverage(manifest, eligibility),
        "producer_readiness": manifest["readiness"],
        "usage_status": "NOT_CHECKED_OFFLINE: confirm AVAILABLE on the server before use",
    }


def class_weights(manifest, eligibility):
    """Balanced inverse-frequency weights from usable train samples (n / (K * n_c))."""
    counts = Counter(
        r["class_index"]
        for r in manifest["rows"]
        if r["split"] == "train" and eligibility[r["sample_id"]] is None
    )
    total = sum(counts.values())
    return [
        total / (len(CLASSES_V2) * counts[i]) if counts[i] else 0.0
        for i in range(len(CLASSES_V2))
    ]


class ConsensusDataset(Dataset):
    """Usable (exact 128x128 RGB8 crop) samples of one split: (x, class_index, weight)."""

    def __init__(self, root, split, manifest, eligibility):
        if split not in SPLITS:
            raise ValueError("SPLIT_REQUIRED")
        self.root = Path(root).absolute()
        self.rows = [
            r
            for r in manifest["rows"]
            if r["split"] == split and eligibility[r["sample_id"]] is None
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        raw = safe_file(self.root, row["crop"]["image_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["crop"]["sha256"]:
            raise ValueError("ASSET_CHANGED_AFTER_VALIDATION")
        with Image.open(io.BytesIO(raw)) as image:
            array = np.asarray(image).copy()
        if array.shape != (128, 128, 3) or array.dtype != np.uint8:
            raise ValueError("ASSET_CHANGED_AFTER_VALIDATION")
        return (
            torch.from_numpy(array.transpose(2, 0, 1).copy()).float().div_(255),
            row["class_index"],
            torch.tensor(float(row["weight"]), dtype=torch.float32),
        )
