"""Strict consumer of the data-tools provider_subtype_point_dataset/v1 folder.

Producer: sic-xrt-data-tools ``provider_subtype_dataset`` (contract document
``docs/provider-subtype-dataset-v1.md`` in that repository). Only the published
folder format (JSON, CSV, uncompressed 128x128 RGB8 TIFF patches) is consumed; no
producer code is imported or copied.

Labels are provider labels from ImageJ ROI file names, not expert ground truth.
Fine labels are direction subtypes on the TIFF pixel grid: a mirrored image would
mirror the subtype, so no spatial transform is ever applied here.
"""

import csv
import hashlib
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .verified import SPLITS, read_json, safe_file

SCHEMA = "provider_subtype_point_dataset"
SCHEMA_VERSION = 1
CLASSES = ["BPD", "TED", "TSD"]
SUBTYPES = {"BPD": [], "TED": ["a", "b", "c", "d", "e", "f"], "TSD": ["a", "b", "c"]}
FINE_LABELS = ["BPD", *(f"{c}_{s}" for c in ("TED", "TSD") for s in SUBTYPES[c])]
# Coarse class index of every fine label (BPD=0, TED=1, TSD=2).
FINE_TO_CLASS = [CLASSES.index(label.split("_")[0]) for label in FINE_LABELS]
# Producer split names -> this repository's split names.
PRODUCER_SPLITS = {"train": "train", "val": "validation", "test": "test"}
PATCH = {
    "size": 128,
    "dtype": "uint8",
    "channels": "RGB",
    "format": "TIFF uncompressed",
    "resized": False,
    "normalized": False,
}
SAMPLE_FIELDS = {
    "patch_id",
    "path",
    "sha256",
    "image_id",
    "area",
    "wafer",
    "stage",
    "split",
    "label",
    "subtype",
    "fine_label",
    "x",
    "y",
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
}
# Display-only files; never model input, never required to be hashed.
PREVIEW_DIR = "preview/"
MAX_ROWS = 200000
DEFAULT_MINIMUM = {"train": 10, "validation": 5, "test": 5}
RARE_SUPPORT = 30


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _area(value):
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError("AREA_ID")
    return value


def _decode_patch(raw):
    """Exact 128x128 uncompressed 8-bit RGB single-frame TIFF -> HWC uint8 array."""
    if raw[:4] not in (b"II*\x00", b"MM\x00*"):
        raise ValueError("TIFF_RGB8_PATCH_REQUIRED")
    with Image.open(io.BytesIO(raw)) as image:
        if (
            image.format != "TIFF"
            or image.mode != "RGB"
            or image.size != (128, 128)
            or getattr(image, "n_frames", 1) != 1
            or image.info.get("compression") != "raw"
        ):
            raise ValueError("TIFF_RGB8_PATCH_REQUIRED")
        array = np.asarray(image).copy()
    if array.shape != (128, 128, 3) or array.dtype != np.uint8:
        raise ValueError("TIFF_RGB8_PATCH_REQUIRED")
    return array


def _rows(raw):
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or not SAMPLE_FIELDS <= set(reader.fieldnames):
        raise ValueError("SAMPLE_FIELDS")
    rows = list(reader)
    if not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("SAMPLE_COUNT")
    return rows


def _check_row(row, manifest):
    try:
        x, y = float(row["x"]), float(row["y"])
        center_x, center_y = int(row["center_x"]), int(row["center_y"])
        left, top = int(row["left"]), int(row["top"])
    except ValueError as exc:
        raise ValueError("SAMPLE_COORDINATES") from exc
    if (
        not re.fullmatch(r"[0-9a-f]{8,64}", row["patch_id"])
        or not re.fullmatch(r"[0-9a-f]{8,64}", row["image_id"])
        or not _sha(row["sha256"])
    ):
        raise ValueError("SAMPLE_IDENTITY")
    if row["labels_source"] != "provider" or row["expert_verified"] != "False":
        raise ValueError("LABEL_PROVENANCE")
    label, subtype = row["label"], row["subtype"]
    if label not in CLASSES or (
        subtype not in SUBTYPES[label] if SUBTYPES[label] else subtype != ""
    ):
        raise ValueError("LABEL_SET")
    if row["fine_label"] != (f"{label}_{subtype}" if subtype else label):
        raise ValueError("LABEL_SET")
    if row["split"] not in PRODUCER_SPLITS:
        raise ValueError("SPLIT_NAME")
    if _area(row["area"]) != row["wafer"]:
        raise ValueError("AREA_WAFER")
    if manifest["split_map"].get(row["area"]) != row["split"]:
        raise ValueError("AREA_SPLIT_MAP")
    if row["path"] != f"patches_128/{row['split']}/{row['fine_label']}/{row['patch_id']}.tif":
        raise ValueError("ASSET_PATH_CONTRACT")
    if (row["size"], row["dtype"], row["channels"]) != ("128", "uint8", "RGB"):
        raise ValueError("MODEL_INPUT_CONTRACT")
    if (
        not math.isfinite(x)
        or not math.isfinite(y)
        or center_x != math.floor(x)
        or center_y != math.floor(y)
        or (left, top) != (center_x - 64, center_y - 64)
        or min(left, top) < 0
    ):
        raise ValueError("SAMPLE_COORDINATES")
    if row["coordinate_transform"] not in ("identity", "hflip"):
        raise ValueError("COORDINATE_TRANSFORM")


@dataclass
class ProviderDataset:
    """Validated folder: manifest, rows (producer order) and, optionally, pixels."""

    root: Path
    manifest: dict
    rows: list
    dataset_hash: str
    output_hashes_sha256: str
    pixels: np.ndarray | None = None  # [N, 3, 128, 128] uint8, same order as rows

    @property
    def synthetic(self):
        return self.manifest.get("synthetic_fixture") is True

    def provenance(self):
        m = self.manifest
        return {
            "schema": m["schema"],
            "schema_version": m["schema_version"],
            "dataset_hash": self.dataset_hash,
            "dataset_hash_basis": "sha256(manifest.json bytes)",
            "samples_sha256": m["samples_sha256"],
            "output_hashes_sha256": self.output_hashes_sha256,
            "producer_code_revision": m["code_revision"],
            "labels_source": m["labels_source"],
            "expert_verified": m["expert_verified"],
            "split_map": m["split_map"],
        }


def validate(root, load_pixels=False):
    root = Path(root).absolute()
    manifest_raw = safe_file(root, "manifest.json", 32 * 1024**2).read_bytes()
    manifest = read_json(manifest_raw)
    if not isinstance(manifest, dict) or (
        manifest.get("schema"),
        manifest.get("schema_version"),
    ) != (SCHEMA, SCHEMA_VERSION):
        raise ValueError("DATASET_SCHEMA")
    # Research-only provider labels. Anything else is a different contract.
    if (
        manifest.get("research_only") is not True
        or manifest.get("labels_source") != "provider"
        or manifest.get("expert_verified") is not False
    ):
        raise ValueError("LABEL_PROVENANCE")
    if (
        manifest.get("classes") != CLASSES
        or manifest.get("subtypes") != SUBTYPES
        or manifest.get("fine_labels") != FINE_LABELS
    ):
        raise ValueError("LABEL_SET")
    patch = manifest.get("patch")
    if not isinstance(patch, dict) or any(patch.get(k) != v for k, v in PATCH.items()):
        raise ValueError("MODEL_INPUT_CONTRACT")
    split_map = manifest.get("split_map")
    if (
        not isinstance(split_map, dict)
        or not split_map
        or any(_area(a) != a or s not in PRODUCER_SPLITS for a, s in split_map.items())
    ):
        raise ValueError("SPLIT_MAP")
    if "synthetic_fixture" in manifest and manifest["synthetic_fixture"] is not True:
        raise ValueError("SYNTHETIC_FLAG")
    hashes_raw = safe_file(root, "output_hashes.json", 64 * 1024**2).read_bytes()
    hashes = read_json(hashes_raw)
    if (
        not isinstance(hashes, dict)
        or not all(isinstance(k, str) and _sha(v) for k, v in hashes.items())
        or "output_hashes.json" in hashes
    ):
        raise ValueError("OUTPUT_HASHES")
    if hashes.get("manifest.json") != hashlib.sha256(manifest_raw).hexdigest():
        raise ValueError("MANIFEST_SHA_MISMATCH")
    samples_raw = safe_file(root, "samples.csv", 256 * 1024**2).read_bytes()
    samples_sha = hashlib.sha256(samples_raw).hexdigest()
    if manifest.get("samples_sha256") != samples_sha or hashes.get("samples.csv") != samples_sha:
        raise ValueError("SAMPLES_SHA_MISMATCH")
    # Every hashed non-display file must exist unchanged; no unlisted files outside preview/.
    for relative, sha in hashes.items():
        if relative.startswith((PREVIEW_DIR, "patches_128/")):
            continue
        if _file_sha(safe_file(root, relative, 256 * 1024**2)) != sha:
            raise ValueError("OUTPUT_SHA_MISMATCH:" + relative)
    for path in root.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("SYMLINK_OR_JUNCTION")
        relative = path.relative_to(root).as_posix()
        if (
            path.is_file()
            and relative != "output_hashes.json"
            and relative not in hashes
            and not relative.startswith(PREVIEW_DIR)
        ):
            raise ValueError("UNREFERENCED_FILE")
    rows = _rows(samples_raw)
    images = read_json(safe_file(root, "images.json", 64 * 1024**2).read_bytes())
    if not isinstance(images, list):
        raise ValueError("IMAGES_JSON")  # noqa: TRY004 -- stable contract error
    image_area = {}
    for image in images:
        if (
            not isinstance(image, dict)
            or not isinstance(image.get("image_id"), str)
            or image["image_id"] in image_area
            or split_map.get(_area(image.get("area"))) != image.get("split")
        ):
            raise ValueError("IMAGES_JSON")
        image_area[image["image_id"]] = image["area"]
    seen_ids, seen_sha, area_splits = set(), set(), {}
    for row in rows:
        _check_row(row, manifest)
        if row["patch_id"] in seen_ids:
            raise ValueError("DUPLICATE_PATCH_ID")
        if row["sha256"] in seen_sha:
            raise ValueError("DUPLICATE_PATCH")
        seen_ids.add(row["patch_id"])
        seen_sha.add(row["sha256"])
        if image_area.get(row["image_id"]) != row["area"]:
            raise ValueError("IMAGE_AREA")
        area_splits.setdefault(row["area"], set()).add(row["split"])
        if hashes.get(row["path"]) != row["sha256"]:
            raise ValueError("OUTPUT_HASHES_PATCH")
    # Area (wafer) is the leakage unit: one Area, one split.
    if any(len(s) != 1 for s in area_splits.values()):
        raise ValueError("AREA_SPLIT_LEAKAGE")
    patch_files = {k for k in hashes if k.startswith("patches_128/")}
    if patch_files != {r["path"] for r in rows}:
        raise ValueError("FILE_INVENTORY")
    counts = {}
    for row in rows:
        by_class = counts.setdefault(row["split"], {}).setdefault(row["label"], {})
        key = row["subtype"] or "-"
        by_class[key] = by_class.get(key, 0) + 1
    recorded = manifest.get("counts")
    if (
        not isinstance(recorded, dict)
        or recorded.get("patches") != len(rows)
        or recorded.get("by_split_class_subtype") != counts
    ):
        raise ValueError("SPLIT_CLASS_COUNTS")
    pixels = (
        np.empty((len(rows), 3, 128, 128), dtype=np.uint8) if load_pixels else None
    )
    for i, row in enumerate(rows):
        raw = safe_file(root, row["path"], 1024**2).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("PATCH_SHA_MISMATCH")
        array = _decode_patch(raw)
        if pixels is not None:
            pixels[i] = array.transpose(2, 0, 1)
    return ProviderDataset(
        root,
        manifest,
        rows,
        hashlib.sha256(manifest_raw).hexdigest(),
        hashlib.sha256(hashes_raw).hexdigest(),
        pixels,
    )


def split_assignment(data, split_map=None):
    """Area -> split name. ``split_map`` overrides the producer map (cross-validation)."""
    areas = sorted({r["area"] for r in data.rows}, key=int)
    if split_map is None:
        return {a: PRODUCER_SPLITS[data.manifest["split_map"][a]] for a in areas}
    if (
        not isinstance(split_map, dict)
        or set(split_map) != set(areas)
        or any(v not in SPLITS for v in split_map.values())
        or set(split_map.values()) != set(SPLITS)
    ):
        raise ValueError("SPLIT_OVERRIDE")
    return {a: split_map[a] for a in areas}


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


def fine_counts(data, assignment):
    return {
        s: {
            f: sum(
                1
                for r in data.rows
                if r["fine_label"] == f and assignment[r["area"]] == s
            )
            for f in FINE_LABELS
        }
        for s in SPLITS
    }


def readiness(data, assignment, synthetic_smoke=False, minimum=None, fold=False):
    """Training gate. ``fold=True`` (Area cross-validation folds): a held-out Area may lack
    a class or subtype, so only train needs every coarse class (gate) and the other
    splits need any sample; absent and rare labels are reported, not blocking.
    """
    if synthetic_smoke and not data.synthetic:
        raise ValueError("SYNTHETIC_OVERRIDE_FORBIDDEN_FOR_REAL_DATA")
    gate = minimum_gate(minimum, synthetic_smoke)
    fine = fine_counts(data, assignment)
    coarse = {
        s: {
            c: sum(n for f, n in fine[s].items() if FINE_TO_CLASS[FINE_LABELS.index(f)] == i)
            for i, c in enumerate(CLASSES)
        }
        for s in SPLITS
    }
    areas = {s: sorted((a for a, v in assignment.items() if v == s), key=int) for s in SPLITS}
    # Every coarse class meets the per-split gate and every fine label is trainable.
    if fold:
        sufficient = all(coarse["train"][c] >= gate["train"] for c in CLASSES) and all(
            sum(coarse[s].values()) for s in SPLITS
        )
    else:
        sufficient = all(coarse[s][c] >= gate[s] for s in SPLITS for c in CLASSES) and all(
            fine["train"][f] >= 1 for f in FINE_LABELS
        )
    ready = (
        sufficient
        and all(areas.values())
        and (synthetic_smoke or not data.synthetic)
    )
    return {
        "status": "TRAINING_PIPELINE_VERIFIED_INPUT"
        if ready and synthetic_smoke
        else "READY"
        if ready
        else "DATA_NOT_READY",
        "ready": bool(ready),
        "synthetic": data.synthetic,
        "dataset_schema": SCHEMA,
        "classes": CLASSES,
        "fine_labels": FINE_LABELS,
        "areas_by_split": areas,
        "class_counts": coarse,
        "fine_counts": fine,
        "absent_fine_labels": {s: [f for f in FINE_LABELS if not fine[s][f]] for s in SPLITS},
        "rare_fine_labels": {
            s: [f for f in FINE_LABELS if 0 < fine[s][f] < RARE_SUPPORT] for s in SPLITS
        },
        "minimum_per_class": gate,
        "mode": "cross_validation_fold" if fold else "standard",
        "labels_source": data.manifest["labels_source"],
        "expert_verified": data.manifest["expert_verified"],
    }


def class_weights(data, assignment, mode):
    """Per fine label loss weights from train counts.

    ``balanced`` = n / (K * n_c); ``sqrt_balanced`` = sqrt of that, renormalised so the
    train-sample-weighted mean weight is 1; ``none`` = None (plain mean CE).
    """
    if mode == "none":
        return None
    counts = fine_counts(data, assignment)["train"]
    total, k = sum(counts.values()), len(FINE_LABELS)
    balanced = [total / (k * counts[f]) if counts[f] else 0.0 for f in FINE_LABELS]
    if mode == "balanced":
        return balanced
    if mode != "sqrt_balanced":
        raise ValueError("CLASS_WEIGHTING")
    root = [math.sqrt(w) for w in balanced]
    mean = sum(w * counts[f] for w, f in zip(root, FINE_LABELS, strict=True)) / total
    return [w / mean for w in root]


def indices(data, assignment, split):
    if split not in SPLITS:
        raise ValueError("SPLIT_REQUIRED")
    return [i for i, r in enumerate(data.rows) if assignment[r["area"]] == split]


def labels(data, chosen):
    return [FINE_LABELS.index(data.rows[i]["fine_label"]) for i in chosen]

