"""One entry point for verified_xrt_dataset v1 (3 classes) and v2 (4 classes).

provider_subtype_point_dataset/v1 (fine subtype labels) is detected here and routed to
its own pipeline (``provider_subtype``, ``training.subtype``) by the CLI.
"""

from dataclasses import dataclass
from pathlib import Path

from . import consensus, provider_subtype, verified


@dataclass
class PreparedDataset:
    root: Path
    schema_version: int
    manifest: dict
    classes: list
    readiness: dict
    eligibility: dict | None = None

    def dataset(self, split):
        if self.schema_version == 1:
            return verified.VerifiedDataset(self.root, split, self.manifest)
        return consensus.ConsensusDataset(
            self.root, split, self.manifest, self.eligibility
        )

    def class_weights(self):
        """None keeps v1 training unchanged; v2 balances the four classes."""
        if self.schema_version == 1:
            return None
        return consensus.class_weights(self.manifest, self.eligibility)

    def identity(self, rows):
        if self.schema_version == 1:
            return [(r["sample_id"], r["class_label"], r["crop_hash"]) for r in rows]
        return [(r["sample_id"], r["label"], r["crop"]["sha256"]) for r in rows]

    def provenance(self):
        m = self.manifest
        result = {
            "schema": m["schema"],
            "schema_version": m["schema_version"],
            "dataset_id": m["dataset_id"],
            "dataset_version": m["dataset_version"],
            "dataset_hash": m["dataset_hash"],
        }
        if self.schema_version == 2:
            result |= {
                "content_hash": m["content_hash"],
                "label_schema": m["label_schema"],
                "label_event_set_hash": m["label_event_set_hash"],
                "parameters": m["parameters"],
            }
        return result


def is_provider_subtype(root):
    """True for a provider_subtype_point_dataset folder (parallel subtype pipeline).

    ``prepare`` keeps accepting only verified_xrt_dataset v1/v2; the subtype dataset
    is validated by ``provider_subtype.validate``.
    """
    raw = verified.safe_file(Path(root).absolute(), "manifest.json", 256 * 1024**2)
    value = verified.read_json(raw.read_bytes())
    return isinstance(value, dict) and value.get("schema") == provider_subtype.SCHEMA


def schema_version(root):
    raw = verified.safe_file(Path(root).absolute(), "manifest.json", 256 * 1024**2)
    value = verified.read_json(raw.read_bytes())
    if not isinstance(value, dict) or value.get("schema") != "verified_xrt_dataset":
        raise ValueError("DATASET_SCHEMA")
    version = value.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("DATASET_SCHEMA")
    return version


def prepare(root, synthetic_smoke=False, min_per_class=None):
    root = Path(root).absolute()
    version = schema_version(root)
    if version == 1:
        if min_per_class is not None:
            raise ValueError("MIN_PER_CLASS_V2_ONLY")
        manifest = verified.validate(root)
        return PreparedDataset(
            root,
            1,
            manifest,
            list(verified.CLASSES),
            verified.readiness(manifest, synthetic_smoke),
        )
    manifest, eligibility = consensus.validate(root)
    return PreparedDataset(
        root,
        2,
        manifest,
        list(consensus.CLASSES_V2),
        consensus.readiness(manifest, eligibility, synthetic_smoke, min_per_class),
        eligibility,
    )


def parse_minimum(text):
    """``"10,5,5"`` -> {"train": 10, "validation": 5, "test": 5}."""
    if text is None:
        return None
    parts = text.split(",")
    if len(parts) != 3 or not all(p.strip().isdigit() for p in parts):
        raise ValueError("MIN_PER_CLASS")
    return consensus.minimum_gate(
        dict(zip(verified.SPLITS, (int(p) for p in parts), strict=True))
    )
