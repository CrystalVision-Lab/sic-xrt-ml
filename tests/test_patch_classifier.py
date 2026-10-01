import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from sic_xrt_ml.training.patch_classifier import (
    CLASSES,
    PatchDataset,
    classification_metrics,
    dataset_info,
    evaluate_test,
    to_tensor,
    train,
)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "기록").mkdir(parents=True)
    rows = []
    for split, wafer in (("train", "1"), ("val", "2"), ("test", "8")):
        for index, label in enumerate(CLASSES):
            path = root / f"{split}_{label}.tif"
            image = np.full((16, 16, 3), 30 + index*70, dtype=np.uint8)
            tifffile.imwrite(path, image, photometric="rgb", metadata=None)
            rows.append({"patch_id": path.stem, "path": path.name,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "point_id": path.stem,
                         "source_id": wafer, "wafer": wafer, "split": split, "label": label,
                         "size": "16", "dtype": "uint8"})
    fields = list(rows[0])
    for name, content in (("samples.csv", rows), ("train_balanced_16.csv", rows[:3])):
        with (root / "기록" / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(content)
    summary = {"schema_version": 1, "task": "center_point_patch_classification_candidates",
               "classes": list(CLASSES), "patch_sizes": [16], "patch_count": 9, "review_status": "pending"}
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "validation.json").write_text(json.dumps({"status": "passed", "files_checked": 9}))
    return root


def test_preprocessing_preserves_rgb_order_and_uint16_scale():
    rgb = np.array([[[0, 127, 255]]], dtype=np.uint8)
    tensor = to_tensor(rgb)
    assert tensor.shape == (3, 1, 1)
    torch.testing.assert_close(tensor[:, 0, 0], torch.tensor([0., 127/255, 1.]))
    gray = to_tensor(np.array([[65535]], dtype=np.uint16))
    assert gray.shape == (3, 1, 1)
    assert torch.all(gray == 1)
    with pytest.raises(ValueError):
        to_tensor(np.array([[1.]], dtype=np.float32))


def test_macro_f1_exposes_majority_class_failure():
    result = classification_metrics(np.array([[0, 10, 0], [0, 80, 0], [0, 10, 0]]))
    assert result["accuracy"] == .8
    assert result["macro_f1"] < .3
    assert result["per_class"]["BPD"]["recall"] == 0


def test_balanced_list_cannot_import_validation_rows(dataset):
    path = dataset / "기록/train_balanced_16.csv"
    with (dataset / "기록/samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[3:6])
    with pytest.raises(ValueError, match="training rows"):
        dataset_info(dataset, 16)


def test_patch_hash_change_is_detected(dataset):
    info = dataset_info(dataset, 16)
    data = PatchDataset(dataset, info["rows"]["train"], 16)
    (dataset / data.rows[0]["path"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="integrity"):
        data[0]


def test_cpu_train_saves_checkpoint_and_test_is_explicit(dataset, tmp_path):
    torch.set_num_threads(2)
    before = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
    result = train(dataset, tmp_path / "runs", epochs=1, size=16, batch_size=3, device="cpu")
    run = Path(result["run_dir"])
    assert result["test_evaluated"] is False
    assert not (run / "test_metrics.json").exists()
    assert (run / "best_model.pt").is_file()
    test = evaluate_test(run, dataset, device="cpu", batch_size=3)
    assert test["metrics"]["samples"] == 3
    assert test["dataset_review_status"] == "pending"
    assert test["research_only"]
    for path, content in before.items():
        assert path.read_bytes() == content
    with pytest.raises(ValueError, match="outside"):
        train(dataset, dataset / "models", size=16, epochs=1, device="cpu")
