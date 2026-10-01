import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from sic_xrt_ml.training.center_comparison import compare_centers
from sic_xrt_ml.training.patch_classifier import (
    CLASSES,
    PatchDataset,
    classification_metrics,
    crop_image,
    dataset_info,
    evaluate_test,
    to_tensor,
    train,
)
from sic_xrt_ml.training.validation_review import generate_review, select_examples


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
                         "size": "16", "dtype": "uint8", "x": "7.5", "y": "7.5", "left": "0", "top": "0"})
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


def test_review_predicts_only_validation_and_keeps_inputs_unchanged(dataset, tmp_path):
    torch.set_num_threads(2)
    result = train(dataset, tmp_path / 'runs', epochs=1, size=16, batch_size=3, device='cpu')
    run = Path(result['run_dir'])
    snapshot = {p: p.read_bytes() for folder in (dataset, run) for p in folder.rglob('*') if p.is_file()}
    output = generate_review(run, dataset, device='cpu')
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['files_predicted'] == 3
    assert summary['split'] == 'val'
    assert summary['test_evaluated'] is False
    assert summary['metrics']['confusion_matrix'] == json.loads((run / 'history.json').read_text())[0]['val']['confusion_matrix']
    assert not (run / 'test_metrics.json').exists()
    with (output / 'val_predictions.csv').open(encoding='utf-8-sig') as stream:
        assert {r['split'] for r in csv.DictReader(stream)} == {'val'}
    page = (output / '검수.html').read_text(encoding='utf-8')
    assert 'data:image/png;base64,' in page and 'PLACEHOLDER_' not in page
    assert '검수 의견 저장' in page
    assert summary['dataset_review_status'] == 'pending'
    for path, contents in snapshot.items():
        assert path.read_bytes() == contents
    again = generate_review(run, dataset, device='cpu')
    assert output != again


def test_review_selection_has_high_score_random_and_correct_controls():
    rows = [{'patch_id': f'p{i}', 'label': 'BPD', 'prediction': 'TSD', 'model_score': i/100} for i in range(100)]
    rows += [{'patch_id': f'c{i}', 'label': 'BPD', 'prediction': 'BPD', 'model_score': .5} for i in range(20)]
    selected = select_examples(rows)
    assert len(selected) == 16
    assert len({r['patch_id'] for r in selected}) == 16
    assert {'p94', 'p95', 'p96', 'p97', 'p98', 'p99'} <= {r['patch_id'] for r in selected}
    assert select_examples(rows) == selected


def test_center_crop_retains_target_and_removes_outer_context():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[64, 64] = (20, 40, 60)
    image[10, 10] = (255, 255, 255)
    smaller = crop_image(image, 32)
    assert smaller.shape == (32, 32, 3)
    assert (smaller[16, 16] == (20, 40, 60)).all()
    assert not (smaller == 255).any()
    assert image[10, 10, 0] == 255
    with pytest.raises(ValueError):
        crop_image(image, 31)


def test_crop_checkpoint_is_applied_in_evaluation_and_review(dataset, tmp_path):
    import base64
    import io

    from PIL import Image

    from sic_xrt_ml.training.validation_review import preview_data

    torch.set_num_threads(2)
    result = train(dataset, tmp_path / 'runs', epochs=1, size=16, center_crop=8, batch_size=3, device='cpu')
    run = Path(result['run_dir'])
    checkpoint = torch.load(run / 'best_model.pt', weights_only=True)
    assert checkpoint['config']['input_size'] == 8
    output = generate_review(run, dataset, device='cpu')
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['metrics']['confusion_matrix'] == checkpoint['validation']['confusion_matrix']
    assert summary['input_size'] == 8
    row = dataset_info(dataset, 16)['rows']['val'][0]
    assert Image.open(io.BytesIO(base64.b64decode(preview_data(dataset, row, 8)))).size == (8, 8)
    assert evaluate_test(run, dataset, device='cpu')['input_size'] == 8
    checkpoint['config']['center_crop'] = None
    torch.save(checkpoint, run / 'best_model.pt')
    with pytest.raises(ValueError, match='contract'):
        generate_review(run, dataset, device='cpu')


def test_fixed_comparison_preserves_baseline_and_never_evaluates_test(dataset, tmp_path):
    torch.set_num_threads(2)
    result = train(dataset, tmp_path / 'runs', epochs=1, size=16, batch_size=3, device='cpu')
    baseline = Path(result['run_dir'])
    # Legacy checkpoints have no crop fields and remain compatible.
    checkpoint = torch.load(baseline / 'best_model.pt', weights_only=True)
    checkpoint['config'].pop('center_crop')
    checkpoint['config'].pop('input_size')
    torch.save(checkpoint, baseline / 'best_model.pt')
    before = {p: p.read_bytes() for folder in (dataset, baseline) for p in folder.rglob('*') if p.is_file()}
    summary = compare_centers(dataset, baseline, tmp_path / 'comparison', crops=(8,), device='cpu')
    assert summary['status'] == 'completed'
    assert [r['input_size'] for r in summary['records']] == [16, 8]
    assert summary['test_evaluated'] is False
    output = Path(summary['output_dir'])
    assert (output / 'comparison.png').is_file()
    plan = json.loads((output / 'plan.json').read_text())
    assert plan['input_sizes'] == [16, 8]
    assert not list(output.rglob('test_metrics.json'))
    for p, content in before.items():
        assert p.read_bytes() == content
