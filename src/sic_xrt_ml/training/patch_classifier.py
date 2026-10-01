"""Research baseline for schema-v1 centered TIFF patches, not a defect detector."""

import argparse
import csv
import hashlib
import json
import random
import time
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CLASSES = ("BPD", "TED", "TSD")
PREPROCESSING = "uint8/255_or_uint16/65535_to_float32_CHW_RGB_grayscale_repeated"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def dataset_info(root: Path, size: int = 128, balanced: bool = True) -> dict:
    root = root.resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    if (summary.get("schema_version") != 1 or summary.get("task") !=
            "center_point_patch_classification_candidates" or (root / "BUILD_FAILED.json").exists()):
        raise ValueError("Expected a complete point-patch dataset with schema_version=1")
    if tuple(summary.get("classes", [])) != CLASSES or size not in summary["patch_sizes"]:
        raise ValueError("Unsupported class order or patch size")
    if validation.get("status") != "passed" or validation.get("files_checked") != summary["patch_count"]:
        raise ValueError("Dataset structural validation must pass first")
    manifest = root / "기록" / "samples.csv"
    manifest_hash = file_hash(manifest)
    provenance = root / "provenance.json"
    if provenance.exists():
        expected = json.loads(provenance.read_text(encoding="utf-8")).get("manifest_sha256")
        if expected and expected != manifest_hash:
            raise ValueError("Manifest changed after dataset validation")
    rows = read_csv(manifest)
    if len(rows) != summary["patch_count"] or len({r["patch_id"] for r in rows}) != len(rows):
        raise ValueError("Invalid manifest row count or duplicate patch IDs")
    groups = defaultdict(set)
    for row in rows:
        groups[row["wafer"]].add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("The same wafer occurs in different splits")
    chosen = [row for row in rows if int(row["size"]) == size]
    splits = {name: [r for r in chosen if r["split"] == name] for name in ("train", "val", "test")}
    train_csv_hash = manifest_hash
    if balanced:
        path = root / "기록" / f"train_balanced_{size}.csv"
        candidate = read_csv(path)
        canonical = {r["patch_id"]: r for r in splits["train"]}
        if (not candidate or len({r["patch_id"] for r in candidate}) != len(candidate)
                or any(canonical.get(r["patch_id"]) != r for r in candidate)):
            raise ValueError("Balanced CSV must contain unique, unchanged training rows only")
        splits["train"] = candidate
        train_csv_hash = file_hash(path)
    for name, items in splits.items():
        if not items or {r["label"] for r in items} != set(CLASSES):
            raise ValueError(f"Every split must contain all classes: {name}")
        for row in items:
            path = (root / row["path"]).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError("Patch is missing or outside dataset root")
    return {"root": root, "size": size, "rows": splits,
            "manifest_sha256": manifest_hash, "train_csv_sha256": train_csv_hash,
            "review_status": summary.get("review_status", "unknown"),
            "counts": {name: dict(Counter(r["label"] for r in items)) for name, items in splits.items()},
            "wafer_groups": {name: sorted({r["wafer"] for r in items}) for name, items in splits.items()}}


def to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError("Only uint8/uint16 patches are supported; no inferred float normalization")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected grayscale YX or RGB YXS")
    normalized = image.astype(np.float32) / np.iinfo(image.dtype).max
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


def validate_crop(size: int, center_crop: int | None):
    if center_crop is not None and (not isinstance(center_crop, int) or center_crop < 8
                                    or center_crop > size or (size-center_crop) % 2):
        raise ValueError("Center crop must be >=8, <= source size, with equal integer margins")


def crop_image(image: np.ndarray, center_crop: int | None = None):
    if image.shape[0] != image.shape[1]:
        raise ValueError("Centered inputs require square patches")
    validate_crop(image.shape[0], center_crop)
    if center_crop is None:
        return image
    start = (image.shape[0]-center_crop)//2
    return image[start:start+center_crop, start:start+center_crop]


def preprocessing_contract(center_crop: int | None = None):
    return PREPROCESSING if center_crop is None else PREPROCESSING + f"_center_crop{center_crop}_pixels_no_resize"


def checkpoint_crop(config: dict):
    center_crop = config.get('center_crop')
    validate_crop(config['size'], center_crop)
    if (tuple(config['classes']) != CLASSES or config['preprocessing'] != preprocessing_contract(center_crop)
            or config['model'] != 'SmallPatchCNN_v1_from_scratch'
            or config.get('input_size', center_crop or config['size']) != (center_crop or config['size'])):
        raise ValueError("Unsupported checkpoint input/model contract")
    return center_crop


class PatchDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], size: int, center_crop: int | None = None):
        validate_crop(size, center_crop)
        self.root, self.rows, self.size = root.resolve(), rows, size
        self.center_crop = center_crop

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["path"]).resolve()
        if self.root not in path.parents or file_hash(path) != row["sha256"]:
            raise ValueError(f"Patch integrity check failed: {row['patch_id']}")
        image = tifffile.imread(path)
        if image.shape[:2] != (self.size, self.size) or str(image.dtype) != row["dtype"]:
            raise ValueError("Patch shape or dtype differs from manifest")
        return to_tensor(crop_image(image, self.center_crop)), CLASSES.index(row["label"])


class SmallPatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for width in (16, 32, 64):
            layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64*4*4, 128),
                                        nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, len(CLASSES)))

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def classification_metrics(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=np.int64)
    per_class = {}
    for index, label in enumerate(CLASSES):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {"accuracy": float(np.trace(matrix)/matrix.sum()) if matrix.sum() else 0.0,
            "macro_f1": float(np.mean([m["f1"] for m in per_class.values()])),
            "per_class": per_class, "confusion_matrix": matrix.tolist(),
            "confusion_axes": "rows=actual,columns=predicted", "class_order": list(CLASSES)}


def run_epoch(model, loader, device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    matrix, total_loss, total = np.zeros((3, 3), dtype=np.int64), 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            if training:
                loss.backward()
                optimizer.step()
            prediction = logits.detach().argmax(dim=1).cpu().numpy()
            actual = targets.cpu().numpy()
            np.add.at(matrix, (actual, prediction), 1)
            total_loss += float(loss.detach()) * len(targets)
            total += len(targets)
    if total == 0:
        raise ValueError("Empty data loader")
    return classification_metrics(matrix) | {"loss": total_loss/total, "samples": total}


def train(dataset_root: Path, output_root: Path, *, epochs=15, batch_size=32,
          learning_rate=0.001, size=128, balanced=True, seed=42, device="cuda",
          center_crop: int | None = None) -> dict:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    info = dataset_info(dataset_root, size, balanced)
    validate_crop(size, center_crop)
    output_root = output_root.resolve()
    if output_root == info["root"] or info["root"] in output_root.parents:
        raise ValueError("Training outputs must be outside the read-only dataset")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select the SiC XRT GPU kernel")
    torch_device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    run_dir = output_root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {"dataset_schema": 1, "dataset_root": str(info["root"]), "size": size,
              "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
              "balanced_train": balanced, "seed": seed, "device": str(torch_device),
              "torch": str(torch.__version__), "manifest_sha256": info["manifest_sha256"],
              "train_csv_sha256": info["train_csv_sha256"], "classes": list(CLASSES),
              "counts": info["counts"], "wafer_groups": info["wafer_groups"],
              "dataset_review_status": info["review_status"], "research_only": True,
              "preprocessing": preprocessing_contract(center_crop), "center_crop": center_crop,
              "input_size": center_crop or size, "model": "SmallPatchCNN_v1_from_scratch",
              "test_used_for_selection": False}
    write_json(run_dir / "config.json", config)
    loaders = {name: DataLoader(PatchDataset(info["root"], info["rows"][name], size, center_crop),
                               batch_size=batch_size, shuffle=name == "train", num_workers=0)
               for name in ("train", "val")}
    model = SmallPatchCNN().to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history, best = [], -1.0
    print(f"GPU: {torch.cuda.get_device_name() if torch_device.type == 'cuda' else 'CPU'}", flush=True)
    print(f"Train {len(loaders['train'].dataset):,} / val {len(loaders['val'].dataset):,} | {run_dir}", flush=True)
    try:
        for epoch in range(1, epochs+1):
            started = time.perf_counter()
            training = run_epoch(model, loaders["train"], torch_device, optimizer)
            validation = run_epoch(model, loaders["val"], torch_device)
            record = {"epoch": epoch, "train": training, "val": validation,
                      "seconds": time.perf_counter() - started}
            history.append(record)
            write_json(run_dir / "history.json", history)
            if validation["macro_f1"] > best:
                best = validation["macro_f1"]
                temporary = run_dir / "best_model.tmp"
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "config": config, "epoch": epoch, "validation": validation}, temporary)
                temporary.replace(run_dir / "best_model.pt")
            print(f"Epoch {epoch}/{epochs} | train loss {training['loss']:.4f} | "
                  f"val loss {validation['loss']:.4f} | val macro F1 {validation['macro_f1']:.3f} | "
                  f"{record['seconds']:.1f}s", flush=True)
        outcome = {"status": "completed", "run_dir": str(run_dir), "epochs_completed": len(history),
                   "best_val_macro_f1": best, "research_only": True,
                   "dataset_review_status": info["review_status"], "test_evaluated": False}
        write_json(run_dir / "result.json", outcome)
        return outcome
    except BaseException as error:
        write_json(run_dir / "result.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt)
                                            else "failed", "epochs_completed": len(history),
                                            "error_type": type(error).__name__, "research_only": True})
        raise


def evaluate_test(run_dir: Path, dataset_root: Path, batch_size=32, device="cuda") -> dict:
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    center_crop = checkpoint_crop(config)
    info = dataset_info(dataset_root, config["size"], config["balanced_train"])
    if info["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("Test dataset differs from training manifest")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model = SmallPatchCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(PatchDataset(info["root"], info["rows"]["test"], config["size"], center_crop),
                        batch_size=batch_size, num_workers=0)
    metrics = run_epoch(model, loader, torch.device(device))
    result = {"split": "test", "selected_epoch": checkpoint["epoch"], "research_only": True,
              "dataset_review_status": info["review_status"], "metrics": metrics,
              "preprocessing": config['preprocessing'], "input_size": center_crop or config['size']}
    write_json(run_dir / "test_metrics.json", result)
    return result


def plot_history(run_dir: Path):
    import matplotlib.pyplot as plt

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for split in ("train", "val"):
        axes[0].plot(epochs, [r[split]["loss"] for r in history], label=split)
        axes[1].plot(epochs, [r[split]["macro_f1"] for r in history], label=split)
    axes[0].set_title("Loss")
    axes[1].set_title("Macro F1")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(run_dir / "learning_curve.png", dpi=150)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--outputs", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    result = train(args.dataset, args.outputs, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    plot_history(Path(result["run_dir"]))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
