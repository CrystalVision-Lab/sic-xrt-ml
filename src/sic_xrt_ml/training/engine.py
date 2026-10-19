"""Bounded CPU/CUDA training, validation-only selection and exact epoch resume."""

import os
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data.loading import prepare
from ..data.verified import CLASSES, canonical, digest, loader
from ..evaluation.metrics import metrics
from .model import ARCHITECTURE, PatchClassifier


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2026
    batch_size: int = 8
    learning_rate: float = 0.001
    epochs: int = 10
    patience: int = 3
    weight_decay: float = 0.0001
    scheduler_step: int = 5
    scheduler_gamma: float = 0.5
    cpu_threads: int = 2
    device: str = "cpu"
    augmentation: str = "none"

    def __post_init__(self):
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or any(
                type(v) is not int or v < 1
                for v in (
                    self.batch_size,
                    self.epochs,
                    self.patience,
                    self.scheduler_step,
                    self.cpu_threads,
                )
            )
            or self.batch_size > 256
            or self.epochs > 10000
            or self.cpu_threads > 8
            or not 0 < self.learning_rate <= 1
            or not 0 <= self.weight_decay <= 1
            or not 0 < self.scheduler_gamma <= 1
            or self.device not in {"cpu", "cuda"}
            or self.augmentation != "none"
        ):
            raise ValueError("TRAIN_CONFIG")


def code_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def setup(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA_NOT_AVAILABLE")
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
    return torch.device(config.device)


def weighted_loss(logits, y, class_weights=None, sample_weights=None):
    """Plain mean CE for v1. v2: class-balanced, consensus-weighted normalized CE."""
    if class_weights is None and sample_weights is None:
        return nn.functional.cross_entropy(logits, y)
    per = nn.functional.cross_entropy(logits, y, reduction="none")
    factor = torch.ones_like(per)
    if class_weights is not None:
        factor = factor * class_weights.to(per.device)[y]
    if sample_weights is not None:
        factor = factor * sample_weights.to(per.device, per.dtype)
    return (per * factor).sum() / factor.sum().clamp_min(1e-12)


def split_batch(batch):
    """v1 batches are (x, y); v2 batches are (x, y, consensus weight)."""
    return batch[0], batch[1], batch[2] if len(batch) > 2 else None


def evaluate_loader(model, data, device, class_weights=None, classes=CLASSES):
    model.eval()
    labels = []
    predictions = []
    losses = []
    with torch.no_grad():
        for batch in data:
            x, y, _ = split_batch(batch)
            logits = model(x.to(device))
            loss = weighted_loss(logits, y.to(device), class_weights)
            losses.extend([float(loss) * len(y)])
            labels.extend(y.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return {
        **metrics(labels, predictions, classes),
        "loss": sum(losses) / len(labels) if labels else None,
    }


def load_checkpoint(path):
    # weights_only rejects executable Pickle globals. Only locally generated primitive/tensor state.
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("architecture") != ARCHITECTURE:
        raise ValueError("CHECKPOINT_ARCHITECTURE")
    return value


def train(
    root, output, config=None, resume=None, synthetic_smoke=False, min_per_class=None
):
    config = TrainConfig() if config is None else config
    prepared = prepare(root, synthetic_smoke, min_per_class)
    manifest = prepared.manifest
    ready = prepared.readiness
    classes = prepared.classes
    weights = prepared.class_weights()
    class_weights = None if weights is None else torch.tensor(weights)
    if not ready["ready"]:
        raise ValueError("DATA_NOT_READY")
    output = Path(output)
    if output.exists():
        raise ValueError("NEW_RUN_DIRECTORY_REQUIRED")
    output.mkdir(parents=True, mode=0o700)
    device = setup(config)
    model = PatchClassifier(len(classes)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config.scheduler_step, config.scheduler_gamma
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_data = loader(
        prepared.dataset("train"), config.batch_size, True, generator
    )
    validation_data = loader(prepared.dataset("validation"), config.batch_size)
    start = 0
    history = []
    best_loss = float("inf")
    bad = 0
    best_state = None
    cfg = asdict(config)
    if resume:
        if (Path(resume).parent / "final-evaluation.json").exists():
            raise ValueError("FINAL_TEST_ALREADY_USED_RESUME_FORBIDDEN")
        checkpoint = load_checkpoint(resume)
        previous = checkpoint["config"]
        if (
            checkpoint["dataset_hash"] != manifest["dataset_hash"]
            or {k: v for k, v in previous.items() if k != "epochs"}
            != {k: v for k, v in cfg.items() if k != "epochs"}
            or checkpoint.get("classes", CLASSES) != classes
            or (
                prepared.schema_version == 2
                and checkpoint.get("minimum_per_class") != ready["minimum_per_class"]
            )
            or config.epochs <= checkpoint["epoch"]
            or checkpoint["early_stopped"]
        ):
            raise ValueError("RESUME_CONTRACT_OR_STOPPED")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        generator.set_state(checkpoint["generator"])
        torch.set_rng_state(checkpoint["rng"])
        random.setstate(checkpoint["python_rng"])
        state = checkpoint["numpy_rng"]
        np.random.set_state(
            (
                state[0],
                np.asarray(state[1], dtype=np.uint32),
                state[2],
                state[3],
                state[4],
            )
        )
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start = checkpoint["epoch"]
        history = checkpoint["history"]
        best_loss = checkpoint["best_loss"]
        bad = checkpoint["bad_epochs"]
        best_state = checkpoint["best_model"]
    (output / "config.json").write_bytes(canonical(cfg))
    for epoch in range(start, config.epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in train_data:
            x, y, sample_weights = split_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = weighted_loss(logits, y.to(device), class_weights, sample_weights)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        validation = evaluate_loader(
            model, validation_data, device, class_weights, classes
        )
        scheduler.step()
        improved = validation["loss"] < best_loss - 1e-8
        if improved:
            best_loss = validation["loss"]
            bad = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            bad += 1
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / n,
                "validation": validation,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        state = np.random.get_state()
        checkpoint = {
            "architecture": ARCHITECTURE,
            "dataset_hash": manifest["dataset_hash"],
            "dataset": prepared.provenance(),
            "classes": classes,
            "class_weights": weights,
            "minimum_per_class": ready["minimum_per_class"],
            "config": cfg,
            "config_hash": digest(cfg),
            "code_revision": code_revision(),
            "torch_version": str(torch.__version__),
            "epoch": epoch + 1,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "generator": generator.get_state(),
            "rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "history": history,
            "best_loss": best_loss,
            "best_model": best_state,
            "bad_epochs": bad,
            "early_stopped": bad >= config.patience,
            "synthetic_smoke": synthetic_smoke,
        }
        staged = output / ".checkpoint.tmp"
        torch.save(checkpoint, staged)
        staged.replace(output / "checkpoint.pt")
        (output / "history.json").write_bytes(canonical(history))
        if bad >= config.patience:
            break
    result = {
        "status": "TRAINING_PIPELINE_VERIFIED"
        if synthetic_smoke
        else "REAL_TRAINING_COMPLETED",
        "architecture": ARCHITECTURE,
        "dataset_hash": manifest["dataset_hash"],
        "dataset": prepared.provenance(),
        "classes": classes,
        "class_weights": weights,
        "config_hash": digest(cfg),
        "code_revision": code_revision(),
        "device": str(device),
        "epochs_completed": len(history),
        "selection": "minimum validation loss; test never read during training",
        "independent_evaluation": "NOT_RUN",
        "improvement": "NOT_VERIFIED",
        "readiness": ready,
    }
    (output / "training-report.json").write_bytes(canonical(result))
    return result


def selected_model(checkpoint):
    model = PatchClassifier(len(checkpoint.get("classes", CLASSES)))
    model.load_state_dict(checkpoint["best_model"])
    model.eval()
    return model
