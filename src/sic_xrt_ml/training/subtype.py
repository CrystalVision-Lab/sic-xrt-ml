"""Fine-label training on provider_subtype_point_dataset/v1.

Same rules as ``engine``: seeded deterministic training, validation-only model
selection (minimum class-weighted validation loss), the test split is never read,
exact epoch resume. Patches stay in memory as uint8 (moved once to the device) and
batches are drawn with a seeded CPU generator.

No spatial augmentation: TED a-f / TSD a-c are direction subtypes, so flips and
rotations would change the label.
"""

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..data import provider_subtype as ps
from ..data.verified import canonical, digest
from ..evaluation.subtype_metrics import summarize
from .engine import code_revision, setup, weighted_loss
from .subtype_model import ARCHITECTURES, MEDIUM, SubtypeClassifier

TASK = "provider_subtype"
WEIGHTINGS = ("none", "balanced", "sqrt_balanced")


@dataclass(frozen=True)
class SubtypeTrainConfig:
    seed: int = 2026
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 30
    patience: int = 6
    weight_decay: float = 0.0001
    scheduler_step: int = 10
    scheduler_gamma: float = 0.5
    cpu_threads: int = 2
    device: str = "cpu"
    augmentation: str = "none"
    architecture: str = MEDIUM
    class_weighting: str = "sqrt_balanced"
    # Exponential moving average of the weights used for validation, selection and
    # export (0 = off). Smooths epoch-to-epoch jumps on the held-out validation Area.
    ema_decay: float = 0.0

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
            or self.batch_size > 512
            or self.epochs > 10000
            or self.cpu_threads > 8
            or not 0 < self.learning_rate <= 1
            or not 0 <= self.weight_decay <= 1
            or not 0 < self.scheduler_gamma <= 1
            or self.device not in {"cpu", "cuda"}
            # Direction subtypes: flips/rotations would change the label.
            or self.augmentation != "none"
            or self.architecture not in ARCHITECTURES
            or self.class_weighting not in WEIGHTINGS
            or isinstance(self.ema_decay, bool)
            or not isinstance(self.ema_decay, (int, float))
            or not 0 <= self.ema_decay < 1
        ):
            raise ValueError("TRAIN_CONFIG")


def device_pixels(data, device):
    """All patches as one uint8 [N,3,128,128] tensor on ``device`` (cached per device)."""
    if data.pixels is None:
        raise ValueError("PIXELS_NOT_LOADED")
    cache = data.__dict__.setdefault("_device_pixels", {})
    key = str(device)
    if key not in cache:
        cache.clear()
        cache[key] = torch.from_numpy(data.pixels).to(device)
    return cache[key]


def predict(model, pixels, chosen, device, batch_size=256):
    """Fine softmax probabilities [len(chosen), 10] as float32 numpy."""
    model.eval()
    out = []
    index = torch.as_tensor(chosen, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            x = pixels[index[start : start + batch_size].to(pixels.device)]
            logits = model(x.to(device).float().div_(255))
            out.append(torch.softmax(logits, 1).cpu())
    if not out:
        return np.zeros((0, len(ps.FINE_LABELS)), np.float32)
    return torch.cat(out).numpy()


def _validation(model, pixels, chosen, labels, device, class_weights):
    model.eval()
    index = torch.as_tensor(chosen, dtype=torch.long)
    logits = []
    with torch.no_grad():
        for start in range(0, len(index), 256):
            x = pixels[index[start : start + 256].to(pixels.device)]
            logits.append(model(x.to(device).float().div_(255)).cpu())
    logits = torch.cat(logits)
    y = torch.as_tensor(labels, dtype=torch.long)
    loss = float(weighted_loss(logits, y, class_weights))
    s = summarize(labels, torch.softmax(logits, 1).numpy())
    return {
        "loss": loss,
        "samples": len(labels),
        "coarse_accuracy": s["coarse"]["accuracy"],
        "coarse_balanced_accuracy": s["coarse"]["balanced_accuracy"],
        "coarse_macro_f1": s["coarse"]["macro_f1"],
        "fine_accuracy": s["fine"]["accuracy"],
        "fine_present_macro_f1": s["fine"]["present_label_macro"]["macro_f1"],
        "fine_present_balanced_accuracy": s["fine"]["present_label_macro"][
            "balanced_accuracy"
        ],
        "subtype_accuracy_given_correct_class": s["within_class"][
            "subtype_accuracy_given_correct_class"
        ],
    }


def load_checkpoint(path):
    # weights_only rejects executable Pickle globals. Only locally generated state.
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("task") != TASK or value.get("architecture") not in ARCHITECTURES:
        raise ValueError("CHECKPOINT_ARCHITECTURE")
    return value


def is_subtype_checkpoint(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    return isinstance(value, dict) and value.get("task") == TASK


def selected_model(checkpoint):
    model = SubtypeClassifier(checkpoint["architecture"])
    model.load_state_dict(checkpoint["best_model"])
    model.eval()
    return model


def _ema_update(ema, model, decay):
    with torch.no_grad():
        for e, v in zip(ema.state_dict().values(), model.state_dict().values(), strict=True):
            if e.dtype.is_floating_point:
                e.mul_(decay).add_(v, alpha=1 - decay)
            else:
                e.copy_(v)


def _restore_rng(checkpoint, generator, device):
    generator.set_state(checkpoint["generator"])
    torch.set_rng_state(checkpoint["rng"])
    random.setstate(checkpoint["python_rng"])
    state = checkpoint["numpy_rng"]
    np.random.set_state(
        (state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4])
    )
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])


def train_subtype(
    root,
    output,
    config=None,
    resume=None,
    synthetic_smoke=False,
    split_map=None,
    minimum=None,
    data=None,
    fold=False,
):
    """Train one run. ``data`` (a validated ProviderDataset with pixels) skips re-reading."""
    config = SubtypeTrainConfig() if config is None else config
    data = ps.validate(root, load_pixels=True) if data is None else data
    assignment = ps.split_assignment(data, split_map)
    ready = ps.readiness(data, assignment, synthetic_smoke, minimum, fold)
    if not ready["ready"]:
        raise ValueError("DATA_NOT_READY")
    output = Path(output)
    if output.exists():
        raise ValueError("NEW_RUN_DIRECTORY_REQUIRED")
    weights = ps.class_weights(data, assignment, config.class_weighting)
    class_weights = None if weights is None else torch.tensor(weights)
    cfg = asdict(config)
    checkpoint = None
    if resume:
        if (Path(resume).parent / "final-evaluation.json").exists():
            raise ValueError("FINAL_TEST_ALREADY_USED_RESUME_FORBIDDEN")
        checkpoint = load_checkpoint(resume)
        previous = checkpoint["config"]
        if (
            checkpoint["dataset_hash"] != data.dataset_hash
            or {k: v for k, v in previous.items() if k != "epochs"}
            != {k: v for k, v in cfg.items() if k != "epochs"}
            or checkpoint["split_assignment"] != assignment
            or checkpoint["minimum_per_class"] != ready["minimum_per_class"]
            or checkpoint.get("readiness_mode", "standard") != ready["mode"]
            or checkpoint["synthetic_smoke"] != synthetic_smoke
            or config.epochs <= checkpoint["epoch"]
            or checkpoint["early_stopped"]
        ):
            raise ValueError("RESUME_CONTRACT_OR_STOPPED")
    output.mkdir(parents=True, mode=0o700)
    device = setup(config)
    model = SubtypeClassifier(config.architecture).to(device)
    ema = None
    if config.ema_decay:
        ema = SubtypeClassifier(config.architecture).to(device)
        ema.load_state_dict(model.state_dict())
        ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config.scheduler_step, config.scheduler_gamma
    )
    generator = torch.Generator().manual_seed(config.seed)
    pixels = device_pixels(data, device)
    train_idx = torch.as_tensor(ps.indices(data, assignment, "train"), dtype=torch.long)
    train_y = torch.as_tensor(ps.labels(data, train_idx.tolist()), dtype=torch.long)
    val_idx = ps.indices(data, assignment, "validation")
    val_y = ps.labels(data, val_idx)
    start, history, best_loss, bad, best_state = 0, [], float("inf"), 0, None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        if ema is not None:
            ema.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        _restore_rng(checkpoint, generator, device)
        start = checkpoint["epoch"]
        history = checkpoint["history"]
        best_loss = checkpoint["best_loss"]
        bad = checkpoint["bad_epochs"]
        best_state = checkpoint["best_model"]
    (output / "config.json").write_bytes(canonical(cfg))
    gpu_weights = None if class_weights is None else class_weights.to(device)
    for epoch in range(start, config.epochs):
        model.train()
        total, n = 0.0, 0
        order = torch.randperm(len(train_idx), generator=generator)
        for s in range(0, len(order), config.batch_size):
            batch = order[s : s + config.batch_size]
            x = pixels[train_idx[batch].to(pixels.device)].to(device).float().div_(255)
            y = train_y[batch].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(model(x), y, gpu_weights)
            loss.backward()
            optimizer.step()
            if ema is not None:
                _ema_update(ema, model, config.ema_decay)
            total += float(loss.detach()) * len(y)
            n += len(y)
        scored = model if ema is None else ema
        validation = _validation(scored, pixels, val_idx, val_y, device, class_weights)
        scheduler.step()
        if validation["loss"] < best_loss - 1e-8:
            best_loss, bad = validation["loss"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in scored.state_dict().items()}
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
            "task": TASK,
            "architecture": config.architecture,
            "dataset_hash": data.dataset_hash,
            "dataset": data.provenance(),
            "classes": ps.CLASSES,
            "fine_labels": ps.FINE_LABELS,
            "split_assignment": assignment,
            "class_weights": weights,
            "minimum_per_class": ready["minimum_per_class"],
            "readiness_mode": ready["mode"],
            "config": cfg,
            "config_hash": digest(cfg),
            "code_revision": code_revision(),
            "torch_version": str(torch.__version__),
            "epoch": epoch + 1,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "ema_model": None
            if ema is None
            else {k: v.detach().cpu() for k, v in ema.state_dict().items()},
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
    best_epoch = min(history, key=lambda h: h["validation"]["loss"])["epoch"]
    result = {
        "status": "TRAINING_PIPELINE_VERIFIED" if synthetic_smoke else "REAL_TRAINING_COMPLETED",
        "task": TASK,
        "architecture": config.architecture,
        "dataset_hash": data.dataset_hash,
        "dataset": data.provenance(),
        "classes": ps.CLASSES,
        "fine_labels": ps.FINE_LABELS,
        "split_assignment": assignment,
        "class_weights": weights,
        "config_hash": digest(cfg),
        "code_revision": code_revision(),
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation": next(h for h in history if h["epoch"] == best_epoch)["validation"],
        "selection": "minimum class-weighted validation loss; test never read during training",
        "independent_evaluation": "NOT_RUN",
        "improvement": "NOT_VERIFIED",
        "readiness": ready,
    }
    (output / "training-report.json").write_bytes(canonical(result))
    return result
