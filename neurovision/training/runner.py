"""Shared training and validation for the 2D and 3D baselines."""

import hashlib
import os
import platform
import random
from collections.abc import Sized
from dataclasses import asdict, dataclass
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.evaluation.segmentation import segmentation_metrics
from neurovision.models.base import BaseSegmentationModel
from neurovision.models.checkpoints import CheckpointManager
from neurovision.models.factory import ModelFactory, MonaiSegmentationModel
from neurovision.training.checkpoints import (
    TrainingError,
    save_history,
)
from neurovision.training.config import AugmentationConfig, ExperimentConfig
from neurovision.training.losses import SegmentationLoss

Batch = tuple[Tensor, Tensor]
MetricSummary = dict[str, dict[str, dict[str, float | int | None]]]


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    training_loss: float
    validation_loss: float
    learning_rate: float
    validation_metrics: MetricSummary


@dataclass(frozen=True, slots=True)
class FitResult:
    model: BaseSegmentationModel
    history: tuple[EpochResult, ...]
    best_epoch: int
    stopped_early: bool


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _epoch_seed(seed: int, epoch: int, stream: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{epoch}:{stream}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def _device(config: ExperimentConfig) -> torch.device:
    settings = config.training
    if settings.device == "cuda" and settings.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is unavailable")
    if (
        settings.device == "cuda"
        and settings.precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise TrainingError("CUDA bfloat16 is unavailable")
    torch.set_num_threads(settings.cpu_threads)
    torch.use_deterministic_algorithms(settings.deterministic)
    torch.backends.cudnn.deterministic = settings.deterministic
    torch.backends.cudnn.benchmark = False
    return torch.device(settings.device)


def _augment(
    images: Tensor,
    labels: Tensor,
    config: AugmentationConfig,
    generator: torch.Generator,
) -> Batch:
    axes = [
        axis
        for axis in range(2, images.ndim)
        if config.flip_probability
        and torch.rand((), generator=generator).item() < config.flip_probability
    ]
    if axes:
        images = images.flip(axes)
        labels = labels.flip([axis - 1 for axis in axes])
    if config.noise_std:
        images = images + config.noise_std * torch.randn(
            images.shape, generator=generator, dtype=images.dtype
        )
    return images, labels


class _Metrics:
    def __init__(self, num_classes: int, spacing: tuple[float, ...] | None) -> None:
        self.num_classes = num_classes
        self.spacing = spacing
        fields = (
            "dice",
            "iou",
            "precision",
            "sensitivity",
            "specificity",
            "hausdorff_mm",
            "hd95_mm",
            "assd_mm",
        )
        self.values = {
            str(label): {key: (0.0, 0) for key in fields}
            for label in range(1, num_classes)
        }

    def update(self, logits: Tensor, target: Tensor) -> None:
        predicted = logits.argmax(dim=1).detach().cpu().numpy()
        expected = target.detach().cpu().numpy()
        for output, label in zip(predicted, expected, strict=True):
            for score in segmentation_metrics(
                output,
                label,
                num_classes=self.num_classes,
                spacing_mm=self.spacing,
            ):
                totals = self.values[str(score.label)]
                for key, value in asdict(score).items():
                    if key in totals and value is not None:
                        total, count = totals[key]
                        totals[key] = (total + value, count + 1)

    def summary(self) -> MetricSummary:
        return {
            label: {
                key: {
                    "mean": total / count if count else None,
                    "defined_samples": count,
                }
                for key, (total, count) in scores.items()
            }
            for label, scores in self.values.items()
        }


def _epoch(
    loader: DataLoader[Batch],
    model: BaseSegmentationModel,
    loss_fn: SegmentationLoss,
    config: ExperimentConfig,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    optimizer: AdamW | None,
    generator: torch.Generator,
) -> tuple[float, MetricSummary]:
    training = optimizer is not None
    model.train(training)
    metrics = (
        None
        if training
        else _Metrics(config.model.num_classes, config.training.validation_spacing_mm)
    )
    total, samples = 0.0, 0
    dtype = torch.float16 if config.training.precision == "fp16" else torch.bfloat16
    for images, labels in loader:
        if (
            images.device.type != "cpu"
            or labels.device.type != "cpu"
            or images.dtype != torch.float32
            or labels.dtype != torch.int64
            or images.ndim != config.model.spatial_dims + 2
            or labels.shape != (images.shape[0], *images.shape[2:])
            or not labels.numel()
            or not bool(torch.isfinite(images).all())
            or labels.min().item() < 0
            or labels.max().item() >= config.model.num_classes
        ):
            raise TrainingError(
                "Expected finite CPU float32 images and valid same-grid int64 labels"
            )
        if optimizer is not None:
            images, labels = _augment(images, labels, config.augmentation, generator)
            optimizer.zero_grad(set_to_none=True)
        images, labels = images.to(device), labels.to(device)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=config.training.precision != "fp32",
            ),
        ):
            logits = cast(Tensor, model(images))
            loss = cast(Tensor, loss_fn(logits, labels))
        value = float(loss.item())
        if not isfinite(value) or not bool(torch.isfinite(logits).all()):
            raise TrainingError("Training or validation produced nonfinite values")
        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if metrics is not None:
            metrics.update(logits, labels)
        total += value * images.shape[0]
        samples += images.shape[0]
    if not samples:
        raise TrainingError(
            "Training and validation datasets must both contain samples"
        )
    return total / samples, metrics.summary() if metrics is not None else {}


def _resume_state(
    checkpoint: Path,
    run: dict[str, Any],
    model: BaseSegmentationModel,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
) -> tuple[list[EpochResult], float, int, int]:
    state = CheckpointManager.load_training(checkpoint)
    if state.get("run") != run:
        raise TrainingError(
            "Checkpoint data, environment, or experiment configuration differs"
        )
    try:
        history = [EpochResult(**record) for record in state["history"]]
        best_loss, best_epoch, stagnant = (
            float(state["best_loss"]),
            int(state["best_epoch"]),
            int(state["stagnant"]),
        )
        if (
            not history
            or not 1 <= best_epoch <= len(history)
            or not isfinite(best_loss)
            or not 0 <= stagnant < len(history)
            or any(record.epoch != index for index, record in enumerate(history, 1))
        ):
            raise ValueError("Invalid checkpoint history")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        return history, best_loss, best_epoch, stagnant
    except (KeyError, RuntimeError, ValueError, TypeError, OverflowError):
        raise TrainingError("Incomplete or incompatible training state") from None


def fit(
    training: Dataset[Batch],
    validation: Dataset[Batch],
    config: ExperimentConfig,
    *,
    output: Path,
    data_id: str,
    resume: bool = False,
    factory: ModelFactory | None = None,
) -> FitResult:
    """Train on finite CPU tensors and checkpoint at complete epoch boundaries."""
    if not data_id.strip() or len(data_id) > 256:
        raise TrainingError("A versioned dataset identifier is required")
    if training is validation:
        raise TrainingError("Training and validation datasets must be separate")
    if (
        not isinstance(training, Sized)
        or not len(training)
        or not isinstance(validation, Sized)
        or not len(validation)
    ):
        raise TrainingError(
            "Nonempty sized training and validation datasets are required"
        )
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise TrainingError("Training output must be empty for a new run")
    settings = config.training
    device = _device(config)
    _seed(settings.seed)
    model = (
        (ModelFactory() if factory is None else factory).create(config.model).to(device)
    )
    loss_fn = SegmentationLoss(config.loss)
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=settings.lr_factor,
        patience=settings.lr_patience,
        threshold=settings.min_delta,
        threshold_mode="abs",
        min_lr=settings.min_lr,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=settings.precision == "fp16")
    recipe = asdict(config)
    recipe["training"].pop("epochs")
    run = {
        "trainer_revision": 1,
        "model_definition": asdict(model.model_config),
        "data_id": data_id,
        "recipe": recipe,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "monai": version("monai")
        if isinstance(model, MonaiSegmentationModel)
        else None,
        "scipy": version("scipy"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
    }
    history: list[EpochResult] = []
    best_loss, best_epoch, stagnant = float("inf"), 0, 0
    if resume:
        history, best_loss, best_epoch, stagnant = _resume_state(
            output / "last.pt", run, model, optimizer, scheduler, scaler
        )
        if len(history) > settings.epochs or not (output / "best.pt").is_file():
            raise TrainingError(
                "Resume requires the best checkpoint and a sufficient epoch budget"
            )
    valid_loader = DataLoader(validation, batch_size=settings.batch_size, shuffle=False)
    for epoch in range(len(history) + 1, settings.epochs + 1):
        if stagnant >= settings.early_stopping_patience:
            break
        _seed(_epoch_seed(settings.seed, epoch, "global"))
        loader_rng = torch.Generator().manual_seed(
            _epoch_seed(settings.seed, epoch, "shuffle")
        )
        augment_rng = torch.Generator().manual_seed(
            _epoch_seed(settings.seed, epoch, "augment")
        )
        train_loader = DataLoader(
            training, batch_size=settings.batch_size, shuffle=True, generator=loader_rng
        )
        training_loss, _ = _epoch(
            train_loader, model, loss_fn, config, device, scaler, optimizer, augment_rng
        )
        validation_loss, scores = _epoch(
            valid_loader, model, loss_fn, config, device, scaler, None, augment_rng
        )
        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(validation_loss)
        history.append(EpochResult(epoch, training_loss, validation_loss, lr, scores))
        improved = validation_loss < best_loss - settings.min_delta
        if improved:
            best_loss, best_epoch, stagnant = validation_loss, epoch, 0
        else:
            stagnant += 1
        state = {
            "schema_version": 1,
            "run": run,
            "history": [asdict(item) for item in history],
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "stagnant": stagnant,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        if improved:
            CheckpointManager.save_training(output / "best.pt", state)
        CheckpointManager.save_training(output / "last.pt", state)
    stopped = stagnant >= settings.early_stopping_patience
    save_history(
        output / "history.json",
        {
            "notice": RESEARCH_NOTICE,
            "run": run,
            "epoch_budget": settings.epochs,
            "best_epoch": best_epoch,
            "stopped_early": stopped,
            "epochs": [asdict(item) for item in history],
        },
    )
    return FitResult(model, tuple(history), best_epoch, stopped)
