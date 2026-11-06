"""Validated experiment configuration for baseline training."""

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

from neurovision.core.errors import ConfigurationError
from neurovision.models.config import ModelConfig
from neurovision.models.unet import UNetConfig
from neurovision.training.losses import LossConfig


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    flip_probability: float = 0
    noise_std: float = 0

    def __post_init__(self) -> None:
        if (
            not 0 <= self.flip_probability <= 1
            or not isfinite(self.noise_std)
            or self.noise_std < 0
        ):
            raise ValueError("Invalid augmentation probability or noise")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 0.001
    device: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    deterministic: bool = True
    cpu_threads: int = 1
    early_stopping_patience: int = 5
    min_delta: float = 0.0
    lr_patience: int = 2
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    validation_spacing_mm: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or type(self.epochs) is not int
            or not 1 <= self.epochs < 2**32
            or type(self.batch_size) is not int
            or self.batch_size < 1
            or type(self.cpu_threads) is not int
            or self.cpu_threads < 1
            or type(self.early_stopping_patience) is not int
            or self.early_stopping_patience < 1
            or type(self.lr_patience) is not int
            or self.lr_patience < 0
            or type(self.deterministic) is not bool
            or not 0 < self.min_lr <= self.learning_rate < float("inf")
            or not isfinite(self.min_delta)
            or self.min_delta < 0
            or not 0 < self.lr_factor < 1
            or self.device not in ("cpu", "cuda")
            or self.precision not in ("fp32", "fp16", "bf16")
            or (self.device == "cpu" and self.precision == "fp16")
        ):
            raise ValueError(
                "Invalid training, device, precision, or schedule configuration"
            )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: UNetConfig | ModelConfig
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    augmentation: AugmentationConfig = AugmentationConfig()

    def __post_init__(self) -> None:
        spacing = self.training.validation_spacing_mm
        if spacing is not None and (
            len(spacing) != self.model.spatial_dims
            or any(not isfinite(value) or value <= 0 for value in spacing)
        ):
            raise ValueError("Validation spacing must match the model axis order")


def load_experiment(path: Path) -> ExperimentConfig:
    """Read one local JSON recipe with explicit model configuration."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError("Experiment configuration exceeds 64 KiB")
        config = json.loads(raw)
        if not isinstance(config, dict) or set(config) - {
            "model",
            "loss",
            "training",
            "augmentation",
        }:
            raise ValueError("Invalid experiment configuration sections")
        model = dict(config["model"])
        training = dict(config.get("training", {}))
        if "features" in model and "architecture" not in model:
            model["features"] = tuple(model["features"])
        if training.get("validation_spacing_mm") is not None:
            training["validation_spacing_mm"] = tuple(training["validation_spacing_mm"])
        return ExperimentConfig(
            model=ModelConfig(**model)
            if "architecture" in model
            else UNetConfig(**model),
            loss=LossConfig(**config.get("loss", {})),
            training=TrainingConfig(**training),
            augmentation=AugmentationConfig(**config.get("augmentation", {})),
        )
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        raise ConfigurationError("Invalid baseline experiment configuration") from None
