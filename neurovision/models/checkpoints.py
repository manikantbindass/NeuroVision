"""Unified training and inference checkpoint operations."""

import hashlib
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from neurovision.models.base import BaseSegmentationModel
from neurovision.models.config import ModelConfig
from neurovision.models.factory import ModelFactory
from neurovision.models.unet import UNetConfig
from neurovision.training.checkpoints import (
    TrainingError,
    load_checkpoint,
    save_checkpoint,
)


class CheckpointManager:
    """Atomically save local weights and reconstruct their declared models."""

    @staticmethod
    def save_training(path: Path, state: dict[str, Any]) -> None:
        save_checkpoint(path, state)

    @staticmethod
    def load_training(path: Path) -> dict[str, Any]:
        state = load_checkpoint(path)
        run = state.get("run")
        if isinstance(run, dict) and "model_definition" not in run:
            # Phase 4 checkpoints predate the architecture definition field.
            try:
                legacy = dict(run["recipe"]["model"])
                legacy["features"] = tuple(legacy["features"])
                config = UNetConfig(**legacy)
                run["model_definition"] = asdict(
                    ModelConfig(
                        "unet",
                        config.spatial_dims,
                        config.in_channels,
                        config.num_classes,
                        {"features": list(config.features)},
                    )
                )
                run["monai"] = None
            except (TypeError, KeyError, ValueError):
                raise TrainingError(
                    "Cannot interpret the legacy baseline checkpoint"
                ) from None
        return state

    @staticmethod
    def export(path: Path, model: BaseSegmentationModel) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        config = model.model_config
        software = {"torch": str(torch.__version__)}
        if config.architecture != "unet":
            software["monai"] = version("monai")
        save_checkpoint(
            path,
            {
                "schema_version": 1,
                "artifact": "segmentation_weights",
                "model_definition": asdict(config),
                "software": software,
                "model": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
            },
        )
        return CheckpointManager.digest(path)

    @staticmethod
    def digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def load_model(
        path: Path,
        *,
        factory: ModelFactory | None = None,
        sha256: str | None = None,
        max_bytes: int = 1024**3,
    ) -> BaseSegmentationModel:
        """Load trusted local weights. The optional digest checks artifact identity."""
        if (
            max_bytes < 1
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > max_bytes
        ):
            raise TrainingError("Checkpoint must be a bounded regular local file")
        if sha256 is not None and CheckpointManager.digest(path) != sha256:
            raise TrainingError("Checkpoint digest mismatch")
        state = load_checkpoint(path)
        try:
            definition = state.get(
                "model_definition", state.get("run", {}).get("model_definition")
            )
            if definition is None and isinstance(state.get("run"), dict):
                state = CheckpointManager.load_training(path)
                definition = state["run"].get("model_definition")
            if not isinstance(definition, dict):
                raise TrainingError("Checkpoint lacks a model definition")
            config = ModelConfig(**definition)
            with torch.random.fork_rng(devices=[]):
                model = (ModelFactory() if factory is None else factory).create(config)
            weights = state["model"]
            expected = model.state_dict()
            if not isinstance(weights, dict) or set(weights) != set(expected):
                raise TrainingError(
                    "Checkpoint parameter names do not match the architecture"
                )
            for name, tensor in weights.items():
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != expected[name].shape
                    or tensor.dtype != expected[name].dtype
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise TrainingError(
                        "Checkpoint tensors do not match the architecture"
                    )
            model.load_state_dict(weights, strict=True)
            return model
        except (KeyError, TypeError, ValueError, RuntimeError):
            raise TrainingError(
                "Checkpoint is incompatible with the declared architecture"
            ) from None
