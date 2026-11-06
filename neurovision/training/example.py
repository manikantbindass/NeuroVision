"""Train the baselines on generated shapes as a reproducible smoke exercise."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from neurovision.core.errors import ConfigurationError
from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.models.unet import UNetConfig
from neurovision.training.checkpoints import TrainingError
from neurovision.training.config import load_experiment
from neurovision.training.runner import Batch, fit

_SIZE = 16


def synthetic_dataset(model: UNetConfig, *, samples: int, seed: int) -> Dataset[Batch]:
    """Generate separate binary phantom cases without filesystem or network input."""
    if model.num_classes != 2 or _SIZE % model.downsampling_factor:
        raise TrainingError(
            "Synthetic examples require two classes and a compatible input grid"
        )
    if samples < 1:
        raise TrainingError("Synthetic dataset requires at least one sample")
    rng = np.random.default_rng(seed)
    shape = (_SIZE,) * model.spatial_dims
    grid = np.indices(shape, dtype=np.float32)
    broadcast = (model.spatial_dims,) + (1,) * model.spatial_dims
    images, labels = [], []
    for _ in range(samples):
        center = rng.uniform(5, 11, size=model.spatial_dims).reshape(broadcast)
        radii = rng.uniform(2.5, 4.5, size=model.spatial_dims).reshape(broadcast)
        mask = np.sum(((grid - center) / radii) ** 2, axis=0) <= 1
        signal = rng.uniform(
            0.7, 1.4, size=(model.in_channels,) + (1,) * model.spatial_dims
        )
        noisy = rng.normal(0, 0.12, size=(model.in_channels, *shape)) + signal * mask
        images.append(noisy.astype(np.float32))
        labels.append(mask.astype(np.int64))
    return cast(
        Dataset[Batch],
        TensorDataset(
            torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(labels))
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a synthetic U-Net training example. This is a smoke test.",
        epilog=RESEARCH_NOTICE,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(RESEARCH_NOTICE)
    try:
        config = load_experiment(args.config)
        spacing = config.training.validation_spacing_mm
        if spacing is not None and any(value != 1 for value in spacing):
            raise TrainingError("Synthetic phantoms use unit spacing")
        seed = config.training.seed
        result = fit(
            synthetic_dataset(config.model, samples=8, seed=2 * seed),
            synthetic_dataset(config.model, samples=4, seed=2 * seed + 1),
            config,
            output=args.output,
            resume=args.resume,
            data_id=f"synthetic-ellipsoids-v1:{_SIZE}:8:4:{seed}",
        )
    except (ConfigurationError, TrainingError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Synthetic smoke training completed {len(result.history)} epoch(s).")
    print("See history.json for synthetic diagnostics. No MRI benchmark was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
