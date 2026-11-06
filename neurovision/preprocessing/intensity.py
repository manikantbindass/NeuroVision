"""Deterministic intensity preparation and optional seeded experiments."""

from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from neurovision.imaging.volume import MRIVolume
from neurovision.preprocessing.contracts import (
    DeterministicTransform,
    Parameters,
    PreprocessingError,
    on_grid,
)
from neurovision.preprocessing.spatial import foreground_mask


@dataclass(frozen=True, slots=True)
class NormalizeIntensity(DeterministicTransform):
    method: Literal["zscore", "percentile"] = "zscore"
    foreground_only: bool = True
    threshold: float = 0
    percentiles: tuple[float, float] = (1, 99)
    min_scale: float = 1e-8

    def parameters(self) -> Parameters:
        return {
            "method": self.method,
            "foreground_only": self.foreground_only,
            "threshold": self.threshold,
            "percentiles": self.percentiles,
            "min_scale": self.min_scale,
        }

    def apply(self, volume: MRIVolume, *, seed: int) -> tuple[MRIVolume, Parameters]:
        mask = (
            foreground_mask(volume.data, self.threshold)
            if self.foreground_only
            else np.ones(volume.data.shape, dtype=np.bool_)
        )
        values = volume.data[mask].astype(np.float64)
        if not isfinite(self.min_scale) or self.min_scale <= 0:
            raise PreprocessingError("Minimum normalization scale must be positive")
        if self.method == "zscore":
            center, scale = float(values.mean()), float(values.std())
        elif self.method == "percentile":
            lower, upper = self.percentiles
            if not 0 <= lower < upper <= 100:
                raise PreprocessingError("Percentiles must increase within 0 to 100")
            center, end = (float(v) for v in np.percentile(values, self.percentiles))
            scale = end - center
        else:
            raise PreprocessingError("Unknown intensity normalization method")
        if not isfinite(scale) or scale < self.min_scale:
            raise PreprocessingError(
                "Intensity normalization has insufficient variation"
            )
        normalized = (values - center) / scale
        if self.method == "percentile":
            normalized = np.clip(normalized, 0, 1)
        output = np.zeros_like(volume.data)
        output[mask] = normalized
        return on_grid(volume, output), {
            "center": center,
            "scale": scale,
            "fitted_voxels": int(values.size),
        }


@dataclass(frozen=True, slots=True)
class GaussianDenoise(DeterministicTransform):
    """Experimental Gaussian smoothing. It can also blur anatomical detail."""

    sigma_mm: float

    def parameters(self) -> Parameters:
        return {"sigma_mm": self.sigma_mm, "boundary": "reflect", "experimental": True}

    def apply(self, volume: MRIVolume, *, seed: int) -> tuple[MRIVolume, Parameters]:
        sigma = tuple(self.sigma_mm / spacing for spacing in volume.spacing_mm)
        if any(not isfinite(v) or not 0 < v <= 32 for v in sigma):
            raise PreprocessingError(
                "Gaussian sigma must be between 0 and 32 voxels per axis"
            )
        data = cast(
            NDArray[np.float32],
            ndimage.gaussian_filter(volume.data, sigma, mode="reflect"),
        )
        return on_grid(volume, data), {"sigma_voxels": sigma}


@dataclass(frozen=True, slots=True)
class RandomIntensity(DeterministicTransform):
    """Training-only seeded gain, offset and Gaussian noise."""

    gain: tuple[float, float] = (0.9, 1.1)
    offset: tuple[float, float] = (-0.05, 0.05)
    noise_std: float = 0

    def parameters(self) -> Parameters:
        return {
            "gain": self.gain,
            "offset": self.offset,
            "noise_std": self.noise_std,
            "training_only": True,
            "generator": "numpy.PCG64",
        }

    def apply(self, volume: MRIVolume, *, seed: int) -> tuple[MRIVolume, Parameters]:
        if (
            not all(isfinite(v) for v in (*self.gain, *self.offset, self.noise_std))
            or not 0 < self.gain[0] <= self.gain[1]
            or self.offset[0] > self.offset[1]
            or self.noise_std < 0
        ):
            raise PreprocessingError("Invalid random intensity parameters")
        rng = np.random.Generator(np.random.PCG64(seed))
        gain, offset = float(rng.uniform(*self.gain)), float(rng.uniform(*self.offset))
        mask = foreground_mask(volume.data, 0)
        values = volume.data[mask].astype(np.float64) * gain + offset
        if self.noise_std:
            values += rng.normal(0, self.noise_std, size=values.size)
        output = np.zeros_like(volume.data)
        output[mask] = values
        return on_grid(volume, output), {"applied_gain": gain, "applied_offset": offset}


@dataclass(frozen=True, slots=True)
class RandomFlip(DeterministicTransform):
    """Training-only voxel mirroring on the existing grid."""

    axes: tuple[int, ...] = (0,)
    probability: float = 0.5

    def parameters(self) -> Parameters:
        return {
            "axes": self.axes,
            "probability": self.probability,
            "training_only": True,
        }

    def apply(self, volume: MRIVolume, *, seed: int) -> tuple[MRIVolume, Parameters]:
        if (
            not 0 <= self.probability <= 1
            or len(set(self.axes)) != len(self.axes)
            or any(axis not in (0, 1, 2) for axis in self.axes)
        ):
            raise PreprocessingError("Invalid flip axes or probability")
        rng = Random(seed)  # noqa: S311 - reproducible augmentation RNG
        chosen = tuple(axis for axis in self.axes if rng.random() < self.probability)
        return on_grid(volume, np.flip(volume.data, axis=chosen)), {
            "flipped_axes": chosen
        }
