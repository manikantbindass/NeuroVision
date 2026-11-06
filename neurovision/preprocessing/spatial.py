"""Spatial transforms that preserve explicit voxel-to-world geometry."""

from dataclasses import dataclass
from math import floor, isfinite, prod
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from neurovision.imaging.limits import DEFAULT_LIMITS
from neurovision.imaging.volume import MRIVolume
from neurovision.preprocessing.contracts import (
    DeterministicTransform,
    Parameters,
    PreprocessingError,
    Shape,
    matrix_values,
    on_grid,
)

Interpolation = Literal["nearest", "linear"]


def check_shape(shape: Shape) -> None:
    if len(shape) != 3 or any(type(size) is not int or size < 1 for size in shape):
        raise PreprocessingError("Expected three positive output dimensions")
    DEFAULT_LIMITS.check_voxels(prod(shape))


def sample(
    data: NDArray[np.float32],
    transform: NDArray[np.float64],
    shape: Shape,
    interpolation: Interpolation,
    fill: float,
) -> NDArray[np.float32]:
    check_shape(shape)
    if interpolation not in ("nearest", "linear") or not isfinite(fill):
        raise PreprocessingError("Invalid interpolation or background value")
    if not np.isfinite(transform).all():
        raise PreprocessingError("The voxel sampling transform is not finite")
    return cast(
        NDArray[np.float32],
        ndimage.affine_transform(
            data,
            transform[:3, :3],
            offset=transform[:3, 3],
            output_shape=shape,
            order=0 if interpolation == "nearest" else 1,
            mode="grid-constant",
            cval=fill,
            prefilter=False,
        ),
    )


def foreground_mask(
    data: NDArray[np.float32], threshold: float, *, largest: bool = False
) -> NDArray[np.bool_]:
    if not isfinite(threshold) or threshold < 0:
        raise PreprocessingError("Foreground threshold must be finite and nonnegative")
    mask = np.abs(data) > threshold
    if not np.any(mask):
        raise PreprocessingError("No foreground voxels meet the configured threshold")
    if not largest:
        return mask
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return cast(NDArray[np.bool_], labels == sizes.argmax())


@dataclass(frozen=True, slots=True)
class OrientRAS(DeterministicTransform):
    def parameters(self) -> Parameters:
        return {"orientation": "RAS", "interpolation": "none"}

    def apply(self, volume: MRIVolume, *, seed: int) -> tuple[MRIVolume, Parameters]:
        return on_grid(volume, volume.data), {}


@dataclass(frozen=True, slots=True)
class Resample(DeterministicTransform):
    spacing_mm: tuple[float, float, float]
    interpolation: Interpolation = "linear"
    fill: float = 0

    def parameters(self) -> Parameters:
        return {
            "spacing_mm": self.spacing_mm,
            "interpolation": self.interpolation,
            "fill": self.fill,
            "grid_anchor": "first_voxel_center",
        }

    def apply(self, volume: MRIVolume, *, seed: int) -> tuple[MRIVolume, Parameters]:
        if len(self.spacing_mm) != 3 or any(
            not isfinite(v) or v <= 0 for v in self.spacing_mm
        ):
            raise PreprocessingError(
                "Target spacing must contain three positive finite values"
            )
        ratios = np.asarray(self.spacing_mm) / np.asarray(volume.spacing_mm)
        ends = (np.asarray(volume.data.shape) - 1) / ratios
        x, y, z = (floor(float(end) + 1e-9) + 1 for end in ends)
        voxel_map = np.diag([*ratios, 1.0])
        data = sample(volume.data, voxel_map, (x, y, z), self.interpolation, self.fill)
        return on_grid(volume, data, volume.affine @ voxel_map), {
            "output_to_input_voxels": matrix_values(voxel_map),
        }


@dataclass(frozen=True, slots=True)
class CropPad(DeterministicTransform):
    shape: Shape
    center: Literal["image", "foreground"] = "image"
    threshold: float = 0
    fill: float = 0

    def parameters(self) -> Parameters:
        return {
            "shape": self.shape,
            "center": self.center,
            "threshold": self.threshold,
            "fill": self.fill,
        }

    def apply(self, volume: MRIVolume, *, seed: int) -> tuple[MRIVolume, Parameters]:
        check_shape(self.shape)
        if not isfinite(self.fill) or abs(self.fill) > np.finfo(np.float32).max:
            raise PreprocessingError(
                "Padding requires a finite float32 background value"
            )
        if self.center == "image":
            centers = [(size - 1) / 2 for size in volume.data.shape]
        elif self.center == "foreground":
            mask = foreground_mask(volume.data, self.threshold)
            centers = []
            for axis in range(3):
                indices = np.flatnonzero(
                    mask.any(axis=tuple(i for i in range(3) if i != axis))
                )
                centers.append(float(indices[0] + indices[-1]) / 2)
        else:
            raise PreprocessingError("Unknown crop center")
        starts = tuple(
            floor(center - (size - 1) / 2)
            for center, size in zip(centers, self.shape, strict=True)
        )
        source = []
        target = []
        for start, size, old in zip(starts, self.shape, volume.data.shape, strict=True):
            lower, upper = max(start, 0), min(start + size, old)
            source.append(slice(lower, upper))
            target.append(slice(lower - start, upper - start))
        data = np.full(self.shape, self.fill, dtype=np.float32)
        data[tuple(target)] = volume.data[tuple(source)]
        voxel_map = np.eye(4)
        voxel_map[:3, 3] = starts
        return on_grid(volume, data, volume.affine @ voxel_map), {"start_voxel": starts}


@dataclass(frozen=True, slots=True)
class ExtractForeground(DeterministicTransform):
    threshold: float = 0
    largest_component: bool = True

    def parameters(self) -> Parameters:
        return {
            "threshold": self.threshold,
            "largest_component": self.largest_component,
            "connectivity": 6,
        }

    def apply(self, volume: MRIVolume, *, seed: int) -> tuple[MRIVolume, Parameters]:
        mask = foreground_mask(
            volume.data, self.threshold, largest=self.largest_component
        )
        return on_grid(volume, np.where(mask, volume.data, np.float32(0))), {
            "foreground_voxels": int(mask.sum()),
        }
