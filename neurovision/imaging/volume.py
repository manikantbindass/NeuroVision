"""Canonical in-memory representation of MRI intensities and geometry."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, cast

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from neurovision.core.errors import NeuroVisionError


class ImagingError(NeuroVisionError):
    """An input failed an imaging format, safety, or geometry check."""


class Modality(StrEnum):
    T1 = "t1"
    T1CE = "t1ce"
    T2 = "t2"
    FLAIR = "flair"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True, eq=False)
class MRIVolume:
    """Read-only float32 voxels and a voxel-to-RAS+ millimetre transform."""

    data: NDArray[np.float32] = field(repr=False)
    affine: NDArray[np.float64] = field(repr=False)
    spacing_mm: tuple[float, float, float]
    source_format: Literal["nifti", "dicom"]
    modality: Modality
    reference_space: str
    assumed_mm: bool = False
    orientation: tuple[str, str, str] = ("R", "A", "S")


def canonical_volume(
    data: NDArray[np.float32],
    affine: NDArray[np.float64],
    *,
    source_format: Literal["nifti", "dicom"],
    modality: Modality,
    reference_space: str,
    assumed_mm: bool = False,
) -> MRIVolume:
    if data.ndim != 3 or not data.size or not np.isfinite(data).all():
        raise ImagingError("MRI voxels must be a nonempty finite 3D array")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ImagingError("MRI affine must be a finite 4 by 4 matrix")
    if not np.array_equal(affine[3], [0, 0, 0, 1]):
        raise ImagingError("MRI affine has an invalid homogeneous row")
    axes = affine[:3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        spacing = np.linalg.norm(axes, axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ImagingError("MRI affine has invalid voxel spacing")
    if abs(np.linalg.det(axes / spacing)) < 1e-6:
        raise ImagingError("MRI affine is singular or nearly singular")

    original = nib.orientations.io_orientation(affine)
    target = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(original, target)
    with np.errstate(over="ignore", invalid="ignore"):
        ras_affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
    if not np.isfinite(ras_affine).all():
        raise ImagingError("Canonical MRI affine is not finite")
    oriented = cast(
        NDArray[np.float32], nib.orientations.apply_orientation(data, transform)
    )
    ras_data = oriented.copy(order="C")
    x, y, z = (float(value) for value in np.linalg.norm(ras_affine[:3, :3], axis=0))
    ras_data.setflags(write=False)
    ras_affine.setflags(write=False)
    return MRIVolume(
        ras_data,
        ras_affine,
        (x, y, z),
        source_format,
        modality,
        reference_space,
        assumed_mm,
    )
