"""Registration, external brain masks, and optional N4 correction."""

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from neurovision.imaging.volume import MRIVolume
from neurovision.preprocessing.contracts import (
    DeterministicTransform,
    Matrix,
    Parameters,
    PreprocessingError,
    matrix_values,
    on_grid,
    same_grid,
    volume_digest,
)
from neurovision.preprocessing.spatial import Interpolation, sample


class RegistrationBackend(Protocol):
    """Estimate a homogeneous moving-to-fixed world transform in RAS+ mm."""

    @property
    def cacheable(self) -> bool: ...

    def parameters(self) -> Parameters: ...

    def moving_to_fixed(
        self, moving: MRIVolume, fixed: MRIVolume, *, seed: int
    ) -> Matrix: ...


@dataclass(frozen=True, slots=True)
class SuppliedAffine(DeterministicTransform):
    matrix: Matrix

    def parameters(self) -> Parameters:
        return {"adapter": "supplied_affine", "moving_to_fixed_ras_mm": self.matrix}

    def moving_to_fixed(
        self, moving: MRIVolume, fixed: MRIVolume, *, seed: int
    ) -> Matrix:
        return self.matrix


@dataclass(frozen=True, slots=True)
class Register:
    fixed: MRIVolume = field(repr=False)
    backend: RegistrationBackend
    interpolation: Interpolation = "linear"
    threshold: float = 0

    @property
    def cacheable(self) -> bool:
        return self.backend.cacheable

    def parameters(self) -> Parameters:
        backend = type(self.backend)
        return {
            "reference_digest": volume_digest(self.fixed),
            "backend": f"{backend.__module__}.{backend.__qualname__}",
            "backend_parameters": tuple(sorted(self.backend.parameters().items())),
            "interpolation": self.interpolation,
            "foreground_threshold": self.threshold,
        }

    def apply(self, volume: MRIVolume, *, seed: int) -> tuple[MRIVolume, Parameters]:
        if not isfinite(self.threshold) or self.threshold < 0:
            raise PreprocessingError(
                "Registration threshold must be finite and nonnegative"
            )
        if self.fixed.data.ndim != 3 or not np.isfinite(self.fixed.data).all():
            raise PreprocessingError(
                "Registration reference must be a finite 3D volume"
            )
        transform = np.asarray(
            self.backend.moving_to_fixed(volume, self.fixed, seed=seed),
            dtype=np.float64,
        )
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.array_equal(transform[3], [0, 0, 0, 1])
            or np.linalg.cond(transform[:3, :3]) > 1e8
        ):
            raise PreprocessingError("Registration backend supplied an invalid affine")
        voxel_map = np.asarray(
            np.linalg.solve(transform @ volume.affine, self.fixed.affine),
            dtype=np.float64,
        )
        x, y, z = self.fixed.data.shape
        data = sample(volume.data, voxel_map, (x, y, z), self.interpolation, 0)
        registered = on_grid(volume, data, self.fixed.affine, reference=self.fixed)
        if not same_grid(registered, self.fixed):
            raise PreprocessingError("Registration reference must use a canonical grid")
        fixed_mask = np.abs(self.fixed.data) > self.threshold
        moving_mask = np.abs(registered.data) > self.threshold
        count = int(fixed_mask.sum()) + int(moving_mask.sum())
        dice = (
            2 * int(np.count_nonzero(fixed_mask & moving_mask)) / count
            if count
            else 0.0
        )
        return registered, {
            "moving_to_fixed_ras_mm": matrix_values(transform),
            "output_to_input_voxels": matrix_values(voxel_map),
            "registration_foreground_dice": dice,
            "registration_linear_determinant": float(np.linalg.det(transform[:3, :3])),
        }


@dataclass(frozen=True, slots=True)
class ApplyBrainMask(DeterministicTransform):
    """Integrate the binary output of an external skull-stripping tool."""

    mask: MRIVolume = field(repr=False)
    method: str
    method_version: str

    def parameters(self) -> Parameters:
        return {
            "mask_digest": volume_digest(self.mask),
            "method": self.method,
            "method_version": self.method_version,
        }

    def apply(self, volume: MRIVolume, *, seed: int) -> tuple[MRIVolume, Parameters]:
        if not self.method.strip() or not self.method_version.strip():
            raise PreprocessingError(
                "An external brain mask needs a method and version"
            )
        if not same_grid(volume, self.mask) or not np.all(
            (self.mask.data == 0) | (self.mask.data == 1)
        ):
            raise PreprocessingError("Brain mask must be binary and match the MRI grid")
        mask = self.mask.data == 1
        count = int(mask.sum())
        if not count:
            raise PreprocessingError("Brain mask is empty")
        return on_grid(volume, np.where(mask, volume.data, np.float32(0))), {
            "mask_voxels": count,
            "mask_fraction": count / mask.size,
        }


@dataclass(frozen=True, slots=True)
class N4BiasCorrection(DeterministicTransform):
    """Fit an N4 field on positive foreground before intensity normalization."""

    iterations: tuple[int, ...] = (30, 20, 10)
    shrink: int = 2
    threshold: float = 0

    def parameters(self) -> Parameters:
        try:
            backend_version = version("SimpleITK")
        except PackageNotFoundError:
            backend_version = "unavailable"
        return {
            "backend": "SimpleITK N4",
            "version": backend_version,
            "iterations": self.iterations,
            "shrink": self.shrink,
            "threshold": self.threshold,
            "work_units": 1,
        }

    def apply(self, volume: MRIVolume, *, seed: int) -> tuple[MRIVolume, Parameters]:
        try:
            import SimpleITK as sitk
        except ImportError:
            raise PreprocessingError("N4 requires the optional bias extra") from None
        if (
            not 1 <= len(self.iterations) <= 4
            or any(type(n) is not int or not 1 <= n <= 200 for n in self.iterations)
            or type(self.shrink) is not int
            or not 1 <= self.shrink <= 8
            or not isfinite(self.threshold)
            or self.threshold < 0
            or min(volume.data.shape) // self.shrink < 4
        ):
            raise PreprocessingError(
                "Invalid N4 parameters or insufficient fit grid size"
            )
        mask = volume.data > self.threshold
        if np.any(volume.data < 0) or int(mask.sum()) < 32:
            raise PreprocessingError(
                "N4 requires nonnegative intensities and positive foreground"
            )
        lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ volume.affine
        direction = lps[:3, :3] / np.asarray(volume.spacing_mm)
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5, rtol=0):
            raise PreprocessingError("N4 requires orthogonal voxel axes")
        image = sitk.GetImageFromArray(volume.data.T)
        image.SetSpacing(volume.spacing_mm)
        image.SetOrigin(tuple(float(v) for v in lps[:3, 3]))
        image.SetDirection(tuple(float(v) for v in direction.flat))
        mask_image = sitk.GetImageFromArray(mask.T.astype(np.uint8))
        mask_image.CopyInformation(image)
        if self.shrink > 1:
            fit = sitk.Shrink(image, [self.shrink] * 3)
            fit_mask = sitk.Shrink(mask_image, [self.shrink] * 3)
        else:
            fit, fit_mask = image, mask_image
        fit_count = int(np.count_nonzero(sitk.GetArrayViewFromImage(fit_mask)))
        if fit_count < 8:
            raise PreprocessingError("N4 fitting mask is too small after shrinking")
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetNumberOfWorkUnits(1)
        corrector.SetMaximumNumberOfIterations(list(self.iterations))
        corrector.Execute(fit, fit_mask)
        field_image = corrector.GetLogBiasFieldAsImage(image)
        corrected = sitk.Divide(image, sitk.Exp(field_image))
        data = cast(NDArray[np.float32], sitk.GetArrayFromImage(corrected).T)
        return on_grid(volume, data), {"fit_voxels": fit_count}
