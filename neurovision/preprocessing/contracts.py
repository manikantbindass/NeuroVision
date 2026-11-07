"""Typed contracts shared by preprocessing components."""

import hashlib
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from neurovision.core.errors import PipelineError
from neurovision.imaging.volume import MRIVolume, canonical_volume

Value: TypeAlias = str | int | float | bool | None | tuple["Value", ...]
Parameters: TypeAlias = dict[str, Value]
Fields: TypeAlias = tuple[tuple[str, Value], ...]
Shape: TypeAlias = tuple[int, int, int]
Matrix: TypeAlias = tuple[tuple[float, ...], ...]


class PreprocessingError(PipelineError):
    """A preprocessing operation could not complete."""


@dataclass(frozen=True, slots=True)
class Grid:
    shape: tuple[int, ...]
    affine: Matrix

    @classmethod
    def from_volume(cls, volume: MRIVolume) -> "Grid":
        return cls(volume.data.shape, matrix_values(volume.affine))


class Transform(Protocol):
    """Provide JSON-safe parameters and deterministic behavior for the given seed."""

    @property
    def cacheable(self) -> bool: ...

    def parameters(self) -> Parameters: ...

    def apply(
        self, volume: MRIVolume, *, seed: int
    ) -> tuple[MRIVolume, Parameters]: ...


class DeterministicTransform:
    """Opt a built-in transform into cache reuse."""

    cacheable = True


def matrix_values(matrix: NDArray[np.float64]) -> Matrix:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def on_grid(
    volume: MRIVolume,
    data: NDArray[np.float32],
    affine: NDArray[np.float64] | None = None,
    *,
    reference: MRIVolume | None = None,
) -> MRIVolume:
    space = volume if reference is None else reference
    return canonical_volume(
        data,
        volume.affine if affine is None else affine,
        source_format=volume.source_format,
        modality=volume.modality,
        reference_space=space.reference_space,
        assumed_mm=space.assumed_mm,
    )


def same_grid(first: MRIVolume, second: MRIVolume) -> bool:
    return (
        first.data.shape == second.data.shape
        and first.reference_space == second.reference_space
        and np.allclose(first.affine, second.affine, atol=1e-3, rtol=0)
    )


def volume_digest(volume: MRIVolume) -> str:
    """Fingerprint intensities and physical geometry for local cache/provenance use."""
    digest = hashlib.sha256()
    for array in (volume.data, volume.affine):
        digest.update(repr((array.shape, array.dtype.str)).encode())
        # NumPy arrays implement the buffer protocol at runtime on Python 3.11.
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))  # type: ignore[arg-type]
    digest.update(
        repr(
            (
                volume.source_format,
                volume.modality.value,
                volume.reference_space,
                volume.spacing_mm,
                volume.assumed_mm,
                volume.orientation,
            )
        ).encode()
    )
    return digest.hexdigest()
