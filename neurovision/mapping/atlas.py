"""Contracts for local, versioned atlases with categorical region labels."""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from neurovision.core.errors import PipelineError
from neurovision.imaging.volume import MRIVolume

Hemisphere = Literal["left", "right", "bilateral", "midline", "unknown"]


class MappingError(PipelineError):
    """Atlas or segmentation geometry and metadata failed validation."""


@dataclass(frozen=True, slots=True)
class AtlasRegion:
    code: int
    name: str
    hemisphere: Hemisphere


@dataclass(frozen=True, slots=True)
class Atlas:
    name: str
    version: str
    license: str
    reference_space: str
    labels: NDArray[np.int32]
    affine: NDArray[np.float64]
    regions: tuple[AtlasRegion, ...]


class AtlasAdapter(Protocol):
    def load(self) -> Atlas:
        """Load an atlas and its curated lookup table."""


@dataclass(frozen=True, slots=True)
class LocalLabelAtlas:
    """Adapt a locally supplied NIfTI label volume loaded by the imaging engine."""

    volume: MRIVolume
    name: str
    version: str
    license: str
    reference_space: str
    regions: tuple[AtlasRegion, ...]

    def load(self) -> Atlas:
        values = self.volume.data
        if (
            self.volume.assumed_mm
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values >= 2**24)
            or np.any(values != np.floor(values))
        ):
            raise MappingError("Atlas requires millimetres and exact integer labels")
        return Atlas(
            self.name,
            self.version,
            self.license,
            self.reference_space,
            values.astype(np.int32),
            self.volume.affine,
            self.regions,
        )
