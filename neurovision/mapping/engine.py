"""Quantify native-grid segmentations and categorical atlas overlap."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import affine_transform

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.imaging.volume import MRIVolume
from neurovision.mapping.atlas import (
    AtlasAdapter,
    AtlasRegion,
    Hemisphere,
    MappingError,
)


@dataclass(frozen=True, slots=True)
class AtlasAssociation:
    region: AtlasRegion
    voxels: int
    volume_mm3: float
    fraction_of_segment: float


@dataclass(frozen=True, slots=True)
class RegionMeasurement:
    code: int
    name: str
    voxels: int
    volume_mm3: float
    fraction_of_foreground: float
    mean_confidence: float | None
    hemisphere: Hemisphere
    atlas_coverage: float
    associations: tuple[AtlasAssociation, ...]


@dataclass(frozen=True, slots=True)
class NeuroMapResult:
    atlas_name: str
    atlas_version: str
    atlas_license: str
    atlas_space: str
    segmentation_space: str
    registration_provenance: str
    segmentation_to_atlas_world: tuple[tuple[float, ...], ...]
    voxel_volume_mm3: float
    measurements: tuple[RegionMeasurement, ...]
    aligned_atlas: NDArray[np.int32]
    notice: str = RESEARCH_NOTICE


def _checked_affine(matrix: NDArray[np.float64]) -> float:
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.array_equal(matrix[3], [0, 0, 0, 1])
    ):
        raise MappingError("Expected a finite homogeneous affine")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        determinant = float(abs(np.linalg.det(matrix[:3, :3])))
    if not np.isfinite(determinant) or determinant <= 0:
        raise MappingError("Affine determinant must be finite and nonzero")
    return determinant


class NeuroMapEngine:
    """Map an externally registered atlas onto a 3D segmentation grid."""

    def __init__(self, adapter: AtlasAdapter, *, max_voxels: int = 16_777_216) -> None:
        if type(max_voxels) is not int or max_voxels < 1:
            raise MappingError("Voxel budget must be a positive integer")
        self.adapter = adapter
        self.max_voxels = max_voxels

    def map(
        self,
        reference: MRIVolume,
        labels: NDArray[np.integer],
        *,
        class_names: Mapping[int, str],
        segmentation_to_atlas_world: NDArray[np.float64],
        registration_provenance: str,
        confidence: NDArray[np.floating] | None = None,
    ) -> NeuroMapResult:
        if (
            labels.ndim != 3
            or labels.shape != reference.data.shape
            or not 0 < labels.size <= self.max_voxels
            or labels.dtype.kind not in "iu"
            or np.any(labels < 0)
            or reference.assumed_mm
            or not reference.reference_space.strip()
            or not registration_provenance.strip()
            or any(
                type(code) is not int or code < 0 or not name.strip()
                for code, name in class_names.items()
            )
        ):
            raise MappingError(
                "Invalid segmentation grid, label schema or registration"
            )
        present = set(int(code) for code in np.unique(labels)) - {0}
        if len(present) > 256:
            raise MappingError("NeuroMap supports at most 256 foreground classes")
        if not present <= class_names.keys():
            raise MappingError(
                "The segmentation contains undeclared foreground classes"
            )
        if confidence is not None and (
            confidence.shape != labels.shape
            or confidence.dtype.kind != "f"
            or not np.isfinite(confidence).all()
            or np.any(confidence < 0)
            or np.any(confidence > 1)
        ):
            raise MappingError("Confidence must match the grid and lie in [0, 1]")

        atlas = self.adapter.load()
        lookup = {region.code: region for region in atlas.regions}
        if (
            not all(
                value.strip()
                for value in (
                    atlas.name,
                    atlas.version,
                    atlas.license,
                    atlas.reference_space,
                )
            )
            or atlas.labels.ndim != 3
            or not 0 < atlas.labels.size <= self.max_voxels
            or atlas.labels.dtype != np.int32
            or len(lookup) != len(atlas.regions)
            or any(
                type(code) is not int
                or code < 1
                or not region.name.strip()
                or region.hemisphere
                not in {"left", "right", "bilateral", "midline", "unknown"}
                for code, region in lookup.items()
            )
            or not (set(np.unique(atlas.labels)) - {0}) <= lookup.keys()
        ):
            raise MappingError("Invalid atlas grid, metadata or region lookup")
        voxel_volume = _checked_affine(reference.affine)
        if not np.isfinite(voxel_volume * labels.size):
            raise MappingError("Physical volume exceeds numerical bounds")
        _checked_affine(atlas.affine)
        _checked_affine(segmentation_to_atlas_world)
        with np.errstate(over="ignore", invalid="ignore"):
            pull = np.linalg.solve(
                atlas.affine, segmentation_to_atlas_world @ reference.affine
            )
        if not np.isfinite(pull).all():
            raise MappingError("The atlas sampling transform is not finite")
        aligned: NDArray[np.int32] = affine_transform(
            atlas.labels,
            pull[:3, :3],
            offset=pull[:3, 3],
            output_shape=labels.shape,
            order=0,
            mode="grid-constant",
            cval=0,
            prefilter=False,
        )
        aligned.setflags(write=False)

        foreground = int(np.count_nonzero(labels))
        measurements = []
        for code in sorted(present):
            mask = labels == code
            voxels = int(np.count_nonzero(mask))
            codes, counts = np.unique(aligned[mask], return_counts=True)
            associations = tuple(
                AtlasAssociation(
                    lookup[int(atlas_code)],
                    int(count),
                    int(count) * voxel_volume,
                    int(count) / voxels,
                )
                for atlas_code, count in zip(codes, counts, strict=True)
                if atlas_code
            )
            coverage = sum(entry.voxels for entry in associations) / voxels
            sides = {entry.region.hemisphere for entry in associations}
            hemisphere: Hemisphere = "unknown"
            if coverage == 1 and "unknown" not in sides:
                if len(sides) == 1:
                    hemisphere = next(iter(sides))
                elif "bilateral" in sides or {"left", "right"} <= sides:
                    hemisphere = "bilateral"
            measurements.append(
                RegionMeasurement(
                    code,
                    class_names[code],
                    voxels,
                    voxels * voxel_volume,
                    voxels / foreground,
                    None
                    if confidence is None
                    else float(confidence[mask].mean(dtype=np.float64)),
                    hemisphere,
                    coverage,
                    associations,
                )
            )
        return NeuroMapResult(
            atlas.name,
            atlas.version,
            atlas.license,
            atlas.reference_space,
            reference.reference_space,
            registration_provenance,
            tuple(
                tuple(float(value) for value in row)
                for row in segmentation_to_atlas_world
            ),
            voxel_volume,
            tuple(measurements),
            aligned,
        )
