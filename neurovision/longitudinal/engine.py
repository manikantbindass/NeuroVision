"""Compare native regional volumes and registered segmentation masks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import cast

import numpy as np
from numpy.typing import NDArray

from neurovision.core.errors import PipelineError
from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.imaging.volume import Modality, MRIVolume
from neurovision.preprocessing.adapters import (
    Register,
    RegistrationBackend,
    SuppliedAffine,
)
from neurovision.preprocessing.contracts import Matrix, matrix_values
from neurovision.preprocessing.spatial import sample

CHANGE_WARNING = (
    "Observed differences may arise from acquisition, preprocessing, registration, "
    "model, or segmentation variation."
)


class TimelineError(PipelineError):
    """Time points could not be compared safely."""


@dataclass(frozen=True, slots=True)
class TimePoint:
    """A study-local time point with labels on the supplied MRI grid."""

    study_key: str
    day: int
    volume: MRIVolume = field(repr=False)
    labels: NDArray[np.integer] = field(repr=False)
    schema: str
    acquisition: str
    preprocessing: str
    model: str
    replicate_volumes_mm3: Mapping[str, Mapping[int, float]] | None = field(
        default=None, repr=False
    )
    replicate_protocol: str = ""


@dataclass(frozen=True, slots=True)
class FollowUp:
    point: TimePoint
    registration: RegistrationBackend
    registration_provenance: str


@dataclass(frozen=True, slots=True)
class ChangeSpread:
    pairs: int
    p05_mm3: float
    median_mm3: float
    p95_mm3: float


@dataclass(frozen=True, slots=True)
class RegionChange:
    code: int
    name: str
    baseline_mm3: float
    follow_up_mm3: float
    delta_mm3: float
    percent_change: float | None
    gained_voxels: int
    lost_voxels: int
    dice: float | None
    spread: ChangeSpread | None


@dataclass(frozen=True, slots=True)
class Comparison:
    day: int
    registration_provenance: str
    moving_to_baseline: Matrix
    field_of_view_fraction: float
    foreground_dice: float
    changed_metadata: tuple[str, ...]
    replicate_protocol: str | None
    regions: tuple[RegionChange, ...]
    registered_volume: MRIVolume = field(repr=False)
    registered_labels: NDArray[np.uint16] = field(repr=False)
    valid: NDArray[np.bool_] = field(repr=False)


@dataclass(frozen=True, slots=True)
class NeuroTimeline:
    baseline_day: int
    baseline_volume: MRIVolume = field(repr=False)
    baseline_labels: NDArray[np.uint16] = field(repr=False)
    comparisons: tuple[Comparison, ...]
    warning: str = CHANGE_WARNING
    notice: str = RESEARCH_NOTICE

    def difference_map(self, follow_up_index: int, class_id: int) -> NDArray[np.int8]:
        """Return -1 loss, +1 gain, 0 unchanged, and 2 outside shared coverage."""
        if not 0 <= follow_up_index < len(self.comparisons):
            raise TimelineError("Unknown follow-up index")
        pair = self.comparisons[follow_up_index]
        if not any(region.code == class_id for region in pair.regions):
            raise TimelineError("Unknown foreground class")
        change: NDArray[np.int8] = (pair.registered_labels == class_id).astype(np.int8)
        change -= self.baseline_labels == class_id
        change[~pair.valid] = 2
        change.setflags(write=False)
        return change


def _validate_point(
    point: TimePoint, class_names: Mapping[int, str], max_voxels: int
) -> float:
    volume, labels = point.volume, point.labels
    if (
        type(point.day) is not int
        or not 0 <= point.day <= 365_000
        or not all(
            value.strip()
            for value in (
                point.study_key,
                point.schema,
                point.acquisition,
                point.preprocessing,
                point.model,
                volume.reference_space,
            )
        )
        or volume.modality == Modality.UNKNOWN
        or volume.assumed_mm
        or volume.data.ndim != 3
        or labels.shape != volume.data.shape
        or not 0 < labels.size <= max_voxels
        or not np.isfinite(volume.data).all()
        or labels.dtype.kind not in "iu"
        or not (set(np.unique(labels)) - {0}) <= class_names.keys()
        or volume.affine.shape != (4, 4)
        or not np.isfinite(volume.affine).all()
        or not np.array_equal(volume.affine[3], [0, 0, 0, 1])
    ):
        raise TimelineError("Invalid time-point metadata, MRI grid or label schema")
    voxel_volume = float(abs(np.linalg.det(volume.affine[:3, :3])))
    if not 0 < voxel_volume or not isfinite(voxel_volume * labels.size):
        raise TimelineError("Invalid physical voxel volume")
    replicates = point.replicate_volumes_mm3
    if replicates is not None and (
        not point.replicate_protocol.strip()
        or not 3 <= len(replicates) <= 256
        or any(
            not key.strip()
            or values.keys() != class_names.keys()
            or any(
                not 0 <= value <= voxel_volume * labels.size
                for value in values.values()
            )
            for key, values in replicates.items()
        )
    ):
        raise TimelineError("Invalid paired replicate region volumes")
    return voxel_volume


def _spread(baseline: TimePoint, moving: TimePoint, code: int) -> ChangeSpread | None:
    first, second = baseline.replicate_volumes_mm3, moving.replicate_volumes_mm3
    if first is None or second is None:
        return None
    changes = [second[key][code] - first[key][code] for key in sorted(first)]
    low, median, high = (
        float(value) for value in np.quantile(changes, [0.05, 0.5, 0.95])
    )
    return ChangeSpread(len(changes), low, median, high)


def build_timeline(
    baseline: TimePoint,
    follow_ups: Sequence[FollowUp],
    *,
    class_names: Mapping[int, str],
    seed: int = 0,
    min_coverage: float = 0.5,
    max_voxels: int = 8_388_608,
) -> NeuroTimeline:
    """Compare each follow-up to baseline with independent reviewed registration."""
    if (
        not 1 <= len(follow_ups) <= 12
        or not 1 <= len(class_names) <= 256
        or any(
            type(code) is not int or not 1 <= code <= 65535 or not name.strip()
            for code, name in class_names.items()
        )
        or type(max_voxels) is not int
        or max_voxels < 1
        or not 0 < min_coverage <= 1
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        raise TimelineError("Invalid timeline limits or foreground classes")
    baseline_voxel = _validate_point(baseline, class_names, max_voxels)
    fixed = baseline.volume
    baseline_labels = baseline.labels.astype(np.uint16, copy=True)
    baseline_labels.setflags(write=False)
    if baseline_labels.size * len(follow_ups) > 4 * max_voxels:
        raise TimelineError("Timeline exceeds the retained output-grid budget")
    baseline_counts = np.bincount(baseline_labels.ravel(), minlength=65536)
    comparisons = []
    previous_day = baseline.day
    for follow_up in follow_ups:
        moving = follow_up.point
        moving_voxel = _validate_point(moving, class_names, max_voxels)
        if (
            moving.day <= previous_day
            or moving.study_key != baseline.study_key
            or moving.schema != baseline.schema
            or moving.volume.modality != fixed.modality
            or not follow_up.registration_provenance.strip()
        ):
            raise TimelineError(
                "Time points must be ordered and share study, schema and modality"
            )
        first, second = baseline.replicate_volumes_mm3, moving.replicate_volumes_mm3
        paired = first is not None and second is not None
        if (
            first is not None
            and second is not None
            and (
                first.keys() != second.keys()
                or baseline.replicate_protocol != moving.replicate_protocol
            )
        ):
            raise TimelineError("Paired replicate IDs and protocols must match")

        transform = np.asarray(
            follow_up.registration.moving_to_fixed(moving.volume, fixed, seed=seed),
            dtype=np.float64,
        )
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.array_equal(transform[3], [0, 0, 0, 1])
            or not np.allclose(
                transform[:3, :3].T @ transform[:3, :3], np.eye(3), rtol=0, atol=1e-5
            )
            or np.linalg.det(transform[:3, :3]) <= 0
        ):
            raise TimelineError(
                "Timeline registration requires a proper rigid transform"
            )
        matrix = matrix_values(transform)
        registered, qc = Register(fixed, SuppliedAffine(matrix)).apply(
            moving.volume, seed=seed
        )
        pull = np.asarray(qc["output_to_input_voxels"], dtype=np.float64)
        x, y, z = fixed.data.shape
        # Negative fill distinguishes out-of-field samples from label zero.
        sampled = sample(
            moving.labels.astype(np.float32), pull, (x, y, z), "nearest", -1
        )
        valid = sampled >= 0
        coverage = float(valid.mean())
        if coverage < min_coverage:
            raise TimelineError(
                "Registration falls below minimum baseline grid coverage"
            )
        labels = np.where(valid, sampled, 0).astype(np.uint16)
        labels.setflags(write=False)
        valid.setflags(write=False)
        moving_counts = np.bincount(
            moving.labels.astype(np.uint16).ravel(), minlength=65536
        )
        regions = []
        for code, name in sorted(class_names.items()):
            before = int(baseline_counts[code]) * baseline_voxel
            after = int(moving_counts[code]) * moving_voxel
            delta = after - before
            percent = 100 * (delta / before) if before else None
            if percent is not None and not isfinite(percent):
                raise TimelineError("Percentage change exceeds numerical bounds")
            fixed_mask = (baseline_labels == code) & valid
            moving_mask = (labels == code) & valid
            shared = int(np.count_nonzero(fixed_mask & moving_mask))
            fixed_count = int(np.count_nonzero(fixed_mask))
            moving_count = int(np.count_nonzero(moving_mask))
            total = fixed_count + moving_count
            regions.append(
                RegionChange(
                    code,
                    name,
                    before,
                    after,
                    delta,
                    percent,
                    moving_count - shared,
                    fixed_count - shared,
                    2 * shared / total if total else None,
                    _spread(baseline, moving, code),
                )
            )
        comparisons.append(
            Comparison(
                moving.day,
                follow_up.registration_provenance,
                matrix,
                coverage,
                cast(float, qc["registration_foreground_dice"]),
                tuple(
                    key
                    for key in ("acquisition", "preprocessing", "model")
                    if getattr(baseline, key) != getattr(moving, key)
                ),
                baseline.replicate_protocol if paired else None,
                tuple(regions),
                registered,
                labels,
                valid,
            )
        )
        previous_day = moving.day
    return NeuroTimeline(baseline.day, fixed, baseline_labels, tuple(comparisons))
