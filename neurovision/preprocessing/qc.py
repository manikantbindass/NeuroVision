"""Configurable research QC checks and machine-readable reports."""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Literal

import numpy as np

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.imaging.volume import Modality, MRIVolume
from neurovision.preprocessing.contracts import Fields, Parameters

Severity = Literal["pass", "warning", "fail"]


@dataclass(frozen=True, slots=True)
class QCPolicy:
    min_axis_voxels: int = 2
    max_voxels: int = 32_000_000
    min_spacing_mm: float = 0.25
    max_spacing_mm: float = 8.0
    blank_range: float = 1e-8
    max_abs_intensity: float = 1e8
    min_registration_dice: float = 0.2

    def __post_init__(self) -> None:
        if (
            self.min_axis_voxels < 1
            or self.max_voxels < 1
            or not 0 < self.min_spacing_mm <= self.max_spacing_mm < float("inf")
            or not isfinite(self.blank_range)
            or self.blank_range < 0
            or not isfinite(self.max_abs_intensity)
            or self.max_abs_intensity <= 0
            or not 0 <= self.min_registration_dice <= 1
        ):
            raise ValueError("Invalid QC policy")


DEFAULT_QC_POLICY = QCPolicy()


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    severity: Severity
    scope: str
    message: str


@dataclass(frozen=True, slots=True)
class QCReport:
    findings: tuple[Finding, ...]
    metrics: Fields
    policy: QCPolicy

    @property
    def status(self) -> Severity:
        if any(item.severity == "fail" for item in self.findings):
            return "fail"
        return "warning" if self.findings else "pass"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "notice": RESEARCH_NOTICE,
            "policy": asdict(self.policy),
            "findings": [asdict(item) for item in self.findings],
            "metrics": dict(self.metrics),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, allow_nan=False) + "\n"


def assess_volume(
    volume: MRIVolume, *, policy: QCPolicy = DEFAULT_QC_POLICY, scope: str = "volume"
) -> QCReport:
    data = volume.data
    findings = []
    metrics: Parameters = {f"{scope}.shape": data.shape}
    if (
        data.ndim != 3
        or data.dtype != np.float32
        or data.size > policy.max_voxels
        or min(data.shape, default=0) < policy.min_axis_voxels
    ):
        findings.append(
            Finding("invalid_dimensions", "fail", scope, "Invalid MRI shape or dtype")
        )
        return QCReport(tuple(findings), tuple(metrics.items()), policy)
    spacing = np.asarray(volume.spacing_mm)
    affine = volume.affine
    with np.errstate(over="ignore", invalid="ignore"):
        actual = (
            np.linalg.norm(affine[:3, :3], axis=0) if affine.shape == (4, 4) else None
        )
        valid = (
            actual is not None
            and spacing.shape == (3,)
            and np.isfinite(spacing).all()
            and np.isfinite(affine).all()
            and np.all(spacing > 0)
            and np.allclose(actual, spacing, rtol=1e-6, atol=0)
            and np.array_equal(affine[3], [0, 0, 0, 1])
            and abs(np.linalg.det(affine[:3, :3] / spacing)) >= 1e-6
        )
    if not valid:
        findings.append(
            Finding("invalid_geometry", "fail", scope, "Invalid affine or spacing")
        )
    else:
        metrics[f"{scope}.spacing_mm"] = volume.spacing_mm
        if np.any(
            (spacing < policy.min_spacing_mm) | (spacing > policy.max_spacing_mm)
        ):
            findings.append(
                Finding(
                    "extreme_spacing", "warning", scope, "Spacing is outside policy"
                )
            )
    if not np.isfinite(data).all():
        findings.append(
            Finding("intensity_anomaly", "fail", scope, "MRI has nonfinite values")
        )
    else:
        low, high = float(data.min()), float(data.max())
        metrics.update(
            {
                f"{scope}.intensity_min": low,
                f"{scope}.intensity_max": high,
                f"{scope}.intensity_range": high - low,
                f"{scope}.nonzero_fraction": int(np.count_nonzero(data)) / data.size,
            }
        )
        if high - low <= policy.blank_range:
            findings.append(
                Finding(
                    "blank_scan", "fail", scope, "MRI has no usable intensity range"
                )
            )
        if max(abs(low), abs(high)) > policy.max_abs_intensity:
            findings.append(
                Finding(
                    "intensity_anomaly",
                    "warning",
                    scope,
                    "Absolute intensity exceeds policy",
                )
            )
    return QCReport(tuple(findings), tuple(metrics.items()), policy)


def assess_case(
    volumes: Mapping[Modality, MRIVolume],
    *,
    required: frozenset[Modality],
    policy: QCPolicy = DEFAULT_QC_POLICY,
) -> QCReport:
    """Run modality completeness and per-volume checks before case preprocessing."""
    missing = tuple(sorted(modality.value for modality in required.difference(volumes)))
    findings = []
    metrics: Parameters = {
        "case.required_modalities": tuple(sorted(item.value for item in required)),
        "case.missing_modalities": missing,
    }
    if missing:
        findings.append(
            Finding(
                "missing_modalities", "fail", "case", "Required MRI modalities missing"
            )
        )
    for modality, volume in volumes.items():
        if volume.modality not in (Modality.UNKNOWN, modality):
            findings.append(
                Finding(
                    "modality_mismatch",
                    "fail",
                    modality.value,
                    "Volume and case modality labels disagree",
                )
            )
        report = assess_volume(volume, policy=policy, scope=modality.value)
        findings.extend(report.findings)
        metrics.update(report.metrics)
    return QCReport(tuple(findings), tuple(metrics.items()), policy)
