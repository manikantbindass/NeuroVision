"""Per-class overlap, confusion, and physical boundary distances."""

from dataclasses import dataclass
from math import isfinite
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    label: int
    dice: float | None
    iou: float | None
    precision: float | None
    sensitivity: float | None
    specificity: float | None
    hausdorff_mm: float | None
    hd95_mm: float | None
    assd_mm: float | None

    @property
    def recall(self) -> float | None:
        return self.sensitivity


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _distances(
    first: NDArray[np.bool_],
    second: NDArray[np.bool_],
    spacing: tuple[float, ...],
) -> tuple[float, float, float]:
    a = first & ~ndimage.binary_erosion(first, border_value=0)
    b = second & ~ndimage.binary_erosion(second, border_value=0)
    ab = cast(
        NDArray[np.float64], ndimage.distance_transform_edt(~b, sampling=spacing)
    )[a]
    ba = cast(
        NDArray[np.float64], ndimage.distance_transform_edt(~a, sampling=spacing)
    )[b]
    return (
        float(max(ab.max(), ba.max())),
        float(max(np.percentile(ab, 95), np.percentile(ba, 95))),
        float((ab.sum() + ba.sum()) / (ab.size + ba.size)),
    )


def segmentation_metrics(
    prediction: NDArray[np.int64],
    target: NDArray[np.int64],
    *,
    num_classes: int,
    spacing_mm: tuple[float, ...] | None = None,
    include_background: bool = False,
) -> tuple[ClassMetrics, ...]:
    """Score same-grid 2D/3D labels. Undefined metrics return None."""
    if (
        prediction.shape != target.shape
        or prediction.ndim not in (2, 3)
        or not prediction.size
        or num_classes < 2
        or prediction.dtype.kind not in "iu"
        or target.dtype.kind not in "iu"
        or min(prediction.min(), target.min()) < 0
        or max(prediction.max(), target.max()) >= num_classes
    ):
        raise ValueError("Metrics require matching integer grids and valid label IDs")
    if spacing_mm is not None and (
        len(spacing_mm) != prediction.ndim
        or any(not isfinite(v) or v <= 0 for v in spacing_mm)
    ):
        raise ValueError("Physical spacing must match the spatial axis order")
    scores = []
    for label in range(0 if include_background else 1, num_classes):
        predicted, actual = prediction == label, target == label
        positives, expected = int(predicted.sum()), int(actual.sum())
        tp = int(np.count_nonzero(predicted & actual))
        fp, fn = positives - tp, expected - tp
        tn = prediction.size - tp - fp - fn
        hd, hd95, assd = (
            _distances(predicted, actual, spacing_mm)
            if spacing_mm is not None and positives and expected
            else (None, None, None)
        )
        scores.append(
            ClassMetrics(
                label,
                _ratio(2 * tp, positives + expected),
                _ratio(tp, tp + fp + fn),
                _ratio(tp, positives),
                _ratio(tp, expected),
                _ratio(tn, tn + fp),
                hd,
                hd95,
                assd,
            )
        )
    return tuple(scores)
