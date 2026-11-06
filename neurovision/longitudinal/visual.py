"""Export an offline, downsampled NeuroTimeline research viewer."""

import base64
import json
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurovision.longitudinal.engine import NeuroTimeline


def _encoded(array: NDArray[np.generic]) -> str:
    return base64.b64encode(array.tobytes()).decode("ascii")


def write_timeline_html(path: Path, timeline: NeuroTimeline) -> None:
    """Embed sampled images and masks, with full-resolution statistics."""
    shape = timeline.baseline_volume.data.shape
    indices = [
        np.arange(0, size, max(1, (size + cap - 1) // cap))
        for size, cap in zip(shape, (64, 64, 32), strict=True)
    ]
    selection = np.ix_(*indices)
    baseline = timeline.baseline_volume.data[selection].astype(np.float64)
    low, high = np.percentile(baseline, [1, 99])
    if low == high:
        low, high = baseline.min(), baseline.max()

    def grayscale(data: NDArray[np.float32]) -> str:
        values = data[selection].astype(np.float64)
        pixels = np.rint(np.clip((values - low) / (high - low or 1), 0, 1) * 255)
        return _encoded(pixels.astype(np.uint8))

    payload = {
        "baselineDay": timeline.baseline_day,
        "notice": timeline.notice,
        "warning": timeline.warning,
        "shape": baseline.shape,
        "slices": indices[2].tolist(),
        "baselineImage": grayscale(timeline.baseline_volume.data),
        "baselineLabels": _encoded(timeline.baseline_labels[selection].astype("<u2")),
        "pairs": [
            {
                "day": pair.day,
                "provenance": pair.registration_provenance,
                "coverage": pair.field_of_view_fraction,
                "foregroundDice": pair.foreground_dice,
                "changedMetadata": pair.changed_metadata,
                "replicateProtocol": pair.replicate_protocol,
                "regions": [asdict(region) for region in pair.regions],
                "image": grayscale(pair.registered_volume.data),
                "labels": _encoded(pair.registered_labels[selection].astype("<u2")),
                "valid": _encoded(pair.valid[selection].astype(np.uint8)),
            }
            for pair in timeline.comparisons
        ],
    }
    data = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    template = files("neurovision.longitudinal").joinpath("neurotimeline.html")
    path.write_text(
        template.read_text(encoding="utf-8").replace("__PAYLOAD__", data),
        encoding="utf-8",
    )
