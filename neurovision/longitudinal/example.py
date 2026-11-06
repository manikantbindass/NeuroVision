"""Generate an interactive NeuroTimeline from three synthetic research visits."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.imaging.volume import Modality, canonical_volume
from neurovision.longitudinal.engine import FollowUp, TimePoint, build_timeline
from neurovision.longitudinal.visual import write_timeline_html
from neurovision.preprocessing.adapters import SuppliedAffine
from neurovision.preprocessing.contracts import matrix_values


def _visit(day: int, shift: int, radius: float) -> TimePoint:
    x, y, z = np.indices((40, 36, 28), dtype=np.float32)
    dist = (x - 19 - shift) ** 2 + (y - 17) ** 2 + (z - 13) ** 2
    volume = canonical_volume(
        np.maximum(0, 1 - dist / 225).astype(np.float32),
        np.eye(4),
        source_format="nifti",
        modality=Modality.T1,
        reference_space="synthetic-visit",
    )
    labels = np.zeros(dist.shape, dtype=np.uint16)
    labels[dist <= radius**2] = 1
    stable = ((x - 12 - shift) ** 2 + (y - 9) ** 2 + (z - 13) ** 2) <= 9
    labels[stable] = 300
    return TimePoint(
        "synthetic-study",
        day,
        volume,
        labels,
        "phantom-v1",
        "synthetic T1",
        "analytic phantom",
        "analytic sphere labels",
        replicate_volumes_mm3={
            key: {
                1: float(np.count_nonzero((dist <= (radius + delta) ** 2) & ~stable)),
                300: float(np.count_nonzero(stable)),
            }
            for key, delta in (("smaller", -0.25), ("nominal", 0.0), ("larger", 0.25))
        },
        replicate_protocol="Matched analytic radius perturbations in the phantom",
    )


def _shift(voxels: int) -> NDArray[np.float64]:
    matrix = np.eye(4)
    matrix[0, 3] = -voxels
    return matrix


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a synthetic longitudinal comparison.", epilog=RESEARCH_NOTICE
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    timeline = build_timeline(
        _visit(0, 0, 5),
        tuple(
            FollowUp(
                _visit(day, shift, radius),
                SuppliedAffine(matrix_values(_shift(shift))),
                f"Known phantom translation of {-shift} mm along RAS x",
            )
            for day, shift, radius in ((90, 3, 6), (180, -2, 4))
        ),
        class_names={1: "Changing sphere", 300: "Stable reference sphere"},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_timeline_html(args.output, timeline)
    print(RESEARCH_NOTICE)
    print(timeline.warning)
    print("Wrote an interactive comparison from three synthetic time points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
