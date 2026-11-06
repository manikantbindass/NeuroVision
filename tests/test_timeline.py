"""Analytic tests for coverage, change metrics and paired replicate spread."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurovision.imaging.volume import Modality, canonical_volume
from neurovision.longitudinal.engine import (
    FollowUp,
    TimelineError,
    TimePoint,
    build_timeline,
)
from neurovision.longitudinal.visual import write_timeline_html
from neurovision.preprocessing.adapters import SuppliedAffine
from neurovision.preprocessing.contracts import matrix_values


def _point(day: int, x_values: list[int], *, z_spacing: float = 2.0) -> TimePoint:
    labels = np.broadcast_to(np.array(x_values)[:, None, None], (5, 2, 2))
    volume = canonical_volume(
        np.ones(labels.shape, dtype=np.float32),
        np.diag([1.0, 1.0, z_spacing, 1.0]),
        source_format="nifti",
        modality=Modality.T1,
        reference_space="scanner",
    )
    return TimePoint(
        "local-key", day, volume, labels, "schema", "sequence", "prep", "model"
    )


def _affine(offset: float = 0) -> NDArray[np.float64]:
    matrix = np.eye(4)
    matrix[0, 3] = offset
    return matrix


def _follow(point: TimePoint, matrix: NDArray[np.float64]) -> FollowUp:
    return FollowUp(point, SuppliedAffine(matrix_values(matrix)), "reviewed <fixture>")


class TimelineTests(unittest.TestCase):
    def test_translation_coverage_and_native_change(self) -> None:
        baseline = _point(0, [1, 1, 0, 0, 0])
        moving = _point(10, [1, 2, 0, 0, 0])
        timeline = build_timeline(
            baseline,
            [_follow(replace(moving, model="changed"), _affine(1))],
            class_names={1: "Existing", 2: "New"},
        )
        pair = timeline.comparisons[0]
        old, new = pair.regions
        self.assertEqual(pair.field_of_view_fraction, 0.8)
        self.assertEqual(pair.changed_metadata, ("model",))
        self.assertEqual(
            (old.baseline_mm3, old.follow_up_mm3, old.delta_mm3), (16, 8, -8)
        )
        self.assertEqual(old.percent_change, -50)
        self.assertEqual((old.gained_voxels, old.lost_voxels, old.dice), (0, 0, 1))
        self.assertIsNone(new.percent_change)
        self.assertEqual(new.gained_voxels, 4)
        np.testing.assert_array_equal(
            timeline.difference_map(0, 1)[:, 0, 0], [2, 0, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            timeline.difference_map(0, 2)[:, 0, 0], [2, 0, 1, 0, 0]
        )

    def test_native_geometry_and_empirical_spread(self) -> None:
        baseline = replace(
            _point(0, [1, 1, 0, 0, 0]),
            replicate_volumes_mm3={"a": {1: 15}, "b": {1: 16}, "c": {1: 19}},
            replicate_protocol="synthetic repeats",
        )
        moving = replace(
            _point(5, [1, 0, 0, 0, 0], z_spacing=4),
            replicate_volumes_mm3={"c": {1: 21}, "b": {1: 16}, "a": {1: 12}},
            replicate_protocol="synthetic repeats",
        )
        timeline = build_timeline(
            baseline, [_follow(moving, _affine())], class_names={1: "A"}
        )
        region = timeline.comparisons[0].regions[0]
        self.assertEqual(region.delta_mm3, 0)
        self.assertEqual(region.lost_voxels, 4)
        spread = region.spread
        assert spread is not None
        self.assertEqual((spread.pairs, spread.median_mm3), (3, 0))
        self.assertAlmostEqual(spread.p05_mm3, -2.7)
        self.assertAlmostEqual(spread.p95_mm3, 1.8)
        with self.assertRaisesRegex(TimelineError, "replicate IDs and protocols"):
            build_timeline(
                baseline,
                [_follow(replace(moving, replicate_protocol="different"), _affine())],
                class_names={1: "A"},
            )
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "timeline.html"
            write_timeline_html(report, timeline)
            html = report.read_text(encoding="utf-8")
            encoded = html.split('id="payload">')[1].split("</script>")[0]
            self.assertNotIn("<fixture>", html)
            payload = json.loads(encoded)
            self.assertEqual(payload["warning"], timeline.warning)
            self.assertEqual(payload["pairs"][0]["provenance"], "reviewed <fixture>")
            self.assertEqual(payload["pairs"][0]["regions"][0]["spread"]["pairs"], 3)

    def test_rejects_incompatible_and_uncovered_comparisons(self) -> None:
        baseline = _point(0, [0, 1, 0, 0, 0])
        moving = _point(2, [0, 1, 0, 0, 0])
        reflection = np.diag([-1.0, 1.0, 1.0, 1.0])
        for visit, transform, message in (
            (replace(moving, study_key="other"), _affine(), "share study"),
            (moving, _affine(40), "minimum baseline grid coverage"),
            (moving, reflection, "proper rigid"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(TimelineError, message),
            ):
                build_timeline(
                    baseline, [_follow(visit, transform)], class_names={1: "A"}
                )
