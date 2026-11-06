"""Analytic segmentation sensitivity and atlas geometry regressions."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from neurovision.explainability.attribution import (
    AttributionError,
    explain_segmentation,
)
from neurovision.imaging.volume import Modality, canonical_volume
from neurovision.inference.engine import InferenceEngine
from neurovision.mapping.atlas import AtlasRegion, LocalLabelAtlas, MappingError
from neurovision.mapping.engine import NeuroMapEngine
from neurovision.mapping.visual import write_comparison_html
from neurovision.uncertainty.example import SyntheticThresholdModel


class NeuroMapTests(unittest.TestCase):
    def test_physical_volume_coverage_hemisphere_and_escaped_report(self) -> None:
        shape = (3, 2, 2)
        geometry = np.diag([2.0, 3.0, 4.0, 1.0])
        geometry[0, 1] = 1.5  # Shear must not inflate physical volume.
        reference = canonical_volume(
            np.ones(shape, dtype=np.float32),
            geometry,
            source_format="nifti",
            modality=Modality.T1,
            reference_space="subject",
        )
        atlas = canonical_volume(
            np.broadcast_to(
                np.array([2, 1, 0], dtype=np.float32)[:, None, None], shape
            ),
            geometry,
            source_format="nifti",
            modality=Modality.UNKNOWN,
            reference_space="atlas",
        )
        adapter = LocalLabelAtlas(
            atlas,
            "<atlas>",
            "1",
            "MIT",
            "atlas",
            (
                AtlasRegion(1, "Left", "left"),
                AtlasRegion(2, "Right", "right"),
            ),
        )
        labels = np.broadcast_to(np.array([1, 2, 0])[:, None, None], shape)
        translation = np.eye(4)
        translation[0, 3] = 2.0
        engine = NeuroMapEngine(adapter)
        result = engine.map(
            reference,
            labels,
            class_names={1: "<foreground>", 2: "Partial"},
            segmentation_to_atlas_world=translation,
            registration_provenance="Verified synthetic +2 mm R shift",
            confidence=np.full(shape, 0.75),
        )
        first, second = result.measurements
        self.assertAlmostEqual(first.volume_mm3, 96.0)
        self.assertEqual(first.voxels, 4)
        self.assertEqual(first.fraction_of_foreground, 0.5)
        self.assertEqual(first.mean_confidence, 0.75)
        self.assertEqual(first.hemisphere, "left")
        self.assertEqual(first.associations[0].region.name, "Left")
        self.assertEqual(first.associations[0].fraction_of_segment, 1.0)
        self.assertEqual(second.hemisphere, "unknown")
        self.assertEqual(second.atlas_coverage, 0.0)
        np.testing.assert_array_equal(result.aligned_atlas[:, 0, 0], [1, 0, 0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.html"
            write_comparison_html(path, reference, labels, result, slice_index=0)
            html = path.read_text(encoding="utf-8")
            for expected in (
                "&lt;atlas&gt;",
                "&lt;foreground&gt;",
                "Aligned atlas",
                result.notice,
            ):
                self.assertIn(expected, html)
            self.assertNotIn("<foreground>", html)
        with self.assertRaises(MappingError):
            engine.map(
                reference,
                labels,
                class_names={1: "Only"},
                segmentation_to_atlas_world=np.eye(4),
                registration_provenance="test",
            )

    def test_fixed_roi_gradient_and_signed_patch_response(self) -> None:
        model = SyntheticThresholdModel(0.5)
        model.train()
        engine = InferenceEngine(model)
        values = torch.ones((1, 1, 2, 2, 2), requires_grad=True)
        roi = torch.zeros((2, 2, 2), dtype=torch.bool)
        roi[0, 0, 0] = True
        with torch.inference_mode():
            result = explain_segmentation(
                engine,
                values,
                class_id=1,
                roi=roi,
                patch_size=(1, 2, 2),
                fill=2.0,
            )
        self.assertEqual(result.baseline_logit, 5.0)
        self.assertEqual(result.roi_voxels, 1)
        self.assertEqual(result.perturbations, 2)
        self.assertEqual(float(result.saliency.sum()), 10.0)
        self.assertEqual(float(result.saliency[0, 0, 0, 0]), 10.0)
        torch.testing.assert_close(
            result.occlusion[0], torch.full((2, 2), -10.0, dtype=torch.float64)
        )
        self.assertEqual(float(result.occlusion[1].sum()), 0.0)
        self.assertTrue(model.training)
        self.assertIsNone(values.grad)
        with self.assertRaisesRegex(AttributionError, "budget"):
            explain_segmentation(
                engine,
                values,
                class_id=1,
                roi=roi,
                patch_size=(1, 1, 1),
                max_patches=7,
            )
