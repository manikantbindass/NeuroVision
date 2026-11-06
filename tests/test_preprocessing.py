import base64
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from numpy.typing import NDArray

from neurovision.core.pipeline import RESEARCH_NOTICE, RunContext
from neurovision.imaging.volume import Modality, MRIVolume, canonical_volume
from neurovision.preprocessing.adapters import (
    ApplyBrainMask,
    N4BiasCorrection,
    Register,
    SuppliedAffine,
)
from neurovision.preprocessing.contracts import matrix_values
from neurovision.preprocessing.intensity import (
    GaussianDenoise,
    NormalizeIntensity,
    RandomFlip,
    RandomIntensity,
)
from neurovision.preprocessing.pipeline import PreprocessingCache, PreprocessingPipeline
from neurovision.preprocessing.qc import QCPolicy, assess_case, assess_volume
from neurovision.preprocessing.spatial import (
    CropPad,
    ExtractForeground,
    OrientRAS,
    Resample,
)
from neurovision.preprocessing.visual import write_qc_montage

_RUN = RunContext("synthetic", 42)


def _volume(
    data: NDArray[np.float32] | None = None, affine: NDArray[np.float64] | None = None
) -> MRIVolume:
    if data is None:
        data = np.zeros((6, 6, 6), dtype=np.float32)
        data[1:5, 1:5, 1:5] = np.arange(1, 65).reshape((4, 4, 4))
    return canonical_volume(
        data,
        np.eye(4) if affine is None else affine,
        source_format="nifti",
        modality=Modality.T1,
        reference_space="scanner",
    )


class SpatialTests(unittest.TestCase):
    def test_nearest_resampling_keeps_discrete_values(self) -> None:
        source = np.zeros((3, 3, 3), dtype=np.float32)
        source[1:, 1:, 1:] = 4
        result = PreprocessingPipeline((Resample((0.5, 0.5, 0.5), "nearest"),)).run(
            _volume(source), context=_RUN
        )
        assert result.volume is not None
        self.assertEqual(set(np.unique(result.volume.data)), {0, 4})
        self.assertEqual(result.volume.data.shape, (5, 5, 5))

    def test_orientation_preserves_world_location(self) -> None:
        volume = _volume()
        flipped_affine = np.diag([-1.0, 1.0, 1.0, 1.0])
        flipped_affine[0, 3] = 5
        native = replace(
            volume,
            data=volume.data[::-1],
            affine=flipped_affine,
            orientation=("L", "A", "S"),
        )
        result = PreprocessingPipeline((OrientRAS(),)).run(native, context=_RUN)
        self.assertIsNotNone(result.volume)
        assert result.volume is not None
        np.testing.assert_array_equal(result.volume.data, volume.data)
        np.testing.assert_array_equal(result.volume.affine, volume.affine)
        self.assertEqual(len(result.steps), 1)
        self.assertNotEqual(result.steps[0].input_digest, result.steps[0].output_digest)

    def test_resample_and_crop_pad_record_affine_and_preserve_samples(self) -> None:
        data = np.broadcast_to(np.arange(5, dtype=np.float32)[:, None, None], (5, 5, 5))
        affine = np.diag([2.0, 1.0, 1.0, 1.0])
        affine[:3, 3] = [10, 11, 12]
        volume = _volume(data, affine)
        result = PreprocessingPipeline((Resample((1, 1, 1)), CropPad((7, 7, 3)))).run(
            volume, context=_RUN
        )
        assert result.volume is not None
        self.assertEqual(result.volume.data.shape, (7, 7, 3))
        np.testing.assert_allclose(result.volume.spacing_mm, [1, 1, 1])
        self.assertAlmostEqual(float(result.volume.data[0, 2, 1]), 0.5)
        self.assertEqual(float(result.volume.data[1, 0, 1]), 0)
        np.testing.assert_allclose(result.volume.affine @ [0, 2, 1, 1], [11, 12, 14, 1])
        self.assertEqual(dict(result.steps[1].details)["start_voxel"], (1, -1, 1))
        self.assertEqual(result.steps[0].output_digest, result.steps[1].input_digest)

    def test_foreground_mask_and_external_skull_strip(self) -> None:
        base = _volume()
        data = base.data.copy()
        data[0, 0, 0] = 100
        mask = np.zeros_like(data)
        mask[1:4, 1:4, 1:4] = 1
        pipeline = PreprocessingPipeline(
            (ExtractForeground(), ApplyBrainMask(_volume(mask), "test", "1"))
        )
        result = pipeline.run(_volume(data), context=_RUN)
        assert result.volume is not None
        self.assertEqual(dict(result.steps[0].details)["foreground_voxels"], 64)
        self.assertEqual(dict(result.steps[1].details)["mask_voxels"], 27)
        np.testing.assert_array_equal(result.volume.data, np.where(mask, data, 0))
        shifted = np.eye(4)
        shifted[0, 3] = 2
        bad = PreprocessingPipeline(
            (ApplyBrainMask(_volume(mask, shifted), "test", "1"),)
        )
        self.assertEqual(bad.run(base, context=_RUN).report.status, "fail")

    def test_registration_applies_moving_to_fixed_and_reports_poor_overlap(
        self,
    ) -> None:
        fixed_data = np.zeros((8, 8, 8), dtype=np.float32)
        fixed_data[1:3, 1:3, 1:3] = 10
        moving_data = np.zeros_like(fixed_data)
        moving_data[4:6, 1:3, 1:3] = 10
        fixed, moving = _volume(fixed_data), _volume(moving_data)
        shift = np.eye(4)
        shift[0, 3] = -3
        corrected = PreprocessingPipeline(
            (Register(fixed, SuppliedAffine(matrix_values(shift))),)
        )
        good = corrected.run(moving, context=_RUN)
        assert good.volume is not None
        np.testing.assert_array_equal(good.volume.data, fixed.data)
        self.assertEqual(
            dict(good.report.metrics)["step.0.registration_foreground_dice"], 1
        )
        wrong = PreprocessingPipeline(
            (Register(fixed, SuppliedAffine(matrix_values(np.eye(4)))),)
        )
        result = wrong.run(moving, context=_RUN)
        self.assertEqual(result.report.status, "warning")
        self.assertIn(
            "registration_overlap", {finding.code for finding in result.report.findings}
        )


class IntensityAndCacheTests(unittest.TestCase):
    def test_percentile_normalization_clips_and_records_fit(self) -> None:
        original = _volume()
        pipeline = PreprocessingPipeline(
            (NormalizeIntensity(method="percentile", percentiles=(25, 75)),)
        )
        result = pipeline.run(original, context=_RUN)
        assert result.volume is not None
        self.assertEqual(result.volume.data.min(), 0)
        self.assertEqual(result.volume.data.max(), 1)
        self.assertEqual(dict(result.steps[0].details)["fitted_voxels"], 64)
        self.assertEqual(dict(result.steps[0].details)["center"], 16.75)

    def test_normalization_preserves_background_and_denoise_smooths_impulse(
        self,
    ) -> None:
        volume = _volume()
        result = PreprocessingPipeline((NormalizeIntensity(),)).run(
            volume, context=_RUN
        )
        assert result.volume is not None
        foreground = result.volume.data[volume.data > 0]
        self.assertAlmostEqual(float(foreground.mean()), 0, places=6)
        self.assertAlmostEqual(float(foreground.std()), 1, places=6)
        self.assertTrue(np.all(result.volume.data[volume.data == 0] == 0))
        impulse = np.zeros((9, 9, 9), dtype=np.float32)
        impulse[4, 4, 4] = 1
        denoised = PreprocessingPipeline((GaussianDenoise(1),)).run(
            _volume(impulse), context=_RUN
        )
        assert denoised.volume is not None
        self.assertLess(float(denoised.volume.data.max()), 1)
        self.assertAlmostEqual(float(denoised.volume.data.sum()), 1, places=5)

    def test_seeded_augmentation_and_cache_key_include_data_recipe_and_seed(
        self,
    ) -> None:
        cache = PreprocessingCache(max_entries=2)
        recipe = (RandomIntensity(noise_std=0.02), RandomFlip(probability=1))
        pipeline = PreprocessingPipeline(recipe, cache=cache)
        original = _volume()
        first = pipeline.run(original, context=_RUN)
        repeat = pipeline.run(original, context=RunContext("second-run", 42))
        replay = PreprocessingPipeline(recipe).run(original, context=_RUN)
        assert (
            first.volume is not None
            and repeat.volume is not None
            and replay.volume is not None
        )
        self.assertTrue(repeat.cache_hit)
        self.assertEqual(repeat.run_id, "second-run")
        self.assertEqual(first.steps, repeat.steps)
        np.testing.assert_array_equal(first.volume.data, repeat.volume.data)
        np.testing.assert_array_equal(first.volume.data, replay.volume.data)
        with self.assertRaises(ValueError):
            repeat.volume.data.setflags(write=True)
        seeded = pipeline.run(original, context=RunContext("new-seed", 43))
        assert seeded.volume is not None
        self.assertFalse(seeded.cache_hit)
        self.assertFalse(np.array_equal(first.volume.data, seeded.volume.data))
        self.assertFalse(
            pipeline.run(_volume(original.data * 2), context=_RUN).cache_hit
        )
        other = PreprocessingPipeline(
            (RandomIntensity(noise_std=0.03), recipe[1]), cache=cache
        )
        self.assertFalse(other.run(original, context=_RUN).cache_hit)
        self.assertFalse(pipeline.run(original, context=_RUN).cache_hit)

    @unittest.skipUnless(
        importlib.util.find_spec("SimpleITK") is not None, "install bias extra"
    )
    def test_optional_n4_fit_on_positive_synthetic_volume(self) -> None:
        data = np.zeros((16, 16, 16), dtype=np.float32)
        data[2:14, 2:14, 2:14] = np.linspace(80, 120, 12, dtype=np.float32)[
            :, None, None
        ]
        original = _volume(data)
        pipeline = PreprocessingPipeline((N4BiasCorrection(iterations=(4,), shrink=2),))
        result = pipeline.run(original, context=_RUN)
        self.assertIsNotNone(result.volume, result.to_json())
        assert result.volume is not None
        np.testing.assert_array_equal(result.volume.affine, original.affine)
        self.assertTrue(np.isfinite(result.volume.data).all())
        self.assertFalse(np.array_equal(result.volume.data, original.data))
        self.assertEqual(dict(result.steps[0].parameters)["backend"], "SimpleITK N4")
        fit_voxels = dict(result.steps[0].details)["fit_voxels"]
        assert isinstance(fit_voxels, int)
        self.assertGreater(fit_voxels, 8)


class QCTests(unittest.TestCase):
    def test_qc_flags_bad_inputs_and_required_modalities(self) -> None:
        valid = _volume()
        invalid = replace(valid, data=np.zeros((3, 3), dtype=np.float32))
        self.assertEqual(assess_volume(invalid).findings[0].code, "invalid_dimensions")
        blank = _volume(np.zeros((3, 3, 3), dtype=np.float32))
        self.assertEqual(assess_volume(blank).findings[0].code, "blank_scan")
        corrupt_data = valid.data.copy()
        corrupt_data[0, 0, 0] = np.nan
        self.assertEqual(
            assess_volume(replace(valid, data=corrupt_data)).status, "fail"
        )
        unusual = _volume(valid.data, np.diag([12.0, 1.0, 1.0, 1.0]))
        report = assess_case(
            {Modality.T1: unusual},
            required=frozenset({Modality.T1, Modality.FLAIR}),
            policy=QCPolicy(max_abs_intensity=25),
        )
        self.assertEqual(report.status, "fail")
        self.assertEqual(
            {f.code for f in report.findings},
            {"missing_modalities", "extreme_spacing", "intensity_anomaly"},
        )

    def test_failure_has_provenance_and_no_partial_output(self) -> None:
        pipeline = PreprocessingPipeline((OrientRAS(), CropPad((500, 500, 500))))
        result = pipeline.run(_volume(), context=_RUN)
        self.assertIsNone(result.volume)
        self.assertEqual(result.report.status, "fail")
        self.assertEqual([step.failed for step in result.steps], [False, True])
        payload = json.loads(result.to_json())
        self.assertEqual(payload["qc"]["notice"], RESEARCH_NOTICE)
        self.assertEqual(payload["qc"]["findings"][-1]["code"], "preprocessing_failure")
        self.assertEqual(payload["steps"][-1]["parameters"]["shape"], [500, 500, 500])

    def test_visual_montage_writes_three_embedded_planes_per_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "qc.svg"
            volume = _volume()
            write_qc_montage(volume, path, comparison=volume)
            svg = ElementTree.parse(path)
            images = svg.findall(".//{http://www.w3.org/2000/svg}image")
            self.assertEqual(len(images), 6)
            for image in images:
                png = base64.b64decode(image.attrib["href"].split(",", 1)[1])
                self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            self.assertIn(RESEARCH_NOTICE, path.read_text(encoding="utf-8"))
