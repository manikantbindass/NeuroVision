import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from neurovision.evaluation.segmentation import segmentation_metrics
from neurovision.imaging.volume import Modality, canonical_volume
from neurovision.models.unet import UNet, UNetConfig
from neurovision.training.checkpoints import TrainingError, load_checkpoint
from neurovision.training.config import (
    AugmentationConfig,
    ExperimentConfig,
    TrainingConfig,
    load_experiment,
)
from neurovision.training.example import synthetic_dataset
from neurovision.training.losses import LossConfig, SegmentationLoss
from neurovision.training.mri import case_tensors
from neurovision.training.runner import fit


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_multiclass_logits_match_2d_and_3d_input_grids(self) -> None:
        for dims in (2, 3):
            model = UNet(
                UNetConfig(dims, in_channels=2, num_classes=3, features=(4, 8, 16))
            )
            self.assertEqual(
                model(torch.zeros((1, 2, *((8,) * dims)))).shape, (1, 3, *((8,) * dims))
            )
            with self.assertRaisesRegex(ValueError, "divisibility"):
                model(torch.zeros((1, 2, 9, *((8,) * (dims - 1)))))

    def test_weighted_loss_matches_components_and_has_finite_gradients(self) -> None:
        logits = torch.tensor(
            [[[[3.0, -1.0], [-1.0, 3.0]], [[-1.0, 3.0], [3.0, -1.0]]]],
            requires_grad=True,
        )
        labels = torch.tensor([[[0, 1], [1, 0]]])
        dice = SegmentationLoss(LossConfig(cross_entropy_weight=0))(logits, labels)
        ce = SegmentationLoss(LossConfig(dice_weight=0))(logits, labels)
        combined = SegmentationLoss(
            LossConfig(dice_weight=0.3, cross_entropy_weight=0.7)
        )(logits, labels)
        torch.testing.assert_close(combined, 0.3 * dice + 0.7 * ce)
        self.assertLess(combined.item(), 0.05)
        combined.backward()
        gradient = logits.grad
        assert gradient is not None
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        with self.assertRaisesRegex(ValueError, "class IDs"):
            SegmentationLoss(LossConfig(dice_weight=0))(logits, labels - 100)

    def test_mri_bridge_preserves_tensor_axis_order_and_spacing(self) -> None:
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        data = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        volume = canonical_volume(
            data,
            affine,
            source_format="nifti",
            modality=Modality.T1,
            reference_space="scanner",
        )
        labels = canonical_volume(
            (data > 11).astype(np.float32),
            affine,
            source_format="nifti",
            modality=Modality.UNKNOWN,
            reference_space="scanner",
        )
        case = case_tensors(
            {Modality.T1: volume},
            labels,
            channels=(Modality.T1,),
            model=UNetConfig(2, features=(4, 8)),
        )
        self.assertEqual(case.images.shape, (3, 1, 4, 2))
        self.assertEqual(case.spacing_mm, (3, 2))
        self.assertEqual(case.images[2, 0, 3, 1].item(), data[1, 3, 2])
        self.assertEqual(case.labels.dtype, torch.int64)


class MetricsTests(unittest.TestCase):
    def test_confusion_metrics_have_known_values(self) -> None:
        actual = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.int64)
        predicted = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.int64)
        (score,) = segmentation_metrics(predicted, actual, num_classes=2)
        self.assertEqual(
            (score.dice, score.precision, score.recall, score.specificity),
            (0.5, 0.5, 0.5, 0.75),
        )
        self.assertAlmostEqual(score.iou or 0, 1 / 3)
        self.assertIsNone(score.hausdorff_mm)

    def test_surface_metrics_respect_anisotropic_spacing_and_undefined_cases(
        self,
    ) -> None:
        actual = np.zeros((3, 4, 5), dtype=np.int64)
        predicted = np.zeros_like(actual)
        actual[1, 1, 1] = 1
        predicted[2, 1, 1] = 1
        score, absent = segmentation_metrics(
            predicted, actual, num_classes=3, spacing_mm=(4, 2, 1)
        )
        self.assertEqual((score.hausdorff_mm, score.hd95_mm, score.assd_mm), (4, 4, 4))
        self.assertIsNone(absent.dice)
        self.assertIsNone(absent.hd95_mm)
        self.assertEqual(absent.specificity, 1)
        (missing,) = segmentation_metrics(
            actual * 0, actual, num_classes=2, spacing_mm=(4, 2, 1)
        )
        self.assertEqual(missing.dice, 0)
        self.assertIsNone(missing.precision)
        self.assertIsNone(missing.assd_mm)

    def test_hd95_is_distinct_from_the_maximum_surface_distance(self) -> None:
        actual = np.zeros((40, 40), dtype=np.int64)
        actual[2:12, 2:12] = 1
        predicted = actual.copy()
        predicted[30, 30] = 1
        (score,) = segmentation_metrics(
            predicted, actual, num_classes=2, spacing_mm=(1, 1)
        )
        self.assertGreater(score.hausdorff_mm or 0, 20)
        self.assertEqual(score.hd95_mm, 0)
        self.assertGreater(score.assd_mm or 0, 0)


class TrainingTests(unittest.TestCase):
    def test_resume_reproduces_uninterrupted_training_and_rejects_different_data(
        self,
    ) -> None:
        cfg = ExperimentConfig(
            UNetConfig(2, features=(4, 8)),
            training=TrainingConfig(epochs=2, validation_spacing_mm=(1, 1)),
            augmentation=AugmentationConfig(flip_probability=0.5, noise_std=0.02),
        )
        train = synthetic_dataset(cfg.model, samples=4, seed=1)
        valid = synthetic_dataset(cfg.model, samples=2, seed=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            uninterrupted = fit(
                train, valid, cfg, output=root / "one", data_id="synthetic"
            )
            short = replace(cfg, training=replace(cfg.training, epochs=1))
            fit(train, valid, short, output=root / "resume", data_id="synthetic")
            with self.assertRaisesRegex(TrainingError, "configuration differs"):
                fit(
                    train,
                    valid,
                    cfg,
                    output=root / "resume",
                    data_id="other",
                    resume=True,
                )
            resumed = fit(
                train,
                valid,
                cfg,
                output=root / "resume",
                data_id="synthetic",
                resume=True,
            )
            self.assertEqual(uninterrupted.history, resumed.history)
            for name, tensor in uninterrupted.model.state_dict().items():
                torch.testing.assert_close(
                    tensor, resumed.model.state_dict()[name], rtol=0, atol=0
                )
            state = load_checkpoint(root / "resume" / "best.pt")
            self.assertEqual(state["best_epoch"], resumed.best_epoch)
            self.assertIn("optimizer", state)
            self.assertIn("scheduler", state)

    def test_3d_training_updates_weights_and_schedules_and_early_stops(self) -> None:
        config = ExperimentConfig(
            UNetConfig(3, features=(4, 8)),
            training=TrainingConfig(
                epochs=4,
                early_stopping_patience=1,
                min_delta=100,
                lr_patience=0,
                validation_spacing_mm=(1, 1, 1),
            ),
        )
        train = synthetic_dataset(config.model, samples=2, seed=3)
        valid = synthetic_dataset(config.model, samples=2, seed=4)
        torch.manual_seed(config.training.seed)
        initial = UNet(config.model).state_dict()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = fit(train, valid, config, output=out, data_id="synthetic-3d")
            self.assertTrue(result.stopped_early)
            self.assertEqual(len(result.history), 2)
            self.assertEqual(result.best_epoch, 1)
            self.assertTrue(
                any(
                    not torch.equal(initial[key], value)
                    for key, value in result.model.state_dict().items()
                )
            )
            state = load_checkpoint(out / "last.pt")
            self.assertEqual(state["optimizer"]["param_groups"][0]["lr"], 0.0005)
            report = json.loads((out / "history.json").read_text(encoding="utf-8"))
            self.assertIn("sensitivity", report["epochs"][0]["validation_metrics"]["1"])
            self.assertIn("Research and educational use only", report["notice"])

    def test_cpu_bfloat16_smoke_and_configuration_validation(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(precision="fp16")
        cfg = ExperimentConfig(
            UNetConfig(2, features=(4, 8)),
            training=TrainingConfig(epochs=1, precision="bf16"),
        )
        train = synthetic_dataset(cfg.model, samples=2, seed=5)
        valid = synthetic_dataset(cfg.model, samples=2, seed=6)
        with tempfile.TemporaryDirectory() as temp:
            result = fit(train, valid, cfg, output=Path(temp), data_id="synthetic-bf16")
            self.assertEqual(len(result.history), 1)
            self.assertTrue(np.isfinite(result.history[0].validation_loss))
            self.assertEqual(load_checkpoint(Path(temp) / "last.pt")["scaler"], {})

    def test_both_repo_recipes_parse(self) -> None:
        root = Path(__file__).resolve().parent.parent
        for name, dims in (("baseline-2d", 2), ("baseline-3d", 3)):
            cfg = load_experiment(root / "configs" / f"{name}.json")
            self.assertEqual(cfg.model.spatial_dims, dims)
