import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from neurovision.core.errors import ConfigurationError
from neurovision.inference.engine import InferenceEngine, InferenceError
from neurovision.models.base import BaseSegmentationModel
from neurovision.models.checkpoints import CheckpointManager
from neurovision.models.config import ModelConfig
from neurovision.models.factory import ModelFactory
from neurovision.models.registry import ModelRegistry
from neurovision.models.reporting import parameter_report, profile_forward
from neurovision.models.unet import UNet, UNetConfig
from neurovision.training.checkpoints import TrainingError
from neurovision.training.config import (
    AugmentationConfig,
    ExperimentConfig,
    TrainingConfig,
    load_experiment,
)
from neurovision.training.example import synthetic_dataset
from neurovision.training.losses import SegmentationLoss
from neurovision.training.runner import fit

_ROOT = Path(__file__).resolve().parent.parent


class ZooTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_each_advanced_architecture_has_same_grid_logits_and_finite_gradients(
        self,
    ) -> None:
        for filename, size in (
            ("segresnet", 16),
            ("attention-unet", 16),
            ("unetr", 32),
            ("swin-unetr", 64),
        ):
            with self.subTest(architecture=filename):
                config = load_experiment(_ROOT / "configs" / f"zoo-{filename}.json")
                model = ModelFactory().create(config.model)
                self.assertIsInstance(model, BaseSegmentationModel)
                inputs = torch.randn(1, 1, size, size, size)
                logits = model(inputs)
                self.assertEqual(logits.shape, (1, 2, size, size, size))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                targets = torch.zeros((1, size, size, size), dtype=torch.int64)
                targets[:, : size // 2] = 1
                loss = SegmentationLoss()(logits, targets)
                loss.backward()
                gradients = [
                    param.grad for param in model.parameters() if param.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(
                    all(bool(torch.isfinite(grad).all()) for grad in gradients)
                )
                self.assertTrue(any(bool(grad.abs().sum() > 0) for grad in gradients))
                self.assertEqual(
                    parameter_report(model).parameters,
                    sum(param.numel() for param in model.parameters()),
                )
                with self.assertRaisesRegex(ValueError, "contract"):
                    model(torch.zeros((1, 1, size - 1, size, size)))

    def test_registry_is_explicit_and_rejects_ambiguous_configuration(self) -> None:
        registry = ModelRegistry()
        registry.register("custom", lambda _: ModelFactory().create(UNetConfig(2)))
        self.assertEqual(registry.names(), ("custom",))
        with self.assertRaises(ConfigurationError):
            registry.register("custom", lambda _: ModelFactory().create(UNetConfig(2)))
        factory = ModelFactory()
        for config in (
            ModelConfig("unknown", 3),
            ModelConfig("unet", 2, parameters={"typo": 1}),
            ModelConfig("unetr", 3, parameters={"img_size": [17, 32, 32]}),
            ModelConfig(
                "segresnet", 3, parameters={"blocks_down": [1, 1], "blocks_up": [1, 1]}
            ),
        ):
            with self.subTest(config=config), self.assertRaises(ConfigurationError):
                factory.create(config)
        with self.assertRaises(ConfigurationError):
            ModelConfig("unet", 2, parameters={"features": [float("nan")]})
        native = factory.create(ModelConfig("unet", 2, parameters={"features": [4, 8]}))
        self.assertEqual(native.input_spec.factor, 2)

    def test_advanced_2d_variants_use_the_same_contract(self) -> None:
        definitions = (
            ("segresnet", {"blocks_down": [1, 1], "blocks_up": [1]}, 16),
            ("attention_unet", {"channels": [4, 8], "strides": [2]}, 16),
            (
                "unetr",
                {
                    "img_size": [32, 32],
                    "feature_size": 4,
                    "hidden_size": 48,
                    "mlp_dim": 96,
                    "num_heads": 4,
                },
                32,
            ),
            (
                "swin_unetr",
                {"feature_size": 12, "depths": [1, 1, 1, 1], "window_size": 4},
                64,
            ),
        )
        for name, parameters, size in definitions:
            with self.subTest(architecture=name):
                model = ModelFactory().create(
                    ModelConfig(name, 2, parameters=parameters)
                )
                result = InferenceEngine(model).predict(torch.zeros(1, 1, size, size))
                self.assertEqual(result.labels.shape, (1, size, size))

    def test_checkpoint_round_trip_inference_and_integrity(self) -> None:
        config = ModelConfig(
            "segresnet",
            3,
            parameters={
                "init_filters": 8,
                "blocks_down": [1, 1],
                "blocks_up": [1],
            },
        )
        model = ModelFactory().create(config)
        inputs = torch.randn(1, 1, 8, 8, 8)
        original = InferenceEngine(model).predict(inputs)
        self.assertTrue(model.training)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            digest = CheckpointManager.export(path, model)
            restored = CheckpointManager.load_model(path, sha256=digest)
            result = InferenceEngine(restored).predict(inputs)
            torch.testing.assert_close(result.logits, original.logits, rtol=0, atol=0)
            torch.testing.assert_close(
                result.probabilities.sum(dim=1), torch.ones((1, 8, 8, 8))
            )
            self.assertEqual(result.labels.dtype, torch.int64)
            with self.assertRaisesRegex(TrainingError, "digest"):
                CheckpointManager.load_model(path, sha256="0" * 64)
            with self.assertRaisesRegex(TrainingError, "bounded"):
                CheckpointManager.load_model(path, max_bytes=1)

    def test_inference_handles_windowing_limits_and_preserves_modes(self) -> None:
        model = ModelFactory().create(UNetConfig(2, features=(4, 8)))
        assert isinstance(model, UNet)
        model.train()
        model.head.eval()
        engine = InferenceEngine(model)
        inputs = torch.randn(1, 1, 16, 16)
        full = engine.predict(inputs)
        window = engine.predict(inputs, roi_size=(16, 16))
        torch.testing.assert_close(full.logits, window.logits, rtol=1e-5, atol=1e-6)
        odd_grid = engine.predict(torch.randn(1, 1, 19, 23), roi_size=(16, 16))
        self.assertEqual(odd_grid.labels.shape, (1, 19, 23))
        self.assertTrue(model.training)
        self.assertFalse(model.head.training)
        for invalid in (
            torch.full((1, 1, 16, 16), float("nan")),
            torch.ones(1, 2, 16, 16),
        ):
            with self.assertRaises(InferenceError):
                engine.predict(invalid)
        with self.assertRaises(InferenceError):
            InferenceEngine(model, max_input_elements=1).predict(inputs)
        with self.assertRaises(InferenceError):
            InferenceEngine(model, max_output_elements=1).predict(inputs)
        with self.assertRaises(ValueError):
            engine.predict(inputs, roi_size=(15, 16))

    def test_forward_profile_reports_scoped_counts_and_restores_mode(self) -> None:
        model = ModelFactory().create(UNetConfig(2, features=(4, 8)))
        model.train()
        report = profile_forward(
            model, torch.zeros(1, 1, 16, 16), warmup=0, repeats=1, count_flops=True
        )
        self.assertTrue(model.training)
        self.assertGreater(report.parameters.parameters, 0)
        self.assertEqual(
            report.parameters.parameter_bytes, 4 * report.parameters.parameters
        )
        self.assertGreater(report.median_forward_ms, 0)
        self.assertIn("partial", report.flop_status)
        self.assertGreater(report.counted_flops or 0, 0)
        self.assertTrue(report.counted_operators)
        json.dumps(report.as_dict(), allow_nan=False)

    def test_advanced_model_trains_resumes_and_loads_for_inference(self) -> None:
        cfg = ExperimentConfig(
            ModelConfig(
                "attention_unet", 2, parameters={"channels": [4, 8], "strides": [2]}
            ),
            training=TrainingConfig(epochs=1),
            augmentation=AugmentationConfig(flip_probability=0.5),
        )
        train = synthetic_dataset(cfg.model, samples=2, seed=10)
        valid = synthetic_dataset(cfg.model, samples=2, seed=11)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            result = fit(train, valid, cfg, output=output, data_id="synthetic-zoo-v1")
            replay = fit(
                train,
                valid,
                cfg,
                output=output,
                data_id="synthetic-zoo-v1",
                resume=True,
            )
            self.assertEqual(result.history, replay.history)
            restored = CheckpointManager.load_model(output / "last.pt")
            inputs, _ = valid[0]
            actual = InferenceEngine(restored).predict(inputs[None])
            expected = InferenceEngine(result.model).predict(inputs[None])
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
            state = CheckpointManager.load_training(output / "last.pt")
            self.assertEqual(state["run"]["model_definition"], asdict(cfg.model))

    def test_training_accepts_an_explicit_registry_extension(self) -> None:
        registry = ModelRegistry()
        registry.register(
            "custom_unet",
            lambda config: UNet(
                UNetConfig(
                    config.spatial_dims, config.in_channels, config.num_classes, (4, 8)
                )
            ),
        )
        config = ExperimentConfig(
            ModelConfig("custom_unet", 2), training=TrainingConfig(epochs=1)
        )
        train = synthetic_dataset(config.model, samples=2, seed=21)
        valid = synthetic_dataset(config.model, samples=2, seed=22)
        with tempfile.TemporaryDirectory() as temp:
            result = fit(
                train,
                valid,
                config,
                output=Path(temp),
                data_id="synthetic-extension",
                factory=ModelFactory(registry),
            )
            self.assertEqual(len(result.history), 1)
