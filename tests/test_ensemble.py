import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import Tensor, nn

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.inference.engine import InferenceEngine, InferenceError
from neurovision.inference.ensemble import (
    EnsembleConfig,
    EnsembleEngine,
    EnsembleMember,
)
from neurovision.models.base import BaseSegmentationModel, InputSpec
from neurovision.models.config import ModelConfig
from neurovision.models.factory import ModelFactory
from neurovision.uncertainty.example import SyntheticThresholdModel
from neurovision.uncertainty.probability import (
    ProbabilityAccumulator,
    UncertaintyError,
    average_probabilities,
)
from neurovision.uncertainty.visual import write_neuroconfidence_html


class _PhantomModel(BaseSegmentationModel):
    def __init__(
        self, bias: float = 0, *, ramp: bool = False, dropout: bool = False
    ) -> None:
        super().__init__()
        self.bias, self.ramp = bias, ramp
        self.drop = nn.Dropout(0.5) if dropout else nn.Identity()
        self.norm = nn.BatchNorm2d(1)

    @property
    def input_spec(self) -> InputSpec:
        return InputSpec(2, 1, 2, 1, 1)

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig(
            "test_phantom", 2, parameters={"bias": self.bias, "ramp": self.ramp}
        )

    def forward(self, inputs: Tensor) -> Tensor:
        value = self.drop(self.norm(inputs)) + self.bias
        if self.ramp:
            value = value + torch.linspace(-2, 2, inputs.shape[2])[None, None, :, None]
        return torch.cat((-value, value), dim=1)


def _member(
    identifier: str,
    model: BaseSegmentationModel,
    *,
    labels: tuple[str, ...] = ("bg", "fg"),
) -> EnsembleMember:
    return EnsembleMember(identifier, InferenceEngine(model), ("t1",), labels)


class ProbabilityTests(unittest.TestCase):
    def test_weighted_average_variance_entropy_and_hard_vote_disagreement(self) -> None:
        first = torch.tensor([0.9, 0.1]).reshape(1, 2, 1, 1).expand(1, 2, 2, 3)
        second = torch.tensor([0.1, 0.9]).reshape(1, 2, 1, 1).expand_as(first)
        result = average_probabilities((first, second), weights=(3, 1))
        torch.testing.assert_close(
            result.probabilities[:, 0], torch.full((1, 2, 3), 0.7)
        )
        torch.testing.assert_close(result.confidence, torch.full((1, 2, 3), 0.7))
        torch.testing.assert_close(result.variance, torch.full((1, 2, 2, 3), 0.12))
        torch.testing.assert_close(result.disagreement, torch.full((1, 2, 3), 0.25))
        expected = -(
            0.7 * torch.log2(torch.tensor(0.7)) + 0.3 * torch.log2(torch.tensor(0.3))
        )
        torch.testing.assert_close(
            result.predictive_entropy, torch.full((1, 2, 3), expected.item())
        )
        self.assertTrue(bool((result.labels == 0).all()))
        balanced = average_probabilities((first, second), weights=(1, 1))
        torch.testing.assert_close(balanced.predictive_entropy, torch.ones(1, 2, 3))
        certain = average_probabilities(
            (torch.tensor([1.0, 0.0]).reshape(1, 2, 1, 1),), weights=(1,)
        )
        self.assertEqual(certain.predictive_entropy.item(), 0)
        self.assertEqual(certain.variance.max().item(), 0)

    def test_invalid_probabilities_weights_and_shapes_fail(self) -> None:
        valid = torch.full((1, 2, 2, 2), 0.5)
        for weights in ((0, 0), (-1, 1), (float("nan"), 1), (float("inf"), 1)):
            with self.subTest(weights=weights), self.assertRaises(UncertaintyError):
                average_probabilities((valid, valid), weights=weights)
        for invalid in (valid * 2, valid * float("nan"), torch.ones(1, 1, 2, 2)):
            with self.assertRaises(UncertaintyError):
                average_probabilities((invalid,), weights=(1,))
        with self.assertRaises(UncertaintyError):
            average_probabilities((valid,), weights=(1, 1))
        with self.assertRaises(UncertaintyError):
            average_probabilities((valid, valid[:, :, :1]), weights=(1, 1))
        with self.assertRaises(UncertaintyError):
            average_probabilities((valid,), weights=(1,), max_elements=2)

    def test_streamed_moments_match_direct_weighted_population_moments(self) -> None:
        rng = torch.Generator().manual_seed(123)
        values = torch.randn((3, 1, 3, 3, 4), generator=rng).softmax(dim=2)
        result = average_probabilities(values, weights=(1e308, 1e308, 0))
        torch.testing.assert_close(result.probabilities, values[:2].mean(dim=0))
        torch.testing.assert_close(result.variance, values[:2].var(dim=0, correction=0))
        acc = ProbabilityAccumulator()
        acc.add(values[0])
        first = acc.summary()
        first.probabilities.zero_()
        torch.testing.assert_close(acc.summary().probabilities, values[0])


class EnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_engine_weights_and_model_disagreement_match_direct_probabilities(
        self,
    ) -> None:
        inputs = torch.zeros(1, 1, 3, 4)
        members = (_member("a", _PhantomModel(1)), _member("b", _PhantomModel(-1)))
        config = EnsembleConfig(
            weights=(3, 1), confidence_threshold=0.6, entropy_threshold=1
        )
        result = EnsembleEngine(members, config).predict(inputs)
        expected = average_probabilities(
            (member.engine.predict(inputs).probabilities for member in members),
            weights=(3, 1),
        )
        torch.testing.assert_close(result.probabilities, expected.probabilities)
        torch.testing.assert_close(result.ensemble_variance, expected.variance)
        torch.testing.assert_close(result.sample_variance, expected.variance)
        torch.testing.assert_close(
            result.model_disagreement, torch.full((1, 3, 4), 0.25)
        )
        self.assertTrue(bool(result.high_confidence.all()))
        metadata = json.loads(result.provenance_json)
        self.assertEqual(metadata["normalized_weights"], [0.75, 0.25])
        self.assertEqual(metadata["notice"], RESEARCH_NOTICE)

    def test_tta_is_reversed_before_probability_averaging(self) -> None:
        member = _member("ramp", _PhantomModel(ramp=True))
        inputs = torch.full((1, 1, 3, 5), 0.3)
        raw = member.engine.predict(inputs).probabilities
        flipped = member.engine.predict(inputs.flip((2,))).probabilities.flip((2,))
        result = EnsembleEngine(
            (member,), EnsembleConfig(tta_flips=((), (0,)))
        ).predict(inputs)
        torch.testing.assert_close(result.probabilities, (raw + flipped) / 2)
        self.assertEqual(result.ensemble_variance.max().item(), 0)
        self.assertGreater(result.sample_variance.max().item(), 0)

    def test_dropout_sampling_is_seeded_and_keeps_batchnorm_and_rng_state(self) -> None:
        model = _PhantomModel(dropout=True)
        model.train()
        model.drop.eval()
        tracked_mean = model.norm.running_mean
        assert tracked_mean is not None
        running_mean = tracked_mean.clone()
        inputs = torch.ones((1, 1, 8, 8))
        engine = EnsembleEngine(
            (_member("dropout", model),), EnsembleConfig(mc_samples=5, seed=7)
        )
        state = torch.get_rng_state().clone()
        first = engine.predict(inputs)
        second = engine.predict(inputs)
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        self.assertTrue(model.training)
        self.assertTrue(model.norm.training)
        self.assertFalse(model.drop.training)
        torch.testing.assert_close(
            model.norm.running_mean, running_mean, rtol=0, atol=0
        )
        torch.testing.assert_close(
            first.probabilities, second.probabilities, rtol=0, atol=0
        )
        self.assertGreater(first.sample_variance.max().item(), 0)
        self.assertEqual(first.ensemble_variance.max().item(), 0)
        self.assertEqual(first.model_disagreement.max().item(), 0)
        other = EnsembleEngine(
            engine.members, EnsembleConfig(mc_samples=5, seed=8)
        ).predict(inputs)
        self.assertFalse(torch.equal(first.probabilities, other.probabilities))

    def test_unsupported_dropout_and_incompatible_schemas_are_rejected(self) -> None:
        member = _member("a", _PhantomModel())
        with self.assertRaises(InferenceError):
            EnsembleEngine((member,), EnsembleConfig(mc_samples=2)).predict(
                torch.ones(1, 1, 3, 3)
            )
        with self.assertRaises(UncertaintyError):
            EnsembleEngine(
                (member, _member("b", _PhantomModel(), labels=("fg", "bg"))),
                EnsembleConfig(),
            )
        with self.assertRaises(UncertaintyError):
            EnsembleEngine((member,), EnsembleConfig(tta_flips=((), (2,))))
        with self.assertRaises(UncertaintyError):
            EnsembleEngine((member,), EnsembleConfig(mc_samples=8, max_passes=2))
        with self.assertRaises(UncertaintyError):
            EnsembleEngine((member,), EnsembleConfig(weights=(1, 2)))
        # The zero-weight member has no dropout, and must never be evaluated.
        result = EnsembleEngine(
            (member, _member("b", _PhantomModel(dropout=True))),
            EnsembleConfig(weights=(0, 1), mc_samples=2),
        ).predict(torch.ones(1, 1, 3, 3))
        self.assertTrue(bool(torch.isfinite(result.probabilities).all()))

    def test_mc_dropout_runs_on_the_segresnet_adapter(self) -> None:
        model = ModelFactory().create(
            ModelConfig(
                "segresnet",
                2,
                parameters={
                    "init_filters": 8,
                    "blocks_down": [1, 1],
                    "blocks_up": [1],
                    "dropout_prob": 0.25,
                },
            )
        )
        engine = EnsembleEngine(
            (_member("segresnet", model),), EnsembleConfig(mc_samples=3)
        )
        inputs = torch.randn((1, 1, 16, 16), generator=torch.Generator().manual_seed(0))
        result = engine.predict(inputs)
        self.assertGreater(result.sample_variance.max().item(), 0)
        torch.testing.assert_close(
            result.probabilities, engine.predict(inputs).probabilities, rtol=0, atol=0
        )

    def test_failed_stochastic_forward_restores_modes_and_random_state(self) -> None:
        class FailedModel(_PhantomModel):
            def forward(self, inputs: Tensor) -> Tensor:
                self.drop(self.norm(inputs))
                raise RuntimeError("synthetic forward failure")

        model = FailedModel(dropout=True)
        model.train()
        model.drop.eval()
        state = torch.get_rng_state().clone()
        modes = [module.training for module in model.modules()]
        with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
            InferenceEngine(model).predict(
                torch.ones(1, 1, 3, 3), mc_dropout=True, seed=7
            )
        self.assertEqual([module.training for module in model.modules()], modes)
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)

    def test_three_dimensional_flip_samples_return_to_the_original_grid(self) -> None:
        member = _member("synthetic-3d", SyntheticThresholdModel(0.4))
        inputs = torch.linspace(0, 1, 64).reshape(1, 1, 4, 4, 4)
        config = EnsembleConfig(tta_flips=((), (0,), (1,), (2,), (0, 1, 2)))
        result = EnsembleEngine((member,), config).predict(inputs)
        expected = member.engine.predict(inputs)
        torch.testing.assert_close(result.probabilities, expected.probabilities)
        self.assertEqual(result.sample_variance.max().item(), 0)

    def test_configuration_and_neuroconfidence_export_escape_untrusted_names(
        self,
    ) -> None:
        root = Path(__file__).resolve().parent.parent
        config = EnsembleConfig.from_json(
            (root / "configs/ensemble-research.json").read_text()
        )
        self.assertEqual(config.tta_flips, ((), (1,)))
        malicious = "</script><script>alert(1)</script>"
        member = _member("a", _PhantomModel(1), labels=("bg", malicious))
        prediction = EnsembleEngine((member,), EnsembleConfig()).predict(
            torch.ones(1, 1, 4, 5)
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "confidence.html"
            write_neuroconfidence_html(torch.ones(1, 1, 4, 5), prediction, path)
            content = path.read_text(encoding="utf-8")
            self.assertIn("High-confidence prediction", content)
            self.assertIn("Low-confidence / uncertain prediction", content)
            self.assertIn(RESEARCH_NOTICE, content)
            self.assertNotIn(malicious, content)
            self.assertIn("\\u003c/script", content)
