"""Sequential ensemble inference with reversible flips and dropout sampling."""

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite, prod

import torch
from torch import Tensor

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.inference.engine import InferenceEngine
from neurovision.uncertainty.probability import (
    ProbabilityAccumulator,
    UncertaintyError,
    normalized_weights,
)


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    weights: tuple[float, ...] | None = None
    tta_flips: tuple[tuple[int, ...], ...] = ((),)
    mc_samples: int = 0
    seed: int = 42
    confidence_threshold: float = 0.8
    entropy_threshold: float = 0.5
    disagreement_threshold: float = 0.25
    max_probability_elements: int = 16_000_000
    max_passes: int = 256

    def __post_init__(self) -> None:
        if self.weights is not None:
            normalized_weights(self.weights)
        if (
            not 1 <= len(self.tta_flips) <= 8
            or len(set(self.tta_flips)) != len(self.tta_flips)
            or () not in self.tta_flips
            or any(
                tuple(sorted(set(axes))) != axes
                or any(type(axis) is not int or axis not in (0, 1, 2) for axis in axes)
                for axes in self.tta_flips
            )
            or type(self.mc_samples) is not int
            or not (self.mc_samples == 0 or 2 <= self.mc_samples <= 128)
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
            or any(
                not isfinite(value) or not 0 <= value <= 1
                for value in (
                    self.confidence_threshold,
                    self.entropy_threshold,
                    self.disagreement_threshold,
                )
            )
            or type(self.max_probability_elements) is not int
            or self.max_probability_elements < 1
            or type(self.max_passes) is not int
            or self.max_passes < 1
        ):
            raise UncertaintyError(
                "Invalid ensemble sampling, limits, or confidence thresholds"
            )

    @classmethod
    def from_json(cls, text: str) -> "EnsembleConfig":
        try:
            if len(text) > 16384:
                raise ValueError("Ensemble configuration exceeds 16 KiB")
            value = json.loads(text)
            if not isinstance(value, dict):
                raise TypeError("Expected an ensemble object")
            if value.get("weights") is not None:
                value["weights"] = tuple(value["weights"])
            if "tta_flips" in value:
                value["tta_flips"] = tuple(tuple(axes) for axes in value["tta_flips"])
            return cls(**value)
        except (TypeError, ValueError, RecursionError):
            raise UncertaintyError(
                "Cannot read a valid ensemble configuration"
            ) from None


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    identifier: str
    engine: InferenceEngine
    channels: tuple[str, ...]
    labels: tuple[str, ...]
    roi_size: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class EnsemblePrediction:
    probabilities: Tensor
    labels: Tensor
    confidence: Tensor
    predictive_entropy: Tensor
    ensemble_variance: Tensor
    sample_variance: Tensor
    model_disagreement: Tensor
    high_confidence: Tensor
    provenance_json: str


class EnsembleEngine:
    """Combine compatible models on a shared channel, label, and voxel grid."""

    def __init__(
        self, members: tuple[EnsembleMember, ...], config: EnsembleConfig
    ) -> None:
        if not 1 <= len(members) <= 16:
            raise UncertaintyError(
                "An ensemble requires between 1 and 16 declared members"
            )
        first = members[0]
        spec = first.engine.model.input_spec
        ids: set[str] = set()
        for member in members:
            current = member.engine.model.input_spec
            if (
                not member.identifier.strip()
                or len(member.identifier) > 256
                or member.identifier in ids
                or member.channels != first.channels
                or member.labels != first.labels
                or len(member.channels) != current.in_channels
                or len(member.labels) != current.num_classes
                or len(set(member.channels)) != len(member.channels)
                or len(set(member.labels)) != len(member.labels)
                or any(not name.strip() for name in (*member.channels, *member.labels))
                or current.spatial_dims != spec.spatial_dims
            ):
                raise UncertaintyError(
                    "Ensemble members require compatible schemas and unique identities"
                )
            ids.add(member.identifier)
        weights = normalized_weights(config.weights or (1.0,) * len(members))
        if len(weights) != len(members):
            raise UncertaintyError("Ensemble weights must match the declared members")
        if any(axis >= spec.spatial_dims for axes in config.tta_flips for axis in axes):
            raise UncertaintyError(
                "TTA axes must exist in the model spatial dimensions"
            )
        draws = max(1, config.mc_samples)
        passes = sum(weight > 0 for weight in weights) * len(config.tta_flips) * draws
        if passes > config.max_passes:
            raise UncertaintyError(
                "Ensemble exceeds the configured inference pass budget"
            )
        self.members, self.config, self.weights = members, config, weights

    def predict(self, inputs: Tensor) -> EnsemblePrediction:
        config = self.config
        spec = self.members[0].engine.model.input_spec
        if (
            inputs.ndim != spec.spatial_dims + 2
            or inputs.shape[1] != spec.in_channels
            or inputs.shape[0] * prod(inputs.shape[2:]) * spec.num_classes
            > config.max_probability_elements
        ):
            raise UncertaintyError(
                "Ensemble input exceeds the declared grid or probability budget"
            )
        # Validate every active model before starting a partially completed experiment.
        for member, weight in zip(self.members, self.weights, strict=True):
            if weight:
                shape = (
                    (1, spec.in_channels, *member.roi_size)
                    if member.roi_size
                    else tuple(inputs.shape)
                )
                member.engine.model.input_spec.validate(shape)
        models = ProbabilityAccumulator(max_elements=config.max_probability_elements)
        samples = ProbabilityAccumulator(max_elements=config.max_probability_elements)
        records = []
        with torch.no_grad():
            for member, weight in zip(self.members, self.weights, strict=True):
                if not weight:
                    continue
                own = ProbabilityAccumulator(
                    max_elements=config.max_probability_elements
                )
                draws = max(1, config.mc_samples)
                sample_weight = 1 / (len(config.tta_flips) * draws)
                for axes in config.tta_flips:
                    dims = tuple(axis + 2 for axis in axes)
                    augmented = inputs.flip(dims) if dims else inputs
                    seeds = []
                    for draw in range(draws):
                        seed = _sample_seed(config.seed, member.identifier, axes, draw)
                        result = member.engine.predict(
                            augmented,
                            roi_size=member.roi_size,
                            mc_dropout=bool(config.mc_samples),
                            seed=seed if config.mc_samples else None,
                        )
                        probabilities = (
                            result.probabilities.flip(dims)
                            if dims
                            else result.probabilities
                        )
                        own.add(probabilities, weight=sample_weight)
                        samples.add(probabilities, weight=weight * sample_weight)
                        seeds.append(seed if config.mc_samples else None)
                    records.append(
                        {
                            "member": member.identifier,
                            "flipped_spatial_axes": axes,
                            "dropout_seeds": seeds,
                        }
                    )
                models.add(own.summary().probabilities, weight=weight)
        between = models.summary()
        total = samples.summary()
        high = (
            (between.confidence >= config.confidence_threshold)
            & (between.predictive_entropy <= config.entropy_threshold)
            & (between.disagreement <= config.disagreement_threshold)
        )
        provenance = (
            json.dumps(
                {
                    "schema_version": 1,
                    "notice": RESEARCH_NOTICE,
                    "interpretation": "Uncalibrated research confidence thresholds",
                    "configuration": asdict(config),
                    "normalized_weights": self.weights,
                    "members": [
                        {
                            "identifier": member.identifier,
                            "channels": member.channels,
                            "labels": member.labels,
                            "model": asdict(member.engine.model.model_config),
                            "roi_size": member.roi_size,
                            "device": str(member.engine.device),
                            "precision": member.engine.precision,
                        }
                        for member in self.members
                    ],
                    "passes": records,
                    "torch": str(torch.__version__),
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        return EnsemblePrediction(
            between.probabilities,
            between.labels,
            between.confidence,
            between.predictive_entropy,
            between.variance,
            total.variance,
            between.disagreement,
            high,
            provenance,
        )


def _sample_seed(seed: int, identifier: str, axes: tuple[int, ...], draw: int) -> int:
    encoded = json.dumps((seed, identifier, axes, draw)).encode()
    digest = hashlib.blake2b(encoded, digest_size=8).digest()
    return int.from_bytes(digest, "little") % 2**63
