"""Streaming weighted probability moments and voxel-level uncertainty."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite, log

import torch
from torch import Tensor

from neurovision.core.errors import PipelineError


class UncertaintyError(PipelineError):
    """An uncertainty experiment has incompatible inputs or invalid settings."""


def normalized_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    if (
        not weights
        or any(not isfinite(w) or w < 0 for w in weights)
        or max(weights) <= 0
    ):
        raise UncertaintyError(
            "Weights must be finite, nonnegative and include a positive value"
        )
    scale = max(weights)
    scaled = tuple(weight / scale for weight in weights)
    total = sum(scaled)
    return tuple(weight / total for weight in scaled)


@dataclass(frozen=True, slots=True)
class ProbabilitySummary:
    probabilities: Tensor
    labels: Tensor
    confidence: Tensor
    predictive_entropy: Tensor
    variance: Tensor
    disagreement: Tensor


class ProbabilityAccumulator:
    """Weighted population moments and hard votes. Store no sample history."""

    def __init__(self, *, max_elements: int = 16_000_000) -> None:
        if type(max_elements) is not int or max_elements < 1:
            raise UncertaintyError("Probability limit must be a positive integer")
        self.max_elements = max_elements
        self._mean: Tensor | None = None
        self._m2: Tensor | None = None
        self._votes: Tensor | None = None
        self._weight = 0.0

    def add(self, probabilities: Tensor, *, weight: float = 1) -> None:
        if not isfinite(weight) or weight < 0:
            raise UncertaintyError("Probability weight must be finite and nonnegative")
        if (
            probabilities.ndim not in (4, 5)
            or probabilities.shape[1] < 2
            or not probabilities.numel()
            or probabilities.numel() > self.max_elements
            or probabilities.device.type != "cpu"
            or not probabilities.is_floating_point()
            or not bool(torch.isfinite(probabilities).all())
            or bool(((probabilities < 0) | (probabilities > 1)).any())
            or not torch.allclose(
                probabilities.sum(dim=1),
                torch.ones_like(probabilities[:, 0]),
                atol=1e-5,
                rtol=1e-5,
            )
        ):
            raise UncertaintyError(
                "Expected bounded finite CPU probability maps that sum to one"
            )
        if self._mean is not None and self._mean.shape != probabilities.shape:
            raise UncertaintyError(
                "All probability maps must share the same class and voxel grid"
            )
        if weight == 0:
            return
        total = self._weight + weight
        if not isfinite(total):
            raise UncertaintyError(
                "Probability weights overflowed; normalize them before reduction"
            )
        with torch.no_grad():
            value = probabilities.detach().to(dtype=torch.float64)
            if self._mean is None:
                self._mean = value.clone()
                self._m2 = torch.zeros_like(value)
                self._votes = torch.zeros_like(value)
            else:
                if self._m2 is None:
                    raise RuntimeError("Probability variance accumulator is missing")
                delta = value - self._mean
                self._mean.add_(delta, alpha=weight / total)
                self._m2.add_(delta * (value - self._mean), alpha=weight)
            if self._votes is None:
                raise RuntimeError("Probability vote accumulator is missing")
            labels = value.argmax(dim=1, keepdim=True)
            self._votes.scatter_add_(
                1, labels, torch.full_like(labels, weight, dtype=torch.float64)
            )
            self._weight = total

    def summary(self) -> ProbabilitySummary:
        if (
            self._mean is None
            or self._m2 is None
            or self._votes is None
            or self._weight <= 0
        ):
            raise UncertaintyError(
                "At least one positive-weight probability sample is required"
            )
        with torch.no_grad():
            mean = self._mean
            entropy = -(mean * mean.clamp_min(torch.finfo(mean.dtype).tiny).log()).sum(
                dim=1
            )
            confidence, labels = mean.max(dim=1)
            disagreement = (1 - self._votes.max(dim=1).values / self._weight).clamp(
                0, 1
            )
            return ProbabilitySummary(
                mean.float(),
                labels,
                confidence.float(),
                (entropy / log(mean.shape[1])).clamp(0, 1).float(),
                (self._m2 / self._weight).clamp_min(0).float(),
                disagreement.float(),
            )


def average_probabilities(
    samples: Iterable[Tensor],
    *,
    weights: tuple[float, ...],
    max_elements: int = 16_000_000,
) -> ProbabilitySummary:
    """Soft voting is the argmax of this weighted mean of class probabilities."""
    normalized = normalized_weights(weights)
    accumulator = ProbabilityAccumulator(max_elements=max_elements)
    count = 0
    for count, sample in enumerate(samples, 1):
        if count > len(normalized):
            raise UncertaintyError("Probability sample count and weights differ")
        accumulator.add(sample, weight=normalized[count - 1])
    if count != len(normalized):
        raise UncertaintyError("Probability sample count and weights differ")
    return accumulator.summary()
