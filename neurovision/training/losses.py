"""Multiclass soft Dice and cross-entropy for integer label maps."""

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class LossConfig:
    dice_weight: float = 1.0
    cross_entropy_weight: float = 1.0
    dice_include_background: bool = False
    smooth: float = 1e-5

    def __post_init__(self) -> None:
        weights = (self.dice_weight, self.cross_entropy_weight)
        if (
            any(not isfinite(weight) or weight < 0 for weight in weights)
            or not any(weights)
            or not isfinite(self.smooth)
            or self.smooth <= 0
        ):
            raise ValueError(
                "Loss weights must be nonnegative with a positive total and smoothing"
            )


DEFAULT_LOSS = LossConfig()


class SegmentationLoss(nn.Module):
    def __init__(self, config: LossConfig = DEFAULT_LOSS) -> None:
        super().__init__()
        self.config = config

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if (
            logits.ndim not in (4, 5)
            or logits.shape[1] < 2
            or targets.shape != (logits.shape[0], *logits.shape[2:])
            or targets.dtype != torch.int64
            or not targets.numel()
        ):
            raise ValueError("Loss expects batched logits and same-grid int64 labels")
        if bool(torch.any((targets < 0) | (targets >= logits.shape[1]))):
            raise ValueError("Loss targets must be valid class IDs")
        # Keep probability reductions in float32 even under autocast.
        scores = logits.float()
        config = self.config
        result = scores.new_zeros(())
        if config.cross_entropy_weight:
            result = config.cross_entropy_weight * F.cross_entropy(scores, targets)
        if config.dice_weight:
            probs = scores.softmax(dim=1)
            encoded = (
                F.one_hot(targets, num_classes=logits.shape[1]).movedim(-1, 1).float()
            )
            if not config.dice_include_background:
                probs, encoded = probs[:, 1:], encoded[:, 1:]
            axes = tuple(range(2, logits.ndim))
            overlap = (probs * encoded).sum(dim=axes)
            total = probs.sum(dim=axes) + encoded.sum(dim=axes)
            dice = (2 * overlap + config.smooth) / (total + config.smooth)
            result = result + config.dice_weight * (1 - dice.mean())
        return result
