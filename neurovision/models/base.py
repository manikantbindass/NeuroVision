"""Common tensor and model-definition contract for segmentation networks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn

from neurovision.models.config import ModelConfig


@dataclass(frozen=True, slots=True)
class InputSpec:
    spatial_dims: int
    in_channels: int
    num_classes: int
    factor: int
    min_size: int
    fixed_shape: tuple[int, ...] | None = None

    def validate(self, shape: tuple[int, ...]) -> None:
        if (
            len(shape) != self.spatial_dims + 2
            or shape[0] < 1
            or shape[1] != self.in_channels
            or any(size < self.min_size or size % self.factor for size in shape[2:])
            or (self.fixed_shape is not None and shape[2:] != self.fixed_shape)
        ):
            raise ValueError(
                "Model input violates its rank, channels, or spatial grid contract"
            )


class BaseSegmentationModel(nn.Module, ABC):
    """Return same-grid multiclass logits with a reproducible model definition."""

    @property
    @abstractmethod
    def input_spec(self) -> InputSpec: ...

    @property
    @abstractmethod
    def model_config(self) -> ModelConfig: ...

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor: ...
