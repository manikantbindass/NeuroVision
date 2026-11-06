"""Explicit architecture registration without dynamic import from configuration."""

from collections.abc import Callable

from neurovision.core.errors import ConfigurationError
from neurovision.models.base import BaseSegmentationModel
from neurovision.models.config import ModelConfig

ModelBuilder = Callable[[ModelConfig], BaseSegmentationModel]


class ModelRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, ModelBuilder] = {}

    def register(self, name: str, builder: ModelBuilder) -> None:
        if name in self._builders:
            raise ConfigurationError("Architecture is already registered")
        ModelConfig(name, 3)
        self._builders[name] = builder

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    def create(self, config: ModelConfig) -> BaseSegmentationModel:
        try:
            builder = self._builders[config.architecture]
        except KeyError:
            raise ConfigurationError("Unknown or unregistered architecture") from None
        try:
            model = builder(config)
        except (ValueError, TypeError, AssertionError):
            raise ConfigurationError(
                "Invalid parameters for selected architecture"
            ) from None
        if not isinstance(model, BaseSegmentationModel):
            raise ConfigurationError(
                "Registered builder must return a segmentation model"
            )
        spec = model.input_spec
        if (spec.spatial_dims, spec.in_channels, spec.num_classes) != (
            config.spatial_dims,
            config.in_channels,
            config.num_classes,
        ):
            raise ConfigurationError(
                "Registered model violates the configured tensor contract"
            )
        return model
