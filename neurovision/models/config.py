"""Portable architecture configuration for the model zoo."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from neurovision.core.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture: str
    spatial_dims: Literal[2, 3]
    in_channels: int = 1
    num_classes: int = 2
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", self.architecture)
            or type(self.spatial_dims) is not int
            or self.spatial_dims not in (2, 3)
            or type(self.in_channels) is not int
            or self.in_channels < 1
            or type(self.num_classes) is not int
            or self.num_classes < 2
            or not isinstance(self.parameters, dict)
        ):
            raise ConfigurationError(
                "Invalid model architecture or channel configuration"
            )
        try:
            serialized = json.dumps(self.parameters, allow_nan=False)
            if len(serialized) > 16384:
                raise ValueError("Model parameters exceed 16 KiB")
            # Detach mutable caller data and normalize JSON sequences.
            object.__setattr__(self, "parameters", json.loads(serialized))
        except (ValueError, TypeError, RecursionError):
            raise ConfigurationError(
                "Model parameters must be finite JSON values"
            ) from None
