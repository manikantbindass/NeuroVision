"""Factory for native U-Nets and optional MONAI architectures."""

from collections.abc import Callable
from dataclasses import asdict
from math import prod
from typing import Any, cast

from torch import Tensor, nn

from neurovision.core.errors import ConfigurationError
from neurovision.models.base import BaseSegmentationModel, InputSpec
from neurovision.models.config import ModelConfig
from neurovision.models.registry import ModelRegistry
from neurovision.models.unet import UNet, UNetConfig


class MonaiSegmentationModel(BaseSegmentationModel):
    def __init__(
        self, network: nn.Module, config: ModelConfig, spec: InputSpec
    ) -> None:
        super().__init__()
        self.network = network
        self._config = config
        self._spec = spec

    @property
    def input_spec(self) -> InputSpec:
        return self._spec

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig(**asdict(self._config))

    def forward(self, inputs: Tensor) -> Tensor:
        self.input_spec.validate(tuple(inputs.shape))
        return cast(Tensor, self.network(inputs))


def _native(config: ModelConfig) -> BaseSegmentationModel:
    if set(config.parameters) - {"features"}:
        raise ConfigurationError("Unsupported native U-Net parameter")
    return UNet(
        UNetConfig(
            config.spatial_dims,
            config.in_channels,
            config.num_classes,
            tuple(config.parameters.get("features", (16, 32, 64))),
        )
    )


def _positive_sequence(value: Any, *, length: int | None = None) -> tuple[int, ...]:
    if (
        not isinstance(value, (tuple, list))
        or not value
        or len(value) > 6
        or (length is not None and len(value) != length)
        or any(type(item) is not int or item < 1 for item in value)
    ):
        raise ConfigurationError("Expected a nonempty positive integer sequence")
    return tuple(value)


def _monai(config: ModelConfig) -> BaseSegmentationModel:
    try:
        from monai.networks.nets.attentionunet import AttentionUnet
        from monai.networks.nets.segresnet import SegResNet
        from monai.networks.nets.swin_unetr import SwinUNETR
        from monai.networks.nets.unetr import UNETR
    except ImportError:
        raise ConfigurationError(
            "Advanced architectures require the optional zoo extra"
        ) from None
    options = dict(config.parameters)
    name = config.architecture
    dims = config.spatial_dims
    common: dict[str, Any] = {
        "spatial_dims": dims,
        "in_channels": config.in_channels,
        "out_channels": config.num_classes,
    }
    fixed: tuple[int, ...] | None = None
    constructor: Callable[..., nn.Module]
    if name == "segresnet":
        allowed = {"init_filters", "blocks_down", "blocks_up", "dropout_prob"}
        down = _positive_sequence(options.get("blocks_down", [1, 2, 2, 4]))
        up = _positive_sequence(options.get("blocks_up", [1, 1, 1]))
        if len(down) < 2 or len(up) != len(down) - 1:
            raise ConfigurationError(
                "SegResNet requires matching encoder and decoder stages"
            )
        options.update(blocks_down=down, blocks_up=up)
        factor, minimum = 1 << len(up), 1 << len(up)
        constructor = SegResNet
    elif name == "attention_unet":
        allowed = {"channels", "strides", "dropout"}
        channels = _positive_sequence(options.get("channels", [16, 32, 64]))
        strides = _positive_sequence(options.get("strides", [2] * (len(channels) - 1)))
        if len(channels) < 2 or len(strides) != len(channels) - 1:
            raise ConfigurationError(
                "Attention U-Net requires matching channels and strides"
            )
        options.update(channels=channels, strides=strides)
        factor, minimum = prod(strides), 2 * prod(strides)
        constructor = AttentionUnet
    elif name == "unetr":
        allowed = {
            "img_size",
            "feature_size",
            "hidden_size",
            "mlp_dim",
            "num_heads",
            "dropout_rate",
            "qkv_bias",
        }
        fixed = _positive_sequence(options.get("img_size", [96] * dims), length=dims)
        if any(size < 32 or size % 16 for size in fixed):
            raise ConfigurationError(
                "UNETR grid axes must be multiples of 16 and at least 32"
            )
        options["img_size"] = fixed
        factor, minimum = 16, 32
        constructor = UNETR
    elif name == "swin_unetr":
        allowed = {
            "feature_size",
            "depths",
            "num_heads",
            "window_size",
            "drop_rate",
            "attn_drop_rate",
            "dropout_path_rate",
            "use_checkpoint",
        }
        if "depths" in options:
            options["depths"] = _positive_sequence(options["depths"], length=4)
        if "num_heads" in options:
            options["num_heads"] = _positive_sequence(options["num_heads"], length=4)
        factor, minimum = 32, 64
        constructor = SwinUNETR
    else:
        raise ConfigurationError("Unsupported MONAI architecture")
    if set(options) - allowed:
        raise ConfigurationError("Unsupported parameter for selected architecture")
    try:
        network = constructor(**common, **options)
    except (ValueError, TypeError, AssertionError, KeyError):
        raise ConfigurationError(
            "Invalid architecture-specific model parameters"
        ) from None
    spec = InputSpec(
        dims, config.in_channels, config.num_classes, factor, minimum, fixed
    )
    return MonaiSegmentationModel(network, config, spec)


def default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("unet", _native)
    for name in ("segresnet", "attention_unet", "unetr", "swin_unetr"):
        registry.register(name, _monai)
    return registry


class ModelFactory:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = default_registry() if registry is None else registry

    def create(self, config: ModelConfig | UNetConfig) -> BaseSegmentationModel:
        if isinstance(config, UNetConfig):
            return UNet(config)
        return self.registry.create(ModelConfig(**asdict(config)))
