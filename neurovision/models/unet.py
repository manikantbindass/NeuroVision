"""Small, auditable 2D and 3D U-Net baselines."""

from dataclasses import dataclass
from typing import Literal, cast

from torch import Tensor, cat, nn

from neurovision.models.base import BaseSegmentationModel, InputSpec
from neurovision.models.config import ModelConfig

SpatialDims = Literal[2, 3]


@dataclass(frozen=True, slots=True)
class UNetConfig:
    spatial_dims: SpatialDims
    in_channels: int = 1
    num_classes: int = 2
    features: tuple[int, ...] = (16, 32, 64)

    def __post_init__(self) -> None:
        if (
            self.spatial_dims not in (2, 3)
            or type(self.in_channels) is not int
            or self.in_channels < 1
            or type(self.num_classes) is not int
            or self.num_classes < 2
            or not 2 <= len(self.features) <= 6
            or any(type(width) is not int or width < 1 for width in self.features)
        ):
            raise ValueError(
                "Invalid U-Net dimensions, channel counts, or feature widths"
            )

    @property
    def downsampling_factor(self) -> int:
        return 1 << (len(self.features) - 1)


class UNet(BaseSegmentationModel):
    """Padded double convolutions, pooling, transposed convolutions, and skip links."""

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config
        conv = nn.Conv2d if config.spatial_dims == 2 else nn.Conv3d
        transpose = (
            nn.ConvTranspose2d if config.spatial_dims == 2 else nn.ConvTranspose3d
        )
        self.pool = nn.MaxPool2d(2) if config.spatial_dims == 2 else nn.MaxPool3d(2)

        def block(source: int, width: int) -> nn.Sequential:
            return nn.Sequential(
                conv(source, width, kernel_size=3, padding=1),
                nn.ReLU(),
                conv(width, width, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        widths = config.features
        self.encoders = nn.ModuleList(
            block(source, width)
            for source, width in zip(
                (config.in_channels, *widths[:-1]), widths, strict=True
            )
        )
        self.upsamplers = nn.ModuleList(
            transpose(source, width, kernel_size=2, stride=2)
            for source, width in zip(
                reversed(widths[1:]), reversed(widths[:-1]), strict=True
            )
        )
        self.decoders = nn.ModuleList(
            block(2 * width, width) for width in reversed(widths[:-1])
        )
        self.head = conv(widths[0], config.num_classes, kernel_size=1)

    @property
    def input_spec(self) -> InputSpec:
        config = self.config
        return InputSpec(
            config.spatial_dims,
            config.in_channels,
            config.num_classes,
            config.downsampling_factor,
            config.downsampling_factor,
        )

    @property
    def model_config(self) -> ModelConfig:
        config = self.config
        return ModelConfig(
            "unet",
            config.spatial_dims,
            config.in_channels,
            config.num_classes,
            {"features": list(config.features)},
        )

    def forward(self, inputs: Tensor) -> Tensor:
        factor = self.config.downsampling_factor
        if (
            inputs.ndim != self.config.spatial_dims + 2
            or inputs.shape[1] != self.config.in_channels
            or any(size < factor or size % factor for size in inputs.shape[2:])
        ):
            raise ValueError(
                "U-Net input channels, rank, or spatial divisibility is invalid"
            )
        x = inputs
        skips = []
        for index, encoder in enumerate(self.encoders):
            if index:
                x = self.pool(x)
            x = encoder(x)
            skips.append(x)
        for up, decoder, skip in zip(
            self.upsamplers, self.decoders, reversed(skips[:-1]), strict=True
        ):
            x = decoder(cat((up(x), skip), dim=1))
        return cast(Tensor, self.head(x))
