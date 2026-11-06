"""Generate a NeuroConfidence preview from explicit synthetic probability phantoms."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.inference.engine import InferenceEngine
from neurovision.inference.ensemble import (
    EnsembleConfig,
    EnsembleEngine,
    EnsembleMember,
)
from neurovision.models.base import BaseSegmentationModel, InputSpec
from neurovision.models.config import ModelConfig
from neurovision.uncertainty.visual import write_neuroconfidence_html


class SyntheticThresholdModel(BaseSegmentationModel):
    """An analytic probability phantom for software demonstrations."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    @property
    def input_spec(self) -> InputSpec:
        return InputSpec(3, 1, 2, 1, 1)

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig(
            "synthetic_threshold", 3, parameters={"threshold": self.threshold}
        )

    def forward(self, inputs: Tensor) -> Tensor:
        foreground = (inputs[:, :1] - self.threshold) * 10
        return torch.cat((-foreground, foreground), dim=1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a synthetic NeuroConfidence Map. No trained model is used.",
        epilog=RESEARCH_NOTICE,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    axes = torch.meshgrid(*(torch.linspace(-1, 1, 48) for _ in range(3)), indexing="ij")
    signal = torch.exp(-4 * (axes[0] ** 2 + 1.3 * axes[1] ** 2 + axes[2] ** 2))
    image = signal[None, None].float()
    members = tuple(
        EnsembleMember(
            f"synthetic-threshold-{index}",
            InferenceEngine(SyntheticThresholdModel(value)),
            ("synthetic_signal",),
            ("background", "synthetic_foreground"),
        )
        for index, value in enumerate((0.35, 0.55))
    )
    prediction = EnsembleEngine(members, EnsembleConfig()).predict(image)
    args.output.mkdir(parents=True, exist_ok=True)
    write_neuroconfidence_html(image, prediction, args.output / "neuroconfidence.html")
    (args.output / "provenance.json").write_text(
        prediction.provenance_json, encoding="utf-8"
    )
    print(RESEARCH_NOTICE)
    print("Wrote a synthetic NeuroConfidence preview and its experiment provenance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
