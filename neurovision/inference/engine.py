"""Bounded full-grid and optional sliding-window segmentation inference."""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from neurovision.core.errors import PipelineError
from neurovision.models.base import BaseSegmentationModel


class InferenceError(PipelineError):
    """Model inference could not produce a valid segmentation."""


@contextmanager
def evaluation_mode(
    model: BaseSegmentationModel,
    *,
    mc_dropout: bool = False,
) -> Iterator[None]:
    states = [(module, module.training) for module in model.modules()]
    dropout = [
        module
        for module in model.modules()
        if isinstance(
            module,
            (
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d,
                nn.AlphaDropout,
                nn.FeatureAlphaDropout,
            ),
        )
        and 0 < module.p < 1
    ]
    if mc_dropout and not dropout:
        raise InferenceError("MC dropout requires explicit active dropout modules")
    model.eval()
    try:
        if mc_dropout:
            for module in dropout:
                module.train(True)
        with torch.inference_mode():
            yield
    finally:
        for saved_module, saved_training in states:
            saved_module.training = saved_training


@dataclass(frozen=True, slots=True)
class Prediction:
    logits: Tensor
    probabilities: Tensor
    labels: Tensor


class InferenceEngine:
    """Own the model device. Call from one worker at a time."""

    def __init__(
        self,
        model: BaseSegmentationModel,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        precision: Literal["fp32", "fp16", "bf16"] = "fp32",
        max_input_elements: int = 32_000_000,
        max_output_elements: int = 64_000_000,
    ) -> None:
        if (
            device not in ("cpu", "cuda")
            or precision not in ("fp32", "fp16", "bf16")
            or (device == "cpu" and precision == "fp16")
            or max_input_elements < 1
            or max_output_elements < 1
        ):
            raise InferenceError("Invalid device, precision or input bound")
        if device == "cuda" and not torch.cuda.is_available():
            raise InferenceError("CUDA inference is unavailable")
        if (
            device == "cuda"
            and precision == "bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise InferenceError("CUDA bfloat16 inference is unavailable")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.precision = precision
        self.max_input_elements = max_input_elements
        self.max_output_elements = max_output_elements

    def predict(
        self,
        inputs: Tensor,
        *,
        roi_size: tuple[int, ...] | None = None,
        overlap: float = 0.25,
        mc_dropout: bool = False,
        seed: int | None = None,
    ) -> Prediction:
        spec = self.model.input_spec
        if mc_dropout and (type(seed) is not int or not 0 <= seed < 2**63):
            raise InferenceError(
                "MC dropout requires an explicit nonnegative 63-bit seed"
            )
        if not mc_dropout and seed is not None:
            raise InferenceError("A sampling seed is valid only with MC dropout")
        if (
            inputs.ndim != spec.spatial_dims + 2
            or inputs.shape[1] != spec.in_channels
            or not inputs.numel()
            or inputs.numel() > self.max_input_elements
            or inputs.numel() // spec.in_channels * spec.num_classes
            > self.max_output_elements
            or inputs.dtype != torch.float32
            or not bool(torch.isfinite(inputs).all())
            or not 0 <= overlap < 1
        ):
            raise InferenceError(
                "Input violates inference rank, dtype, channels or size constraints"
            )
        if roi_size is None:
            spec.validate(tuple(inputs.shape))
        else:
            spec.validate((1, spec.in_channels, *roi_size))
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        devices = (
            []
            if self.device.type == "cpu"
            else [
                torch.cuda.current_device()
                if self.device.index is None
                else self.device.index
            ]
        )
        with (
            torch.random.fork_rng(devices=devices) if mc_dropout else nullcontext(),
            evaluation_mode(self.model, mc_dropout=mc_dropout),
            torch.autocast(
                self.device.type, dtype=dtype, enabled=self.precision != "fp32"
            ),
        ):
            if seed is not None:
                torch.set_rng_state(
                    torch.Generator(device="cpu").manual_seed(seed).get_state()
                )
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(seed)
            if roi_size is None:
                logits = self.model(inputs.to(self.device))
            else:
                try:
                    from monai.inferers.utils import sliding_window_inference
                except ImportError:
                    raise InferenceError(
                        "Sliding-window inference requires the zoo extra"
                    ) from None
                # Accumulate on CPU so only a window needs to reside on the device.
                logits = sliding_window_inference(
                    inputs.cpu(),
                    roi_size,
                    1,
                    self.model,
                    overlap=overlap,
                    mode="gaussian",
                    sw_device=self.device,
                    device=torch.device("cpu"),
                )
            if (
                not isinstance(logits, Tensor)
                or logits.shape
                != (inputs.shape[0], spec.num_classes, *inputs.shape[2:])
                or not bool(torch.isfinite(logits).all())
            ):
                raise InferenceError("Model returned invalid segmentation logits")
            scores = logits.detach().float().cpu()
            probs = scores.softmax(dim=1)
            return Prediction(scores, probs, scores.argmax(dim=1))
