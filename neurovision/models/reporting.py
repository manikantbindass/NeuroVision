"""Exact parameter storage counts and explicitly scoped forward profiling."""

import platform
import statistics
import time
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.inference.engine import evaluation_mode
from neurovision.models.base import BaseSegmentationModel


@dataclass(frozen=True, slots=True)
class ParameterReport:
    parameters: int
    trainable_parameters: int
    parameter_bytes: int
    buffer_bytes: int


def parameter_report(model: BaseSegmentationModel) -> ParameterReport:
    return ParameterReport(
        sum(param.numel() for param in model.parameters()),
        sum(param.numel() for param in model.parameters() if param.requires_grad),
        sum(param.numel() * param.element_size() for param in model.parameters()),
        sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()),
    )


@dataclass(frozen=True, slots=True)
class ForwardProfile:
    parameters: ParameterReport
    shape: tuple[int, ...]
    dtype: str
    device: str
    torch_version: str
    platform: str
    threads: int
    warmup: int
    repeats: int
    median_forward_ms: float
    counted_flops: int | None
    counted_operators: tuple[str, ...]
    flop_status: str

    def as_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "notice": RESEARCH_NOTICE,
            "timing_scope": "eval forward only with resident input",
            "memory_scope": "parameter and buffer storage only; activations excluded",
        }


def profile_forward(
    model: BaseSegmentationModel,
    example: Tensor,
    *,
    warmup: int = 2,
    repeats: int = 5,
    count_flops: bool = False,
) -> ForwardProfile:
    """Measure the supplied model/device only. FLOP counts cover supported operators."""
    if warmup < 0 or not 1 <= repeats <= 100:
        raise ValueError("Invalid profiling repetitions")
    model.input_spec.validate(tuple(example.shape))
    if not example.is_floating_point() or not bool(torch.isfinite(example).all()):
        raise ValueError("Profiling requires finite floating-point input")
    if any(parameter.device != example.device for parameter in model.parameters()):
        raise ValueError("Profiling model and input must share a device")

    def sync() -> None:
        if example.device.type == "cuda":
            torch.cuda.synchronize(example.device)

    durations = []
    flops = None
    operators: tuple[str, ...] = ()
    status = "not requested"
    with evaluation_mode(model):
        for _ in range(warmup):
            model(example)
        sync()
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            model(example)
            sync()
            durations.append((time.perf_counter() - start) * 1000)
        if count_flops:
            try:
                from torch.utils.flop_counter import FlopCounterMode

                with FlopCounterMode(display=False) as counter:
                    model(example)
                flops = int(counter.get_total_flops())
                operators = tuple(
                    sorted(str(op) for op in counter.get_flop_counts()["Global"])
                )
                status = "partial; PyTorch-supported operator formulas only"
            except (ImportError, RuntimeError, NotImplementedError, KeyError):
                status = "unavailable for this graph and PyTorch backend"
    return ForwardProfile(
        parameter_report(model),
        tuple(example.shape),
        str(example.dtype),
        str(example.device),
        str(torch.__version__),
        platform.platform(),
        torch.get_num_threads(),
        warmup,
        repeats,
        statistics.median(durations),
        flops,
        operators,
        status,
    )
