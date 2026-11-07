"""Compare input gradients and bounded patch occlusions for a fixed target."""

import math
from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor

from neurovision.core.errors import PipelineError
from neurovision.inference.engine import InferenceEngine, evaluation_mode


class AttributionError(PipelineError):
    """An attribution target, input, or result failed validation."""


@dataclass(frozen=True, slots=True)
class Attribution:
    class_id: int
    roi_voxels: int
    baseline_logit: float
    saliency: Tensor
    occlusion: Tensor
    patch_size: tuple[int, ...]
    fill: float
    perturbations: int


def explain_segmentation(
    engine: InferenceEngine,
    inputs: Tensor,
    *,
    class_id: int,
    roi: Tensor,
    patch_size: tuple[int, ...],
    fill: float = 0.0,
    max_patches: int = 128,
) -> Attribution:
    """Explain the mean target logit in a fixed, nonempty output ROI.

    Return absolute input gradients per channel and signed logit drops
    from jointly occluding all input channels in nonoverlapping patches.
    """
    spec = engine.model.input_spec
    shape = tuple(inputs.shape[2:])
    if (
        inputs.ndim != spec.spatial_dims + 2
        or inputs.shape[0] != 1
        or type(class_id) is not int
        or not 0 <= class_id < spec.num_classes
        or roi.dtype != torch.bool
        or tuple(roi.shape) != shape
        or not bool(roi.any())
        or len(patch_size) != len(shape)
        or any(type(size) is not int or size < 1 for size in patch_size)
        or type(max_patches) is not int
        or not 1 <= max_patches <= 1024
        or not math.isfinite(fill)
        or abs(fill) > torch.finfo(torch.float32).max
        or engine.precision != "fp32"
    ):
        raise AttributionError("Expected a single FP32 input and a valid fixed ROI")
    count = math.prod(
        (length + size - 1) // size
        for length, size in zip(shape, patch_size, strict=True)
    )
    if count > max_patches:
        raise AttributionError("Patch count exceeds the occlusion budget")

    roi = roi.detach().cpu().clone()
    baseline = engine.predict(inputs)
    score = float(baseline.logits[0, class_id][roi].double().mean())
    with (
        torch.inference_mode(False),
        torch.enable_grad(),
        evaluation_mode(engine.model, gradients=True),
        torch.autocast(engine.device.type, enabled=False),
    ):
        features = inputs.detach().to(engine.device).clone().requires_grad_(True)
        # Clone within inference_mode(False) to support callers using inference mode.
        mask = roi.to(engine.device).clone()
        target = engine.model(features)[0, class_id][mask].double().mean()
        gradient = (
            torch.autograd.grad(target, features, allow_unused=True)[0]
            if target.requires_grad
            else None
        )
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AttributionError("The model did not return finite input gradients")
        saliency = gradient.detach().abs().cpu()[0]

    occlusion = torch.empty(shape, dtype=torch.float64)
    starts = (
        range(0, length, size) for length, size in zip(shape, patch_size, strict=True)
    )
    for start in product(*starts):
        patch = tuple(
            slice(index, index + size)
            for index, size in zip(start, patch_size, strict=True)
        )
        occluded = inputs.detach().clone()
        occluded[(slice(None), slice(None), *patch)] = fill
        logits = engine.predict(occluded).logits
        occlusion[patch] = score - float(logits[0, class_id][roi].double().mean())
    return Attribution(
        class_id, int(roi.sum()), score, saliency, occlusion, patch_size, fill, count
    )
