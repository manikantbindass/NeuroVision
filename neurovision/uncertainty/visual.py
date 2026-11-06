"""Self-contained, interactive NeuroConfidence Map for controlled local research use."""

import base64
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.inference.ensemble import EnsemblePrediction
from neurovision.uncertainty.probability import UncertaintyError


def write_neuroconfidence_html(
    inputs: Tensor,
    prediction: EnsemblePrediction,
    path: Path,
    *,
    batch_index: int = 0,
    channel_index: int = 0,
    max_side: int = 128,
    max_slices: int = 64,
) -> None:
    """Embed bounded previews. The resulting HTML contains research image pixels."""
    if (
        inputs.ndim not in (4, 5)
        or inputs.dtype != torch.float32
        or not 0 <= batch_index < inputs.shape[0]
        or not 0 <= channel_index < inputs.shape[1]
        or not 8 <= max_side <= 256
        or not 1 <= max_slices <= 128
        or prediction.labels.shape != (inputs.shape[0], *inputs.shape[2:])
        or prediction.probabilities.ndim != inputs.ndim
        or not 2 <= prediction.probabilities.shape[1] <= 256
    ):
        raise UncertaintyError(
            "NeuroConfidence preview requires matching MRI and prediction grids"
        )
    for value in (
        inputs,
        prediction.confidence,
        prediction.predictive_entropy,
        prediction.model_disagreement,
    ):
        if not bool(torch.isfinite(value).all()):
            raise UncertaintyError(
                "NeuroConfidence preview requires finite image and map values"
            )
    maps = (
        prediction.confidence,
        prediction.predictive_entropy,
        prediction.model_disagreement,
    )
    if any(value.shape != prediction.labels.shape for value in maps):
        raise UncertaintyError("NeuroConfidence maps must match the segmentation grid")
    if any(bool(((value < 0) | (value > 1)).any()) for value in maps):
        raise UncertaintyError("NeuroConfidence maps must lie between zero and one")
    classes = prediction.probabilities.shape[1]
    if prediction.labels.dtype != torch.int64 or bool(
        ((prediction.labels < 0) | (prediction.labels >= classes)).any()
    ):
        raise UncertaintyError("NeuroConfidence labels must be valid integer class IDs")
    metadata = json.loads(prediction.provenance_json)
    config = metadata["configuration"]
    labels = metadata["members"][0]["labels"]
    if len(labels) != classes or not all(isinstance(label, str) for label in labels):
        raise UncertaintyError(
            "NeuroConfidence provenance must declare the label schema"
        )

    image = inputs[batch_index, channel_index].detach().cpu().numpy()
    if image.ndim == 2:
        image = image[None]
    depth, height, width = image.shape
    indices = (
        np.linspace(0, depth - 1, min(depth, max_slices), dtype=np.intp),
        np.linspace(0, height - 1, min(height, max_side), dtype=np.intp),
        np.linspace(0, width - 1, min(width, max_side), dtype=np.intp),
    )
    selection = np.ix_(*indices)
    image = image[selection].astype(np.float64)
    low, high = (float(value) for value in np.percentile(image, [1, 99]))
    gray = np.rint(
        np.clip((image - low) / (high - low if high > low else 1), 0, 1) * 255
    )

    def encoded(value: Tensor, *, floating: bool) -> str:
        array = value[batch_index].detach().cpu().numpy()
        if array.ndim == 2:
            array = array[None]
        dtype = "<f4" if floating else "u1"
        return base64.b64encode(array[selection].astype(dtype).tobytes()).decode(
            "ascii"
        )

    content = {
        "notice": RESEARCH_NOTICE,
        "shape": image.shape,
        "sliceIndices": indices[0].tolist(),
        "image": base64.b64encode(gray.astype(np.uint8).tobytes()).decode("ascii"),
        "labels": encoded(prediction.labels, floating=False),
        "confidence": encoded(prediction.confidence, floating=True),
        "entropy": encoded(prediction.predictive_entropy, floating=True),
        "disagreement": encoded(prediction.model_disagreement, floating=True),
        "classNames": labels,
        "thresholds": {
            "confidence": config["confidence_threshold"],
            "entropy": config["entropy_threshold"],
            "disagreement": config["disagreement_threshold"],
        },
        "experiment": {
            "seed": config["seed"],
            "mc_samples": config["mc_samples"],
            "tta_count": len(config["tta_flips"]),
            "weights": metadata["normalized_weights"],
        },
    }
    payload = (
        json.dumps(content, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    path.write_text(
        files("neurovision.uncertainty")
        .joinpath("neuroconfidence.html")
        .read_text(encoding="utf-8")
        .replace("__PAYLOAD__", payload),
        encoding="utf-8",
    )
