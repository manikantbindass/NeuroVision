"""Bridge aligned MRI volumes and label maps to the U-Net tensor convention."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from neurovision.imaging.volume import Modality, MRIVolume
from neurovision.models.unet import UNetConfig
from neurovision.preprocessing.contracts import same_grid
from neurovision.training.checkpoints import TrainingError


@dataclass(frozen=True, slots=True)
class TensorCase:
    images: Tensor
    labels: Tensor
    spacing_mm: tuple[float, ...]


def case_tensors(
    volumes: Mapping[Modality, MRIVolume],
    labels: MRIVolume,
    *,
    channels: tuple[Modality, ...],
    model: UNetConfig,
) -> TensorCase:
    """Build 2D axial samples or one 3D sample on a shared orthogonal RAS grid."""
    if (
        len(channels) != model.in_channels
        or len(set(channels)) != len(channels)
        or Modality.UNKNOWN in channels
        or any(channel not in volumes for channel in channels)
    ):
        raise TrainingError("MRI channels must be explicit, unique, and available")
    first = volumes[channels[0]]
    if any(
        not same_grid(first, volumes[channel])
        or volumes[channel].modality not in (Modality.UNKNOWN, channel)
        for channel in channels
    ) or not same_grid(first, labels):
        raise TrainingError("MRI channels and labels must share the same physical grid")
    direction = first.affine[:3, :3] / np.asarray(first.spacing_mm)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5, rtol=0):
        raise TrainingError("Baseline tensors require an orthogonal physical grid")
    if (
        not np.isfinite(labels.data).all()
        or np.any(labels.data != np.floor(labels.data))
        or np.any(labels.data < 0)
        or np.any(labels.data >= model.num_classes)
    ):
        raise TrainingError("Labels must be integer class IDs in the model range")
    # RAS volume (i, j, k) becomes tensor spatial order (k, j, i).
    images = np.stack(
        [volumes[channel].data.transpose(2, 1, 0) for channel in channels]
    )
    target = labels.data.transpose(2, 1, 0).astype(np.int64)
    spacing: tuple[float, ...]
    if model.spatial_dims == 2:
        images = images.swapaxes(0, 1)
        spacing = first.spacing_mm[1::-1]
    else:
        images, target = images[None], target[None]
        spacing = first.spacing_mm[::-1]
    factor = model.downsampling_factor
    if any(size < factor or size % factor for size in images.shape[2:]):
        raise TrainingError(
            "MRI spatial dimensions must be divisible by the U-Net pooling factor"
        )
    return TensorCase(
        torch.from_numpy(np.ascontiguousarray(images)),
        torch.from_numpy(np.ascontiguousarray(target)),
        spacing,
    )
