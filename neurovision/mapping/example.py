"""Generate a local NeuroMap comparison from synthetic volumes and logits."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.explainability.attribution import explain_segmentation
from neurovision.imaging.volume import Modality, canonical_volume
from neurovision.inference.engine import InferenceEngine
from neurovision.mapping.atlas import AtlasRegion, LocalLabelAtlas
from neurovision.mapping.engine import NeuroMapEngine
from neurovision.mapping.visual import write_comparison_html
from neurovision.uncertainty.example import SyntheticThresholdModel


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a synthetic NeuroMap report without MRI or trained weights.",
        epilog=RESEARCH_NOTICE,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    axes = np.meshgrid(*(np.linspace(-1, 1, 24) for _ in range(3)), indexing="ij")
    image = np.exp(-3 * (axes[0] ** 2 + 1.4 * axes[1] ** 2 + axes[2] ** 2))
    phantom = np.where(axes[0] < 0, 1, 2)
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    # Use the canonical representation without constructing any MRI files.
    reference = canonical_volume(
        image.astype(np.float32),
        affine,
        source_format="nifti",
        modality=Modality.UNKNOWN,
        reference_space="synthetic-space",
    )
    atlas_volume = canonical_volume(
        phantom.astype(np.float32),
        affine,
        source_format="nifti",
        modality=Modality.UNKNOWN,
        reference_space="synthetic-space",
    )
    adapter = LocalLabelAtlas(
        atlas_volume,
        "Synthetic halves",
        "1",
        "MIT",
        "synthetic-space",
        (
            AtlasRegion(1, "Synthetic left", "left"),
            AtlasRegion(2, "Synthetic right", "right"),
        ),
    )
    engine = InferenceEngine(SyntheticThresholdModel(0.36))
    inputs = torch.from_numpy(image.astype(np.float32))[None, None]
    prediction = engine.predict(inputs)
    labels = prediction.labels[0].numpy()
    confidence = prediction.probabilities[0].amax(dim=0).numpy()
    result = NeuroMapEngine(adapter).map(
        reference,
        labels,
        class_names={1: "Synthetic foreground"},
        segmentation_to_atlas_world=np.eye(4),
        registration_provenance="Identity on a shared synthetic grid",
        confidence=confidence,
    )
    attribution = explain_segmentation(
        engine, inputs, class_id=1, roi=prediction.labels[0] == 1, patch_size=(8, 8, 8)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_comparison_html(
        args.output,
        reference,
        labels,
        result,
        slice_index=12,
        saliency=attribution.saliency[0].numpy(),
        occlusion=attribution.occlusion.numpy(),
        attribution_note=(
            f"Synthetic foreground class 1, fixed ROI of {attribution.roi_voxels} "
            f"predicted foreground voxels, channel 0, zero-filled 8³ patches."
        ),
    )
    print(RESEARCH_NOTICE)
    print("Wrote a comparison report from synthetic data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
