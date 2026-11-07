# Explainable segmentation and NeuroMap

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

**Experimental.** Atlas mapping and attribution have been tested on synthetic inputs. **Benchmark pending.**

Phase 7 provides local segmentation sensitivity experiments and atlas overlap on an externally registered 3D grid. The example uses synthetic data and an analytic threshold model.

```sh
python -m pip install ".[imaging,training]"
python -m neurovision.mapping.example --output artifacts/neuromap-synthetic/comparison.html
```

Open the HTML locally to compare the same voxel-grid slice across MRI, predicted labels, aligned atlas, input-gradient magnitude, and patch occlusion. The table summarizes the full volume. The report embeds the displayed image and results. Store or share it only within the data permissions for the source volume.

## Explain a fixed segmentation target

Call `explain_segmentation` with an existing FP32 `InferenceEngine`, a single batch of 2D or 3D float32 inputs, a class ID and a nonempty boolean output ROI. The target is the spatial mean of the selected class **logit** inside that ROI. Keep the mask fixed across perturbations. If a mask was derived from the original prediction, record that choice when interpreting or comparing results.

- **Gradient saliency** returns the absolute derivative of the target with respect to each input channel and voxel. It describes local sensitivity in the units of the preprocessed input. [Simonyan et al.](https://arxiv.org/abs/1312.6034)
- **Occlusion sensitivity** replaces all channels in each nonoverlapping patch with a caller-supplied constant and records baseline minus perturbed score. A negative value means the score increased under that perturbation. The patch size is in voxels and edge patches may be smaller. [Zeiler and Fergus](https://arxiv.org/abs/1311.2901)

These methods can disagree. Gradients may be noisy or locally flat, and a constant fill can produce out-of-distribution inputs. The demo uses full-grid inference on one model, with no TTA, sliding windows, or MC dropout. Attention visualization and ensemble attribution are outside this implementation. Sensitivity maps provide no causal, clinical, or anatomical validation. Report the model and checkpoint, preprocessing, target ROI rule, patch size and fill. The demo rescales panels for display. Compare raw maps using a common scale for quantitative experiments.

## Map a local atlas

Implement `AtlasAdapter.load()` for a source that supplies categorical labels, an affine, a version, license terms and a curated code-to-region lookup. For an authorized local NIfTI atlas, use `load_nifti(path, label_map=True)` with explicit spatial units, then pass the resulting canonical volume, verified atlas version, reference space and lookup table to `LocalLabelAtlas`. This repository ships no third-party atlas. Verify that the atlas and derived output may be used and shared under its terms.

```python
from neurovision.mapping.engine import NeuroMapEngine

result = NeuroMapEngine(adapter).map(
    reference,                     # MRIVolume on the prediction grid
    labels,                        # 3D integer class map, 0 = background
    class_names={1: "Research target"},
    segmentation_to_atlas_world=transform,  # 4×4 segmentation RAS+ mm -> atlas RAS+ mm
    registration_provenance="Run identifier and reviewed registration QC reference",
    confidence=confidence,         # optional matching 3D probabilities
)
```

For a single prediction, `labels` can come from `prediction.labels[0].numpy()` and `confidence` from `prediction.probabilities[0].amax(dim=0).numpy()`. For an ensemble, use its labels and voxel confidence for the same case and grid. The MRI reference must be on that exact grid. If an external registration is expressed in LPS or the opposite direction, convert it to the documented direction and RAS+ millimetres first. An identity transform is appropriate only when both world coordinate systems are known to coincide. The engine checks geometry and uses nearest-neighbor label sampling. It neither estimates registration nor determines whether alignment is anatomically acceptable. Review registration QC before interpreting atlas overlap. [NiBabel affine conventions](https://nipy.org/nibabel/coordinate_systems.html)

Physical volume equals the voxel count times the absolute determinant of the segmentation affine's linear part. Atlas association volumes are measured on this same grid. The share of foreground uses all nonzero segmentation voxels as its denominator. Atlas overlap uses all voxels of the segmented class as its denominator, including atlas background and out-of-field voxels. Hemisphere comes from atlas annotations and becomes unknown when coverage is incomplete or the annotations do not determine a side. Confidence is the mean supplied value within each class and remains uncalibrated unless the caller validates calibration separately.

The report compares one voxel slice without correcting for physical aspect ratio. Its segmentation and atlas panels each use their own label vocabulary. Matching colors can occur across vocabularies and imply no relationship. The engine accepts affine transforms and hard labels. Probabilistic atlases, deformable mapping, and anatomical benchmarks remain future work.
