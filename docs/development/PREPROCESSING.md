# Preprocessing and quality control

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

Install the optional imaging and preprocessing extras with `python -m pip install ".[imaging,preprocessing]"`. Include `bias` for N4 correction. The catalog CLI still works with the standard library.

## Configure a recipe

Compose an ordered tuple of transforms. The correct order depends on the acquisition and experiment. Run N4 before intensity normalization and apply fitted transforms consistently to the intended grid.

```python
from pathlib import Path

from neurovision.core.pipeline import RunContext
from neurovision.imaging.nifti import load_nifti
from neurovision.imaging.volume import Modality
from neurovision.preprocessing.intensity import NormalizeIntensity
from neurovision.preprocessing.pipeline import PreprocessingCache, PreprocessingPipeline
from neurovision.preprocessing.spatial import CropPad, OrientRAS, Resample
from neurovision.preprocessing.visual import write_qc_montage

source = load_nifti(Path("data/reviewed-t1.nii.gz"), modality=Modality.T1)
pipeline = PreprocessingPipeline(
    (OrientRAS(), Resample((1, 1, 1)), CropPad((128, 128, 128)), NormalizeIntensity()),
    cache=PreprocessingCache(max_entries=4),
)
result = pipeline.run(source, context=RunContext("research-run-01", seed=42))
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/qc.json").write_text(result.to_json(), encoding="utf-8")
if result.volume is not None:
    write_qc_montage(result.volume, Path("artifacts/qc.svg"))
```

| Transform | Behavior and configuration |
| --- | --- |
| `OrientRAS` | Reorder and flip voxel axes to the closest RAS orientation. Retain oblique geometry and apply no interpolation. Loaded MRI is already canonical. |
| `Resample` | Set target spacing and choose linear or nearest sampling. Keep the first voxel center and sample toward the last source center. Use nearest for discrete masks. Downsampling does not apply an anti-alias filter. [SciPy affine resampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html) |
| `CropPad` | Crop or pad to a requested shape around the image or a thresholded foreground. Set the fill value. Shift the affine so retained voxels keep their physical positions. |
| `ExtractForeground` | Threshold absolute intensity and optionally keep the largest 6-connected component. This is an intensity heuristic. |
| `NormalizeIntensity` | Apply z-score or percentile scaling. Fit on the whole image or thresholded foreground. Foreground-only mode writes zero outside that foreground. |
| `GaussianDenoise` | Experimental smoothing with a physical sigma. It can blur anatomical boundaries. Compare its effect on your task before using it. |
| `N4BiasCorrection` | Optional SimpleITK adapter for nonnegative intensities with sufficient positive foreground and orthogonal voxel axes. It fits a shrinkable grid and applies the estimated field at full resolution. [SimpleITK N4 example](https://simpleitk.readthedocs.io/en/latest/link_N4BiasFieldCorrection_docs.html) |
| `Register` | Sample a moving volume on a fixed canonical grid. A `RegistrationBackend` supplies the moving-to-fixed affine in RAS+ millimetres. Use `SuppliedAffine` to apply an already computed transform. Automatic registration estimation is not bundled. |
| `ApplyBrainMask` | Integrate an externally generated binary skull-stripping mask on the same grid. Pass the method and version. The adapter checks shape, affine, and label values. Brain mask generation and anatomical validation remain with the external tool and reviewer. |
| `RandomIntensity`, `RandomFlip` | Seeded training-only augmentation. Intensity augmentation samples gain, offset, and optional noise. Flips mirror voxels on the existing grid. Keep validation and test recipes deterministic without augmentation. Apply identical spatial augmentation to any paired labels. This phase does not orchestrate paired samples. |

## Inspect QC and provenance

`assess_case` checks required modalities and assesses each volume. `PreprocessingPipeline.run` checks the input and final volume and reports failed operations. Invalid dimensions or geometry, nonfinite voxels, blank scans, and failed transforms block output. Extreme spacing, excessive absolute intensities, poor post-registration foreground overlap, and reflections raise warnings. `QCPolicy` controls thresholds, and the report includes the applied policy and measured values.

Foreground Dice is a warning heuristic. It does not establish anatomical alignment and can be misleading across modalities or after masking. Inspect images and registration results manually. `write_qc_montage` writes center axial, coronal, and sagittal slices to a self-contained SVG. Pass a same-grid comparison to view two rows with one shared display window. Direction labels describe the closest axes for oblique inputs.

Each run records the software versions, run seed, derived step seeds, operation order and parameters, measured details, input and output grids, and local content fingerprints. Failed operations appear in the report without exposing arbitrary backend exception text. Use the same input, recipe, versions, and seed when replaying a run. Numerical equivalence across platforms and future library versions requires separate validation.

The optional LRU cache lives in memory and is bounded by entry count and stored array bytes. Its key covers input data and geometry, ordered parameters, software versions, seed, and declared adapter inputs. Cache hits reuse the transform records and evaluate QC for the current run. Custom steps and registration backends should opt in only when deterministic and include all external state and backend versions in their parameters. Keep reports, digests, and montages in controlled research storage. A montage contains image pixels.
