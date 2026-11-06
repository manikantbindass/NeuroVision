# NeuroTimeline longitudinal MRI experiments

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.
>
> Observed differences may arise from acquisition, preprocessing, registration, model, or segmentation variation.

```sh
python -m pip install ".[imaging,preprocessing]"
python -m neurovision.longitudinal.example --output artifacts/neurotimeline-synthetic/timeline.html
```

Open the HTML locally and choose a follow-up day, segmentation region, and slice. The three panels show the baseline MRI, registered follow-up, and a class-specific difference map. The table tracks that region at every follow-up relative to baseline. The demo uses synthetic scans, known translations, and matched phantom perturbations. Reports embed sampled MRI intensities and labels. Keep reports from research scans within the access controls for the source data.

## Inputs and registration

Build a `TimePoint` for each visit from a canonical `MRIVolume` with explicit millimetre units and a 3D hard-label array on its exact voxel grid. Use the same study-local pseudonymous key, known modality and class schema across visits. Supply increasing relative days, region names, and nonidentifying acquisition, preprocessing and model revision descriptors. The engine flags changed descriptors. The caller must independently verify subject linkage, mask grid and class semantics.

```python
from neurovision.longitudinal.engine import FollowUp, build_timeline
from neurovision.preprocessing.adapters import SuppliedAffine
from neurovision.preprocessing.contracts import matrix_values

timeline = build_timeline(
    baseline,
    [FollowUp(
        follow_up,
        SuppliedAffine(matrix_values(reviewed_rigid_transform)),
        "Research run ID and registration QC reference",
    )],
    class_names={1: "Research region"},
)
change = timeline.difference_map(follow_up_index=0, class_id=1)
```

Supply a reviewed proper rigid transform or implement the existing `RegistrationBackend` contract to estimate one. The transform maps follow-up RAS+ millimetres to baseline RAS+ millimetres. The engine registers intensities with linear interpolation and samples labels with nearest-neighbor interpolation. It rejects reflection, scale, and shear in the registration transform. An identity transform requires a verified shared coordinate frame. Automatic registration estimation requires an external backend. The engine checks minimum baseline grid coverage and reports an image-foreground overlap proxy. Grid coverage and foreground Dice cannot establish anatomical alignment. Review landmarks, overlap and acquisition differences before interpreting changes. [SciPy sampling convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html)

## Read the results

- **Native volume change.** Each class volume uses the full native mask and the absolute determinant of its own voxel-to-world affine. The change is follow-up minus baseline. The percentage uses baseline volume as the denominator and is undefined when it is zero. Different fields of view can affect native volumes.
- **Spatial change.** Gain and loss are measured on the baseline grid within shared field of view. The map uses −1 for loss, +1 for gain, 0 for unchanged, and 2 for outside shared coverage. The output also reports shared-field per-class Dice, undefined when both masks are empty there. Registered spatial counts and native volume differences need not agree.
- **Replicate spread.** Optionally supply 3 to 256 matched replicate IDs with per-class native volumes in mm³ and a common `replicate_protocol` describing how runs were paired. For example, pair each ensemble member with itself across time. The engine subtracts volumes by replicate ID and reports the 5th, 50th and 95th percentiles. Without paired data, spread is unavailable. This empirical summary is sensitive to the sample and omits variability not represented by the replicate protocol. It provides no confidence interval or threshold for meaningful change.

The display subsamples the baseline grid and shares its intensity scale across panels. Measurements use the full arrays. The analysis is baseline-referenced. It does not build a symmetric within-subject template, and interpolation can contribute to apparent spatial change. The present validation uses analytic phantoms. [Reuter et al. on longitudinal processing asymmetry](https://pmc.ncbi.nlm.nih.gov/articles/PMC3260043/)
