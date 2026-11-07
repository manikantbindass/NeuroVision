# MRI ingestion and dataset validation

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

Install the optional imaging extra using the interpreter from the [README setup](../../README.md):

```text
python -m pip install ".[imaging]"
```

The stage catalog CLI still works with the standard library. NumPy holds typed voxel arrays, NiBabel interprets NIfTI geometry and orientation, and pydicom parses DICOM elements. For development and synthetic tests, install `.[dev,imaging]`.

## Load a local volume

```python
from pathlib import Path

from neurovision.imaging.nifti import load_nifti
from neurovision.imaging.volume import Modality

volume = load_nifti(Path("data/reviewed-t1.nii.gz"), modality=Modality.T1)
print(volume.data.shape, volume.spacing_mm, volume.reference_space)
```

`MRIVolume` carries the data, voxel-to-world affine, spacing, closest axis orientation, declared reference space, source format, and modality. Its arrays are read-only for normal callers. Index the array in voxel axis order `(i, j, k)`. Multiply the affine by `[i, j, k, 1]` to get RAS+ millimetres. The reference space field distinguishes scanner coordinates from an aligned or template space. [NiBabel coordinate systems and affines](https://nipy.org/nibabel/coordinate_systems.html)

## Discover and load reviewed DICOM

```python
from pathlib import Path

from neurovision.imaging.dicom import discover_dicom_series, load_dicom_series

staging = Path("data/reviewed-series")
series = discover_dicom_series(staging, privacy_reviewed=True)
volume = load_dicom_series(series[0].files, privacy_reviewed=True)
```

Set `privacy_reviewed=True` only after trusted prior review. Never derive it from a public upload field. Discovery examines a single directory for `.dcm`, `.dicom`, or extensionless regular files, groups by study and series, and checks slice geometry. The loader rechecks files, sorts by projected position, verifies spacing and frame of reference, and maps patient LPS coordinates to RAS. Discovery reports the native grid shape before reorientation. Compressed or deflated transfer syntaxes, multiframe and mosaic objects, sequences, irregular stacks, and single slices are outside the supported subset. [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/2023a/output/chtml/part03/sect_C.7.6.2.html)

## Ingestion limits

`read_nifti(stream, filename=...)` is the bounded binary-stream entry point for a future upload handler. It uses the filename suffix as a format hint and does not open it as a path. It validates format magic, geometry, scalar datatype, dimensionality, payload length, and finite voxels. Gzip decoding reads through EOF with an output cap and detects truncated streams. Local loaders expect regular files in trusted staging locations.

| Default limit | Value |
| --- | --- |
| Encoded bytes per NIfTI or DICOM operation | 128 MiB |
| Decoded bytes per operation | 256 MiB |
| Voxel count per operation | 32 million |
| DICOM bytes per file | 16 MiB |
| DICOM directory entries or files per load | 1,024 |

Use `IngestionLimits` to tighten these bounds. Peak memory also includes temporary decode arrays, the original bytes, and copies for reorientation. Enforce request size, worker memory, timeouts, authentication, staging permissions, and concurrency limits in any future upload service. This phase exposes a local Python library.

## Create and validate a dataset manifest

```python
from pathlib import Path

from neurovision.datasets.brats import BRATS_MODALITIES, Brats2021Adapter
from neurovision.datasets.manifest import validate_dataset

root = Path("data/brats2021")
manifest = Brats2021Adapter(split="train").build_manifest(root)
summary = validate_dataset(manifest, root, required_modalities=frozenset(BRATS_MODALITIES))
(root / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
print(summary)
```

The adapter expects immediate child directories such as `BraTS2021_00000` with files named `BraTS2021_00000_t1.nii.gz`, `_t1ce`, `_t2`, `_flair`, and optionally `_seg` using `.nii` or `.nii.gz`. Use `read_manifest(path)` to reload a version 1 manifest. A case ID may appear only once across splits, and resolved file paths must be unique and stay within the dataset root. The caller is responsible for using one ID per subject and controlling access to manifests.

Use a local dataset obtained under its access terms. [Data card](../../DATA_CARD.md)
