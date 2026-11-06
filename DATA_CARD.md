# Data card

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

## Supported inputs and modalities

- Single-file, scalar 3D NIfTI-1 and NIfTI-2 volumes in `.nii` and `.nii.gz` with a valid coded sform or qform. Headers, extensions, and filenames are excluded from returned volume metadata.
- Pre-reviewed classic MR Image Storage DICOM series staged in a trusted directory. The supported subset uses native little-endian transfer syntax, single-frame 16-bit monochrome slices, consistent geometry, and at least two positions.
- Explicit modality labels are T1, contrast-enhanced T1 (T1ce), T2, and FLAIR. Generic NIfTI files default to unknown. The DICOM loader only recognizes an exact allowlist of series labels and returns unknown for conflicting or ambiguous labels. It does not infer the sequence from image appearance.
- Version 1 local manifests associate one or more NIfTI modalities and an optional NIfTI label map with each research case. The BraTS 2021 adapter expects all four modalities in the challenge naming convention.

## Preprocessing assumptions and validation

Ingestion applies stored intensity scaling, reports float32 data in the closest RAS orientation, and retains the physical transform and spacing in millimetres. Orientation changes permute or flip voxel axes without interpolation. Oblique scans can retain oblique affines. NIfTI prefers a coded sform over qform and flags unspecified spatial units as assumed millimetres.

Manifest validation requires aligned images and labels within each case. It checks shape, declared reference space, and affine agreement to 0.001 mm. Label maps must contain integer IDs from 0 through 65535. This phase does not perform registration, skull stripping, intensity normalization, clinical QC, or label ontology validation. Matching grids cannot establish anatomical alignment or dataset completeness.

## Privacy and licensing

Use only data approved for your research workflow. Keep source MRI, manifests with study-local IDs, and derived volumes outside Git and public logs. DICOM needs an authorized operator's prior review of metadata, burned-in text, and recognizable anatomy. The loader checks required flags and rejects common identifiers, private elements, sequences, curves, and overlays. These checks do not detect facial anatomy or establish anonymization or DICOM profile conformance. NIfTI voxels can also contain recognizable features. [Imaging review and retention](docs/architecture/THREAT_MODEL.md#research-imaging-review-and-retention) · [DICOM PS3.15 confidentiality options](https://dicom.nema.org/medical/dicom/current/output/chtml/part15/sect_e.3.html)

The repository contains no patient MRI, dataset copies, or download automation. Obtain BraTS data through its official access process, follow the applicable terms and citation requirements, and stage the files yourself. The software license does not grant rights to source data. [Official BraTS 2021 challenge and data access](https://www.med.upenn.edu/cbica/brats2021/)

## Biases and limits

BraTS 2021 focuses on glioma and reflects the scanners, acquisition protocols, participating institutions, and inclusion criteria of that challenge. It does not represent the full distribution of brain MRI or other conditions. Explicit modality names can be wrong, and a strict DICOM subset excludes acquisitions with different encodings or privacy records. Preserve cohort and acquisition provenance when reporting results.

Tests generate synthetic arrays, NIfTI files, and DICOM slices locally. They validate software behavior and do not measure data quality, representativeness, or performance on human MRI. [MRI ingestion guide](docs/development/IMAGING.md)
