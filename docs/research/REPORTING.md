# AI-Assisted Research Segmentation Report

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

Reports are structured research records. Schema version 2 combines explicit metadata snapshots with reproducibility fingerprints. The JSON, HTML and optional PDF exports derive from the same saved snapshot.

## Generate a synthetic example

Follow the project CPU PyTorch installation steps, then install the API and research extras.

```sh
python -m pip install ".[api,imaging,training,pdf]"
python -m neurovision.reporting.example --output artifacts/research-report-synthetic --pdf
```

The command runs a local in-memory workflow from synthetic NIfTI data through QC, inference and reference evaluation. It writes JSON, HTML and PDF files. Remove `--pdf` and omit the `pdf` extra when using JSON and HTML only.

## Report contents

| Section | Recorded information |
| --- | --- |
| Report record | Report ID, timezone-aware generation timestamp, schema version, software versions and content fingerprint |
| Scan metadata | Canonical shape, affine, spacing, modality, reference space and whether millimetres were assumed |
| Preprocessing | Selected steps, parameters, seeds, grids, software and transformation provenance |
| QC | Findings, severity, measurements, policy and availability |
| Models | Enabled model ID, revision, architecture configuration, state fingerprint, synthetic status, actual device and precision |
| Classes and volumes | Declared label names, per-class voxel counts, physical volumes, and fractions of total foreground |
| Confidence | Per-predicted-class mean maximum probability and fraction at or above 0.8 |
| Uncertainty | Per-class mean normalized entropy and whole-grid mean confidence and entropy, with explicit scope and calibration status |
| Evaluation | Reference-label metrics when a metrics artifact is selected, including distance availability |
| Atlas associations | Measurements and registration metadata when a NeuroMap artifact is selected |
| Limitations | Platform limitations and any operator-supplied model limitations |
| Reproducibility | Source IDs, timestamps, metadata fingerprints, volume and label fingerprints, model state and software |
| Research notice | The required notice in every output |

Physical volume uses voxel count times the absolute determinant of the affine linear part. When millimetres were assumed, volume in mm³ is unavailable. An empty predicted class has a zero voxel count and unavailable mean confidence and entropy. Foreground fractions are unavailable when the entire foreground is empty. Background contributes to whole-grid uncertainty summaries and is excluded from foreground volume fractions.

## API and web exports

POST `{"artifact_ids":["<result UUID>"]}` to `/api/v1/reports` and poll the returned job ID. The service follows owned source links, with a bound of 32 total artifacts. A metrics source therefore includes its linked segmentation, preprocessing and reference scan if those artifacts still exist. Only explicitly selected or linked artifacts are included. An inference artifact alone cannot supply evaluation results that were never run or selected.

The returned report remains a snapshot after a source is removed. Inference captures its input scan and preprocessing metadata so these can be reported when the original input has been removed. Missing fields remain unavailable. Reports retain copied metadata until the report job is deleted separately.

- `GET /api/v1/reports/{id}` returns the saved document.
- `GET /api/v1/reports/{id}/exports` lists installed export formats.
- `GET /api/v1/reports/{id}/export.json` returns the validated JSON snapshot.
- `GET /api/v1/reports/{id}/export.html` downloads escaped, self-contained HTML.
- `GET /api/v1/reports/{id}/export.pdf` downloads PDF if ReportLab is installed, otherwise returns a sanitized `503 pdf_unavailable`.

The connected Reports page exposes these exports. The browser-only demonstration exports a deterministic phantom's measurements in JSON or HTML and labels unrun QC and content hashing as unavailable. PDF export uses the connected API.

## Reproducibility and limits

The report digest is SHA-256 of canonical JSON excluding the digest itself. This detects accidental changes to a snapshot. It provides no authentication signature. Source metadata digests use the same canonical encoding. MRI input digests reuse the preprocessing volume fingerprint. Label digests use little-endian uint16 label bytes on the recorded grid. Model state digests cover sorted tensor names, shapes, datatypes and bytes. Architecture and model revision are recorded separately. Model entries are provisioned at startup and must remain fixed for the lifetime of the service.

Generation timestamps and runtime software are captured explicitly. Re-rendering a saved JSON report preserves its recorded contents. PDF container timestamps can change between exports, and bitwise model replay depends on hardware, software and algorithm settings. Reports do not bundle datasets, patient identifiers, source paths or weights. Keep research reports within the source data's access and retention controls because even derived measurements and scan geometry may be sensitive.

Report generation accepts up to 1 MB of source metadata before assembling sections and exports. Schema-2 exporters validate snapshot integrity. The PDF renderer uses ReportLab's bundled Vera font, which covers the Latin and Greek text used in the sample. Additional scripts may require a deployment-owned font and validation. Evaluation, atlas associations, uncertainty experiments and model performance require independent scientific review.
