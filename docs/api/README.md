# NeuroVision Research API

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

The FastAPI app lives at `apps.api.main:create_app`. Install the API, imaging, and training extras in the project environment. Follow the README's CPU PyTorch setup first if needed.

```sh
python -m pip install ".[api,imaging,training]"
export NEUROVISION_API_TOKEN='<operator-provisioned random credential, at least 32 ASCII characters>'
python -m uvicorn apps.api.main:create_app --factory --host 127.0.0.1 --port 8000 --no-access-log
```

In PowerShell, set the variable with `$env:NEUROVISION_API_TOKEN = '<operator-provisioned credential>'`. In Windows Command Prompt, use `set "NEUROVISION_API_TOKEN=<operator-provisioned credential>"`. Keep the value outside source control.

Interactive docs are at `/docs`, with the schema at `/api/v1/openapi.json`. Health and documentation are public. Research routes require `Authorization: Bearer <credential>`. The app denies those routes when no credential is configured. Each upload, job, and result belongs to the authenticated principal. Other principals receive a 404 for private IDs.

| Route under `/api/v1` | Behavior |
| --- | --- |
| `GET /health` | Liveness and research notice |
| `POST /scans`, `GET /scans`, `GET/DELETE /scans/{id}` | Validated NIfTI upload, owner-scoped IDs, metadata, and removal |
| `GET /scans/{id}/volume.nii`, `GET /preprocess/{id}/volume.nii` | Owner-scoped canonical NIfTI-2 volume for synchronized viewing |
| `POST /preprocess`, `GET /preprocess/{id}` | QC, optional intensity normalization, and provenance. A failed QC result has `usable: false`. A usable result ID can be passed to inference as `scan_id`. |
| `GET /models` | Enabled, operator-registered inference models, revision and class schema |
| `POST /inference` | Queue a full-grid prediction for one channel and one 3D scan |
| `GET /segmentations`, `GET /segmentations/{id}`, `GET /segmentations/{id}/labels.nii` | IDs, prediction metadata and label voxels, and NIfTI-2 label export |
| `POST /metrics`, `GET /metrics/{id}` | Compare a segmentation to an uploaded label map on the same grid. Physical distances require explicit millimetres and orthogonal axes. |
| `POST /maps`, `GET /maps/{id}` | Run NeuroMap with an authorized local label atlas uploaded through `/scans`, a region table, and a reviewed segmentation-to-atlas RAS+ transform. Returns region statistics. |
| `GET /uncertainty`, `GET /uncertainty/{id}`, `GET /uncertainty/{id}/{confidence\|entropy}.nii` | Single-prediction maximum probability and normalized entropy summaries and exports. These values are uncalibrated. |
| `POST /reports`, `GET /reports/{id}` | Structured research report from up to 12 selected artifacts and their available dependencies |
| `GET /reports/{id}/exports`, `GET /reports/{id}/export.{json\|html\|pdf}` | Available formats and integrity-checked report downloads. PDF requires the optional `pdf` extra. [Reporting guide](../research/REPORTING.md) |
| `GET /jobs`, `GET/DELETE /jobs/{id}` | List and poll owned jobs. Deleting a terminal job removes its primary result. |

### Upload and run an experiment

The upload body is raw `.nii` or `.nii.gz` bytes with `Content-Type: application/octet-stream`. Choose `format=nii` or `format=nii.gz` and provide `X-Deidentified-Research: true` after authorized upstream de-identification and review. Set `role=labels` for an atlas or reference segmentation. The header is an attestation. The service does not de-identify pixel data. Upload only reviewed data permitted for the research purpose. The API does not accept DICOM, archives, file paths, or checkpoint uploads. It retains decoded arrays and limited imaging metadata in the local store, and discards the original upload container.

```sh
curl -X POST 'http://127.0.0.1:8000/api/v1/scans?format=nii&modality=t1' \
  -H "Authorization: Bearer $NEUROVISION_API_TOKEN" \
  -H 'X-Deidentified-Research: true' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @synthetic.nii
```

Then POST JSON such as `{"scan_id":"<uploaded UUID>","model_id":"synthetic-threshold"}` to `/api/v1/inference`. A `202` returns a job ID. Poll `/api/v1/jobs/{id}` with backoff. On success, follow `result_url`. Jobs report `queued`, `running`, `succeeded`, or `failed`. A failed job includes a sanitized error and its submission request ID. The built-in model is an analytic synthetic threshold fixture. Configure validated local weights at startup by injecting a `models` mapping of `ModelEntry` objects backed by the existing `InferenceEngine`. The HTTP service currently exposes single-model inference. Ensemble, attribution, and longitudinal research interfaces remain available in Python.

Errors use `{"error":{"code":"…","message":"…","request_id":"…"}}`. Responses carry a server-generated `X-Request-ID`. Request and job events contain allowlisted fields, without credentials, request bodies, raw URLs, or tracebacks. Keep Uvicorn access logging disabled when processing research data.

### Limits and deployment boundary

The local adapters use a 16 MiB upload limit, 32 MiB decoded NIfTI limit, 2 million voxel limit, and 64 KiB JSON request limit. The server bounds concurrent protected requests, queued work, job history and retained artifacts. The default rate limit is 120 requests per minute per principal. Queue saturation returns 503, excess request rate returns 429 with `Retry-After`, and exhausted retained storage or job history returns 507. Operators can customize the adapters through `create_app`. Remove obsolete scans, terminal jobs, and report snapshots as needed. Deleting one source does not delete reports that already captured a metadata snapshot.

Use one Uvicorn worker for the default local adapters. Queued jobs, the artifact store, and rate-limit counters are process-local. A restart loses local state. Graceful shutdown waits for the local queue. The distributed factory `apps.api.deployed:create_app` adds shared SQL artifacts and jobs with Redis rate limits. The separate worker runs `python -m apps.api.worker`. See the [deployment guide](../deployment/README.md) for schema initialization, secret references and operational limits.
