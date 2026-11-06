# Capability roadmap and technology choices

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

**Available today:** the foundation CLI, [local MRI data engine](IMAGING.md), [preprocessing and QC](PREPROCESSING.md), and [baseline segmentation training](BASELINE_TRAINING.md). A research experiment earns an **Experimental** label only when it can run end to end with documented inputs, constraints, and validation results. An interface without a runnable implementation is a **Stub**. The catalog CLI displays stage names.

| Capability | Status | Boundary and evidence required before enabling |
| --- | --- | --- |
| NIfTI MRI and T1, T1ce, T2, FLAIR | Available locally | Bounded NIfTI-1 and NIfTI-2 decoding validates dimensions, scalar datatype, finite intensities and physical geometry. Modalities are explicit or unknown. |
| Multimodal MRI | Manifest validation available | Check required sequences and a shared grid for each case. Model channel order and registration are planned. |
| DICOM ingestion | Available locally, privacy gated | Load reviewed, classic uncompressed MR series from a trusted staging directory. Public uploads require a separate design and privacy review. |
| Skull stripping, bias-field correction, normalization, resampling, registration and automatic QC | Local transforms and adapters available | Normalize, resample, run optional N4, apply external brain masks and affine registration, and review QC reports. Automatic brain mask and registration estimation are deferred. [Preprocessing guide](PREPROCESSING.md) |
| 2D, 3D and multi-class segmentation for anatomy and lesion or tumor experiments | Trainable baselines available | PyTorch U-Nets, synthetic training examples, and per-class overlap and surface metrics are available. Task-specific trained models and MRI benchmarks remain pending. [Baseline guide](BASELINE_TRAINING.md) |
| Ensemble inference and uncertainty | Local research experiments available | Weighted probabilities, reversible flip TTA, supported MC dropout, disagreement and uncertainty maps, and a local NeuroConfidence viewer. Calibration remains planned. [Ensemble guide](../research/ENSEMBLE_INTELLIGENCE.md) |
| Volumetry, hemisphere analysis and atlas mapping | Local research experiments available | Measure physical volumes and categorical atlas overlaps after externally reviewed affine registration. [NeuroMap guide](../research/NEUROMAP.md) |
| Explainability | Local research experiments available | Compare fixed-ROI input-gradient saliency and patch occlusion for one FP32 model. [NeuroMap guide](../research/NEUROMAP.md) |
| Longitudinal comparison | Planned research experiment | Use controlled study-local linkage and relative time points. Verify within-subject alignment and report acquisition or pipeline changes that could affect comparisons. |
| 2D viewer and 3D visualization | Planned | Verify slice orientation, overlay alignment and label identity before adding surfaces. Record any mesh simplification used for display. |
| Reproducible pipelines, experiment tracking and model registry | Planned | Record code revision, split manifest, parameters, seed, artifact hashes and software versions. Resolve models by immutable version or digest with a declared modality and label contract. |
| REST API, asynchronous inference and GPU inference | Planned | Separate HTTP requests from long jobs. Test authorization, bounded uploads, duplicate submissions, retries, cancellation, worker failure and device selection. Record actual device and precision in run metadata. |
| Containers, Kubernetes and observability | Planned | Verify a local container build and smoke test before orchestration. Test readiness, worker resource limits, telemetry redaction, and end-to-end operation before claiming deployment support. |

## Technology choices by layer

Dependencies are added in the phase that exercises them. Keep the core stage protocol independent of imaging, model, database and deployment libraries.

| Layer | Preferred technology and adoption trigger |
| --- | --- |
| Imaging | Use NiBabel and NumPy for NIfTI and volume arrays. Adopt SimpleITK when implementing registration, resampling or bias correction. Consider pydicom for inspecting DICOM metadata when the isolated ingestion path is ready. Add TorchIO for a demonstrated patch sampling or augmentation need and SciPy or scikit-image for operations that require them. |
| Models | PyTorch baselines and optional MONAI SegResNet, UNETR, Swin UNETR, and Attention U-Net integrations are available for experiments. Promote a candidate based on comparable accuracy, runtime and memory measurements. [Model zoo](../research/MODEL_ZOO.md) |
| Backend and persistence | Use FastAPI and Pydantic for versioned request, job and result schemas. Add SQLAlchemy and PostgreSQL when durable job and provenance metadata are needed. Keep volume bytes in a restricted artifact store and keep metadata references opaque. |
| Jobs and GPU | Start with a single worker contract and persisted job state. Evaluate Redis with RQ or Celery once durable long-running jobs require a broker. Pass artifact IDs through the queue. Give GPU workers explicit memory and concurrency limits. |
| Web | Use React and TypeScript for a responsive research workspace. Consider Cornerstone for orientation-aware slice views and Three.js for mesh rendering when the 3D workflow is ready. Add Next.js only if server rendering or its routing model solves a documented need. |
| Experiments | Begin with a versioned run manifest and a model registry interface. Adopt MLflow when shared tracking and model promotion are required. Consider DVC for manifests or approved datasets with a controlled remote. |
| Delivery | Use GitHub Actions for checks and image builds. Add Docker for reproducible API and worker environments, then Compose when local services need coordination. Consider Kubernetes and Helm for multi-worker deployments and Terraform when infrastructure provisioning needs repeatable configuration. AWS remains an optional target. |
| Observability | Start with structured logs and correlation IDs. Add Prometheus-compatible job, latency and GPU metrics when services run persistently. Use Grafana dashboards for operational views and consider OpenTelemetry when traces must cross API and worker boundaries. |
| Testing and security | Introduce pytest as imaging fixtures and API integrations grow. Add frontend component tests and browser tests when the viewer exists. Run secret and dependency scans in CI once it is configured. Keep test fixtures synthetic. |

## Boundaries to preserve

- **Geometry.** Carry voxel-to-world affine, units, reference space and transform history through each imaging artifact. Specify the canonical world convention and test adapters at RAS/LPS boundaries. [NiBabel coordinate systems and affines](https://nipy.org/nibabel/coordinate_systems.html)
- **Privacy.** Bound input bytes and decoded size. Use server-generated identifiers, restricted storage, explicit retention, and access-controlled artifact delivery. Exclude filenames, headers, pixel data and identifiers from logs, metrics, traces, reports and experiment trackers. Define a DICOM profile and test attribute, pixel and recognizable-feature safeguards independently before enabling ingestion. [DICOM PS3.15 Annex E confidentiality options](https://dicom.nema.org/medical/dicom/current/output/chtml/part15/sect_e.3.html)
- **Reproducibility.** Freeze dataset splits by subject and keep related scans together. Record model weights by hash, software versions, seed and relevant hardware. Report the dataset, task and evaluation assumptions beside metrics.
- **Job and artifact ownership.** Scope access to the submitting workspace. Use idempotent job submission and publish results only after successful artifact creation. Keep retries bounded and distinguish failed, cancelled and successful jobs.

## Suggested implementation order

1. Implement bounded local NIfTI loading and geometry checks against synthetic volumes.
2. Add preprocessing and QC, then baseline segmentation and reproducible evaluation.
3. Add quantification, research experiments and report generation against validated artifacts.
4. Add the API, worker and viewer. Introduce deployment and observability tools as each service needs them. Treat DICOM ingestion as a separate privacy gate before accepting DICOM uploads.
