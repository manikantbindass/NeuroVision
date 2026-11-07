# NeuroVision-2028 final verification matrix

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

**Audited code:** `ac1005a0ef4573178cc4fb493e54d2c6454db22d` on the integrated Phase 15 branch in [PR #18](https://github.com/manikantbindass/NeuroVision/pull/18). This is branch evidence. At the time of verification, GitHub `main` was at `03f0006a0e25733fa3fa3b378de23641654d4da4`, the audited branch was 22 commits ahead and none behind, and the 18 phase pull requests were open. The requested PowerPoint and technical PDF were checked as local files.

**Disposition:** The synthetic research demo passes. Real MRI checkpoint serving and benchmark publication remain blocked by the failed checks below. GPU, live Kubernetes, and AWS execution remain unverified.

| Status | Meaning |
| --- | --- |
| PASS | The stated scope was implemented, documented, and verified with the cited evidence. |
| PARTIAL | A subset passed, or an important scenario, integration, or runtime remains unverified. |
| FAIL | A reproducible defect violates the intended contract. |
| NOT APPLICABLE | The capability is outside the project's stated research scope. |

## Component matrix

| Component | Implemented | Tested | Documented | Status |
| --- | --- | --- | --- | --- |
| Foundation, package and CLI | Installable package, settings, logging, seed and stage contracts | Clean Python 3.12 wheel install, isolated imports and CLI startup, and [foundation tests](../../tests/test_foundation.py) | [Foundation](FOUNDATION.md) | PASS |
| Architecture and coupling | Modular imaging, research, and application layers with typed boundaries | AST scan of 80 modules found no top-level import cycles. Worker imports HTTP defaults and fingerprints registered model state for every job. | [Architecture](ARCHITECTURE.md) | PARTIAL |
| NIfTI ingestion and canonical geometry | Bounded scalar 3D NIfTI-1/2 decoding, affine checks, closest RAS orientation | Synthetic valid, corrupt and oversized fixtures in [imaging tests](../../tests/test_imaging.py) | [Data card](../../DATA_CARD.md) | PASS |
| Reviewed local DICOM subset | Bounded series discovery, geometry and privacy checks for supported classic MR slices | Synthetic series, privacy rejection and geometry tests in [imaging tests](../../tests/test_imaging.py) | [Imaging guide](IMAGING.md) | PASS |
| Dataset manifests and BraTS adapter | Local authorized inputs, modality declarations, path containment and shared-grid validation | [Dataset tests](../../tests/test_datasets.py) cover manifests, traversal, and fixtures. No real BraTS cohort was used. | [Imaging guide](IMAGING.md) | PASS |
| Benchmark split integrity | Unique paths and case IDs | Reproduced byte-identical NIfTI content accepted in both train and test under different IDs and paths. Subject-level disjointness is not enforced. | [Training methodology](BASELINE_TRAINING.md) | FAIL |
| Preprocessing, QC, provenance and caching | Configurable spatial and intensity operations, QC reports and cache | Synthetic geometry, intensity, QC, cache, provenance and optional N4 checks in [preprocessing tests](../../tests/test_preprocessing.py) | [Preprocessing guide](PREPROCESSING.md) | PASS |
| General discrete-mask resampling and registration | Nearest neighbor is selectable. The public spatial transforms default to linear and do not distinguish masks from intensities. | Half-voxel resampling introduced class 1 into a mask containing only classes 0 and 2. Explicit nearest preserved the class set. | [Preprocessing guide](PREPROCESSING.md) | FAIL |
| Registration estimation and skull stripping | Reviewed transform and brain-mask adapters | Synthetic supplied-transform and mask checks passed. External estimation and mask generation were not validated. | [Preprocessing adapters](PREPROCESSING.md) | PARTIAL |
| Baseline training and evaluation on CPU | 2D and 3D U-Net, configurable losses, schedules, checkpoint resume, overlap and physical surface metrics | Synthetic training, deterministic CPU replay, BF16 and metric checks in [segmentation tests](../../tests/test_segmentation.py) | [Baseline training](BASELINE_TRAINING.md) | PASS |
| Advanced model zoo and Python inference on CPU | SegResNet, Attention U-Net, UNETR and Swin UNETR, factory, checkpoint validation and supported windowing | [Model zoo tests](../../tests/test_model_zoo.py) cover shape, checkpoint, inference and train/eval contracts | [Model zoo](../research/MODEL_ZOO.md) | PASS |
| Training-to-API spatial contract | Python training uses `(k,j,i)`; API model inputs use `(i,j,k)` | A noncubic `(4,6,8)` phantom produced model inputs of `(1,1,8,6,4)` and `(1,1,4,6,8)` respectively. The synthetic threshold fixture does not detect the difference. | [Training tensor conventions](BASELINE_TRAINING.md) and [API guide](../api/README.md) | FAIL |
| Ensembles and uncertainty | Probability weighting, reversible flip TTA, supported MC dropout, entropy, variance and disagreement | [Ensemble tests](../../tests/test_ensemble.py) cover analytic results, seeded sampling and mode restoration | [Ensemble guide](../research/ENSEMBLE_INTELLIGENCE.md) | PASS |
| NeuroConfidence and calibration | Synthetic confidence viewer and uncalibrated scores | Offline controls passed. Real-data calibration remains unmeasured. | [NeuroConfidence](../research/ENSEMBLE_INTELLIGENCE.md) | PARTIAL |
| Explainability and NeuroMap | Fixed-target saliency, occlusion, quantitative mapping and nearest-neighbor atlas labels | Synthetic sensitivity, overlap and physical-volume checks passed. External atlas alignment remains unvalidated. [NeuroMap tests](../../tests/test_neuromap.py) | [NeuroMap](../research/NEUROMAP.md) | PARTIAL |
| NeuroTimeline | Native volumes, registered difference maps and paired replicate spread, with nearest-neighbor mask alignment | Analytic rigid shift, coverage and paired spread tests passed. Real longitudinal cohorts and registration estimation remain unverified. [Timeline tests](../../tests/test_timeline.py) | [NeuroTimeline](../research/NEUROTIMELINE.md) | PARTIAL |
| FastAPI, OpenAPI and synthetic jobs | Versioned routes, bounded upload, owner isolation, error contracts and synthetic jobs | [API tests](../../tests/test_api.py), browser integration and container smoke passed | [API guide](../api/README.md) | PASS |
| Durable worker, PostgreSQL and Redis | Leased jobs, owner-scoped artifacts and shared rate limiting | [Distributed tests](../../tests/test_distributed.py), CI PostgreSQL service and real Compose worker smoke | [Deployment](../deployment/README.md) | PASS |
| React build and synthetic workspace | Responsive guided demo, linked slices, overlays, report downloads and sampled 3D preview | Frontend lint, format and build passed. All three Chromium scenarios passed. | [Web workspace](WEB_WORKSPACE.md) | PASS |
| Viewer geometry and accessibility | Native voxel views, labelled controls and keyboard navigation in the tested demo | Oblique geometry reproducer showed a voxel plane labelled axial was not world axial. Full accessibility conformance remains unverified. | [Web workspace](WEB_WORKSPACE.md) | PARTIAL |
| Browser scan listing and job recovery | Fetches up to 24 scan records and polls active jobs | Code review found silent truncation after 24 records and no recovery from HTTP 429 polling responses. Short synthetic flows passed. | [Web workspace](WEB_WORKSPACE.md) and [API guide](../api/README.md) | PARTIAL |
| Research reports | Integrity-checked structured reports in JSON, HTML and optional PDF | [Report tests](../../tests/test_reporting.py), generated outputs and browser export passed | [Reporting guide](../research/REPORTING.md) | PASS |
| MLOps and reproducibility | Experiment identities, checkpoint provenance, local MLflow and registry adapters, synthetic DVC workflow | [MLOps tests](../../tests/test_mlops.py) and isolated DVC reproduction passed | [MLOps guide](MLOPS.md) and [model card](../../MODEL_CARD.md) | PASS |
| Docker images and Compose | CLI, API and frontend images, separate worker, private backing services and loopback frontend | Compose config and Linux CI image builds passed. CI ran end-to-end synthetic inference through the frontend proxy and worker. | [Deployment](../deployment/README.md) | PASS |
| Kubernetes and Helm | Manifests, probes, resource limits, ingress, policy and optional autoscaling | Helm lint, render and Kustomize passed. Live cluster behavior remains unverified. | [Deployment](../deployment/README.md) | PARTIAL |
| AWS Terraform | Optional private EKS, RDS, Redis and S3 architecture | Format and provider schema validation passed. No apply or live AWS workload was tested. | [AWS deployment](../deployment/README.md#optional-aws-terraform-architecture) | PARTIAL |
| GPU training and inference | Configurable CUDA and optional GPU worker | CPU paths and GPU chart rendering passed. CUDA execution and GPU scheduling remain unverified. | [Deployment](../deployment/README.md#images-and-gpus) | PARTIAL |
| Security controls and scanners | Bounded parsing, owner scoping, secure headers, dependency and secret scans | Upload abuse and owner isolation tests passed. Redacted Git history, Python and npm scans passed for the audited revision. | [Threat model](../architecture/THREAT_MODEL.md) and [security policy](../../SECURITY.md) | PASS |
| Multi-user identity and imaging privacy | Owner scoping, single-principal deployment and prior-review safeguards | Synthetic authorization and DICOM rejection tested. External identity integration and institutional de-identification validation are outside the shipped workflow. | [Threat model](../architecture/THREAT_MODEL.md) | PARTIAL |
| Observability | Health and readiness, audit events, private metrics, optional tracing hooks | API metrics, worker metrics and Compose smoke passed. External trace and hardware exporters remain unverified. | [Observability](../deployment/README.md#private-observability) | PARTIAL |
| QA and GitHub Actions | Lint, format, typing, test matrix, browser, deployment and security checks | 83 Python tests passed locally at 81% statement and branch coverage; three browser tests passed. All jobs passed on the audited commit. | [Quality gates](QUALITY.md) | PASS |
| README and visuals | Setup for Bash, PowerShell and CMD, diagrams, local animations and synthetic screenshot | README checks passed. GitHub rendering preserved the banners and Mermaid blocks. The release sweep resolved 114 local link targets. | [README](../../README.md) | PASS |
| Project presentation and technical report | 24-slide PowerPoint and 14-page PDF prepared locally | Both rendered and were checked against the repository and measured validation results. Repository publication is pending phase merges. | [Phase status](PHASE_STATUS.md) | PARTIAL |
| Real MRI benchmarking | Metrics and synthetic protocols are available | Real MRI accuracy, calibration and generalization have not been measured. Split integrity also needs correction. | [Benchmarking](../../README.md#benchmarking-and-evaluation) | PARTIAL |
| Clinical diagnosis or treatment | Outside scope | No clinical performance or approval claimed | [Research use notice](../../README.md) | NOT APPLICABLE |
| Integration into `main` | Reviewed as a stack of phase branches | Remote branch comparison and open PRs verified. The stack has not merged. | [Phase status](PHASE_STATUS.md) | PARTIAL |

## GitHub workflow

1. Fetch the remote and inspect `origin/main`. Confirm the intended base and existing phase branches before changing them.
2. Create or reuse the appropriate phase branch. For stacked work, record the preceding phase as the PR base and confirm its ancestry.
3. Implement and document the phase. Run the relevant tests and record known limits.
4. Commit the verified work with a descriptive Conventional Commit such as `docs: record final verification matrix`. Push normally and open or update the phase PR when appropriate.
5. Merge only after checks pass on the current head and any blocking review findings are resolved. Preserve Git history. Force pushes require explicit authorization.
6. Sync `main` after each merge, confirm the merged revision, and update the next phase's base before beginning work. Publish the final PDF and PowerPoint after the phase merges.

## Evidence

- [Python, browser, PostgreSQL and Docker CI](https://github.com/manikantbindass/NeuroVision/actions/runs/42325780749)
- [Deployment definition checks](https://github.com/manikantbindass/NeuroVision/actions/runs/42325780724)
- [Dependency and secret scans](https://github.com/manikantbindass/NeuroVision/actions/runs/42325780751)
- [Phase 15 review and integration status](https://github.com/manikantbindass/NeuroVision/pull/18)
