# Repository map

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

| Location | Responsibility |
| --- | --- |
| `apps/api` | FastAPI routes, request boundaries, service orchestration, and local adapters. |
| `apps/web` | React and TypeScript research workspace, linked MRI viewer, API client, and Playwright tests. |
| `neurovision/core` | Stage contracts shared by all future pipeline components. `neurovision.pipeline` remains a public import path. |
| `neurovision/imaging`, `neurovision/datasets` | Local MRI readers, canonical volume contracts, and dataset manifests or adapters. |
| `neurovision/preprocessing` | Configurable transforms, provenance and caching, registration adapters, QC reports, and slice montages. |
| `neurovision/models`, `neurovision/training`, `neurovision/evaluation` | Native U-Nets, the MONAI model zoo, architecture and checkpoint management, training, profiling, and metrics. |
| `neurovision/inference` | Full-grid and sliding-window inference, weighted ensembles, and reversible TTA. |
| `neurovision/uncertainty` | Weighted probability statistics, confidence maps, and a local interactive viewer. |
| `neurovision/explainability`, `mapping` | Fixed-target segmentation sensitivity, registered atlas overlaps, and local comparison reports. |
| `neurovision/longitudinal` | Registered mask comparisons, native regional changes, and the NeuroTimeline viewer. |
| `neurovision/reporting` | Structured report snapshots, reproducibility fingerprints, JSON/HTML/PDF exporters, and a synthetic example. |
| `neurovision/mlops` | Dataset and configuration identity, training lifecycle artifacts, MLflow tracking, and immutable trained-model registry adapters. |
| `configs`, `scripts`, `tests`, `notebooks`, `examples` | Configuration, developer utilities, tests, and synthetic examples. |
| `docs/` | Architecture, research, API, deployment, development, presentations, and reports. |
| `assets/` | README animations and reserved diagrams, screenshots, and branding. |
| `deployment/` | Docker CLI image and reserved Kubernetes, Helm, and Terraform definitions. |
| `.github/` | CI workflows and issue and pull request templates. |

Directories containing only `.gitkeep` are reserved for future implementation. They are excluded from the Python distribution. Application code will consume the pipeline package through public contracts. Pipeline components should stay independent of application servers, UI code, and infrastructure.

See the [pipeline architecture](../development/ARCHITECTURE.md), [capability roadmap](../development/CAPABILITIES.md), and [container setup](../deployment/README.md).
