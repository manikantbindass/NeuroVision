# Repository map

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

| Location | Responsibility |
| --- | --- |
| `apps/api`, `apps/web` | Reserved for the API and research viewer. |
| `neurovision/core` | Stage contracts shared by all future pipeline components. `neurovision.pipeline` remains a public import path. |
| `neurovision/imaging`, `neurovision/datasets` | Local MRI readers, canonical volume contracts, and dataset manifests or adapters. |
| `neurovision/preprocessing` | Configurable transforms, provenance and caching, registration adapters, QC reports, and slice montages. |
| `neurovision/models`, `training`, `inference`, `evaluation` | Reserved for model definitions, fitting, prediction, and benchmarking. |
| `neurovision/uncertainty`, `explainability`, `mapping`, `longitudinal`, `reporting` | Reserved for research analysis and report generation. |
| `configs`, `scripts`, `tests`, `notebooks`, `examples` | Configuration, developer utilities, tests, and synthetic examples. |
| `docs/` | Architecture, research, API, deployment, development, presentations, and reports. |
| `assets/` | README animations and reserved diagrams, screenshots, and branding. |
| `deployment/` | Docker CLI image and reserved Kubernetes, Helm, and Terraform definitions. |
| `.github/` | CI workflows and issue and pull request templates. |

Directories containing only `.gitkeep` are reserved for future implementation. They are excluded from the Python distribution. Application code will consume the pipeline package through public contracts. Pipeline components should stay independent of application servers, UI code, and infrastructure.

See the [pipeline architecture](../development/ARCHITECTURE.md), [capability roadmap](../development/CAPABILITIES.md), and [container setup](../deployment/README.md).
