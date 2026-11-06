<img src="assets/animations/header.svg" alt="NeuroVision-2028 animated MRI scan illustration" width="100%">

# NeuroVision-2028

**AI-Driven Precision Brain Segmentation & Mapping for MRI**

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

NeuroVision-2028 is a research platform under development for reproducible brain MRI segmentation experiments, model benchmarking, uncertainty and explainability studies, and longitudinal analysis prototypes. Its planned outputs include quantitative measurements, interactive 2D and 3D views, and structured reports served through an API and web interface.

The foundation provides shared pipeline contracts, validated runtime settings, structured logs, reproducible Python random streams, and a CLI that lists the reference stages. The optional MRI engine loads NIfTI and reviewed DICOM locally and validates dataset manifests. [Architecture](docs/development/ARCHITECTURE.md) · [Phase status](docs/development/PHASE_STATUS.md) · [Changelog](CHANGELOG.md)

The preprocessing engine adds configurable transforms, provenance, and QC. Trainable 2D and 3D U-Net baselines include reproducible synthetic examples. The next phases cover benchmarking, measurements and research reports, then the API, worker, and viewer. [Preprocessing guide](docs/development/PREPROCESSING.md) · [Baseline training guide](docs/development/BASELINE_TRAINING.md) · [Capability roadmap](docs/development/CAPABILITIES.md)

The optional model zoo integrates SegResNet, Attention U-Net, UNETR, and Swin UNETR with the training loop and a tensor inference engine. [Model zoo guide](docs/research/MODEL_ZOO.md)

Ensemble experiments add weighted probability averaging, reversible flip TTA, model disagreement, predictive entropy, variance, and supported MC dropout sampling. The local NeuroConfidence viewer distinguishes threshold-based high-confidence predictions from low-confidence or uncertain predictions. [Ensemble and confidence guide](docs/research/ENSEMBLE_INTELLIGENCE.md)

NeuroMap measures segmented volumes and overlaps with registered local atlases. The comparison report shows the MRI, labels, aligned atlas, gradient saliency, and patch occlusion side by side. Run `python -m neurovision.mapping.example --output artifacts/neuromap-synthetic/comparison.html` after setup for a synthetic demonstration. [NeuroMap research guide](docs/research/NEUROMAP.md)

NeuroTimeline compares regional volumes and registered segmentation masks across MRI time points. Run `python -m neurovision.longitudinal.example --output artifacts/neurotimeline-synthetic/timeline.html` after setup for an interactive synthetic demonstration. [NeuroTimeline research guide](docs/research/NEUROTIMELINE.md)

The FastAPI service exposes authenticated MRI uploads, queued research jobs, segmentation exports, and OpenAPI documentation. [Backend setup and API guide](docs/api/README.md)

The React research workspace adds synchronized MRI slices, segmentation and uncertainty overlays, a sampled 3D preview, and a guided synthetic case. Start it with `npm ci --prefix apps/web` and `npm run dev --prefix apps/web`. [Workspace setup and connected workflows](docs/development/WEB_WORKSPACE.md)

Research reports capture scan and preprocessing metadata, QC, model versions, class volumes, uncertainty, selected evaluation results, and reproducibility fingerprints. Export JSON, HTML, or optional PDF from the Reports page. [Reporting guide](docs/research/REPORTING.md)

Training artifacts record dataset and configuration fingerprints, Git revision, dependency snapshots, and run-specific model cards. Optional local MLflow tracking and artifact registries support versioned experiments, with a DVC-compatible synthetic workflow. [MLOps guide](docs/development/MLOPS.md)

The distributed deployment adds a separate research worker, PostgreSQL-backed jobs and artifacts, Redis rate limits, and a frontend proxy. Compose, Kubernetes/Helm, and an optional AWS Terraform architecture are included. [Deployment guide](docs/deployment/README.md)

**MRI benchmark metrics: Not yet benchmarked.**

## Setup

You need Git, Python 3.11 or later, and pip. The setup below installs the optional libraries and CPU PyTorch for the synthetic tests and baseline examples. Include `bias` in the extras for the N4 test. For the stage catalog CLI alone, install `.`.

### Bash terminal

```bash
git clone https://github.com/manikantbindass/NeuroVision.git
cd NeuroVision
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-deps "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install ".[imaging,preprocessing,training]"
.venv/bin/python -m neurovision
.venv/bin/python -m unittest discover -s tests -v
```

### Windows PowerShell

```powershell
git clone https://github.com/manikantbindass/NeuroVision.git
Set-Location NeuroVision
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-deps "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install ".[imaging,preprocessing,training]"
.\.venv\Scripts\python.exe -m neurovision
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Windows Command Prompt

```bat
git clone https://github.com/manikantbindass/NeuroVision.git
cd NeuroVision
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --no-deps "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install ".[imaging,preprocessing,training]"
.venv\Scripts\python.exe -m neurovision
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The CLI also installs as `neurovision`. It uses defaults when the process environment has no overrides. See the [environment and dependency guide](docs/development/FOUNDATION.md) for Bash, PowerShell, and Command Prompt configuration examples.

See the [repository map](docs/architecture/REPOSITORY.md), [contributor guide](CONTRIBUTING.md), and [local CLI container setup](docs/deployment/README.md).

See the [MRI ingestion guide](docs/development/IMAGING.md) for code examples and the [data card](DATA_CARD.md) for input support, access terms, and limitations.

<img src="assets/animations/footer.svg" alt="Animated flow from MRI to a research report" width="100%">
