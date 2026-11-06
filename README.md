<img src="assets/animations/header.svg" alt="NeuroVision-2028 animated MRI scan illustration" width="100%">

# NeuroVision-2028

**AI-Driven Precision Brain Segmentation & Mapping for MRI**

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

NeuroVision-2028 is a research platform under development for reproducible brain MRI segmentation experiments, model benchmarking, uncertainty and explainability studies, and longitudinal analysis prototypes. Its planned outputs include quantitative measurements, interactive 2D and 3D views, and structured reports served through an API and web interface.

The foundation provides shared pipeline contracts, validated runtime settings, structured logs, reproducible Python random streams, and a CLI that lists the reference stages. [Architecture](docs/development/ARCHITECTURE.md) · [Phase status](docs/development/PHASE_STATUS.md) · [Changelog](CHANGELOG.md)

The roadmap starts with bounded NIfTI loading and geometry validation, then adds preprocessing, QC, and reproducible segmentation benchmarks. Later phases cover measurements and research reports, followed by the API, worker, and viewer. DICOM ingestion has a separate privacy gate. [Capability roadmap](docs/development/CAPABILITIES.md)

## Setup

You need Git, Python 3.11 or later, and pip. The CLI and tests use the Python standard library. The installation step may download the build backend.

### Bash terminal

```bash
git clone https://github.com/manikantbindass/NeuroVision.git
cd NeuroVision
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m neurovision
.venv/bin/python -m unittest discover -s tests -v
```

### Windows PowerShell

```powershell
git clone https://github.com/manikantbindass/NeuroVision.git
Set-Location NeuroVision
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m neurovision
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Windows Command Prompt

```bat
git clone https://github.com/manikantbindass/NeuroVision.git
cd NeuroVision
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install .
.venv\Scripts\python.exe -m neurovision
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The CLI also installs as `neurovision`. It uses defaults when the process environment has no overrides. See the [environment and dependency guide](docs/development/FOUNDATION.md) for Bash, PowerShell, and Command Prompt configuration examples.

See the [repository map](docs/architecture/REPOSITORY.md), [contributor guide](CONTRIBUTING.md), and [local CLI container setup](docs/deployment/README.md).

<img src="assets/animations/footer.svg" alt="Animated flow from MRI to a research report" width="100%">
