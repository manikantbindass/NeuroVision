<img src="assets/animations/header.svg" alt="NeuroVision-2028 animated MRI scan illustration" width="100%">

# NeuroVision-2028

**AI-Driven Precision Brain Segmentation & Mapping for MRI**

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

NeuroVision-2028 is a research platform under development for reproducible brain MRI segmentation experiments, model benchmarking, uncertainty and explainability studies, and longitudinal analysis prototypes. Its planned outputs include quantitative measurements, interactive 2D and 3D views, and structured reports served through an API and web interface.

The current package defines the pipeline interfaces and provides a CLI that lists the reference stages. [Architecture](docs/development/ARCHITECTURE.md) · [Capability roadmap](docs/development/CAPABILITIES.md) · [Phase status](docs/development/PHASE_STATUS.md) · [Changelog](CHANGELOG.md)

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

The CLI requires no configuration or MRI data. To check formatting and lint locally, install Ruff in your development environment and run `python -m ruff format --check .` and `python -m ruff check .` with that environment's interpreter.

See the [repository map](docs/architecture/REPOSITORY.md), [contributor guide](CONTRIBUTING.md), and [local CLI container setup](docs/deployment/README.md).

<img src="assets/animations/footer.svg" alt="Animated flow from MRI to a research report" width="100%">
