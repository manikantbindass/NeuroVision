# Foundation configuration

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

## Environment

The CLI reads settings from the process environment at startup. Set overrides in the current shell or pass them from the deployment platform. [`.env.example`](../../.env.example) documents the defaults. The CLI reads process variables only. Docker Compose can read a local `.env` file for its own interpolation, and this service passes only the declared variables into the container.

| Setting | Default | Accepted value |
| --- | --- | --- |
| `NEUROVISION_SEED` | `42` | An ASCII decimal integer from 0 through 4294967295. |
| `NEUROVISION_LOG_LEVEL` | `WARNING` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`, case insensitive. |

For a local CLI run with logging enabled:

```bash
export NEUROVISION_SEED=7 NEUROVISION_LOG_LEVEL=INFO
.venv/bin/python -m neurovision
```

```powershell
$env:NEUROVISION_SEED = '7'
$env:NEUROVISION_LOG_LEVEL = 'INFO'
.\.venv\Scripts\python.exe -m neurovision
```

```bat
set "NEUROVISION_SEED=7"
set "NEUROVISION_LOG_LEVEL=INFO"
.venv\Scripts\python.exe -m neurovision
```

The CLI writes its stage listing to stdout and sends operational events as newline-delimited JSON to stderr. Invalid configuration exits with code 2 and reports the invalid field without echoing its value. Logging call sites should use fixed event names and omit paths, images, raw metadata, and subject identifiers.

## Reproducibility and errors

Create a `RunContext` with an independently generated run ID and the configured seed. Call `rng()` once per operation that needs a fresh stream, then retain the returned generator while drawing samples.

```python
from neurovision.core.settings import Settings
from neurovision.pipeline import RunContext

settings = Settings.from_env()
context = RunContext(run_id="synthetic-demo", seed=settings.seed)
rng = context.rng()
sample = rng.random()
```

Each call to `rng()` starts a fresh Python stream at the same seed and leaves global random state untouched. Future NumPy and PyTorch adapters will seed their own generators and record backend settings and hardware. A Python seed alone cannot guarantee repeatability across model backends or devices.

`NeuroVisionError` is the base for expected application failures. `ConfigurationError` represents invalid settings and `PipelineError` is reserved for failures from future stage implementations. Translate expected exceptions at the application boundary.

## Dependency strategy

Install the Python 3.11+ runtime with `python -m pip install .`. The package uses the standard library. Setuptools builds the wheel in an isolated build environment.

Install `.[dev,imaging,preprocessing,bias,training,zoo]` in a virtual environment for repository-wide checks. The development extra provides Ruff, mypy, and pre-commit. Imaging and preprocessing extras cover volume handling and transforms. Training uses PyTorch and SciPy. The zoo extra pins MONAI for advanced architectures and adds einops for transformer rearrangements. Install the CPU wheel first for local development and record resolved versions for experiments. [Contributor setup](../../CONTRIBUTING.md) · [Dependency adoption criteria](CAPABILITIES.md)
