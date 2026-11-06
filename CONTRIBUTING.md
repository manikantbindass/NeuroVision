# Contributing

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

Read the [repository map](docs/architecture/REPOSITORY.md) and [phase status](docs/development/PHASE_STATUS.md) before starting a change. Branch from the intended review base and keep each pull request focused on one phase or fix. For a stacked pull request, name the preceding PR and target its branch until it merges.

Follow the virtual environment setup in the [README](README.md) and activate the environment in your shell before installing or running Git hooks. Run these commands with that environment's Python:

```text
python -m pip install --upgrade pip
python -m pip install --no-deps "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[dev,imaging,preprocessing,bias,training,zoo,api,api-test,pdf,report-test,mlops,distributed,distributed-test]"
python -m pre_commit install
python -m pre_commit run --all-files
python -m ruff check neurovision tests
python -m ruff format --check neurovision tests
python -m mypy
python -m unittest discover -s tests -v
python -m neurovision.uncertainty.example --output artifacts/neuroconfidence-synthetic
node tests/check_neuroconfidence.cjs artifacts/neuroconfidence-synthetic/neuroconfidence.html
python -m pip check
python -m neurovision
```

With GNU Make, run `make install`, `make hooks`, and `make check`. Run `make viewer-test` for the offline JavaScript viewer checks, which require Node.js 24 or later. Set `PYTHON` to the environment's interpreter if it is outside your shell path. The hooks use the active shell's Python so they run the same tool versions as the developer commands and CI.

Use synthetic data for tests. Document any new dependency, format assumptions, geometry conventions and validation limits. Update the changelog and phase status when completing a phase. Follow the [code of conduct](CODE_OF_CONDUCT.md) and report security issues using the [security policy](SECURITY.md).
