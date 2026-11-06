# Phase status

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

| Phase | Implementation status | Validation |
| --- | --- | --- |
| Project vision and pipeline architecture | Stage catalog, typed component interface, experiment context, CLI, architecture notes, and animated README assets | Unit tests cover component substitution, reference order, and notice visibility through the CLI. |
| Advanced capabilities and technology direction | [Capability roadmap](CAPABILITIES.md) maps proposed features to validation gates and dependency choices | Documentation review and existing CLI regression checks. |
| Monorepo architecture | [Repository layout](../architecture/REPOSITORY.md), compatible core imports, contributor files, CI, and CLI container definition | Import compatibility, CLI checks, wheel installation, and Compose configuration validation. |
| Phase 1 foundation | [Runtime settings](FOUNDATION.md), JSON logging, seed handling, exceptions, CLI entry point, and developer tooling | Clean installation, unit and CLI integration checks, Ruff, mypy, pre-commit, and Compose configuration. |
| Phase 2 MRI data engine | [MRI ingestion](IMAGING.md), reviewed DICOM discovery and loading, RAS volume contracts, dataset validation, and local BraTS 2021 adapter | Synthetic NIfTI and DICOM inputs, safety rejection cases, and dataset manifest integration tests. |
| Phase 3 preprocessing and QC | [Preprocessing recipes](PREPROCESSING.md), provenance, registration and skull-mask adapters, optional N4, cache, JSON reports, and QC montages | Synthetic geometry and intensity checks, seeded replay, cache isolation, N4 smoke test, and QC failure cases. |
| Phase 4 baseline segmentation | [2D and 3D U-Nets](BASELINE_TRAINING.md), configurable training and losses, checkpoints, and overlap and surface metrics | Synthetic training, exact CPU checkpoint replay, CPU BF16, schedule and stopping behavior, and analytic metric tests. MRI metrics are not yet benchmarked. |
| MRI processing, models, and application | Planned for later phases | Validation will be defined with each implementation. |
| Final PDF and PowerPoint | Planned after implementation phases are merged | Check the artifacts against the merged project. |
