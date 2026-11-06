# Changelog

## Unreleased

- Add schema-versioned AI-Assisted Research Segmentation Reports with input, model, QC, volume, uncertainty, evaluation, and reproducibility metadata.
- Add integrity-checked JSON, escaped HTML, optional PDF exports, and report download controls in the web workspace.

- Add a responsive React MRI research workspace with linked planes, segmentation and uncertainty overlays, before/after views, a sampled 3D preview, and a guided synthetic case.
- Connect the web workspace to authenticated research jobs and add owner-scoped canonical NIfTI image exports.

- Add a versioned FastAPI service for MRI uploads, preprocessing, inference, segmentation exports, metrics, atlas maps, uncertainty, reports, and asynchronous jobs.
- Add request IDs, sanitized errors, structured logs, upload quotas, and replaceable authentication, rate limiting, storage, and queue adapters.

- Add baseline-referenced longitudinal MRI mapping, native regional volume changes, shared-coverage difference maps, paired replicate spread, and the interactive NeuroTimeline viewer.

- Add fixed-ROI gradient saliency and bounded patch occlusion for segmentation experiments.
- Add NeuroMap atlas overlap, physical volumes, hemisphere annotations, confidence summaries, and a local comparison report.

- Add weighted ensemble probability reduction, reversible flip TTA, model disagreement, and supported Monte Carlo dropout sampling.
- Add predictive entropy, probability variance, voxel confidence maps, and the interactive local NeuroConfidence research viewer.

- Integrate MONAI SegResNet, Attention U-Net, UNETR, and Swin UNETR through a common model contract, factory, and architecture registry.
- Add architecture-aware checkpoint reconstruction, tensor inference, parameter counts, and scoped forward profiling.

- Add 2D and 3D U-Net baselines, configurable Dice and cross-entropy training, and reproducible synthetic experiment recipes.
- Add epoch checkpoints, mixed precision, scheduling, early stopping, and per-class overlap and physical surface metrics.

- Add reproducible preprocessing recipes, optional N4 bias correction, affine registration and brain-mask adapters, and bounded caching.
- Record per-step provenance and generate configurable QC reports and visual slice montages.

- Add bounded NIfTI and privacy-gated DICOM ingestion, canonical RAS volumes, dataset manifests and validation, and a local BraTS 2021 adapter.

- Add validated environment settings, JSON CLI logs, isolated seeded random streams, and shared application exceptions.
- Add the installed CLI entry point, pre-commit hooks, strict type checking, and Mermaid architecture diagrams.

- Organize the monorepo with shared core contracts, reserved component directories, contribution and research metadata, CI, and local CLI container tooling.

- Document planned MRI, research, application and deployment capabilities with validation gates and dependency selection criteria.

- Define the NeuroVision Intelligence Pipeline stage catalog and typed component interface.
- Add a dependency-free CLI that lists the reference stages and displays the research use notice.
- Document the architecture, implementation status, and setup commands for Bash, PowerShell, and Command Prompt.
