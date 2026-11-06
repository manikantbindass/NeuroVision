# Model card

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

**MRI benchmark status: Not yet benchmarked.** Trainable architectures include the native 2D and 3D U-Nets and MONAI SegResNet, Attention U-Net, UNETR, and Swin UNETR integrations. The repository distributes no trained weights or patient images.

The baselines accept explicitly ordered channels on a common physical grid and predict mutually exclusive integer label classes. The training examples use generated ellipsoids for smoke testing. They have no human MRI training or evaluation cohort.

The model zoo provides explicit architecture configurations, tensor input contracts, a factory and registry, checkpoint reconstruction, and local inference. Backend-specific input-size and normalization constraints apply. [Model zoo comparison guide](docs/research/MODEL_ZOO.md)

Ensemble uncertainty statistics and confidence categories are experimental research outputs. Their calibration and association with MRI segmentation errors have not been established. The NeuroConfidence demo uses analytic probability phantoms. [Ensemble intelligence guide](docs/research/ENSEMBLE_INTELLIGENCE.md)

Results can be affected by sequence selection, scan quality, label mapping, spacing, preprocessing, class imbalance, domain shift, and slice or patch selection. Assess each model on a documented subject-level split, retain checkpoint provenance, and report the metric conventions with the results. [Baseline training guide](docs/development/BASELINE_TRAINING.md)

## Artifact identity and reproducibility

Training now emits a run-specific model card and an `artifact.json` alongside the selected checkpoint. The record contains architecture and hyperparameters, a versioned dataset reference, preprocessing declarations, observed validation metrics, Git revision and dirty status, installed dependency versions, and content fingerprints. Missing provenance is listed in `reproducibility.json`. The static card here describes repository capabilities. Each trained artifact requires its own reviewed card.

## Data and evaluation requirements

Record the licensed dataset release, subject-level splits, modality and channel order, label ontology, preprocessing recipe, known cohort exclusions, and evaluation conventions. Report calibration and subgroup results only when evaluated. Dataset hashes detect changed inputs and cannot establish subject independence or annotation validity.

## Registry and intended use

The architecture registry selects implementations. The trained-artifact registry resolves immutable checkpoint versions and their metadata. Registration alone provides no scientific validation. Review performance on the intended research task, acquisition domain and population before enabling a model in the API. [MLOps lifecycle guide](docs/development/MLOPS.md)
