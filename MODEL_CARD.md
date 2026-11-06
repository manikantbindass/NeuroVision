# Model card

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

**MRI benchmark status: Not yet benchmarked.** Trainable architectures include the native 2D and 3D U-Nets and MONAI SegResNet, Attention U-Net, UNETR, and Swin UNETR integrations. The repository distributes no trained weights or patient images.

The baselines accept explicitly ordered channels on a common physical grid and predict mutually exclusive integer label classes. The training examples use generated ellipsoids for smoke testing. They have no human MRI training or evaluation cohort.

The model zoo provides explicit architecture configurations, tensor input contracts, a factory and registry, checkpoint reconstruction, and local inference. Backend-specific input-size and normalization constraints apply. [Model zoo comparison guide](docs/research/MODEL_ZOO.md)

Ensemble uncertainty statistics and confidence categories are experimental research outputs. Their calibration and association with MRI segmentation errors have not been established. The NeuroConfidence demo uses analytic probability phantoms. [Ensemble intelligence guide](docs/research/ENSEMBLE_INTELLIGENCE.md)

Results can be affected by sequence selection, scan quality, label mapping, spacing, preprocessing, class imbalance, domain shift, and slice or patch selection. Assess each model on a documented subject-level split, retain checkpoint provenance, and report the metric conventions with the results. [Baseline training guide](docs/development/BASELINE_TRAINING.md)
