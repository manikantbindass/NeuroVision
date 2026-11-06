# Model card

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

**MRI benchmark status: Not yet benchmarked.** This repository provides trainable 2D and 3D U-Net architectures. It distributes no trained weights or patient images.

The baselines accept explicitly ordered channels on a common physical grid and predict mutually exclusive integer label classes. The training examples use generated ellipsoids for smoke testing. They have no human MRI training or evaluation cohort.

Results can be affected by sequence selection, scan quality, label mapping, spacing, preprocessing, class imbalance, domain shift, and slice or patch selection. Assess each model on a documented subject-level split, retain checkpoint provenance, and report the metric conventions with the results. [Baseline training guide](docs/development/BASELINE_TRAINING.md)
