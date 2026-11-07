# NeuroVision research novelty register

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

The names below identify NeuroVision workflows and research proposals. Their differences are implementation or study-design choices. The cited work provides comparison points. This register makes no claim of global uniqueness, scientific superiority, or clinical effectiveness.

**Evidence levels.** **E0 — Proposal** means there is no dedicated implementation or end-to-end evidence. **E1 — Synthetic verification** means analytic fixtures, synthetic tests, or a demo exercise the mechanism. **E2 — Retrospective MRI evaluation** requires an authorized, subject-disjoint holdout and a documented comparison. **E3 — Independent replication** requires external data or an independent team to reproduce the results. None of these levels denotes clinical approval. No concept in this register has reached E2 or E3.

## NeuroConfidence — Spatial uncertainty visualization

- **Problem addressed.** A segmentation overlay alone gives a reviewer little guidance about which regions deserve closer inspection.
- **Mechanism.** Show segmentation alongside maximum mean class probability, normalized predictive entropy and available disagreement. The experimental review mask uses configurable thresholds. The viewer renders supplied scores and does not calibrate them. [Implementation and synthetic demo](ENSEMBLE_INTELLIGENCE.md)
- **Related existing approaches.** Pixel-level uncertainty modeling and deep ensembles already provide ways to estimate or summarize prediction uncertainty. [Kendall and Gal, 2017](https://arxiv.org/abs/1703.04977) · [Lakshminarayanan et al., 2017](https://arxiv.org/abs/1612.01474)
- **Difference.** NeuroVision combines several signals with the mask and slice context in one review interface. It does not introduce an uncertainty estimator.
- **Hypothesis.** Given the same review time, an uncertainty overlay helps reviewers locate more reference-defined segmentation errors than a mask-only viewer.
- **Validation methodology.** Randomize the order of mask-only and overlay review. Use the same frozen predictions and independent reference annotations. Measure subject-level error recall, review time, and false flags. Compare individual signals and assess risk–coverage and calibration separately.
- **Limitations.** Models can agree on an error. Scores can vary by class and scanner. Background can dominate whole-volume summaries, and thresholding can obscure small regions.
- **Current evidence level.** **E1.** Synthetic probability calculations and viewer interactions are verified. No MRI calibration or reader study has been run.

## NeuroMap — Segmentation-to-atlas mapping

- **Problem addressed.** Model class IDs and voxel counts lack anatomical context across atlas versions, spatial grids, and fields of view.
- **Mechanism.** Use an authorized categorical atlas and a reviewed transform. Sample atlas labels with nearest neighbor on the segmentation grid. Report overlap, physical volume, coverage, atlas-defined hemisphere, and optional confidence summaries. [Implementation and synthetic comparison](NEUROMAP.md)
- **Related existing approaches.** FreeSurfer's automated anatomical labelling uses probabilistic information from labelled data and registration. [Fischl et al., 2002](https://pubmed.ncbi.nlm.nih.gov/11832223/)
- **Difference.** NeuroMap measures overlap between an existing segmentation and a caller-supplied atlas. The atlas association describes the segmentation in a declared reference space.
- **Hypothesis.** Explicit coverage and atlas version fields reduce reviewer errors when interpreting regional overlap and hemisphere.
- **Validation methodology.** First verify laterality, volumes, and out-of-field handling with asymmetric phantoms. Then compare region assignments and volumes against an independent reviewed reference on authorized MRI. Record overlap, absolute volume error, reviewer agreement, and sensitivity to perturbing the transform.
- **Limitations.** Results depend on registration, atlas licence and granularity, coverage, and the source mask. Overlap alone cannot establish anatomical correctness.
- **Current evidence level.** **E1.** Synthetic atlas overlap, asymmetric geometry, and physical volumetry were verified. External atlas alignment and MRI performance remain untested.

## NeuroTimeline — Longitudinal segmentation comparison

- **Problem addressed.** A volume difference can reflect acquisition or processing variation as well as structural change.
- **Mechanism.** Compare native volumes and baseline-grid gain and loss within shared coverage using a reviewed rigid transform. An optional paired-replicate protocol reports empirical spread. Mask resampling uses nearest neighbor. [Implementation and synthetic comparison](NEUROTIMELINE.md)
- **Related existing approaches.** Longitudinal FreeSurfer uses a within-subject template to reduce asymmetry and improve consistency across time points. [Reuter et al., 2012](https://pubmed.ncbi.nlm.nih.gov/22430496/)
- **Difference.** NeuroTimeline compares existing masks in a baseline-referenced workflow. It exposes native volume change separately from registered spatial differences and does not estimate a symmetric subject template.
- **Hypothesis.** Reporting shared coverage and paired-replicate spread reduces spurious change calls compared with a raw volume-difference summary.
- **Validation methodology.** Use controlled rigid perturbations and known volume changes, then evaluate authorized scan–rescan and longitudinal data. Compare raw change, the baseline-referenced workflow, and a symmetric-template baseline where compatible. Measure false change on scan–rescan, error on controlled changes, and sensitivity to reversing the reference direction.
- **Limitations.** Acquisition, field of view, registration, interpolation and segmentation can all affect apparent change. Replicate quantiles describe the sampled protocol and are not confidence intervals.
- **Current evidence level.** **E1.** Analytic translation, shared-coverage, and paired-replicate tests passed. Real longitudinal performance and automatic registration estimation remain unverified.

## NeuroConsensus — Multi-model consensus and disagreement

- **Problem addressed.** One model can hide model-specific errors, while an averaged prediction can hide where models disagree.
- **Mechanism.** The ensemble subsystem reduces aligned class probabilities using uniform or configured weights. Reversible flip TTA and supported MC dropout are optional. It retains member-level disagreement and variance alongside the combined segmentation. [Implemented ensemble subsystem](ENSEMBLE_INTELLIGENCE.md)
- **Related existing approaches.** Deep ensembles aggregate predictive distributions. STAPLE combines segmentation labels and estimates contributor performance. [Lakshminarayanan et al., 2017](https://arxiv.org/abs/1612.01474) · [Warfield et al., 2004](https://pubmed.ncbi.nlm.nih.gov/15250643/)
- **Difference.** NeuroConsensus is the research name for NeuroVision's existing probability reducer and disagreement outputs. Weights are configured by the experiment. It does not estimate contributor performance from agreement as STAPLE does.
- **Hypothesis.** On the same held-out cases, a preselected ensemble reduces segmentation error or better identifies error regions compared with its strongest single member.
- **Validation methodology.** Freeze members, weights, calibration, and augmentation on development data. Compare the strongest validation-selected member, uniform averaging, configured weights, and separate TTA and dropout ablations. Report paired subject-level Dice, HD95, calibration, error detection, runtime and peak memory.
- **Limitations.** Members must share geometry, channel order and class semantics. Correlated errors can survive aggregation. Flips can invalidate laterality, and sampling increases runtime.
- **Current evidence level.** **E1.** Synthetic weighted reduction, disagreement, and seeded sampling tests passed. No comparative MRI benchmark has been run.

## NeuroQC — MRI preprocessing quality control

- **Problem addressed.** Input or transform failures can enter the experiment before a segmentation is inspected.
- **Mechanism.** The implemented rule-based QC layer checks geometry, required modalities, blank or nonfinite data, intensity and spacing thresholds, and available registration evidence. It records provenance, warnings, and failures. Learned QC is a potential extension. [Implemented preprocessing and QC](../development/PREPROCESSING.md)
- **Related existing approaches.** MRIQC extracts image quality metrics and evaluates automatic quality prediction across acquisition sites. [Esteban et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28945803/)
- **Difference.** NeuroQC is a proposed name for QC tied to a particular preprocessing run, including checks after transforms and recorded thresholds. The current system has no trained image-quality classifier.
- **Hypothesis.** Combining checks before and after preprocessing detects more unusable outputs than intake checks alone at an acceptable false rejection rate.
- **Validation methodology.** Inject known affine, modality, intensity, and registration failures. Then compare intake-only, full rule-based, and relevant image-quality metric baselines against blinded reviewer labels on authorized, site-held-out data. Measure sensitivity by failure type and false rejection by site and protocol.
- **Limitations.** Thresholds may fail on unusual acquisitions. Foreground overlap is a proxy for registration quality. The rules cannot replace visual review or identify every motion or scanner artifact.
- **Current evidence level.** **E1** for implemented rules and provenance. **E0** for learned QC. Synthetic fixtures exercise warnings and failures, and no cross-site study has been run.

## NeuroTwin — EXPERIMENTAL subject-specific research representation

**Experimental.** This is an unimplemented research proposal.

- **Problem addressed.** A subject's segmentation, region measurements, time points and uncertainty may live in separate outputs, making their provenance difficult to reconstruct.
- **Mechanism.** Propose a versioned, research-only record that links an opaque subject key to time-point segmentations, regional volumes, longitudinal measurements, uncertainty and provenance. Declare units, coordinate frames and missing values. There is no dedicated NeuroTwin schema, service or interface.
- **Related existing approaches.** BIDS Derivatives organizes processed outputs and metadata for reuse. Longitudinal FreeSurfer provides subject-level structural measurements across visits. [BIDS Derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html) · [Reuter et al., 2012](https://pubmed.ncbi.nlm.nih.gov/22430496/)
- **Difference.** The proposed record would link NeuroVision artifacts into an auditable subject timeline. It would describe observed derivatives and their uncertainty, with no simulated or predicted subject state. Use the term subject-specific research representation.
- **Hypothesis.** A linked record reduces manual reconciliation errors and the time required to regenerate a longitudinal research summary compared with separate exports.
- **Validation methodology.** Define a schema and synthetic multi-visit fixtures. Verify round trips, units, versioning, missing visits and exact regeneration of metrics. In a controlled comparison, measure reconstruction time, provenance completeness and reconciliation errors. Review retention and access before using authorized real data.
- **Limitations.** Longitudinal linkage can increase re-identification risk. Upstream errors and uncertainty propagate to the record. Interoperability, schema stability and workflow value remain untested.
- **Current evidence level.** **E0.** Individual inputs exist as separate modules. A combined representation has neither been implemented nor evaluated.

## Shared study prerequisites

Use authorized data, verify subject-level split integrity, freeze preprocessing and analysis choices before evaluating a holdout, and report missing data and negative results. Resolve the mask resampling and training-to-API axis issues before studies that depend on those paths. No study design in this document grants dataset access or implies that validation has already occurred. [Final verification matrix](../development/FINAL_AUDIT.md) · [Data card](../../DATA_CARD.md)
