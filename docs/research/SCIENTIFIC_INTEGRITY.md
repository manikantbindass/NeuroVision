# Scientific integrity

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

Use the following wording in the README, application, generated reports, model cards, presentations, and release materials.

| Label | When to use it |
| --- | --- |
| **Benchmark pending.** | No benchmark has been run for the stated data, task, and evaluation protocol. Keep synthetic test results and software coverage separate from MRI performance. |
| **Experimental.** | A feature or research concept still needs validation. State whether it is a runnable prototype or an unimplemented proposal and describe the evidence available. |
| **Planned.** | Architecture or future work that is not yet implemented. Distinguish a deployment definition or adapter from a running integration. |

Publish Dice, accuracy, sensitivity, specificity, dataset sizes, patient counts, or performance comparisons only when traceable to actual measurements. Record the dataset and version, inclusion and split criteria, unit of analysis, evaluated count, reference labels, metric definitions, method settings, and software revision. Label synthetic data and software tests explicitly. A benchmark on one cohort does not establish superiority outside the tested conditions.

Claims about clinical or radiologist validation, FDA approval, CE certification, or peer-reviewed publication require directly applicable, verifiable evidence for the specific claim. Cite actual publications with their publication status. Cite prior work as prior work. Do not draw medical conclusions from the project's synthetic examples, confidence scores, or research reports.

See the [novelty register](NOVELTY.md) for concept evidence levels and the [final verification matrix](../development/FINAL_AUDIT.md) for measured software checks and open defects.
