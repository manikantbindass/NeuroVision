import { compareMasks, demo, NOTICE } from "./demo";

export function syntheticResearchReport() {
  const fixture = demo();
  const rows = fixture.variants.map((variant) => {
    const volumes = [1, 2].map((code) => {
      const score = compareMasks(variant.labels, fixture.truth, code);
      let confidence = 0,
        entropy = 0;
      variant.labels.values.forEach((value, i) => {
        if (value === code) {
          confidence += variant.confidence.values[i];
          entropy += variant.entropy.values[i];
        }
      });
      return {
        class_id: code,
        name: variant.classes[code],
        voxel_count: score.found,
        volume_mm3: score.found * 1.4,
        dice: score.dice,
        iou: score.iou,
        mean_confidence: score.found ? confidence / score.found : null,
        mean_normalized_entropy: score.found ? entropy / score.found : null,
      };
    });
    return {
      model: variant.label,
      revision: "analytic-phantom-v1",
      classes: variant.classes,
      measurements: volumes,
    };
  });
  return {
    schema_version: 2,
    title: "AI-Assisted Research Segmentation Report",
    created_at: new Date().toISOString(),
    notice: NOTICE,
    source: "browser-generated analytic phantom",
    software: { workspace: "0.1.0" },
    scan_metadata: {
      shape: fixture.baseline.image.shape,
      spacing_mm: fixture.baseline.image.spacing,
      affine: fixture.baseline.image.affine,
      modality: "synthetic MRI-like intensities",
    },
    preprocessing_configuration: {
      method: "procedural phantom generation",
      synthetic_bias_field: true,
      applied_api_pipeline: false,
    },
    qc_findings: {
      availability: "unavailable",
      reason: "API QC was not run for this browser fixture",
    },
    experiments: rows,
    uncertainty_summary: {
      method: "binary boundary-margin score and entropy",
      calibrated: false,
      scope: "per predicted foreground class",
    },
    model_limitations: [
      "Analytic teaching masks without trained MRI weights.",
      "Metrics use a procedurally generated reference and do not establish MRI performance.",
      "Confidence is synthetic and uncalibrated.",
    ],
    reproducibility: {
      generator: "apps/web/src/demo.ts",
      version: "analytic-phantom-v1",
      randomness: "deterministic coordinate formulas",
      content_sha256: null,
    },
  };
}
