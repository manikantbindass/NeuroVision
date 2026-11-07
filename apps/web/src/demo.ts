import { indexAt, type Point, type Volume } from "./volume";

export type OverlaySet = {
  id: string;
  label: string;
  labels: Volume;
  confidence: Volume;
  entropy: Volume;
  classes: string[];
};
export type Case = {
  image: Volume;
  name: string;
  source: "synthetic" | "api";
  id?: string;
  overlay?: OverlaySet;
};

export const NOTICE =
  "Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.";
export const VARIATION =
  "Observed differences may arise from acquisition, preprocessing, registration, model, or segmentation variation.";

let cached: ReturnType<typeof createDemo> | undefined;
export function demo() {
  return (cached ??= createDemo());
}

function createDemo() {
  const shape: Point = [84, 92, 64];
  const spacing: Point = [1, 1, 1.4];
  const affine = [
    [1, 0, 0, -42],
    [0, 1, 0, -46],
    [0, 0, 1.4, -44.8],
    [0, 0, 0, 1],
  ];
  const count = shape[0] * shape[1] * shape[2];
  const volume = (values: Volume["values"]): Volume => ({
    shape,
    spacing,
    affine,
    values,
  });
  const image = new Float32Array(count);
  const truth = new Uint16Array(count);
  const atlas = new Uint16Array(count);
  const processed = new Float32Array(count);
  const followup = new Float32Array(count);
  const followLabels = new Uint16Array(count);
  const labels = [new Uint16Array(count), new Uint16Array(count)];
  const confidences = [new Float32Array(count), new Float32Array(count)];
  const entropies = [new Float32Array(count), new Float32Array(count)];
  const classes = ["Background", "Synthetic lesion", "Ventricular phantom"];
  const ellipsoid = (
    x: number,
    y: number,
    z: number,
    a: number,
    b: number,
    c: number,
  ) => (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2;
  for (let z = 0; z < shape[2]; z++) {
    for (let y = 0; y < shape[1]; y++) {
      for (let x = 0; x < shape[0]; x++) {
        const i = indexAt(shape, x, y, z);
        const u = (x - 41.5) / 39,
          v = (y - 45.5) / 42,
          w = (z - 31.5) / 29;
        const outer = ellipsoid(u, v, w, 0.98, 1.01, 1.01);
        const brain = ellipsoid(u, v + 0.015, w, 0.86, 0.93, 0.88);
        const fissure =
          Math.abs(u) < 0.018 + 0.012 * Math.sin(v * 11) && Math.abs(v) > 0.24;
        const gyral =
          0.036 * Math.sin(u * 28 + v * 17 + w * 8) +
          0.026 * Math.sin(v * 32 - u * 13) * Math.cos(w * 17);
        const vent = ellipsoid(
          Math.abs(u) - 0.17,
          v + 0.04 + 0.18 * Math.abs(u),
          w + 0.02,
          0.088,
          0.34,
          0.17,
        );
        const lesion = ellipsoid(
          u - 0.39,
          v - 0.14,
          w - 0.04,
          0.184,
          0.203,
          0.26,
        );
        const next = ellipsoid(
          u - 0.37,
          v - 0.12,
          w - 0.04,
          0.221,
          0.235,
          0.28,
        );
        const noise = Math.sin(x * 47.2 + y * 21.8 + z * 67.9) * 0.022;
        let intensity = 0;
        if (outer < 1) intensity = outer > 0.94 ? 0.44 + noise : 0.12 + noise;
        if (brain < 1) {
          const cortex = brain > 0.69 + gyral * 2;
          intensity = (cortex ? 0.62 : 0.76) + gyral * 2 + noise;
          if (
            cortex &&
            Math.sin(Math.atan2(v, u) * 17 + w * 9 + brain * 12) > 0.8
          )
            intensity *= 0.62;
          if (fissure || vent < 1) intensity = 0.21 + noise;
          atlas[i] = (u < 0 ? 1 : 2) + (v < 0 ? 2 : 0);
        }
        const signal = Math.max(
          0,
          intensity + (lesion < 1.45 ? 0.3 * Math.exp(-lesion) : 0),
        );
        image[i] = signal * (1 + 0.22 * u + 0.1 * v);
        processed[i] = signal;
        followup[i] = Math.max(
          0,
          intensity + (next < 1.45 ? 0.3 * Math.exp(-next) : 0),
        );
        truth[i] = lesion < 1 ? 1 : vent < 1 ? 2 : 0;
        followLabels[i] = next < 1 ? 1 : vent < 1 ? 2 : 0;
        for (let m = 0; m < 2; m++) {
          const score =
            m === 0
              ? lesion + gyral * 3 - 0.04
              : ellipsoid(u - 0.43, v - 0.12, w - 0.04, 0.155, 0.2, 0.24) +
                gyral * 4;
          labels[m][i] = score < 1 ? 1 : vent < (m === 0 ? 0.95 : 1.22) ? 2 : 0;
          const margin = Math.min(Math.abs(score - 1), Math.abs(vent - 1));
          const p = 0.5 + 0.499 * (1 - Math.exp(-3.3 * margin));
          confidences[m][i] = p;
          entropies[m][i] = -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
        }
      }
    }
  }
  const variants: OverlaySet[] = labels.map((values, i) => ({
    id: `phantom-${i ? "b" : "a"}`,
    label: i ? "Analytic mask B" : "Analytic mask A",
    labels: volume(values),
    confidence: volume(confidences[i]),
    entropy: volume(entropies[i]),
    classes,
  }));
  const baseline: Case = {
    image: volume(image),
    name: "NV-001 · Synthetic brain phantom",
    source: "synthetic",
    overlay: variants[0],
  };
  return {
    baseline,
    processed: volume(processed),
    variants,
    truth: volume(truth),
    atlas: volume(atlas),
    followup: volume(followup),
    followLabels: volume(followLabels),
  };
}

export function compareMasks(prediction: Volume, target: Volume, code: number) {
  let found = 0,
    expected = 0,
    overlap = 0;
  for (let i = 0; i < prediction.values.length; i++) {
    const a = prediction.values[i] === code;
    const b = target.values[i] === code;
    found += Number(a);
    expected += Number(b);
    overlap += Number(a && b);
  }
  return {
    found,
    expected,
    gained: found - overlap,
    lost: expected - overlap,
    dice: found + expected ? (2 * overlap) / (found + expected) : null,
    iou:
      found + expected - overlap
        ? overlap / (found + expected - overlap)
        : null,
  };
}
