import { Box, Crosshair, Layers3, SplitSquareHorizontal } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { OverlaySet } from "./demo";
import {
  classColor,
  indexAt,
  intensityWindow,
  midpoint,
  PLANES,
  rgb,
  sameGrid,
  type Plane,
  type Point,
  type Volume,
} from "./volume";

export type OverlayMode =
  "labels" | "confidence" | "entropy" | "difference" | "none";
type Comparison = { image: Volume; labels?: Volume; title: string };

type SliceProps = {
  volume: Volume;
  point: Point;
  onPoint?: (point: Point) => void;
  plane: Plane;
  window: [number, number];
  overlay?: OverlaySet;
  comparison?: Comparison;
  mode: OverlayMode;
  opacity: number;
  hidden: number[];
  crosshair?: boolean;
};

export function Slice({
  volume,
  point,
  onPoint,
  plane,
  window,
  overlay,
  comparison,
  mode,
  opacity,
  hidden,
  crosshair = true,
}: SliceProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const columns = volume.shape[plane.horizontal];
  const rows = volume.shape[plane.vertical];
  const width = Math.min(512, columns),
    height = Math.min(512, rows);
  const aspect =
    (columns * volume.spacing[plane.horizontal]) /
    (rows * volume.spacing[plane.vertical]);
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    const frame = context.createImageData(width, height);
    const coordinate = [...point] as Point;
    for (let v = 0; v < height; v++) {
      coordinate[plane.vertical] =
        rows - 1 - Math.floor(((v + 0.5) * rows) / height);
      for (let u = 0; u < width; u++) {
        coordinate[plane.horizontal] = Math.floor(
          ((u + 0.5) * columns) / width,
        );
        const i = indexAt(volume.shape, ...coordinate);
        const gray = Math.max(
          0,
          Math.min(
            235,
            ((volume.values[i] - window[0]) / (window[1] - window[0])) * 235,
          ),
        );
        const id = overlay?.labels.values[i] ?? 0;
        let color: readonly number[] | undefined;
        let alpha = opacity;
        if (mode === "labels" && id && !hidden.includes(id))
          color = classColor(id);
        if (mode === "confidence" && id && !hidden.includes(id) && overlay) {
          color =
            overlay.confidence.values[i] >= 0.8
              ? [75, 230, 190]
              : [255, 175, 78];
        }
        if (mode === "entropy" && overlay && volume.values[i] !== 0) {
          const value = overlay.entropy.values[i];
          color = [235 + 20 * value, 72 + 134 * value, 174 - 120 * value];
          alpha *= value;
        }
        if (mode === "difference" && overlay && comparison?.labels) {
          const after = comparison.labels.values[i];
          if (after !== id) {
            if (after && !hidden.includes(after)) color = [75, 230, 190];
            else if (id && !hidden.includes(id)) color = [255, 171, 78];
          }
        }
        const offset = (v * width + u) * 4;
        for (let c = 0; c < 3; c++)
          frame.data[offset + c] = color
            ? gray * (1 - alpha) + color[c] * alpha
            : gray;
        frame.data[offset + 3] = 255;
      }
    }
    context.putImageData(frame, 0, 0);
  }, [
    volume,
    point,
    plane,
    window,
    overlay,
    comparison,
    mode,
    opacity,
    hidden,
    width,
    height,
    rows,
    columns,
  ]);
  const left = ((point[plane.horizontal] + 0.5) / columns) * 100;
  const top = (1 - (point[plane.vertical] + 0.5) / rows) * 100;
  return (
    <div className="slice-stage">
      <div
        className="slice-image"
        style={{ aspectRatio: Math.max(0.35, Math.min(2.85, aspect)) }}
        onPointerDown={
          onPoint
            ? (event) => {
                const box = event.currentTarget.getBoundingClientRect();
                const position = [...point] as Point;
                position[plane.horizontal] = Math.max(
                  0,
                  Math.min(
                    columns - 1,
                    Math.floor(
                      ((event.clientX - box.left) / box.width) * columns,
                    ),
                  ),
                );
                position[plane.vertical] =
                  rows -
                  1 -
                  Math.max(
                    0,
                    Math.min(
                      rows - 1,
                      Math.floor(
                        ((event.clientY - box.top) / box.height) * rows,
                      ),
                    ),
                  );
                onPoint(position);
              }
            : undefined
        }
      >
        <canvas
          ref={canvas}
          width={width}
          height={height}
          role="img"
          aria-label={`${plane.name} MRI slice ${point[plane.axis] + 1}`}
        />
        {crosshair && (
          <>
            <span className="crosshair-v" style={{ left: `${left}%` }} />
            <span className="crosshair-h" style={{ top: `${top}%` }} />
          </>
        )}
      </div>
      {crosshair && (
        <>
          <span className="orientation top">{plane.top}</span>
          <span className="orientation bottom">{plane.bottom}</span>
          <span className="orientation left">{plane.left}</span>
          <span className="orientation right">{plane.right}</span>
        </>
      )}
    </div>
  );
}

function Surface({ volume, hidden }: { volume: Volume; hidden: number[] }) {
  const [angle, setAngle] = useState(35);
  const [tilt, setTilt] = useState(22);
  const ref = useRef<HTMLCanvasElement>(null);
  const points = useMemo(() => {
    const result: { xyz: Point; id: number }[] = [];
    const shape = volume.shape;
    const step = Math.max(1, Math.ceil(Math.max(...shape) / 64));
    const at = (x: number, y: number, z: number) =>
      x < 0 || y < 0 || z < 0 || x >= shape[0] || y >= shape[1] || z >= shape[2]
        ? 0
        : volume.values[indexAt(shape, x, y, z)];
    const scale = Math.max(...shape.map((n, i) => n * volume.spacing[i]));
    for (let z = 0; z < shape[2]; z += step)
      for (let y = 0; y < shape[1]; y += step)
        for (let x = 0; x < shape[0]; x += step) {
          const id = at(x, y, z);
          if (!id || hidden.includes(id)) continue;
          if (
            [
              at(x - step, y, z),
              at(x + step, y, z),
              at(x, y - step, z),
              at(x, y + step, z),
              at(x, y, z - step),
              at(x, y, z + step),
            ].every((other) => other === id)
          )
            continue;
          result.push({
            xyz: [x, y, z].map(
              (p, i) => ((p - shape[i] / 2) * volume.spacing[i]) / scale,
            ) as Point,
            id,
          });
        }
    return result;
  }, [volume, hidden]);
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, 360, 250);
    const a = (angle * Math.PI) / 180,
      b = (tilt * Math.PI) / 180;
    const projected = points
      .map(({ xyz: [x, y, z], id }) => {
        const xx = x * Math.cos(a) - y * Math.sin(a);
        const yy = x * Math.sin(a) + y * Math.cos(a);
        return {
          x: xx,
          y: z * Math.cos(b) - yy * Math.sin(b),
          depth: yy * Math.cos(b) + z * Math.sin(b),
          id,
        };
      })
      .sort((a, b) => a.depth - b.depth);
    for (const point of projected) {
      ctx.globalAlpha = Math.max(
        0.25,
        Math.min(0.9, 0.65 + point.depth * 0.45),
      );
      ctx.fillStyle = rgb(classColor(point.id));
      ctx.fillRect(180 + point.x * 205, 125 - point.y * 205, 3, 3);
    }
    ctx.globalAlpha = 1;
  }, [angle, tilt, points]);
  return (
    <div className="surface">
      <canvas
        width={360}
        height={250}
        ref={ref}
        role="img"
        aria-label="Projected 3D segmentation boundary voxels"
      />
      <div className="surface-controls">
        <label>
          Rotation{" "}
          <input
            type="range"
            min="0"
            max="360"
            value={angle}
            onChange={(e) => setAngle(Number(e.target.value))}
          />
        </label>
        <label>
          Tilt{" "}
          <input
            type="range"
            min="-90"
            max="90"
            value={tilt}
            onChange={(e) => setTilt(Number(e.target.value))}
          />
        </label>
      </div>
      <p className="muted">Sampled boundary voxels · projected preview</p>
    </div>
  );
}

export function Viewer({
  volume,
  overlay,
  comparison,
  initialMode = "labels",
}: {
  volume: Volume;
  overlay?: OverlaySet;
  comparison?: Comparison;
  initialMode?: OverlayMode;
}) {
  const [point, setPoint] = useState(() => midpoint(volume));
  const [mode, setMode] = useState<OverlayMode>(initialMode);
  const [opacity, setOpacity] = useState(0.63);
  const [hidden, setHidden] = useState<number[]>([]);
  const [display, setDisplay] = useState<"planes" | "compare" | "3d">("planes");
  const window = useMemo(() => intensityWindow(volume), [volume]);
  const aligned =
    comparison &&
    sameGrid(volume, comparison.image) &&
    (!comparison.labels || sameGrid(volume, comparison.labels))
      ? comparison
      : undefined;
  const safeOverlay =
    overlay &&
    [overlay.labels, overlay.confidence, overlay.entropy].every((item) =>
      sameGrid(volume, item),
    )
      ? overlay
      : undefined;
  const props = {
    volume,
    point,
    onPoint: setPoint,
    window,
    overlay: safeOverlay,
    comparison: aligned,
    mode,
    opacity,
    hidden,
  };
  const setAxis = (axis: number, value: number) =>
    setPoint((current) => {
      const next = [...current] as Point;
      next[axis] = value;
      return next;
    });
  return (
    <section className="viewer panel" aria-label="MRI viewer">
      <div className="viewer-heading">
        <div>
          <span className="eyebrow">
            <Crosshair size={14} /> Image viewer
          </span>
          <h2>Synchronized slices</h2>
        </div>
        <div className="segmented" aria-label="Viewer layout">
          <button
            aria-pressed={display === "planes"}
            onClick={() => setDisplay("planes")}
          >
            <Layers3 size={15} /> 2D
          </button>
          {aligned && (
            <button
              aria-pressed={display === "compare"}
              onClick={() => setDisplay("compare")}
            >
              <SplitSquareHorizontal size={15} /> Compare
            </button>
          )}
          {safeOverlay && (
            <button
              aria-pressed={display === "3d"}
              onClick={() => setDisplay("3d")}
            >
              <Box size={15} /> 3D
            </button>
          )}
        </div>
      </div>
      {display === "planes" && (
        <div className="view-grid">
          {PLANES.map((plane) => (
            <figure className="view-pane" key={plane.name}>
              <figcaption>
                <span>{plane.name}</span>
                <code>
                  {point[plane.axis] + 1} / {volume.shape[plane.axis]}
                </code>
              </figcaption>
              <Slice {...props} plane={plane} />
              <label className="slice-range">
                <span className="sr-only">{plane.name} slice</span>
                <input
                  type="range"
                  min="0"
                  max={volume.shape[plane.axis] - 1}
                  value={point[plane.axis]}
                  onChange={(e) => setAxis(plane.axis, Number(e.target.value))}
                />
              </label>
            </figure>
          ))}
        </div>
      )}
      {display === "compare" && aligned && (
        <div className="compare-wrap">
          <div className="view-grid compare-grid">
            <figure className="view-pane">
              <figcaption>Before · baseline</figcaption>
              <Slice {...props} plane={PLANES[0]} mode="none" />
            </figure>
            <figure className="view-pane">
              <figcaption>After · {aligned.title}</figcaption>
              <Slice
                {...props}
                volume={aligned.image}
                plane={PLANES[0]}
                mode={mode === "difference" ? mode : "none"}
              />
            </figure>
          </div>
          <label className="compare-range">
            Linked axial slice{" "}
            <input
              type="range"
              min="0"
              max={volume.shape[2] - 1}
              value={point[2]}
              onChange={(e) => setAxis(2, Number(e.target.value))}
            />
          </label>
        </div>
      )}
      {display === "3d" && safeOverlay && (
        <Surface volume={safeOverlay.labels} hidden={hidden} />
      )}
      {safeOverlay && (
        <div className="viewer-controls">
          <label>
            Overlay{" "}
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as OverlayMode)}
            >
              <option value="labels">Segmentation</option>
              <option value="confidence">Confidence</option>
              <option value="entropy">Uncertainty heatmap</option>
              {aligned?.labels && (
                <option value="difference">Gains and losses</option>
              )}
              <option value="none">Off</option>
            </select>
          </label>
          <label>
            Opacity{" "}
            <input
              type="range"
              min="0"
              max="100"
              value={Math.round(opacity * 100)}
              onChange={(e) => setOpacity(Number(e.target.value) / 100)}
            />
            <output>{Math.round(opacity * 100)}%</output>
          </label>
          <div className="class-toggles" aria-label="Visible classes">
            {safeOverlay.classes.slice(1).map((name, index) => {
              const id = index + 1;
              return (
                <label key={id}>
                  <input
                    type="checkbox"
                    checked={!hidden.includes(id)}
                    onChange={() =>
                      setHidden((current) =>
                        current.includes(id)
                          ? current.filter((code) => code !== id)
                          : [...current, id],
                      )
                    }
                  />
                  <span
                    className="swatch"
                    style={{ background: rgb(classColor(id)) }}
                  />
                  {name}
                </label>
              );
            })}
          </div>
        </div>
      )}
      {mode === "confidence" && safeOverlay && (
        <p className="viewer-note">
          Mint ≥ 0.80 · Amber &lt; 0.80 · Uncalibrated model score
        </p>
      )}
      {mode === "entropy" && safeOverlay && (
        <p className="viewer-note">
          Brighter warm colors indicate higher normalized entropy.
        </p>
      )}
      {mode === "difference" && safeOverlay && (
        <p className="viewer-note">
          Mint: gained or changed class · Amber: lost class
        </p>
      )}
      {overlay && !safeOverlay && (
        <p className="viewer-note">
          Overlay hidden because the MRI and prediction grids differ.
        </p>
      )}
      {comparison && !aligned && (
        <p className="viewer-note">Comparison requires matching voxel grids.</p>
      )}
      <div className="viewer-footer">
        <span>
          <Crosshair size={14} /> Voxel {point.join(" / ")}
        </span>
        <span>
          {volume.shape.join(" × ")} voxels ·{" "}
          {volume.spacing.map((value) => Number(value.toFixed(2))).join(" × ")}{" "}
          grid spacing
        </span>
      </div>
    </section>
  );
}
