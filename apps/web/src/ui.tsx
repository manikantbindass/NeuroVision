import {
  ArrowRight,
  AudioWaveform,
  BrainCircuit,
  CircleHelp,
  FlaskConical,
} from "lucide-react";
import { useMemo, type ReactNode } from "react";
import { demo } from "./demo";
import { Slice } from "./viewer";
import { intensityWindow, midpoint, PLANES } from "./volume";

export function Panel({
  title,
  kicker,
  children,
  className = "",
}: {
  title: string;
  kicker?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel content-panel ${className}`}>
      {kicker && <span className="eyebrow">{kicker}</span>}
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="empty">
      <CircleHelp size={21} /> <p>{children}</p>
    </div>
  );
}

export function PageTitle({
  section,
  title,
  description,
  action,
}: {
  section: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="page-title">
      <div>
        <p className="eyebrow">
          <span className="dot" /> {section}
        </p>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}

export function Metric({
  label,
  value,
  caption,
}: {
  label: string;
  value: ReactNode;
  caption: string;
}) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{caption}</small>
    </div>
  );
}

export function downloadJson(value: unknown, filename: string) {
  downloadBlob(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
    filename,
  );
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1_000);
}

export const display = (value: number | null | undefined, digits = 3) =>
  value == null
    ? "—"
    : value.toLocaleString("en-US", { maximumFractionDigits: digits });

export function Landing({
  launch,
  tour,
}: {
  launch: () => void;
  tour: () => void;
}) {
  const phantom = demo().baseline;
  const window = useMemo(() => intensityWindow(phantom.image), [phantom.image]);
  return (
    <>
      <section className="hero panel">
        <div className="hero-copy">
          <span className="eyebrow">
            <span className="dot" /> OPEN RESEARCH WORKBENCH
          </span>
          <h1>
            A closer look
            <br />
            at the <em>human brain.</em>
          </h1>
          <p>
            Explore brain MRI, compare segmentation outputs, and trace
            uncertainty and change in one research workspace.
          </p>
          <div className="button-row">
            <button className="primary" onClick={launch}>
              Open workspace <ArrowRight size={17} />
            </button>
            <button className="secondary" onClick={tour}>
              Start guided demo
            </button>
          </div>
          <div className="hero-meta">
            <FlaskConical size={16} /> Synthetic case ready · No sign-in
            required
          </div>
        </div>
        <div className="hero-visual" aria-label="Synthetic brain MRI preview">
          <div className="hero-image">
            <Slice
              volume={phantom.image}
              point={midpoint(phantom.image)}
              plane={PLANES[0]}
              window={window}
              overlay={phantom.overlay}
              mode="labels"
              opacity={0.66}
              hidden={[]}
            />
          </div>
          <div className="image-tag">
            <AudioWaveform size={15} />
            <div>
              <strong>NV–001</strong>
              <small>AXIAL / SYNTHETIC</small>
            </div>
          </div>
          <span className="hero-index">FIG. 01 · PHANTOM</span>
        </div>
      </section>
      <div className="feature-grid">
        {[
          [
            BrainCircuit,
            "Explore anatomy",
            "Three linked planes, class overlays and a sampled 3D boundary preview.",
          ],
          [
            AudioWaveform,
            "Inspect uncertainty",
            "Read confidence and normalized entropy alongside the segmentation.",
          ],
          [
            FlaskConical,
            "Compare experiments",
            "Review atlas overlap, model masks and synthetic follow-up changes.",
          ],
        ].map(([Icon, title, copy]) => {
          const FeatureIcon = Icon as typeof BrainCircuit;
          return (
            <article className="panel feature" key={String(title)}>
              <FeatureIcon size={22} />
              <h2>{String(title)}</h2>
              <p>{String(copy)}</p>
            </article>
          );
        })}
      </div>
    </>
  );
}
