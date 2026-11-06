import { ArrowRight, Download, FileUp, Play, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import type { Api, Maps, Metrics, Model, QC, Report, Scan } from "./api";
import {
  compareMasks,
  demo,
  NOTICE,
  VARIATION,
  type Case,
  type OverlaySet,
} from "./demo";
import {
  display,
  downloadBlob,
  downloadJson,
  Empty,
  Metric,
  Panel,
} from "./ui";
import { Viewer } from "./viewer";
import { syntheticResearchReport } from "./report";

export function Upload({
  connected,
  busy,
  onConnect,
  onUpload,
}: {
  connected: boolean;
  busy: boolean;
  onConnect: () => void;
  onUpload: (file: File, modality: string, role: string) => void;
}) {
  const [file, setFile] = useState<File>();
  const [modality, setModality] = useState("t1");
  const [role, setRole] = useState("image");
  const [reviewed, setReviewed] = useState(false);
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (file && reviewed && connected) onUpload(file, modality, role);
  };
  return (
    <div className="page-grid">
      <Panel title="Add a NIfTI volume" kicker="SECURE RESEARCH INGEST">
        {!connected && (
          <div className="inline-alert">
            Connect the local API to upload a reviewed scan.{" "}
            <button className="text-button" onClick={onConnect}>
              Connect API <ArrowRight size={14} />
            </button>
          </div>
        )}
        <form className="form-stack" onSubmit={submit}>
          <label className="file-field">
            <FileUp size={26} />
            <strong>{file ? file.name : "Choose a NIfTI file"}</strong>
            <span>.nii or .nii.gz · Maximum 16 MiB</span>
            <input
              type="file"
              aria-label="NIfTI file"
              accept=".nii,.gz"
              onChange={(e) => {
                setFile(e.target.files?.[0]);
                setReviewed(false);
              }}
            />
          </label>
          <div className="field-row">
            <label>
              Modality{" "}
              <select
                value={modality}
                onChange={(e) => setModality(e.target.value)}
              >
                {["t1", "t1ce", "t2", "flair", "unknown"].map((value) => (
                  <option key={value} value={value}>
                    {value.toUpperCase()}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Input role{" "}
              <select value={role} onChange={(e) => setRole(e.target.value)}>
                <option value="image">MRI image</option>
                <option value="labels">Reference or atlas labels</option>
              </select>
            </label>
          </div>
          <label className="check-field">
            <input
              type="checkbox"
              checked={reviewed}
              onChange={(e) => setReviewed(e.target.checked)}
            />
            I confirm that this volume was reviewed and de-identified for
            authorized research use.
          </label>
          <button
            type="submit"
            className="primary"
            disabled={!file || !reviewed || !connected || busy}
          >
            <FileUp size={16} /> Upload volume
          </button>
        </form>
      </Panel>
      <Panel title="Input requirements">
        <ul className="guidance">
          <li>Use a single 3D scalar MRI or a categorical label map.</li>
          <li>Upload labels separately from the source MRI.</li>
          <li>
            The server validates file size, decoded size, geometry and voxel
            data.
          </li>
          <li>
            The application does not remove identifying information from the
            pixels. Complete privacy review before upload.
          </li>
        </ul>
      </Panel>
    </div>
  );
}

export function Datasets({
  scans,
  connected,
  busy,
  load,
  open,
  refresh,
}: {
  scans: Scan[];
  connected: boolean;
  busy: boolean;
  load: () => void;
  open: (scan: Scan) => void;
  refresh: () => void;
}) {
  return (
    <>
      <Panel title="NeuroVision synthetic phantom" kicker="BUILT-IN CASE">
        <p>
          Generated 3D MRI-like intensities, two analytic masks, a reference
          label map, a four-region phantom atlas and a simulated follow-up.
        </p>
        <div className="chip-row">
          <span>84 × 92 × 64</span>
          <span>2 foreground classes</span>
          <span>2 time points</span>
        </div>
        <button className="primary" onClick={load}>
          Open synthetic case <ArrowRight size={16} />
        </button>
      </Panel>
      <Panel title="API scan library" kicker="YOUR RESEARCH SESSION">
        {connected && (
          <button className="secondary" disabled={busy} onClick={refresh}>
            <RefreshCw size={15} /> Refresh scans
          </button>
        )}
        {!scans.length ? (
          <Empty>
            {connected
              ? "Upload a volume to populate the library."
              : "Connect the local API to list your uploaded volumes."}
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Role</th>
                  <th>Modality</th>
                  <th>Grid</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {scans.map((scan) => (
                  <tr key={scan.id}>
                    <td>
                      <code>{scan.id.slice(0, 8)}</code>
                    </td>
                    <td>{scan.role}</td>
                    <td>{scan.modality.toUpperCase()}</td>
                    <td>{scan.shape.join(" × ")}</td>
                    <td>
                      {scan.role === "image" ? (
                        <button
                          className="text-button"
                          disabled={busy}
                          onClick={() => open(scan)}
                        >
                          Open MRI
                        </button>
                      ) : (
                        "Available as reference"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </>
  );
}

export function Preprocessing({
  current,
  processed,
  qc,
  busy,
  run,
}: {
  current: Case;
  processed?: Case;
  qc?: QC;
  busy: boolean;
  run: (normalization: string) => void;
}) {
  const [recipe, setRecipe] = useState("zscore");
  if (current.source === "synthetic")
    return (
      <>
        <div className="metrics-grid">
          <Metric
            label="Case input"
            value="Synthetic"
            caption="Known analytic geometry"
          />
          <Metric
            label="Voxel spacing"
            value="1 × 1 × 1.4"
            caption="Millimetres"
          />
          <Metric
            label="Comparison"
            value="Bias field"
            caption="Known gradient removed in demo"
          />
        </div>
        <p className="inline-alert">
          This teaching example compares a synthetic bias gradient with the
          original phantom signal. It is separate from the live API
          normalization recipe.
        </p>
        <Viewer
          key="demo-qc"
          volume={current.image}
          comparison={{
            image: demo().processed,
            title: "synthetic bias removed",
          }}
        />
      </>
    );
  return (
    <>
      <Panel title="Run preprocessing and QC">
        <div className="field-row">
          <label>
            Intensity recipe{" "}
            <select value={recipe} onChange={(e) => setRecipe(e.target.value)}>
              <option value="zscore">Z-score</option>
              <option value="percentile">Percentile</option>
              <option value="none">QC only</option>
            </select>
          </label>
          <button
            className="primary"
            disabled={busy}
            onClick={() => run(recipe)}
          >
            <Play size={16} /> Queue preprocessing
          </button>
        </div>
        {qc && (
          <div className="result-block">
            <strong>QC status: {qc.provenance.qc.status}</strong>
            {qc.provenance.qc.findings.length ? (
              <ul>
                {qc.provenance.qc.findings.map((finding, i) => (
                  <li key={`${finding.code}-${i}`}>
                    {finding.severity}: {finding.message}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No findings in the selected QC policy.</p>
            )}
          </div>
        )}
      </Panel>
      <Viewer
        key={processed?.id ?? current.id}
        volume={current.image}
        comparison={
          processed
            ? { image: processed.image, title: "preprocessed" }
            : undefined
        }
      />
    </>
  );
}

export function Studio({
  current,
  models,
  processed,
  busy,
  run,
}: {
  current: Case;
  models: Model[];
  processed?: Case;
  busy: boolean;
  run: (model: string, processed: boolean) => void;
}) {
  const [selection, setSelection] = useState("");
  const [useProcessed, setUseProcessed] = useState(false);
  const selected = selection || models[0]?.id || "";
  const [demoModel, setDemoModel] = useState(0);
  const overlay =
    current.source === "synthetic"
      ? demo().variants[demoModel]
      : current.overlay;
  return (
    <>
      <Panel
        title="Segmentation run"
        kicker={
          current.source === "synthetic"
            ? "SYNTHETIC TEACHING MASKS"
            : "API MODEL REGISTRY"
        }
      >
        {current.source === "synthetic" ? (
          <label className="narrow">
            Analytic mask{" "}
            <select
              value={demoModel}
              onChange={(e) => setDemoModel(Number(e.target.value))}
            >
              {demo().variants.map((variant, index) => (
                <option key={variant.id} value={index}>
                  {variant.label}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <div className="field-row">
            <label>
              Model{" "}
              <select
                value={selected}
                onChange={(e) => setSelection(e.target.value)}
              >
                {!models.length && <option value="">No enabled models</option>}
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.id}
                    {model.synthetic ? " · synthetic fixture" : ""}
                  </option>
                ))}
              </select>
            </label>
            {processed && (
              <label className="check-field">
                <input
                  type="checkbox"
                  checked={useProcessed}
                  onChange={(e) => setUseProcessed(e.target.checked)}
                />{" "}
                Use preprocessed volume
              </label>
            )}
            <button
              className="primary"
              disabled={busy || !selected}
              onClick={() => run(selected, useProcessed)}
            >
              <Play size={16} /> Queue inference
            </button>
          </div>
        )}
      </Panel>
      <Viewer
        key={overlay?.id ?? current.id}
        volume={current.image}
        overlay={overlay}
      />
    </>
  );
}

export function ModelComparison({
  current,
  runs,
}: {
  current: Case;
  runs: OverlaySet[];
}) {
  const variants = current.source === "synthetic" ? demo().variants : runs;
  const [index, setIndex] = useState(0);
  const chosen = variants[index] ?? variants[0];
  const synthetic = current.source === "synthetic";
  const baseline = variants[0];
  const compatible =
    chosen &&
    baseline &&
    chosen.id !== baseline.id &&
    chosen.classes.join("\0") === baseline.classes.join("\0");
  return (
    <>
      <Panel
        title="Prediction runs"
        kicker={synthetic ? "ANALYTIC DEMONSTRATION" : "CURRENT CASE"}
      >
        {!variants.length ? (
          <Empty>
            Run inference in Segmentation Studio to compare predictions for this
            case.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Classes</th>
                  <th>Reference Dice · class 1</th>
                  <th>Viewer</th>
                </tr>
              </thead>
              <tbody>
                {variants.map((variant, i) => (
                  <tr key={variant.id}>
                    <td>
                      {variant.label}{" "}
                      <small>{synthetic ? "" : variant.id.slice(0, 8)}</small>
                    </td>
                    <td>{variant.classes.length - 1}</td>
                    <td>
                      {synthetic
                        ? display(
                            compareMasks(variant.labels, demo().truth, 1).dice,
                          )
                        : "Use Metrics with a reference"}
                    </td>
                    <td>
                      <button
                        className="text-button"
                        aria-pressed={chosen?.id === variant.id}
                        onClick={() => setIndex(i)}
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {synthetic && (
          <p className="muted">
            Scores are computed against the generated phantom reference. Real
            MRI metrics are not yet benchmarked.
          </p>
        )}
      </Panel>
      {chosen && (
        <Viewer
          key={chosen.id}
          volume={current.image}
          overlay={compatible ? baseline : chosen}
          comparison={
            compatible
              ? {
                  image: current.image,
                  labels: chosen.labels,
                  title: chosen.label,
                }
              : undefined
          }
          initialMode={compatible ? "difference" : "labels"}
        />
      )}
    </>
  );
}

export function Confidence({ current }: { current: Case }) {
  const scores = useMemo(() => {
    if (!current.overlay) return null;
    let count = 0,
      high = 0,
      entropy = 0;
    current.overlay.labels.values.forEach((code, index) => {
      if (!code || !current.overlay) return;
      count++;
      high += Number(current.overlay.confidence.values[index] >= 0.8);
      entropy += current.overlay.entropy.values[index];
    });
    return { count, high, entropy: count ? entropy / count : null };
  }, [current.overlay]);
  return (
    <>
      <p className="inline-alert">
        Model scores are uncalibrated. A high-confidence color does not
        establish correctness or clinical certainty.
      </p>
      {scores ? (
        <div className="metrics-grid">
          <Metric
            label="Foreground voxels"
            value={display(scores.count, 0)}
            caption="Predicted labels above background"
          />
          <Metric
            label="At or above 0.80"
            value={
              scores.count
                ? `${display((scores.high / scores.count) * 100, 1)}%`
                : "—"
            }
            caption="Foreground confidence threshold"
          />
          <Metric
            label="Mean entropy"
            value={display(scores.entropy)}
            caption="Normalized foreground score"
          />
        </div>
      ) : (
        <Empty>Run a segmentation to inspect confidence and entropy.</Empty>
      )}
      <Viewer
        key={current.overlay?.id ?? current.id}
        volume={current.image}
        overlay={current.overlay}
        initialMode="confidence"
      />
    </>
  );
}

export function MapPage({
  current,
  scans,
  result,
  busy,
  run,
}: {
  current: Case;
  scans: Scan[];
  result?: Maps;
  busy: boolean;
  run: (payload: object) => void;
}) {
  const [atlas, setAtlas] = useState("");
  const [reviewed, setReviewed] = useState(false);
  const [formError, setFormError] = useState("");
  const rows = useMemo(() => {
    if (current.source !== "synthetic") return result?.measurements ?? [];
    const fixture = demo();
    const labels = fixture.baseline.overlay!.labels.values;
    return ["Synthetic lesion", "Ventricular phantom"].map((name, index) => {
      const code = index + 1;
      const bins = [0, 0, 0, 0, 0];
      let count = 0;
      labels.forEach((label, i) => {
        if (label === code) {
          bins[fixture.atlas.values[i]]++;
          count++;
        }
      });
      const names = [
        "Outside atlas",
        "Left anterior",
        "Right anterior",
        "Left posterior",
        "Right posterior",
      ];
      return {
        code,
        name,
        voxels: count,
        volume_mm3: count * 1.4,
        hemisphere: bins[0]
          ? "unknown"
          : bins[1] + bins[3] && bins[2] + bins[4]
            ? "bilateral"
            : bins[1] + bins[3]
              ? "left"
              : "right",
        associations: bins.slice(1).flatMap((n, i) =>
          n
            ? [
                {
                  region: { name: names[i + 1] },
                  fraction_of_segment: n / count,
                },
              ]
            : [],
        ),
      };
    });
  }, [current.source, result]);
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!atlas || !reviewed || !current.overlay) return;
    setFormError("");
    const data = new FormData(event.currentTarget);
    try {
      run({
        segmentation_id: current.overlay.id,
        atlas_scan_id: atlas,
        atlas_name: data.get("name"),
        atlas_version: data.get("version"),
        atlas_license: data.get("license"),
        atlas_space: data.get("space"),
        registration_provenance: data.get("provenance"),
        segmentation_to_atlas_world: JSON.parse(String(data.get("transform"))),
        regions: JSON.parse(String(data.get("regions"))),
      });
    } catch {
      setFormError("Enter valid JSON for the affine and region table.");
    }
  };
  return (
    <>
      {current.source === "api" && (
        <Panel title="Map against an uploaded label atlas">
          {!current.overlay ? (
            <Empty>Run segmentation before mapping regions.</Empty>
          ) : (
            <form onSubmit={submit} className="form-stack">
              <label>
                Atlas label map{" "}
                <select
                  required
                  value={atlas}
                  onChange={(e) => {
                    setAtlas(e.target.value);
                    setReviewed(false);
                  }}
                >
                  <option value="">Choose uploaded label map</option>
                  {scans
                    .filter((scan) => scan.role === "labels")
                    .map((scan) => (
                      <option key={scan.id} value={scan.id}>
                        {scan.id.slice(0, 8)} · {scan.shape.join(" × ")}
                      </option>
                    ))}
                </select>
              </label>
              <div className="field-row">
                {[
                  ["name", "Atlas name"],
                  ["version", "Version"],
                  ["license", "License"],
                  ["space", "Reference space"],
                ].map(([name, title]) => (
                  <label key={name}>
                    {title}
                    <input name={name} maxLength={160} required />
                  </label>
                ))}
              </div>
              <label>
                Registration QC reference{" "}
                <input name="provenance" maxLength={160} required />
              </label>
              <label>
                Segmentation-to-atlas RAS+ affine (JSON){" "}
                <textarea
                  name="transform"
                  required
                  defaultValue="[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"
                />
              </label>
              <label>
                Atlas regions (JSON){" "}
                <textarea
                  name="regions"
                  required
                  placeholder='[{"code":1,"name":"Region A","hemisphere":"left"}]'
                />
              </label>
              <label className="check-field">
                <input
                  type="checkbox"
                  checked={reviewed}
                  onChange={(e) => setReviewed(e.target.checked)}
                />
                I reviewed the transform and am authorized to use the atlas.
              </label>
              {formError && <p role="alert">{formError}</p>}
              <button
                className="primary"
                disabled={!reviewed || busy || !atlas}
                type="submit"
              >
                Queue NeuroMap
              </button>
            </form>
          )}
        </Panel>
      )}
      <Panel
        title="Regional overlap"
        kicker={
          current.source === "synthetic"
            ? "SYNTHETIC ATLAS · SHARED GRID"
            : "NEUROMAP RESULTS"
        }
      >
        {rows.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Region</th>
                  <th>Voxels</th>
                  <th>Volume mm³</th>
                  <th>Hemisphere</th>
                  <th>Atlas overlap</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.code}>
                    <td>{row.name}</td>
                    <td>{display(row.voxels, 0)}</td>
                    <td>{display(row.volume_mm3, 1)}</td>
                    <td>{row.hemisphere}</td>
                    <td>
                      {row.associations
                        .map(
                          (a) =>
                            `${a.region.name} ${display(a.fraction_of_segment * 100, 1)}%`,
                        )
                        .join(" · ") || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>No atlas measurements available for this case.</Empty>
        )}
      </Panel>
      <Viewer
        key={current.overlay?.id ?? current.id}
        volume={current.image}
        overlay={current.overlay}
      />
    </>
  );
}

export function Timeline() {
  const fixture = demo();
  const delta = useMemo(
    () => compareMasks(fixture.followLabels, fixture.truth, 1),
    [fixture],
  );
  const referenceOverlay = { ...fixture.variants[0], labels: fixture.truth };
  return (
    <>
      <p className="inline-alert">{VARIATION}</p>
      <div className="metrics-grid">
        <Metric
          label="Day 0 · lesion"
          value={display(delta.expected * 1.4, 1)}
          caption="Baseline mm³"
        />
        <Metric
          label="Day 90 · lesion"
          value={display(delta.found * 1.4, 1)}
          caption="Follow-up mm³"
        />
        <Metric
          label="Native change"
          value={`+${display((delta.found - delta.expected) * 1.4, 1)}`}
          caption={`mm³ · +${display((delta.found / delta.expected - 1) * 100, 1)}%`}
        />
      </div>
      <Viewer
        volume={fixture.baseline.image}
        overlay={referenceOverlay}
        initialMode="difference"
        comparison={{
          image: fixture.followup,
          labels: fixture.followLabels,
          title: "day 90",
        }}
      />
      <p className="muted">
        The two phantom visits share a known grid. Live longitudinal comparisons
        use the Python NeuroTimeline workflow.
      </p>
    </>
  );
}

export function MetricsPage({
  current,
  scans,
  result,
  run,
  busy,
}: {
  current: Case;
  scans: Scan[];
  result?: Metrics;
  run: (id: string) => void;
  busy: boolean;
}) {
  const [target, setTarget] = useState("");
  const rows = useMemo(
    () =>
      current.source === "synthetic"
        ? demo().variants.flatMap((variant) =>
            [1, 2].map((label) => ({
              key: `${variant.id}-${label}`,
              name: variant.label,
              region: variant.classes[label],
              ...compareMasks(variant.labels, demo().truth, label),
              hd95_mm: null,
            })),
          )
        : (result?.classes ?? []).map((row) => ({
            key: String(row.label),
            name: current.overlay?.label ?? "Prediction",
            region: current.overlay?.classes[row.label] ?? String(row.label),
            ...row,
          })),
    [current, result],
  );
  return (
    <Panel
      title="Overlap and distance metrics"
      kicker={
        current.source === "synthetic"
          ? "GENERATED REFERENCE MASK"
          : "REFERENCE LABEL COMPARISON"
      }
    >
      {current.source === "api" && (
        <div className="field-row">
          <label>
            Target labels{" "}
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">Choose uploaded reference</option>
              {scans
                .filter((scan) => scan.role === "labels")
                .map((scan) => (
                  <option key={scan.id} value={scan.id}>
                    {scan.id.slice(0, 8)} · {scan.shape.join(" × ")}
                  </option>
                ))}
            </select>
          </label>
          <button
            className="primary"
            disabled={busy || !target || !current.overlay}
            onClick={() => run(target)}
          >
            Queue metrics
          </button>
        </div>
      )}
      {rows.length ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Region</th>
                <th>Dice</th>
                <th>IoU</th>
                <th>HD95 mm</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.key}>
                  <td>{row.name}</td>
                  <td>{row.region}</td>
                  <td>{display(row.dice)}</td>
                  <td>{display(row.iou)}</td>
                  <td>{display(row.hd95_mm)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>
          Queue metrics with a prediction and a matching reference label map.
        </Empty>
      )}
    </Panel>
  );
}

export function Reports({
  current,
  report,
  sources,
  busy,
  run,
  api,
}: {
  current: Case;
  report?: Report;
  sources: string[];
  busy: boolean;
  run: () => void;
  api?: Api;
}) {
  const [formats, setFormats] = useState<string[]>(["json", "html"]);
  const [exportError, setExportError] = useState("");
  const [exporting, setExporting] = useState(false);
  useEffect(() => {
    let active = true;
    if (api && report) {
      void api
        .json<{ formats: string[] }>(`/api/v1/reports/${report.id}/exports`)
        .then((value) => {
          if (active) setFormats(value.formats);
        })
        .catch(() => {
          if (active) setFormats(["json", "html"]);
        });
    }
    return () => {
      active = false;
    };
  }, [api, report]);
  const syntheticReport = useMemo(
    () =>
      current.source === "synthetic" ? syntheticResearchReport() : undefined,
    [current.source],
  );
  const document = syntheticReport ?? report;
  return (
    <Panel
      title="AI-Assisted Research Segmentation Report"
      kicker="RESEARCH EXPORT"
    >
      {current.source === "api" && (
        <>
          <p>
            {sources.length} available case artifacts will be included in the
            snapshot.
          </p>
          <button
            className="primary"
            disabled={busy || !sources.length}
            onClick={run}
          >
            Queue report
          </button>
        </>
      )}
      {document ? (
        <>
          <p>Review the captured metadata and export this research snapshot.</p>
          <div className="button-row">
            <button
              className="secondary"
              onClick={() =>
                downloadJson(document, "neurovision-research-report.json")
              }
            >
              <Download size={16} /> Download JSON
            </button>
            {(syntheticReport
              ? ["html"]
              : formats.filter((format) => format !== "json")
            ).map((format) => (
              <button
                key={format}
                className="secondary"
                disabled={exporting}
                onClick={() => {
                  setExportError("");
                  if (syntheticReport) {
                    const escaped = JSON.stringify(syntheticReport, null, 2)
                      .replaceAll("&", "&amp;")
                      .replaceAll("<", "&lt;")
                      .replaceAll(">", "&gt;");
                    const html = `<!doctype html><html lang="en"><meta charset="utf-8"><title>${syntheticReport.title}</title><body><h1>${syntheticReport.title}</h1><p>${NOTICE}</p><pre style="white-space:pre-wrap;overflow-wrap:anywhere">${escaped}</pre></body></html>`;
                    downloadBlob(
                      new Blob([html], { type: "text/html" }),
                      "research-report.html",
                    );
                  } else if (api && report) {
                    setExporting(true);
                    void api
                      .exportReport(report.id, format)
                      .then((blob) =>
                        downloadBlob(blob, `research-report.${format}`),
                      )
                      .catch((error: unknown) =>
                        setExportError(
                          error instanceof Error
                            ? error.message
                            : "Report export failed.",
                        ),
                      )
                      .finally(() => setExporting(false));
                  }
                }}
              >
                <Download size={16} /> Download {format.toUpperCase()}
              </button>
            ))}
          </div>
          {exportError && <p role="alert">{exportError}</p>}
          <pre className="report-preview">
            {JSON.stringify(document, null, 2)}
          </pre>
        </>
      ) : (
        <Empty>No report has been generated for this case.</Empty>
      )}
    </Panel>
  );
}
