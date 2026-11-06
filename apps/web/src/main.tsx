import {
  Activity,
  ArrowLeft,
  ArrowRight,
  AudioWaveform,
  BarChart3,
  BrainCircuit,
  CircleHelp,
  Database,
  FileBarChart2,
  FlaskConical,
  Grid2X2,
  House,
  Layers3,
  Map,
  Menu,
  Plug,
  ScanLine,
  Settings2,
  Timer,
  Upload as UploadIcon,
  Workflow,
  X,
} from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { createRoot } from "react-dom/client";
import {
  Api,
  type Job,
  type Maps,
  type Metrics,
  type Model,
  type QC,
  type Report,
  type Scan,
} from "./api";
import { demo, NOTICE, type Case, type OverlaySet } from "./demo";
import {
  Confidence,
  Datasets,
  MapPage,
  MetricsPage,
  ModelComparison,
  Preprocessing,
  Reports,
  Studio,
  Timeline,
  Upload,
} from "./pages";
import { Empty, Landing, Metric, PageTitle, Panel } from "./ui";
import { Viewer } from "./viewer";
import "./style.css";

const navigation = [
  {
    title: "WORKSPACE",
    pages: [
      ["home", "Overview", House],
      ["workspace", "Research Workspace", Grid2X2],
      ["upload", "MRI Upload", UploadIcon],
      ["datasets", "Dataset Explorer", Database],
    ],
  },
  {
    title: "PIPELINE",
    pages: [
      ["qc", "Preprocessing / QC", Workflow],
      ["segmentation", "Segmentation Studio", ScanLine],
      ["comparison", "Model Comparison", Layers3],
      ["confidence", "NeuroConfidence", Activity],
      ["map", "NeuroMap", Map],
      ["timeline", "NeuroTimeline", Timer],
    ],
  },
  {
    title: "RESEARCH",
    pages: [
      ["metrics", "Metrics", BarChart3],
      ["reports", "Reports", FileBarChart2],
      ["system", "System / Models", Settings2],
    ],
  },
] as const;

type Page = (typeof navigation)[number]["pages"][number][0];
const descriptions: Record<Exclude<Page, "home">, [string, string, string]> = {
  workspace: [
    "Research workspace",
    "MRI Workspace",
    "Inspect a research case across linked views and examine its segmentation overlays.",
  ],
  upload: [
    "Data intake",
    "MRI Upload",
    "Add a reviewed, de-identified NIfTI volume to your research session.",
  ],
  datasets: [
    "Data library",
    "Dataset Explorer",
    "Open the built-in phantom or browse scans owned by your API session.",
  ],
  qc: [
    "Pipeline / 01",
    "Preprocessing & QC",
    "Inspect scan preparation, quality findings, and linked before-and-after slices.",
  ],
  segmentation: [
    "Pipeline / 02",
    "Segmentation Studio",
    "Select a registered model or inspect an analytic demo mask.",
  ],
  comparison: [
    "Pipeline / 03",
    "Model Comparison",
    "Compare prediction runs on a common grid and inspect differences by class.",
  ],
  confidence: [
    "Pipeline / 04",
    "NeuroConfidence",
    "Inspect confidence and uncertainty together with the predicted anatomy.",
  ],
  map: [
    "Pipeline / 05",
    "NeuroMap",
    "Measure segmented volumes and compare their overlap with an atlas.",
  ],
  timeline: [
    "Pipeline / 06",
    "NeuroTimeline",
    "Explore segmentation changes between simulated research visits.",
  ],
  metrics: [
    "Research outputs",
    "Metrics",
    "Evaluate predictions against reference masks and inspect class-level scores.",
  ],
  reports: [
    "Research outputs",
    "Reports",
    "Review and export structured snapshots of research findings.",
  ],
  system: [
    "Environment",
    "System & Models",
    "Manage the local API connection and inspect registered models and research jobs.",
  ],
};

const tourSteps: { page: Page; title: string; copy: string }[] = [
  {
    page: "workspace",
    title: "Navigate linked slices",
    copy: "Select a point in any image or use a slice slider. The other planes follow the shared voxel cursor.",
  },
  {
    page: "segmentation",
    title: "Inspect the prediction",
    copy: "Change the analytic mask, hide a class, adjust overlay opacity, or switch to the 3D boundary preview.",
  },
  {
    page: "confidence",
    title: "Read uncertainty",
    copy: "Switch between the thresholded confidence view and the uncertainty heatmap. The synthetic scores are uncalibrated.",
  },
  {
    page: "timeline",
    title: "Explore change over time",
    copy: "Switch to Compare and use the linked axial slider to inspect the simulated follow-up.",
  },
];

function readPage(): Page {
  const path = window.location.hash.slice(1);
  return navigation.some((group) => group.pages.some(([key]) => key === path))
    ? (path as Page)
    : "home";
}

function App() {
  const [page, setPage] = useState<Page>(readPage);
  const [menu, setMenu] = useState(false);
  const [tour, setTour] = useState<number>();
  const [current, setCurrent] = useState<Case>(() => demo().baseline);
  const [api, setApi] = useState<Api>();
  const [token, setToken] = useState("");
  const [scans, setScans] = useState<Scan[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [runs, setRuns] = useState<OverlaySet[]>([]);
  const [processed, setProcessed] = useState<Case>();
  const [qc, setQC] = useState<QC>();
  const [metrics, setMetrics] = useState<Metrics>();
  const [maps, setMaps] = useState<Maps>();
  const [report, setReport] = useState<Report>();
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [progress, setProgress] = useState("");
  const operation = useRef<AbortController | null>(null);
  const main = useRef<HTMLElement>(null);

  useEffect(() => {
    const sync = () => {
      setPage(readPage());
      setMenu(false);
    };
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);
  useEffect(() => {
    document.title =
      page === "home"
        ? "NeuroVision 2028 · MRI research workspace"
        : `${descriptions[page][1]} · NeuroVision 2028`;
  }, [page]);

  function go(target: Page) {
    window.location.hash = target;
    setPage(target);
    setMenu(false);
    window.scrollTo(0, 0);
    main.current?.focus({ preventScroll: true });
  }
  function resetCase(next: Case) {
    setCurrent(next);
    setRuns([]);
    setProcessed(undefined);
    setQC(undefined);
    setMetrics(undefined);
    setMaps(undefined);
    setReport(undefined);
  }
  function cancelOperation() {
    operation.current?.abort();
    operation.current = null;
    setBusy("");
    setError("");
    setProgress("");
  }
  function loadDemo() {
    cancelOperation();
    resetCase(demo().baseline);
    go("workspace");
  }
  function startTour() {
    loadDemo();
    setTour(0);
  }
  function stepTour(index: number) {
    if (index >= tourSteps.length) {
      setTour(undefined);
      return;
    }
    setTour(index);
    go(tourSteps[index].page);
  }
  function run<T>(
    label: string,
    task: (signal: AbortSignal, update: (job: Job) => void) => Promise<T>,
    apply: (value: T) => void,
  ) {
    if (operation.current) return;
    const controller = new AbortController();
    operation.current = controller;
    setBusy(label);
    setError("");
    setProgress("");
    const active = () =>
      operation.current === controller && !controller.signal.aborted;
    void task(controller.signal, (job) => {
      if (!active()) return;
      setJobs((previous) => [
        job,
        ...previous.filter((item) => item.id !== job.id),
      ]);
      setProgress(`${job.kind} · ${job.status}`);
    })
      .then((value) => {
        if (active()) apply(value);
      })
      .catch((failure: unknown) => {
        if (active())
          setError(
            failure instanceof Error
              ? failure.message
              : "The research operation failed.",
          );
      })
      .finally(() => {
        if (active()) {
          operation.current = null;
          setBusy("");
        }
      });
  }
  function connect(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const client = new Api(token.trim());
    setToken("");
    run(
      "Connecting",
      async () => {
        const catalog = await client.json<{ models: Model[] }>(
          "/api/v1/models",
        );
        return { models: catalog.models, scans: await client.scans() };
      },
      (result) => {
        setApi(client);
        setModels(result.models);
        setScans(result.scans);
      },
    );
  }
  function disconnect() {
    cancelOperation();
    setApi(undefined);
    setToken("");
    setScans([]);
    setModels([]);
    setJobs([]);
    resetCase(demo().baseline);
  }
  function upload(file: File, modality: string, role: string) {
    if (!api) return;
    run(
      "Uploading volume",
      async () => {
        const scan = await api.upload(file, modality, role);
        return {
          scan,
          opened: scan.role === "image" ? await api.openScan(scan) : undefined,
        };
      },
      ({ scan, opened }) => {
        setScans((previous) => [scan, ...previous]);
        setTour(undefined);
        if (opened) {
          resetCase(opened);
          go("workspace");
        } else go("datasets");
      },
    );
  }
  function openScan(scan: Scan) {
    if (!api) return;
    run(
      "Opening volume",
      () => api.openScan(scan),
      (opened) => {
        setTour(undefined);
        resetCase(opened);
        go("workspace");
      },
    );
  }
  function preprocess(normalization: string) {
    if (!api || !current.id) return;
    run(
      "Running preprocessing",
      async (signal, update) => {
        const result = await api.job<QC>(
          "preprocess",
          { scan_id: current.id, normalization },
          update,
          signal,
        );
        const image = result.usable
          ? await api.volume(`/api/v1/preprocess/${result.id}/volume.nii`)
          : undefined;
        return { result, image };
      },
      ({ result, image }) => {
        setQC(result);
        setProcessed(
          image
            ? { source: "api", id: result.id, name: "Preprocessed", image }
            : undefined,
        );
      },
    );
  }
  function inference(model: string, onProcessed: boolean) {
    if (!api || !current.id) return;
    run(
      "Running inference",
      async (signal, update) => {
        const result = await api.job<{ id: string }>(
          "inference",
          {
            scan_id: onProcessed ? processed?.id : current.id,
            model_id: model,
          },
          update,
          signal,
        );
        return api.overlay(result.id);
      },
      (overlay) => {
        setRuns((previous) => [...previous.slice(-5), overlay]);
        setCurrent((previous) => ({ ...previous, overlay }));
        setMetrics(undefined);
        setMaps(undefined);
        setReport(undefined);
      },
    );
  }
  const sources = [
    ...new Set(
      [current.id, qc?.id, current.overlay?.id, metrics?.id, maps?.id].filter(
        (id): id is string => typeof id === "string",
      ),
    ),
  ];
  const activeStep = tour === undefined ? undefined : tourSteps[tour];

  let content;
  switch (page) {
    case "home":
      content = <Landing launch={() => go("workspace")} tour={startTour} />;
      break;
    case "workspace":
      content = (
        <>
          <div className="workspace-strip">
            <div>
              <span className="dot" />
              <strong>{current.name}</strong>
            </div>
            <span className="pill">
              {current.source === "synthetic"
                ? "Synthetic research case"
                : "API research case"}
            </span>
          </div>
          <Viewer
            key={current.overlay?.id ?? current.id ?? "synthetic"}
            volume={current.image}
            overlay={current.overlay}
          />
          <div className="metrics-grid">
            <Metric
              label="Input grid"
              value={current.image.shape.join(" × ")}
              caption="Canonical voxel dimensions"
            />
            <Metric
              label="Foreground classes"
              value={current.overlay ? current.overlay.classes.length - 1 : "—"}
              caption="Current prediction schema"
            />
            <Metric
              label="Viewer modes"
              value="2D / 3D"
              caption="Linked slices and sampled boundaries"
            />
          </div>
        </>
      );
      break;
    case "upload":
      content = (
        <Upload
          connected={!!api}
          busy={!!busy}
          onConnect={() => go("system")}
          onUpload={upload}
        />
      );
      break;
    case "datasets":
      content = (
        <Datasets
          connected={!!api}
          busy={!!busy}
          scans={scans}
          load={loadDemo}
          open={openScan}
          refresh={() =>
            api && run("Refreshing scans", () => api.scans(), setScans)
          }
        />
      );
      break;
    case "qc":
      content = (
        <Preprocessing
          key={current.id ?? "demo"}
          current={current}
          processed={processed}
          qc={qc}
          busy={!!busy}
          run={preprocess}
        />
      );
      break;
    case "segmentation":
      content = (
        <Studio
          key={current.id ?? "demo"}
          current={current}
          models={models}
          processed={processed}
          busy={!!busy}
          run={inference}
        />
      );
      break;
    case "comparison":
      content = (
        <ModelComparison
          key={current.id ?? "demo"}
          current={current}
          runs={runs}
        />
      );
      break;
    case "confidence":
      content = <Confidence current={current} />;
      break;
    case "map":
      content = (
        <MapPage
          key={current.overlay?.id ?? current.id ?? "demo"}
          current={current}
          scans={scans}
          result={maps}
          busy={!!busy}
          run={(payload) =>
            api &&
            run(
              "Running NeuroMap",
              (signal, update) =>
                api.job<Maps>("maps", payload, update, signal),
              setMaps,
            )
          }
        />
      );
      break;
    case "timeline":
      content = <Timeline />;
      break;
    case "metrics":
      content = (
        <MetricsPage
          key={current.id ?? "demo"}
          current={current}
          scans={scans}
          result={metrics}
          busy={!!busy}
          run={(id) =>
            api &&
            current.overlay &&
            run(
              "Computing metrics",
              (signal, update) =>
                api.job<Metrics>(
                  "metrics",
                  { segmentation_id: current.overlay?.id, target_scan_id: id },
                  update,
                  signal,
                ),
              setMetrics,
            )
          }
        />
      );
      break;
    case "reports":
      content = (
        <Reports
          current={current}
          report={report}
          sources={sources}
          busy={!!busy}
          api={api}
          run={() =>
            api &&
            run(
              "Creating report",
              (signal, update) =>
                api.job<Report>(
                  "reports",
                  { artifact_ids: sources },
                  update,
                  signal,
                ),
              setReport,
            )
          }
        />
      );
      break;
    case "system":
      content = (
        <>
          <div className="page-grid">
            <Panel title="Local API connection" kicker="SESSION SETTINGS">
              <p>
                The bearer token stays in this tab’s memory. The development
                proxy forwards API requests to the configured local FastAPI
                server.
              </p>
              {api ? (
                <button className="secondary" onClick={disconnect}>
                  Disconnect API
                </button>
              ) : (
                <form className="form-stack" onSubmit={connect}>
                  <label>
                    Bearer token
                    <input
                      type="password"
                      value={token}
                      onChange={(e) => setToken(e.target.value)}
                      autoComplete="off"
                      required
                      minLength={32}
                    />
                  </label>
                  <button className="primary" type="submit" disabled={!!busy}>
                    <Plug size={16} /> Connect API
                  </button>
                </form>
              )}
            </Panel>
            <Panel title="Research environment">
              <p>
                The synthetic case runs entirely in this tab. Connected jobs and
                artifacts are managed by the API.
              </p>
              {api && (
                <a
                  className="text-button"
                  href="/api/v1/openapi.json"
                  target="_blank"
                  rel="noreferrer"
                >
                  Open API schema <ArrowRight size={15} />
                </a>
              )}
            </Panel>
          </div>
          <Panel title="Registered models">
            {models.length ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Architecture</th>
                      <th>Revision</th>
                      <th>Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((model) => (
                      <tr key={model.id}>
                        <td>{model.id}</td>
                        <td>{model.architecture}</td>
                        <td>{model.revision}</td>
                        <td>
                          {model.synthetic
                            ? "Synthetic fixture"
                            : "Operator-registered"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty>
                Connect the API to inspect its model registry. The guided demo
                uses two generated analytic masks.
              </Empty>
            )}
          </Panel>
          <Panel title="Research jobs" kicker="ASYNC QUEUE">
            {api && (
              <button
                className="secondary"
                disabled={!!busy}
                onClick={() =>
                  run(
                    "Refreshing jobs",
                    () => api.json<Job[]>("/api/v1/jobs"),
                    setJobs,
                  )
                }
              >
                Refresh jobs
              </button>
            )}
            {jobs.length ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Job</th>
                      <th>Operation</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobs.map((job) => (
                      <tr key={job.id}>
                        <td>
                          <code>{job.id.slice(0, 8)}</code>
                        </td>
                        <td>{job.kind}</td>
                        <td>{job.status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty>No jobs have been submitted in this session.</Empty>
            )}
          </Panel>
        </>
      );
      break;
  }

  return (
    <div className="app-shell">
      <a
        className="skip-link"
        href="#main-content"
        onClick={(event) => {
          event.preventDefault();
          main.current?.focus();
        }}
      >
        Skip to content
      </a>
      <aside
        className={menu ? "sidebar is-open" : "sidebar"}
        id="site-navigation"
      >
        <button className="brand" onClick={() => go("home")}>
          <div className="brand-icon">
            <BrainCircuit size={23} />
          </div>
          <div>
            <strong>
              NeuroVision<span> 2028</span>
            </strong>
            <small>MRI RESEARCH LAB</small>
          </div>
        </button>
        <nav aria-label="Research pages">
          {navigation.map((group) => (
            <div className="nav-group" key={group.title}>
              <p>{group.title}</p>
              {group.pages.map(([key, label, Icon]) => (
                <a
                  key={key}
                  href={`#${key}`}
                  aria-current={page === key ? "page" : undefined}
                  onClick={(e) => {
                    e.preventDefault();
                    go(key);
                  }}
                >
                  <Icon size={17} />
                  <span>{label}</span>
                  {page === key && <span className="nav-active-dot" />}
                </a>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <FlaskConical size={16} />
          <span>Research use only</span>
          <span className="dot" />
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div>
            <button
              className="icon-button mobile-menu"
              aria-label={menu ? "Close navigation" : "Open navigation"}
              aria-expanded={menu}
              aria-controls="site-navigation"
              onClick={() => setMenu(!menu)}
            >
              {menu ? <X size={22} /> : <Menu size={22} />}
            </button>
            <span className="topbar-label">
              <AudioWaveform size={18} /> NeuroVision Intelligence Pipeline
            </span>
          </div>
          <div className="top-actions">
            <button className="connection" onClick={() => go("system")}>
              <span className="dot" />
              {api ? "API connected" : "Local demo"}
            </button>
            <button
              className="icon-button"
              onClick={startTour}
              aria-label="Start guided demo"
            >
              <CircleHelp size={20} />
            </button>
          </div>
        </header>
        <main
          id="main-content"
          className="main-content"
          ref={main}
          tabIndex={-1}
        >
          {page !== "home" && (
            <PageTitle
              section={descriptions[page][0]}
              title={descriptions[page][1]}
              description={descriptions[page][2]}
              action={
                page === "workspace" ? (
                  <button className="secondary" onClick={startTour}>
                    <FlaskConical size={16} /> Guided demo
                  </button>
                ) : undefined
              }
            />
          )}
          {activeStep && (
            <section className="tour panel" aria-label="Guided demo">
              <div>
                <span className="eyebrow">
                  GUIDED DEMO · {(tour ?? 0) + 1} / {tourSteps.length}
                </span>
                <h2>{activeStep.title}</h2>
                <p>{activeStep.copy}</p>
              </div>
              <div className="button-row">
                {!!tour && (
                  <button
                    className="secondary"
                    onClick={() => stepTour(tour - 1)}
                  >
                    <ArrowLeft size={16} /> Back
                  </button>
                )}
                <button
                  className="primary"
                  onClick={() => stepTour((tour ?? 0) + 1)}
                >
                  {tour === tourSteps.length - 1 ? "Finish" : "Next"}{" "}
                  <ArrowRight size={16} />
                </button>
                <button
                  className="icon-button"
                  aria-label="Close guided demo"
                  onClick={() => setTour(undefined)}
                >
                  <X size={18} />
                </button>
              </div>
            </section>
          )}
          {error && (
            <div className="inline-alert error" role="alert">
              {error}
            </div>
          )}
          {(busy || progress) && (
            <div className="job-status" role="status">
              <span className="dot" /> {busy || progress}
              {busy && progress ? ` · ${progress}` : ""}
            </div>
          )}
          {content}
          <footer className="page-footer">
            <span>{NOTICE}</span>
            <span>NEUROVISION / RESEARCH BUILD</span>
          </footer>
        </main>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
