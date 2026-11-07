import type { Case, OverlaySet } from "./demo";
import { decodeVolume } from "./volume";

export type Scan = {
  id: string;
  role: "image" | "labels";
  modality: string;
  shape: number[];
  spacing_mm: number[];
  reference_space: string;
  assumed_mm: boolean;
};
export type Model = {
  id: string;
  architecture: string;
  revision: string;
  class_names: string[];
  modality: string | null;
  synthetic: boolean;
};
export type JobKind =
  "preprocess" | "inference" | "metrics" | "maps" | "reports";
export type Job = {
  id: string;
  kind: JobKind;
  status: "queued" | "running" | "succeeded" | "failed";
  result_url: string | null;
  error: { message: string; request_id: string } | null;
};
export type QC = {
  id: string;
  usable: boolean;
  provenance: {
    qc: {
      status: string;
      findings: { code: string; severity: string; message: string }[];
    };
  };
};
export type MetricRow = {
  label: number;
  dice: number | null;
  iou: number | null;
  precision: number | null;
  sensitivity: number | null;
  hd95_mm: number | null;
};
export type Metrics = {
  id: string;
  classes: MetricRow[];
  physical_distances: boolean;
};
export type Maps = {
  id: string;
  measurements: {
    code: number;
    name: string;
    voxels: number;
    volume_mm3: number;
    hemisphere: string;
    mean_confidence: number | null;
    atlas_coverage: number;
    associations: { region: { name: string }; fraction_of_segment: number }[];
  }[];
};
export type Report = {
  id: string;
  notice: string;
  schema_version: number;
  title?: string;
  created_at?: string;
  sections?: Record<string, unknown>;
  content_sha256?: string;
  sources: { kind: string; summary: Record<string, unknown> }[];
};
type Segmentation = {
  id: string;
  class_names: string[];
  model_id: string;
  scan_id: string;
};

export class Api {
  constructor(private readonly token: string) {}

  async exportReport(id: string, format: string): Promise<Blob> {
    return (
      await this.request(`/api/v1/reports/${id}/export.${format}`)
    ).blob();
  }

  private async request(path: string, init: RequestInit = {}) {
    if (!path.startsWith("/api/v1/"))
      throw new Error("Unexpected research API address.");
    const headers = new Headers(init.headers);
    headers.set("Authorization", `Bearer ${this.token}`);
    let response: Response;
    try {
      response = await fetch(path, {
        ...init,
        headers,
        cache: "no-store",
        redirect: "error",
      });
    } catch {
      throw new Error(
        "Could not reach the research API. Check that the local server is running.",
      );
    }
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      const requestId = response.headers.get("X-Request-ID");
      const message =
        body?.error?.message ?? `API request failed (${response.status}).`;
      throw new Error(requestId ? `${message} Request ${requestId}.` : message);
    }
    return response;
  }

  async json<T>(path: string, init?: RequestInit): Promise<T> {
    return (await this.request(path, init)).json() as Promise<T>;
  }

  async scans() {
    const { ids } = await this.json<{ ids: string[] }>("/api/v1/scans");
    const result: Scan[] = [];
    // Keep parallel requests below the server's protected-request limit.
    for (let i = 0; i < Math.min(ids.length, 24); i += 4) {
      result.push(
        ...(await Promise.all(
          ids
            .slice(i, Math.min(i + 4, 24))
            .map((id) => this.json<Scan>(`/api/v1/scans/${id}`)),
        )),
      );
    }
    return result;
  }

  async upload(file: File, modality: string, role: string) {
    if (!/\.nii(\.gz)?$/i.test(file.name))
      throw new Error("Choose a .nii or .nii.gz file.");
    if (!file.size || file.size > 16 * 1024 ** 2)
      throw new Error("Choose a file between 1 byte and 16 MiB.");
    const query = new URLSearchParams({
      format: /\.gz$/i.test(file.name) ? "nii.gz" : "nii",
      modality,
      role,
    });
    return this.json<Scan>(`/api/v1/scans?${query}`, {
      method: "POST",
      body: file,
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Deidentified-Research": "true",
      },
    });
  }

  async volume(path: string) {
    const response = await this.request(path);
    if (Number(response.headers.get("Content-Length")) > 8_001_000) {
      throw new Error("The exported volume exceeds the viewer limit.");
    }
    return decodeVolume(await response.arrayBuffer());
  }

  async openScan(scan: Scan): Promise<Case> {
    return {
      id: scan.id,
      image: await this.volume(`/api/v1/scans/${scan.id}/volume.nii`),
      name: `${scan.modality.toUpperCase()} · ${scan.id.slice(0, 8)}`,
      source: "api",
    };
  }

  async overlay(id: string): Promise<OverlaySet> {
    const [metadata, labels, confidence, entropy] = await Promise.all([
      this.json<Segmentation>(`/api/v1/segmentations/${id}`),
      this.volume(`/api/v1/segmentations/${id}/labels.nii`),
      this.volume(`/api/v1/uncertainty/${id}/confidence.nii`),
      this.volume(`/api/v1/uncertainty/${id}/entropy.nii`),
    ]);
    return {
      id,
      label: metadata.model_id,
      labels,
      confidence,
      entropy,
      classes: metadata.class_names,
    };
  }

  async job<T>(
    kind: JobKind,
    payload: object,
    onProgress: (job: Job) => void,
    signal: AbortSignal,
  ) {
    let job = await this.json<Job>(`/api/v1/${kind}`, {
      method: "POST",
      signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    onProgress(job);
    const deadline = Date.now() + 120_000;
    while (job.status === "queued" || job.status === "running") {
      await new Promise((resolve) => setTimeout(resolve, 650));
      signal.throwIfAborted();
      if (Date.now() > deadline)
        throw new Error("Polling timed out. Find the job in System / Models.");
      job = await this.json<Job>(`/api/v1/jobs/${job.id}`, { signal });
      onProgress(job);
    }
    if (job.status === "failed" || !job.result_url) {
      throw new Error(
        job.error
          ? `${job.error.message} Request ${job.error.request_id}.`
          : "The research job failed.",
      );
    }
    return this.json<T>(job.result_url, { signal });
  }
}
