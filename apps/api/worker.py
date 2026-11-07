"""Leased research worker with graceful shutdown and internal health probes."""

import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CONTENT_TYPE_LATEST

from apps.api.contracts import APIError, ErrorDetail
from apps.api.deployed import setting
from apps.api.durable import Database, Lease, SQLArtifactStore, SQLJobs
from apps.api.main import DEFAULT_LIMITS
from apps.api.middleware import log_event
from apps.api.service import ModelEntry, ResearchService, demo_models
from apps.api.telemetry import Telemetry
from neurovision.inference.engine import InferenceEngine


def run_lease(
    database: Database,
    queue: SQLJobs,
    lease: Lease,
    logger: logging.Logger,
    models: dict[str, ModelEntry] | None = None,
    on_heartbeat: Callable[[], None] | None = None,
    telemetry: Telemetry | None = None,
) -> None:
    store = SQLArtifactStore(database, lease=lease)
    service = ResearchService(
        store, models if models is not None else demo_models(), DEFAULT_LIMITS
    )
    finished, lost = threading.Event(), threading.Event()

    def heartbeat() -> None:
        while not finished.wait(15):
            try:
                if not queue.renew(lease):
                    lost.set()
                    return
                if on_heartbeat is not None:
                    on_heartbeat()
            except Exception:
                lost.set()
                return

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    error = None
    try:
        try:
            store.get(lease.job.owner, lease.job.id)
        except APIError as missing:
            if missing.status != 404:
                raise
            if telemetry is None:
                service.run(lease.job)
            else:
                with telemetry.job(lease.job.kind, lease.job.request_id):
                    service.run(lease.job)
    except Exception as exc:
        if isinstance(exc, APIError):
            error = ErrorDetail(
                code=exc.code, message=exc.message, request_id=lease.job.request_id
            )
        else:
            error = ErrorDetail(
                code="job_failed",
                message="Research operation failed.",
                request_id=lease.job.request_id,
            )
    finally:
        finished.set()
        thread.join(timeout=3)
    published = not lost.is_set() and queue.complete(lease, error=error)
    log_event(
        logger,
        "worker_attempt_complete",
        job_id=str(lease.job.id),
        request_id=lease.job.request_id,
        lease_lost=not published,
        status=("failed" if error else "completed") if published else "superseded",
    )


def main() -> None:
    database_url = setting("NEUROVISION_DATABASE_URL")
    database, stop = Database(database_url), threading.Event()
    queue = SQLJobs(database)
    logger = logging.Logger("neurovision.worker", logging.INFO)
    logger.addHandler(logging.StreamHandler())
    models = demo_models()
    device = os.environ.get("NEUROVISION_WORKER_DEVICE", "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Worker device must be cpu or cuda")
    if device == "cuda":
        models = {
            name: ModelEntry(
                InferenceEngine(entry.engine.model, device="cuda"),
                entry.classes,
                entry.revision,
                entry.modality,
                entry.synthetic,
                entry.limitations,
            )
            for name, entry in models.items()
        }
    last_ready = [0.0]
    active = [False]
    telemetry = Telemetry()

    def ready_pulse() -> None:
        last_ready[0] = time.monotonic()

    class Health(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/metrics":
                try:
                    counts = queue.counts()
                    metrics = f"neurovision_worker_active {int(active[0])}\n"
                    for state in ("queued", "running", "succeeded", "failed"):
                        metrics += (
                            f'neurovision_jobs{{status="{state}"}} '
                            f"{counts.get(state, 0)}\n"
                        )
                    data, status = metrics.encode() + telemetry.exposition(), 200
                except Exception:
                    data, status = b"# backend unavailable\n", 503
                content_type = CONTENT_TYPE_LATEST
            else:
                ready = time.monotonic() - last_ready[0] < 65
                status = (
                    200
                    if self.path == "/health" or (self.path == "/ready" and ready)
                    else 503
                )
                data, content_type = (
                    json.dumps({"ready": ready}).encode(),
                    "application/json",
                )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            pass

    port = int(os.environ.get("NEUROVISION_WORKER_HEALTH_PORT", "9000"))
    # The port is internal to the container network for health and scraping.
    server = ThreadingHTTPServer(("0.0.0.0", port), Health)  # noqa: S104
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        while not stop.is_set():
            try:
                lease = queue.claim()
                last_ready[0] = time.monotonic()
                if lease is None:
                    stop.wait(1)
                    continue
                active[0] = True
                try:
                    run_lease(
                        database, queue, lease, logger, models, ready_pulse, telemetry
                    )
                finally:
                    active[0] = False
            except Exception as exc:
                log_event(
                    logger, "worker_backend_unavailable", exception=type(exc).__name__
                )
                stop.wait(2)
    finally:
        server.shutdown()
        server.server_close()
        database.engine.dispose()


if __name__ == "__main__":
    main()
