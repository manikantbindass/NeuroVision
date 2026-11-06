"""Process-local metrics and opt-in tracing with bounded, non-patient labels."""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from opentelemetry import trace
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

_TRACER = trace.get_tracer("neurovision.research")
_METHODS = frozenset({"GET", "HEAD", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"})


def method_label(method: str) -> str:
    return method if method in _METHODS else "OTHER"


class Telemetry:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "neurovision_http_requests",
            "HTTP responses by route template and status class.",
            ("method", "route", "status_class"),
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "neurovision_http_request_duration_seconds",
            "HTTP request duration.",
            ("route",),
            registry=self.registry,
        )
        self.jobs = Counter(
            "neurovision_job_executions",
            "Research job executions by kind and outcome.",
            ("kind", "outcome"),
            registry=self.registry,
        )
        self.job_latency = Histogram(
            "neurovision_job_execution_duration_seconds",
            "Research job execution duration excluding queue wait.",
            ("kind",),
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 15, 30, 60, 120, 300),
            registry=self.registry,
        )

    def request(self, method: str, route: str, status: int, elapsed: float) -> None:
        self.requests.labels(method, route, f"{status // 100}xx").inc()
        self.request_latency.labels(route).observe(elapsed)

    @contextmanager
    def job(self, kind: str, request_id: str) -> Iterator[None]:
        start = perf_counter()
        outcome = "failed"
        with _TRACER.start_as_current_span(
            "research.job",
            attributes={"job.kind": kind, "neurovision.request_id": request_id},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield
                outcome = "succeeded"
            finally:
                span.set_attribute("job.outcome", outcome)
                self.jobs.labels(kind, outcome).inc()
                self.job_latency.labels(kind).observe(perf_counter() - start)

    def exposition(self) -> bytes:
        return generate_latest(self.registry)


def request_span(request_id: str) -> trace.Span:
    # The request boundary records an independent span without trusting external
    # trace headers or making the entire ASGI scope available to exporters.
    return _TRACER.start_span(
        "research.http", attributes={"neurovision.request_id": request_id}
    )
