"""Process-local adapters for authorization, quotas, storage, and work scheduling."""

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hmac import compare_digest
from threading import Lock
from time import monotonic
from typing import Literal, Protocol
from uuid import UUID, uuid4

from apps.api.contracts import APIError, ErrorDetail, JobInput, JobView, Kind


class Authenticator(Protocol):
    def authenticate(self, authorization: str | None) -> str | None: ...


class BearerTokens:
    """Map operator-provisioned credentials to stable principal IDs."""

    def __init__(self, tokens: Mapping[str, str]) -> None:
        if any(
            len(token) < 32 or not token.isascii() or not principal.strip()
            for token, principal in tokens.items()
        ):
            raise ValueError(
                "Bearer credentials need 32+ ASCII characters and an owner"
            )
        self._tokens = {key.encode("ascii"): value for key, value in tokens.items()}

    def authenticate(self, authorization: str | None) -> str | None:
        if authorization is None:
            return None
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.isascii() or len(token) > 1024:
            return None
        candidate = token.encode("ascii")
        return next(
            (
                owner
                for key, owner in self._tokens.items()
                if compare_digest(candidate, key)
            ),
            None,
        )


class RateLimiter(Protocol):
    def retry_after(self, owner: str) -> int | None: ...


class FixedWindow:
    def __init__(self, limit: int = 120, seconds: int = 60) -> None:
        if limit < 1 or seconds < 1:
            raise ValueError("Rate-limit settings must be positive")
        self.limit, self.seconds = limit, seconds
        self._window = -1
        self._counts: dict[str, int] = {}
        self._lock = Lock()

    def retry_after(self, owner: str) -> int | None:
        with self._lock:
            now = int(monotonic())
            window, elapsed = divmod(now, self.seconds)
            if window != self._window:
                self._window = window
                self._counts.clear()
            count = self._counts.get(owner, 0)
            if count >= self.limit:
                return self.seconds - elapsed
            self._counts[owner] = count + 1
            return None


ArtifactKind = Literal["scan", "preprocess", "inference", "metrics", "maps", "reports"]


@dataclass(frozen=True, slots=True)
class Artifact:
    kind: ArtifactKind
    payload: object
    size: int


class ArtifactStore(Protocol):
    def put(self, owner: str, key: UUID, artifact: Artifact) -> None: ...
    def get(self, owner: str, key: UUID) -> Artifact: ...
    def remove(self, owner: str, key: UUID) -> None: ...
    def keys(self, owner: str, kind: ArtifactKind) -> list[UUID]: ...


class MemoryStore:
    """Keep bounded typed artifacts under opaque, owner-scoped keys."""

    def __init__(self, max_bytes: int = 256 * 1024**2, max_items: int = 256) -> None:
        if max_bytes < 1 or max_items < 1:
            raise ValueError("Storage limits must be positive")
        self._items: dict[tuple[str, UUID], Artifact] = {}
        self._max_bytes, self._max_items = max_bytes, max_items
        self._used = 0
        self._lock = Lock()

    def put(self, owner: str, key: UUID, artifact: Artifact) -> None:
        with self._lock:
            if (owner, key) in self._items:
                raise APIError(409, "artifact_exists", "Artifact already exists.")
            if (
                not 0 <= artifact.size <= self._max_bytes - self._used
                or len(self._items) >= self._max_items
            ):
                raise APIError(
                    507, "storage_full", "Research artifact storage is full."
                )
            self._items[owner, key] = artifact
            self._used += artifact.size

    def get(self, owner: str, key: UUID) -> Artifact:
        with self._lock:
            if (artifact := self._items.get((owner, key))) is None:
                raise APIError(404, "not_found", "Artifact not found.")
            return artifact

    def remove(self, owner: str, key: UUID) -> None:
        with self._lock:
            if (artifact := self._items.pop((owner, key), None)) is not None:
                self._used -= artifact.size

    def keys(self, owner: str, kind: ArtifactKind) -> list[UUID]:
        with self._lock:
            return [
                key
                for (principal, key), artifact in self._items.items()
                if principal == owner and artifact.kind == kind
            ]


@dataclass(frozen=True, slots=True)
class Job:
    owner: str
    id: UUID
    kind: Kind
    payload: JobInput
    request_id: str


JobRunner = Callable[[Job], str]
JobEvent = Callable[[str, Job, str | None], None]


class JobQueue(Protocol):
    def submit(
        self, owner: str, kind: Kind, body: JobInput, request_id: str
    ) -> JobView: ...
    def get(self, owner: str, key: UUID) -> JobView: ...
    def remove(self, owner: str, key: UUID) -> None: ...
    def close(self) -> None: ...
    def list(self, owner: str) -> list[JobView]: ...


class LocalJobs:
    """Bound queued work and retained records, with one worker for model safety."""

    def __init__(
        self, run: JobRunner, event: JobEvent, *, pending: int = 8, retained: int = 256
    ) -> None:
        if not 1 <= pending <= retained:
            raise ValueError("Invalid job limits")
        self._run, self._event = run, event
        self._pending, self._retained = pending, retained
        self._active = 0
        self._closed = False
        self._records: dict[tuple[str, UUID], JobView] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="neurovision"
        )

    def submit(
        self, owner: str, kind: Kind, body: JobInput, request_id: str
    ) -> JobView:
        with self._lock:
            if self._closed or self._active >= self._pending:
                raise APIError(503, "queue_busy", "Research job queue is busy.")
            if len(self._records) >= self._retained:
                raise APIError(
                    507, "job_history_full", "Delete finished jobs to free history."
                )
            job = Job(owner, uuid4(), kind, body, request_id)
            view = JobView(id=job.id, kind=kind, status="queued")
            self._records[owner, job.id] = view
            self._active += 1
            try:
                self._executor.submit(self._execute, job)
            except RuntimeError:
                self._active -= 1
                del self._records[owner, job.id]
                raise APIError(
                    503, "queue_busy", "Research job queue is unavailable."
                ) from None
            return view

    def _execute(self, job: Job) -> None:
        key = (job.owner, job.id)
        with self._lock:
            self._records[key] = JobView(id=job.id, kind=job.kind, status="running")
        failure: str | None
        try:
            url = self._run(job)
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, APIError)
                else APIError(500, "job_failed", "Research operation failed.")
            )
            finished = JobView(
                id=job.id,
                kind=job.kind,
                status="failed",
                error=ErrorDetail(
                    code=error.code, message=error.message, request_id=job.request_id
                ),
            )
            failure = type(exc).__name__
        else:
            finished = JobView(
                id=job.id, kind=job.kind, status="succeeded", result_url=url
            )
            failure = None
        with self._lock:
            self._records[key] = finished
            self._active -= 1
        self._event(finished.status, job, failure)

    def get(self, owner: str, key: UUID) -> JobView:
        with self._lock:
            if (view := self._records.get((owner, key))) is None:
                raise APIError(404, "not_found", "Job not found.")
            return view

    def remove(self, owner: str, key: UUID) -> None:
        with self._lock:
            view = self._records.get((owner, key))
            if view is None:
                raise APIError(404, "not_found", "Job not found.")
            if view.status in {"queued", "running"}:
                raise APIError(
                    409, "job_active", "Wait for the job to finish before deleting it."
                )
            del self._records[owner, key]

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)

    def list(self, owner: str) -> list[JobView]:
        with self._lock:
            return [
                view
                for (principal, _), view in self._records.items()
                if principal == owner
            ]
