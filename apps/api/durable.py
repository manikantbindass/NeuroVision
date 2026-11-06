"""Shared SQL artifacts and leased jobs for separate API and inference processes."""

import hashlib
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import (
    Column,
    Float,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from apps.api.adapters import Artifact, ArtifactKind, Job, JobQueue
from apps.api.contracts import (
    APIError,
    ErrorDetail,
    InferenceRequest,
    JobInput,
    JobView,
    Kind,
    MapRequest,
    MetricsRequest,
    PreprocessRequest,
    ReportRequest,
)
from apps.api.serialization import decode, encode

metadata = MetaData()
quota = Table(
    "nv_quota",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("bytes", Integer, nullable=False),
    Column("items", Integer, nullable=False),
)
artifacts = Table(
    "nv_artifacts",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("owner", String(160), nullable=False, index=True),
    Column("kind", String(20), nullable=False),
    Column("size", Integer, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("payload", LargeBinary, nullable=False),
)
jobs = Table(
    "nv_jobs",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("owner", String(160), nullable=False, index=True),
    Column("kind", String(20), nullable=False),
    Column("payload", Text, nullable=False),
    Column("request_id", String(64), nullable=False),
    Column("status", String(20), nullable=False, index=True),
    Column("created", Float, nullable=False),
    Column("lease_until", Float, nullable=False),
    Column("token", String(36)),
    Column("attempts", Integer, nullable=False),
    Column("result_url", Text),
    Column("error", Text),
)
MODELS: dict[
    str,
    type[
        PreprocessRequest
        | InferenceRequest
        | MetricsRequest
        | MapRequest
        | ReportRequest
    ],
] = {
    "preprocess": PreprocessRequest,
    "inference": InferenceRequest,
    "metrics": MetricsRequest,
    "maps": MapRequest,
    "reports": ReportRequest,
}


class Database:
    def __init__(self, url: str) -> None:
        if not url.startswith(("postgresql+psycopg://", "sqlite:///")):
            raise ValueError("Use PostgreSQL or a local SQLite test database.")
        self.engine = create_engine(url, pool_pre_ping=True)

    def initialize(self) -> None:
        metadata.create_all(self.engine)
        try:
            with self.engine.begin() as db:
                db.execute(insert(quota).values(id=1, bytes=0, items=0))
        except IntegrityError:
            pass

    def ready(self) -> bool:
        try:
            with self.engine.connect() as db:
                return bool(
                    db.execute(select(quota.c.id).where(quota.c.id == 1)).scalar() == 1
                )
        except Exception:
            return False

    @staticmethod
    def lock(db: Connection) -> None:
        # A transactional row update serializes quota admission and lease changes.
        if db.execute(update(quota).where(quota.c.id == 1).values(id=1)).rowcount != 1:
            raise APIError(
                503, "schema_unavailable", "Research database is not initialized."
            )

    @staticmethod
    def now(db: Connection) -> float:
        return float(
            db.execute(
                select(func.extract("epoch", func.current_timestamp()))
            ).scalar_one()
        )


@dataclass(frozen=True, slots=True)
class Lease:
    job: Job
    token: str


def result_url(kind: str, key: UUID) -> str:
    return f"/api/v1/{'segmentations' if kind == 'inference' else kind}/{key}"


def remove_artifact(db: Connection, owner: str, key: UUID) -> None:
    size = db.execute(
        select(artifacts.c.size).where(
            artifacts.c.id == str(key),
            artifacts.c.owner == owner,
        )
    ).scalar()
    if size is not None:
        db.execute(
            artifacts.delete().where(
                artifacts.c.id == str(key),
                artifacts.c.owner == owner,
            )
        )
        db.execute(
            update(quota)
            .where(quota.c.id == 1)
            .values(
                bytes=quota.c.bytes - size,
                items=quota.c["items"] - 1,
            )
        )


class SQLArtifactStore:
    """Byte-bounded, owner-scoped arrays and metadata. No pickle deserialization."""

    def __init__(
        self,
        database: Database,
        *,
        max_bytes: int = 512 * 1024**2,
        max_items: int = 512,
        lease: Lease | None = None,
    ) -> None:
        if max_bytes < 1 or max_items < 1:
            raise ValueError("Storage limits must be positive")
        self.database, self.max_bytes, self.max_items, self.lease = (
            database,
            max_bytes,
            max_items,
            lease,
        )

    def put(self, owner: str, key: UUID, artifact: Artifact) -> None:
        data = encode(artifact)
        size = max(len(data), artifact.size)
        with self.database.engine.begin() as db:
            self.database.lock(db)
            if self.lease is not None:
                active = db.execute(
                    select(jobs.c.id).where(
                        jobs.c.id == str(key),
                        jobs.c.owner == owner,
                        jobs.c.token == self.lease.token,
                        jobs.c.status == "running",
                        jobs.c.lease_until > self.database.now(db),
                    )
                ).scalar()
                if key != self.lease.job.id or active is None:
                    raise APIError(
                        409, "lease_lost", "Worker lease is no longer active."
                    )
            current = db.execute(
                select(artifacts.c.id).where(artifacts.c.id == str(key))
            ).scalar()
            if current is not None:
                raise APIError(409, "artifact_exists", "Artifact already exists.")
            used = db.execute(select(quota)).mappings().one()
            if size > self.max_bytes - used["bytes"] or used["items"] >= self.max_items:
                raise APIError(
                    507, "storage_full", "Research artifact storage is full."
                )
            db.execute(
                insert(artifacts).values(
                    id=str(key),
                    owner=owner,
                    kind=artifact.kind,
                    size=size,
                    sha256=hashlib.sha256(data).hexdigest(),
                    payload=data,
                )
            )
            db.execute(
                update(quota)
                .where(quota.c.id == 1)
                .values(
                    bytes=used["bytes"] + size,
                    items=used["items"] + 1,
                )
            )

    def get(self, owner: str, key: UUID) -> Artifact:
        with self.database.engine.connect() as db:
            row = (
                db.execute(
                    select(artifacts).where(
                        artifacts.c.id == str(key), artifacts.c.owner == owner
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise APIError(404, "not_found", "Artifact not found.")
        data = bytes(row["payload"])
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise APIError(
                500, "artifact_corrupt", "Stored artifact failed integrity validation."
            )
        return decode(data)

    def remove(self, owner: str, key: UUID) -> None:
        with self.database.engine.begin() as db:
            self.database.lock(db)
            remove_artifact(db, owner, key)

    def keys(self, owner: str, kind: ArtifactKind) -> list[UUID]:
        with self.database.engine.connect() as db:
            return [
                UUID(key)
                for key in db.execute(
                    select(artifacts.c.id).where(
                        artifacts.c.owner == owner,
                        artifacts.c.kind == kind,
                    )
                ).scalars()
            ]


class SQLJobs(JobQueue):
    def __init__(
        self, database: Database, *, pending: int = 32, retained: int = 512
    ) -> None:
        if not 1 <= pending <= retained:
            raise ValueError("Job limits must be positive and ordered")
        self.database, self.pending, self.retained = database, pending, retained

    def submit(
        self, owner: str, kind: Kind, body: JobInput, request_id: str
    ) -> JobView:
        key = uuid4()
        with self.database.engine.begin() as db:
            self.database.lock(db)
            active = db.execute(
                select(func.count())
                .select_from(jobs)
                .where(jobs.c.status.in_(["queued", "running"]))
            ).scalar_one()
            count = db.execute(select(func.count()).select_from(jobs)).scalar_one()
            if active >= self.pending:
                raise APIError(503, "queue_busy", "Research worker queue is busy.")
            if count >= self.retained:
                raise APIError(
                    507, "job_history_full", "Delete finished jobs to free history."
                )
            db.execute(
                insert(jobs).values(
                    id=str(key),
                    owner=owner,
                    kind=kind,
                    payload=body.model_dump_json(),
                    request_id=request_id,
                    status="queued",
                    created=self.database.now(db),
                    lease_until=0,
                    attempts=0,
                )
            )
        return JobView(id=key, kind=kind, status="queued")

    @staticmethod
    def view(row: Any) -> JobView:
        return JobView(
            id=UUID(row["id"]),
            kind=row["kind"],
            status=row["status"],
            result_url=row["result_url"],
            error=None
            if row["error"] is None
            else ErrorDetail.model_validate_json(row["error"]),
        )

    def get(self, owner: str, key: UUID) -> JobView:
        with self.database.engine.connect() as db:
            row = (
                db.execute(
                    select(jobs).where(jobs.c.id == str(key), jobs.c.owner == owner)
                )
                .mappings()
                .first()
            )
        if row is None:
            raise APIError(404, "not_found", "Job not found.")
        return self.view(row)

    def list(self, owner: str) -> list[JobView]:
        with self.database.engine.connect() as db:
            return [
                self.view(row)
                for row in db.execute(
                    select(jobs)
                    .where(jobs.c.owner == owner)
                    .order_by(jobs.c.created.desc())
                ).mappings()
            ]

    def remove(self, owner: str, key: UUID) -> None:
        with self.database.engine.begin() as db:
            self.database.lock(db)
            row = (
                db.execute(
                    select(jobs).where(jobs.c.id == str(key), jobs.c.owner == owner)
                )
                .mappings()
                .first()
            )
            if row is None:
                raise APIError(404, "not_found", "Job not found.")
            if row["status"] in {"queued", "running"}:
                raise APIError(
                    409, "job_active", "Wait for the job to finish before deleting it."
                )
            db.execute(jobs.delete().where(jobs.c.id == str(key)))
            remove_artifact(db, owner, key)

    def close(self) -> None:
        pass

    def counts(self) -> dict[str, int]:
        with self.database.engine.connect() as db:
            return {
                str(status): int(count)
                for status, count in db.execute(
                    select(jobs.c.status, func.count()).group_by(jobs.c.status)
                )
            }

    def claim(self, *, seconds: int = 60) -> Lease | None:
        with self.database.engine.begin() as db:
            self.database.lock(db)
            now = self.database.now(db)
            row = (
                db.execute(
                    select(jobs)
                    .where(
                        or_(
                            jobs.c.status == "queued",
                            and_(jobs.c.status == "running", jobs.c.lease_until <= now),
                        )
                    )
                    .order_by(jobs.c.created, jobs.c.id)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            published = db.execute(
                select(artifacts.c.id).where(
                    artifacts.c.id == row["id"],
                    artifacts.c.owner == row["owner"],
                    artifacts.c.kind == row["kind"],
                )
            ).scalar()
            if published is not None:
                db.execute(
                    update(jobs)
                    .where(jobs.c.id == row["id"])
                    .values(
                        status="succeeded",
                        result_url=result_url(row["kind"], UUID(row["id"])),
                    )
                )
                return None
            if row["attempts"] >= 3:
                error = ErrorDetail(
                    code="worker_expired",
                    message="Worker attempts expired.",
                    request_id=row["request_id"],
                )
                db.execute(
                    update(jobs)
                    .where(jobs.c.id == row["id"])
                    .values(status="failed", error=error.model_dump_json())
                )
                return None
            token = str(uuid4())
            db.execute(
                update(jobs)
                .where(jobs.c.id == row["id"])
                .values(
                    status="running",
                    lease_until=now + seconds,
                    token=token,
                    attempts=row["attempts"] + 1,
                )
            )
            job = Job(
                row["owner"],
                UUID(row["id"]),
                cast(Kind, row["kind"]),
                MODELS[row["kind"]].model_validate_json(row["payload"]),
                row["request_id"],
            )
            return Lease(job, token)

    def renew(self, lease: Lease) -> bool:
        with self.database.engine.begin() as db:
            self.database.lock(db)
            now = self.database.now(db)
            return bool(
                db.execute(
                    update(jobs)
                    .where(
                        jobs.c.id == str(lease.job.id),
                        jobs.c.token == lease.token,
                        jobs.c.status == "running",
                        jobs.c.lease_until > now,
                    )
                    .values(lease_until=now + 60)
                ).rowcount
                == 1
            )

    def complete(self, lease: Lease, *, error: ErrorDetail | None = None) -> bool:
        with self.database.engine.begin() as db:
            self.database.lock(db)
            return bool(
                db.execute(
                    update(jobs)
                    .where(
                        jobs.c.id == str(lease.job.id),
                        jobs.c.token == lease.token,
                        jobs.c.status == "running",
                        jobs.c.lease_until > self.database.now(db),
                    )
                    .values(
                        status="failed" if error else "succeeded",
                        error=None if error is None else error.model_dump_json(),
                        result_url=None
                        if error
                        else result_url(lease.job.kind, lease.job.id),
                    )
                ).rowcount
                == 1
            )
