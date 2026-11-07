"""Immutable trained-model versions with local and MLflow registry adapters."""

import json
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from neurovision.mlops.provenance import LifecycleError, file_sha256
from neurovision.reporting.research import canonical_json, fingerprint


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    name: str
    version: str
    artifact_uri: str
    checkpoint_sha256: str
    metadata: dict[str, Any]


class ArtifactRegistry(Protocol):
    def register(
        self, name: str, checkpoint: Path, metadata: Path
    ) -> RegisteredModel: ...
    def resolve(self, name: str, version: str) -> RegisteredModel: ...


def _name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise LifecycleError("Registry names require a bounded simple identifier.")


def _metadata(checkpoint: Path, path: Path) -> dict[str, Any]:
    if path.is_symlink() or path.stat().st_size > 1_000_000:
        raise LifecycleError("Model metadata must be a bounded local file.")
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        unsigned = {
            key: value for key, value in data.items() if key != "metadata_sha256"
        }
        if (
            data["schema_version"] != 1
            or fingerprint(unsigned) != data["metadata_sha256"]
            or file_sha256(checkpoint) != data["checkpoint"]["sha256"]
        ):
            raise ValueError
        return data
    except (ValueError, TypeError, KeyError):
        raise LifecycleError(
            "Model metadata or checkpoint fingerprint differs."
        ) from None


class LocalArtifactRegistry:
    """SQLite versions and content-addressed checkpoint copies in a local directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / "objects"
        self.objects.mkdir(exist_ok=True)
        self.database = self.root / "registry.db"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS models "
                "(name TEXT, version INTEGER, digest TEXT, metadata TEXT, "
                "PRIMARY KEY(name, version), UNIQUE(name, digest, metadata))"
            )

    def register(self, name: str, checkpoint: Path, metadata: Path) -> RegisteredModel:
        _name(name)
        data = _metadata(checkpoint, metadata)
        digest = data["checkpoint"]["sha256"]
        destination = self.objects / (digest + ".pt")
        with tempfile.NamedTemporaryFile(
            dir=self.objects, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(checkpoint, temporary)
            if file_sha256(temporary) != digest:
                raise LifecycleError("Checkpoint changed during registry copy.")
            # Copy before committing a visible model version.
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        encoded = canonical_json(data)
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            found = db.execute(
                "SELECT version FROM models WHERE name=? AND digest=? AND metadata=?",
                (name, digest, encoded),
            ).fetchone()
            if found is None:
                latest = db.execute(
                    "SELECT COALESCE(MAX(version),0) FROM models WHERE name=?", (name,)
                ).fetchone()[0]
                number = int(latest) + 1
                db.execute(
                    "INSERT INTO models VALUES(?,?,?,?)",
                    (name, number, digest, encoded),
                )
            else:
                number = int(found[0])
        return self.resolve(name, str(number))

    def resolve(self, name: str, version: str) -> RegisteredModel:
        _name(name)
        if not version.isdecimal() or len(version) > 10 or int(version) < 1:
            raise LifecycleError("Resolve an explicit positive model version.")
        with closing(sqlite3.connect(self.database)) as db, db:
            row = db.execute(
                "SELECT digest, metadata FROM models WHERE name=? AND version=?",
                (name, int(version)),
            ).fetchone()
        if row is None:
            raise LifecycleError("Model version was not found.")
        checkpoint = self.objects / (row[0] + ".pt")
        if file_sha256(checkpoint) != row[0]:
            raise LifecycleError("Registered checkpoint failed integrity validation.")
        metadata = json.loads(row[1])
        unsigned = {
            key: value for key, value in metadata.items() if key != "metadata_sha256"
        }
        if fingerprint(unsigned) != metadata.get("metadata_sha256"):
            raise LifecycleError("Registered metadata failed integrity validation.")
        return RegisteredModel(name, version, checkpoint.as_uri(), row[0], metadata)


class MLflowArtifactRegistry:
    """Register explicit checkpoint versions in the same local MLflow store."""

    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker
        self.client = tracker.client

    def register(self, name: str, checkpoint: Path, metadata: Path) -> RegisteredModel:
        from mlflow.exceptions import MlflowException

        _name(name)
        data = _metadata(checkpoint, metadata)
        run_id = self.tracker.run_id
        if not run_id or self.client.get_run(run_id).info.status != "FINISHED":
            raise LifecycleError("Finish a tracked run before registry publication.")
        try:
            self.client.create_registered_model(name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        info = self.client.get_run(run_id).info
        # Publication may use a caller's best checkpoint from the tracked run.
        artifact_path = "registered/" + data["metadata_sha256"]
        self.client.log_artifact(run_id, str(checkpoint), artifact_path)
        self.client.log_artifact(run_id, str(metadata), artifact_path)
        entry = self.client.create_model_version(
            name,
            source=info.artifact_uri.rstrip("/")
            + "/"
            + artifact_path
            + "/"
            + checkpoint.name,
            run_id=run_id,
            tags={
                "checkpoint_sha256": data["checkpoint"]["sha256"],
                "metadata_sha256": data["metadata_sha256"],
                "metadata_path": artifact_path + "/" + metadata.name,
                "checkpoint_path": artifact_path + "/" + checkpoint.name,
            },
        )
        return self.resolve(name, str(entry.version))

    def resolve(self, name: str, version: str) -> RegisteredModel:
        _name(name)
        if not version.isdecimal():
            raise LifecycleError("Resolve an explicit model version.")
        entry = self.client.get_model_version(name, version)
        metadata_path = Path(
            self.client.download_artifacts(entry.run_id, entry.tags["metadata_path"])
        )
        checkpoint = Path(
            self.client.download_artifacts(entry.run_id, entry.tags["checkpoint_path"])
        )
        data = _metadata(checkpoint, metadata_path)
        if data["metadata_sha256"] != entry.tags["metadata_sha256"]:
            raise LifecycleError("Registered metadata failed integrity validation.")
        return RegisteredModel(
            name,
            version,
            str(entry.source),
            entry.tags["checkpoint_sha256"],
            data,
        )
