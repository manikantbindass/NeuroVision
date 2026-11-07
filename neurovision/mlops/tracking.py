"""Opt-in MLflow experiment tracking through a replaceable lifecycle interface."""

import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from neurovision.mlops.provenance import LifecycleError
from neurovision.reporting.research import canonical_json


class ExperimentTracker(Protocol):
    def start(self, metadata: dict[str, Any]) -> str: ...
    def metrics(self, epoch: int, values: Mapping[str, float]) -> None: ...
    def finish(self, status: str, artifacts: tuple[Path, ...]) -> None: ...


@contextmanager
def tracking_attempt(
    tracker: ExperimentTracker | None,
    metadata: dict[str, Any],
    output: Path,
) -> Iterator[str | None]:
    if tracker is None:
        yield None
        return
    try:
        run_id = tracker.start(metadata)
    except Exception:
        raise LifecycleError("Experiment tracking could not start.") from None
    try:
        yield run_id
    except BaseException:
        try:
            tracker.finish("FAILED", ())
        except Exception:  # noqa: S110 - preserve the original training failure
            pass
        raise
    else:
        artifacts = tuple(
            output / name
            for name in (
                "best.pt",
                "artifact.json",
                "history.json",
                "environment.json",
                "configuration.json",
                "reproducibility.json",
                "MODEL_CARD.md",
            )
            if (output / name).is_file()
        )
        try:
            tracker.finish("FINISHED", artifacts)
        except Exception:
            raise LifecycleError(
                "Training artifacts are saved, but tracker finalization failed."
            ) from None


class MLflowTracker:
    """One local SQLite-backed run per instance. No process-global active run."""

    def __init__(
        self, directory: Path, *, experiment: str = "neurovision-research"
    ) -> None:
        os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
        os.environ.setdefault("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", "NullPool")
        try:
            from mlflow.tracking import MlflowClient
        except ImportError:
            raise LifecycleError("MLflow tracking requires the mlops extra.") from None
        self.root = directory.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = "sqlite:///" + (self.root / "mlflow.db").as_posix()
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        found = self.client.get_experiment_by_name(experiment)
        self.experiment_id = (
            found.experiment_id
            if found is not None
            else self.client.create_experiment(
                experiment, artifact_location=(self.root / "artifacts").as_uri()
            )
        )
        # Reject an existing experiment whose artifact destination was changed.
        info = self.client.get_experiment(self.experiment_id)
        if info.artifact_location != (self.root / "artifacts").as_uri():
            raise LifecycleError(
                "MLflow experiment must use this local artifact directory."
            )
        self.run_id: str | None = None

    def start(self, metadata: dict[str, Any]) -> str:
        if self.run_id is not None:
            raise LifecycleError("Use a fresh MLflow tracker per training attempt.")
        tags = {"neurovision.research_only": "true"}
        git = metadata.get("identity", {}).get("git", {})
        if git.get("commit_sha"):
            tags["mlflow.source.git.commit"] = str(git["commit_sha"])
        run = self.client.create_run(self.experiment_id, tags=tags)
        self.run_id = str(run.info.run_id)
        try:
            self.client.log_text(
                self.run_id, canonical_json(metadata), "provenance/run.json"
            )
        except Exception:
            self.client.set_terminated(self.run_id, status="FAILED")
            raise
        return self.run_id

    def metrics(self, epoch: int, values: Mapping[str, float]) -> None:
        if self.run_id is None:
            raise LifecycleError("Start tracking before logging metrics.")
        for key, value in values.items():
            if not math.isfinite(value):
                raise LifecycleError("Tracking metrics must be finite.")
            self.client.log_metric(self.run_id, key, value, step=epoch)

    def finish(self, status: str, artifacts: tuple[Path, ...]) -> None:
        if self.run_id is None or status not in {"FINISHED", "FAILED"}:
            raise LifecycleError("Invalid tracking completion state.")
        try:
            for path in artifacts:
                self.client.log_artifact(self.run_id, str(path), "research")
        except Exception:
            self.client.set_terminated(self.run_id, status="FAILED")
            raise
        self.client.set_terminated(self.run_id, status=status)
