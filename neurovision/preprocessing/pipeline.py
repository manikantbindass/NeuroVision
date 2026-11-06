"""Run an ordered preprocessing recipe with provenance, QC, and bounded cache."""

import hashlib
import json
import platform
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from threading import Lock

import numpy as np

from neurovision.core.pipeline import RunContext, Stage
from neurovision.imaging.volume import ImagingError, MRIVolume
from neurovision.preprocessing.contracts import (
    Fields,
    Grid,
    PreprocessingError,
    Transform,
    volume_digest,
)
from neurovision.preprocessing.qc import (
    DEFAULT_QC_POLICY,
    Finding,
    QCPolicy,
    QCReport,
    assess_volume,
)

_STEP_FAILURES = (
    PreprocessingError,
    ImagingError,
    ValueError,
    TypeError,
    RuntimeError,
    ArithmeticError,
)


@dataclass(frozen=True, slots=True)
class StepRecord:
    index: int
    operation: str
    seed: int
    parameters: Fields
    input_grid: Grid | None
    input_digest: str | None
    output_grid: Grid | None = None
    output_digest: str | None = None
    details: Fields = ()
    failed: bool = False


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    volume: MRIVolume | None
    report: QCReport
    steps: tuple[StepRecord, ...]
    software: tuple[tuple[str, str], ...]
    run_id: str
    seed: int
    cache_hit: bool = False

    def to_json(self) -> str:
        """Serialize QC and provenance for storage with the research run."""
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "seed": self.seed,
                    "cache_hit": self.cache_hit,
                    "software": dict(self.software),
                    "qc": self.report.as_dict(),
                    "steps": [
                        {
                            **asdict(step),
                            "parameters": dict(step.parameters),
                            "details": dict(step.details),
                        }
                        for step in self.steps
                    ],
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )


@dataclass(frozen=True, slots=True)
class _Cached:
    volume: MRIVolume
    records: tuple[StepRecord, ...]

    @property
    def size(self) -> int:
        return self.volume.data.nbytes + self.volume.affine.nbytes


class PreprocessingCache:
    """Process-local LRU. Limits cover entries and stored array bytes."""

    def __init__(self, *, max_entries: int = 8, max_bytes: int = 256 * 1024**2) -> None:
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("Cache limits must be positive")
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._size = 0
        self._items: OrderedDict[str, _Cached] = OrderedDict()
        self._lock = Lock()

    def _get(self, key: str) -> _Cached | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is not None:
                self._items.move_to_end(key)
            return entry

    def _put(self, key: str, entry: _Cached) -> None:
        if entry.size > self._max_bytes:
            return
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return
            while self._items and (
                self._size + entry.size > self._max_bytes
                or len(self._items) >= self._max_entries
            ):
                self._size -= self._items.popitem(last=False)[1].size
            volume = entry.volume
            # Byte-backed arrays prevent a caller from changing a shared cached volume.
            frozen = replace(
                volume,
                data=np.frombuffer(volume.data.tobytes(), dtype=np.float32).reshape(
                    volume.data.shape
                ),
                affine=np.frombuffer(volume.affine.tobytes(), dtype=np.float64).reshape(
                    (4, 4)
                ),
            )
            self._items[key] = _Cached(frozen, entry.records)
            self._size += entry.size


def _software() -> tuple[tuple[str, str], ...]:
    versions = [("python", platform.python_version())]
    for package in ("neurovision-2028", "numpy", "scipy", "nibabel", "SimpleITK"):
        try:
            versions.append((package, version(package)))
        except PackageNotFoundError:
            versions.append((package, "unavailable"))
    return tuple(versions)


@dataclass(frozen=True, slots=True)
class PreprocessingPipeline:
    steps: tuple[Transform, ...]
    policy: QCPolicy = DEFAULT_QC_POLICY
    cache: PreprocessingCache | None = None

    stage = Stage.PREPROCESSING

    def run(self, inputs: MRIVolume, *, context: RunContext) -> PreprocessingResult:
        software = _software()
        before = assess_volume(inputs, policy=self.policy, scope="input")
        if before.status == "fail":
            return PreprocessingResult(
                None, before, (), software, context.run_id, context.seed
            )

        rng = context.rng()
        start_digest = volume_digest(inputs)
        recipe = []
        for index, step in enumerate(self.steps):
            name = f"{type(step).__module__}.{type(step).__qualname__}"
            seed = rng.getrandbits(63)
            try:
                parameters = tuple(sorted(step.parameters().items()))
                json.dumps(parameters, allow_nan=False)
            except _STEP_FAILURES:
                failed = StepRecord(index, name, seed, (), None, None, failed=True)
                return self._failed(before, (failed,), software, context)
            recipe.append((name, seed, parameters))
        key = None
        if self.cache is not None and all(step.cacheable for step in self.steps):
            payload = json.dumps(
                (1, software, start_digest, context.seed, recipe), allow_nan=False
            )
            key = hashlib.sha256(payload.encode()).hexdigest()
            cached = self.cache._get(key)
            if cached is not None:
                return self._finish(
                    cached.volume, cached.records, before, software, context, True
                )

        volume, digest = inputs, start_digest
        records = []
        for index, (step, (name, seed, parameters)) in enumerate(
            zip(self.steps, recipe, strict=True)
        ):
            record = StepRecord(
                index, name, seed, parameters, Grid.from_volume(volume), digest
            )
            try:
                output, details = step.apply(volume, seed=seed)
                json.dumps(details, allow_nan=False)
                next_digest = volume_digest(output)
                output_grid = Grid.from_volume(output)
                json.dumps(asdict(output_grid), allow_nan=False)
            except _STEP_FAILURES:
                records.append(replace(record, failed=True))
                return self._failed(before, tuple(records), software, context)
            records.append(
                replace(
                    record,
                    output_grid=output_grid,
                    output_digest=next_digest,
                    details=tuple(sorted(details.items())),
                )
            )
            volume, digest = output, next_digest
        result = self._finish(volume, tuple(records), before, software, context, False)
        if key is not None and self.cache is not None and result.volume is not None:
            self.cache._put(key, _Cached(result.volume, result.steps))
        return result

    def _failed(
        self,
        before: QCReport,
        records: tuple[StepRecord, ...],
        software: tuple[tuple[str, str], ...],
        context: RunContext,
    ) -> PreprocessingResult:
        finding = Finding(
            "preprocessing_failure",
            "fail",
            f"step.{records[-1].index}",
            "Preprocessing could not complete the recorded operation",
        )
        report = QCReport((*before.findings, finding), before.metrics, self.policy)
        return PreprocessingResult(
            None, report, records, software, context.run_id, context.seed
        )

    def _finish(
        self,
        volume: MRIVolume,
        records: tuple[StepRecord, ...],
        before: QCReport,
        software: tuple[tuple[str, str], ...],
        context: RunContext,
        cache_hit: bool,
    ) -> PreprocessingResult:
        after = assess_volume(volume, policy=self.policy, scope="output")
        findings = [*before.findings, *after.findings]
        metrics = [*before.metrics, *after.metrics]
        for record in records:
            dice = dict(record.details).get("registration_foreground_dice")
            if isinstance(dice, (float, int)):
                scope = f"step.{record.index}"
                metrics.append((f"{scope}.registration_foreground_dice", dice))
                if dice < self.policy.min_registration_dice:
                    findings.append(
                        Finding(
                            "registration_overlap",
                            "warning",
                            scope,
                            "Foreground overlap is below the registration threshold",
                        )
                    )
            determinant = dict(record.details).get("registration_linear_determinant")
            if isinstance(determinant, (float, int)):
                scope = f"step.{record.index}"
                metrics.append(
                    (f"{scope}.registration_linear_determinant", determinant)
                )
                if determinant < 0:
                    findings.append(
                        Finding(
                            "registration_reflection",
                            "warning",
                            scope,
                            "Registration reverses spatial handedness",
                        )
                    )
        report = QCReport(tuple(findings), tuple(metrics), self.policy)
        return PreprocessingResult(
            volume if report.status != "fail" else None,
            report,
            records,
            software,
            context.run_id,
            context.seed,
            cache_hit,
        )
