"""Versioned local dataset manifests and validation."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from neurovision.core.errors import NeuroVisionError
from neurovision.imaging.limits import (
    DEFAULT_LIMITS,
    IngestionLimits,
    bounded_read,
    regular_file_size,
)
from neurovision.imaging.nifti import load_nifti, nifti_suffix
from neurovision.imaging.volume import ImagingError, Modality, MRIVolume

_CASE_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}")
_SPLITS = {"train", "validation", "test", "unassigned"}
_MAX_CASES = 10_000
_MANIFEST_BYTES = 8 * 1024 * 1024


class DatasetError(NeuroVisionError):
    """A manifest or dataset failed validation."""


@dataclass(frozen=True, slots=True)
class VolumeEntry:
    modality: Modality
    relative_path: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class DatasetCase:
    case_id: str = field(repr=False)
    volumes: tuple[VolumeEntry, ...]
    split: str = "unassigned"
    label: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    cases: tuple[DatasetCase, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.cases) <= _MAX_CASES:
            raise DatasetError("Manifest must contain between 1 and 10000 cases")
        ids: set[str] = set()
        paths: set[str] = set()
        for case in self.cases:
            if not _CASE_ID.fullmatch(case.case_id) or case.case_id.casefold() in ids:
                raise DatasetError(
                    "Manifest case IDs must be valid and unique across splits"
                )
            ids.add(case.case_id.casefold())
            if case.split not in _SPLITS:
                raise DatasetError("Manifest split is invalid")
            modalities = {entry.modality for entry in case.volumes}
            if (
                not modalities
                or Modality.UNKNOWN in modalities
                or len(modalities) != len(case.volumes)
            ):
                raise DatasetError(
                    "Each case needs distinct, explicitly named MRI modalities"
                )
            for path in [*(entry.relative_path for entry in case.volumes), case.label]:
                if path is None:
                    continue
                _validate_path(path)
                key = path.casefold()
                if key in paths:
                    raise DatasetError("A manifest file is assigned more than once")
                paths.add(key)

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "cases": [
                        {
                            "id": case.case_id,
                            "split": case.split,
                            "volumes": {
                                entry.modality.value: entry.relative_path
                                for entry in case.volumes
                            },
                            **({"label": case.label} if case.label is not None else {}),
                        }
                        for case in self.cases
                    ],
                },
                indent=2,
            )
            + "\n"
        )


class DatasetAdapter(Protocol):
    """Discover a manifest for a dataset already staged on local storage."""

    def build_manifest(self, root: Path) -> DatasetManifest: ...


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    cases: int
    volumes: int
    labels: int


def read_manifest(path: Path) -> DatasetManifest:
    try:
        if regular_file_size(path) > _MANIFEST_BYTES:
            raise DatasetError("Dataset manifest exceeds the configured byte limit")
        with path.open("rb") as stream:
            content = bounded_read(stream, _MANIFEST_BYTES)
        return _parse_manifest(
            json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object)
        )
    except (OSError, ImagingError, ValueError, RecursionError):
        raise DatasetError("Cannot read a valid dataset manifest") from None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError("Dataset manifest contains duplicate JSON keys")
        result[key] = value
    return result


def _parse_manifest(value: object) -> DatasetManifest:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "cases"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or not isinstance(value["cases"], list)
    ):
        raise DatasetError("Expected a version 1 dataset manifest")
    if not 1 <= len(value["cases"]) <= _MAX_CASES:
        raise DatasetError("Manifest must contain between 1 and 10000 cases")
    cases = []
    for entry in value["cases"]:
        if (
            not isinstance(entry, dict)
            or set(entry) - {"id", "split", "volumes", "label"}
            or not isinstance(entry.get("id"), str)
            or not isinstance(entry.get("volumes"), dict)
            or not isinstance(entry.get("split", "unassigned"), str)
            or ("label" in entry and not isinstance(entry["label"], str))
        ):
            raise DatasetError("Invalid dataset case entry")
        volumes = []
        for key, path in entry["volumes"].items():
            if not isinstance(path, str):
                raise DatasetError("Manifest volume paths must be strings")
            try:
                modality = Modality(key)
            except ValueError:
                raise DatasetError(
                    "Manifest contains an unsupported modality"
                ) from None
            volumes.append(VolumeEntry(modality, path))
        cases.append(
            DatasetCase(
                entry["id"],
                tuple(volumes),
                entry.get("split", "unassigned"),
                entry.get("label"),
            )
        )
    return DatasetManifest(tuple(cases))


def _validate_path(path: str) -> None:
    parts = path.split("/")
    if (
        len(path) > 1024
        or "\\" in path
        or ":" in path
        or any(ord(c) < 32 for c in path)
        or any(
            not part or part in (".", "..") or part.endswith((" ", "."))
            for part in parts
        )
        or not nifti_suffix(path)
    ):
        raise DatasetError(
            "Manifest paths must be relative NIfTI paths inside the dataset root"
        )


def validate_dataset(
    manifest: DatasetManifest,
    root: Path,
    *,
    required_modalities: frozenset[Modality] = frozenset(),
    limits: IngestionLimits = DEFAULT_LIMITS,
) -> DatasetSummary:
    """Validate bounded file decoding and a shared physical grid for each case."""
    seen: set[Path] = set()
    try:
        root = root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise DatasetError("Dataset root is inaccessible") from None
    if not root.is_dir():
        raise DatasetError("Dataset root must be a directory")
    total_volumes = total_labels = 0
    for case in manifest.cases:
        if not required_modalities.issubset(entry.modality for entry in case.volumes):
            raise DatasetError("A dataset case is missing required modalities")
        shape: tuple[int, ...] | None = None
        affine = None
        reference_space = None
        entries = [
            (entry.relative_path, entry.modality, False) for entry in case.volumes
        ]
        if case.label is not None:
            entries.append((case.label, Modality.UNKNOWN, True))
        for relative, modality, is_label in entries:
            volume = _load(root, relative, modality, limits, seen, label_map=is_label)
            if shape is None:
                shape, affine, reference_space = (
                    volume.data.shape,
                    volume.affine,
                    volume.reference_space,
                )
            elif (
                affine is None
                or shape != volume.data.shape
                or reference_space != volume.reference_space
                or not np.allclose(affine, volume.affine, rtol=0, atol=1e-3)
            ):
                raise DatasetError(
                    "Dataset case volumes and labels do not share a physical grid"
                )
            if is_label:
                total_labels += 1
            else:
                total_volumes += 1
    return DatasetSummary(len(manifest.cases), total_volumes, total_labels)


def resolve_dataset_file(root: Path, relative: str) -> Path:
    """Resolve a validated manifest path without leaving its dataset root."""
    _validate_path(relative)
    try:
        root = root.resolve(strict=True)
        path = (root / relative).resolve(strict=True)
    except (OSError, RuntimeError):
        raise DatasetError("Manifest references an inaccessible dataset file") from None
    if not root.is_dir() or not path.is_relative_to(root):
        raise DatasetError("Dataset file must stay inside its dataset root")
    return path


def _load(
    root: Path,
    relative: str,
    modality: Modality,
    limits: IngestionLimits,
    seen: set[Path],
    *,
    label_map: bool,
) -> MRIVolume:
    path = resolve_dataset_file(root, relative)
    if path in seen:
        raise DatasetError(
            "Dataset files must be unique and stay inside the dataset root"
        )
    seen.add(path)
    try:
        return load_nifti(path, modality=modality, limits=limits, label_map=label_map)
    except ImagingError as error:
        raise DatasetError(f"Dataset volume failed validation: {error}") from None
