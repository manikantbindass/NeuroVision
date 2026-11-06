"""Content identities and environment metadata without paths or credentials."""

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from importlib.metadata import distributions
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import TensorDataset

from neurovision.core.errors import PipelineError
from neurovision.reporting.research import canonical_json, fingerprint

if TYPE_CHECKING:
    from neurovision.datasets.manifest import DatasetManifest


class LifecycleError(PipelineError):
    """Lifecycle metadata or a tracking adapter failed validation."""


def file_sha256(path: Path, *, max_bytes: int = 4 * 1024**3) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise LifecycleError("Artifact exceeds its fingerprint size limit")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_dataset_sha256(dataset: object) -> str | None:
    """Fingerprint stored tensors in order, without evaluating dataset transforms."""
    if not isinstance(dataset, TensorDataset):
        return None
    digest = hashlib.sha256()
    for tensor in dataset.tensors:
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise LifecycleError("Dataset fingerprint requires dense CPU tensors")
        value = tensor.detach().contiguous()
        digest.update(
            canonical_json(
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            ).encode()
        )
        raw = value.reshape(-1).view(torch.uint8).numpy()
        # Chunk iteration bounds transient hashing buffers.
        for offset in range(0, raw.size, 1024 * 1024):
            digest.update(raw[offset : offset + 1024 * 1024].tobytes())
    return digest.hexdigest()


def manifest_fingerprint(root: Path, manifest: "DatasetManifest") -> dict[str, Any]:
    """Bind local file bytes, modality, case membership, split and canonical paths."""
    from neurovision.datasets.manifest import resolve_dataset_file

    records = []
    for case in sorted(manifest.cases, key=lambda item: item.case_id):
        members = [
            (entry.modality.value, entry.relative_path) for entry in case.volumes
        ]
        if case.label is not None:
            members.append(("labels", case.label))
        records.append(
            {
                "case": case.case_id,
                "split": case.split,
                "members": [
                    {
                        "role": role,
                        "path": relative,
                        "sha256": file_sha256(resolve_dataset_file(root, relative)),
                    }
                    for role, relative in sorted(members)
                ],
            }
        )
    # Only hashes and counts leave this function, not case names or file paths.
    return {
        "method": "manifest-and-file-sha256-v1",
        "manifest_sha256": fingerprint(json.loads(manifest.to_json())),
        "content_sha256": fingerprint({"cases": records}),
        "cases": len(records),
        "splits": {
            split: fingerprint(
                {"cases": [row for row in records if row["split"] == split]}
            )
            for split in sorted({case.split for case in manifest.cases})
        },
    }


def git_snapshot(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()

    try:
        sha = git("rev-parse", "--verify", "HEAD")
        if not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", sha):
            raise ValueError
        dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"commit_sha": None, "dirty": None, "available": False}
    return {"commit_sha": sha, "dirty": dirty, "available": True}


def environment_snapshot() -> dict[str, str]:
    return dict(
        sorted(
            (str(dist.metadata["Name"]).lower(), dist.version)
            for dist in distributions()
            if dist.metadata["Name"]
        )
    )


@dataclass(frozen=True, slots=True)
class DataProvenance:
    reference: str
    version: str
    fingerprint: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(canonical_json(asdict(self)))
        if (
            not self.reference.strip()
            or not self.version.strip()
            or len(canonical_json(result).encode()) > 65_536
        ):
            raise LifecycleError("Expected bounded, versioned dataset provenance")
        return result


def capture_identity(
    recipe: dict[str, Any],
    data_id: str,
    training: object,
    validation: object,
    provenance: DataProvenance | None,
    code_root: Path,
) -> dict[str, Any]:
    declared = (
        provenance.as_dict()
        if provenance is not None
        else {
            "reference": data_id,
            "version": "caller-declared",
            "fingerprint": {},
            "preprocessing": {"availability": "not supplied"},
        }
    )
    return {
        "schema_version": 1,
        "configuration_sha256": fingerprint(recipe),
        "dataset": declared,
        "training_tensors_sha256": tensor_dataset_sha256(training),
        "validation_tensors_sha256": tensor_dataset_sha256(validation),
        "git": git_snapshot(code_root),
        "dependencies": environment_snapshot(),
    }
