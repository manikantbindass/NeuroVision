"""Atomic local artifacts and restricted checkpoint loading."""

import json
import os
import pickle
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from neurovision.core.errors import PipelineError


class TrainingError(PipelineError):
    """Training could not proceed with the supplied configuration or data."""


def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    _atomic(path, lambda temporary: torch.save(state, temporary))


def save_history(path: Path, state: dict[str, Any]) -> None:
    content = json.dumps(state, indent=2, allow_nan=False) + "\n"

    def write(temporary: Path) -> None:
        temporary.write_text(content, encoding="utf-8")

    _atomic(path, write)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted local checkpoint using PyTorch's restricted tensor loader."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
        raise TrainingError("Cannot read a compatible local checkpoint") from None
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise TrainingError("Unsupported checkpoint schema")
    return state
