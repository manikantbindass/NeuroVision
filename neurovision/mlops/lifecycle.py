"""Persist self-describing training artifacts and research model cards."""

from pathlib import Path
from typing import Any

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.mlops.provenance import file_sha256
from neurovision.reporting.research import fingerprint, timestamp
from neurovision.training.checkpoints import save_history


def artifact_metadata(
    output: Path,
    state: dict[str, Any],
    *,
    epoch_budget: int,
) -> dict[str, Any]:
    run = state["run"]
    identity = run.get("mlops", {})
    history = state["history"]
    best = history[state["best_epoch"] - 1]
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "notice": RESEARCH_NOTICE,
        "created_at": timestamp(),
        "tracking_run_id": state.get("tracking_run_id"),
        "architecture": run["model_definition"],
        "hyperparameters": run["recipe"],
        "epoch_budget": epoch_budget,
        "completed_epochs": len(history),
        "selected_epoch": state["best_epoch"],
        "dataset_reference": identity.get("dataset", {"reference": run["data_id"]}),
        "preprocessing": identity.get("dataset", {}).get("preprocessing"),
        "metrics": best,
        "commit": identity.get("git"),
        "dependencies": identity.get("dependencies"),
        "identity": identity,
        "checkpoint": {
            "filename": "best.pt",
            "sha256": file_sha256(output / "best.pt"),
        },
    }
    metadata["metadata_sha256"] = fingerprint(metadata)
    return metadata


def persist_lifecycle(
    output: Path,
    state: dict[str, Any],
    *,
    epoch_budget: int,
) -> dict[str, Any]:
    metadata = artifact_metadata(output, state, epoch_budget=epoch_budget)
    identity = metadata["identity"]
    warnings = []
    if not identity.get("git", {}).get("available"):
        warnings.append("Git revision unavailable.")
    elif identity["git"]["dirty"]:
        warnings.append("Working tree contains uncommitted changes.")
    if not identity.get("training_tensors_sha256") or not identity.get(
        "validation_tensors_sha256"
    ):
        if not identity.get("dataset", {}).get("fingerprint", {}).get("content_sha256"):
            warnings.append("Dataset content identity is incomplete.")
    if (
        not metadata["preprocessing"]
        or metadata["preprocessing"].get("availability") == "not supplied"
    ):
        warnings.append("Preprocessing provenance was not supplied.")
    save_history(output / "artifact.json", metadata)
    save_history(
        output / "environment.json", {"dependencies": metadata["dependencies"]}
    )
    save_history(
        output / "configuration.json",
        {
            "schema_version": 1,
            "configuration": state["run"]["recipe"],
            "configuration_sha256": identity.get("configuration_sha256"),
            "epoch_budget": epoch_budget,
        },
    )
    save_history(
        output / "reproducibility.json",
        {
            "schema_version": 1,
            "notice": RESEARCH_NOTICE,
            "artifact_metadata_sha256": metadata["metadata_sha256"],
            "warnings": warnings,
            "checkpoint_sha256": metadata["checkpoint"]["sha256"],
            "configuration_sha256": identity.get("configuration_sha256"),
            "training_tensors_sha256": identity.get("training_tensors_sha256"),
            "validation_tensors_sha256": identity.get("validation_tensors_sha256"),
            "commit": metadata["commit"],
            "created_at": metadata["created_at"],
        },
    )
    # JSON is the authoritative complete record. The card summarizes known facts.
    card = (
        "# Research model card\n\n" + RESEARCH_NOTICE + "\n\n"
        "**MRI benchmark status: Not yet benchmarked.**\n\n"
        "## Architecture and selection\n\n"
        f"Architecture: {metadata['architecture']['architecture']}. "
        f"Selected epoch: {metadata['selected_epoch']} by validation loss.\n\n"
        "## Dataset and preprocessing\n\n"
        "The complete dataset reference, fingerprints, configuration and preprocessing "
        "declarations are recorded in artifact.json. Review reproducibility.json for "
        "missing provenance and code-state warnings.\n\n"
        "## Evaluation and limitations\n\n"
        "artifact.json records observed validation metrics with defined-sample counts. "
        "These measurements describe only the supplied experiment. Synthetic fixtures "
        "cannot establish MRI accuracy. Calibration, domain shift and subgroup "
        "performance require independent research validation.\n\n"
        "## Reproducibility\n\n"
        f"Checkpoint SHA-256: {metadata['checkpoint']['sha256']}\n\n"
        "Dependencies are captured as package names and versions. Hardware differences "
        "and nondeterministic operations can affect replay.\n"
    )
    (output / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    return metadata
