"""Run a versioned synthetic experiment with optional local MLflow tracking."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.mlops.provenance import DataProvenance
from neurovision.mlops.registry import LocalArtifactRegistry, MLflowArtifactRegistry
from neurovision.mlops.tracking import MLflowTracker
from neurovision.training.checkpoints import load_checkpoint, save_history
from neurovision.training.config import load_experiment
from neurovision.training.example import synthetic_dataset
from neurovision.training.runner import fit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a versioned synthetic experiment.", epilog=RESEARCH_NOTICE
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mlflow", type=Path)
    parser.add_argument("--register")
    args = parser.parse_args(argv)
    config = load_experiment(args.config)
    training = synthetic_dataset(config.model, samples=4, seed=2 * config.training.seed)
    validation = synthetic_dataset(
        config.model, samples=2, seed=2 * config.training.seed + 1
    )
    tracker = MLflowTracker(args.mlflow) if args.mlflow is not None else None
    result = fit(
        training,
        validation,
        config,
        output=args.output,
        data_id="synthetic-ellipsoids-v1",
        provenance=DataProvenance(
            "neurovision.synthetic.ellipsoids",
            "1",
            preprocessing={
                "generator": "synthetic_dataset",
                "size": 16,
                "seed": config.training.seed,
                "training_samples": 4,
                "validation_samples": 2,
                "external_steps": [],
            },
        ),
        tracker=tracker,
    )
    best = load_checkpoint(args.output / "best.pt")
    save_history(
        args.output / "metrics.json",
        {
            "best_validation_loss": best["best_loss"],
            "best_epoch": result.best_epoch,
            "completed_epochs": len(result.history),
        },
    )
    if args.register:
        registry = LocalArtifactRegistry(args.output / "registry")
        local = registry.register(
            args.register, args.output / "best.pt", args.output / "artifact.json"
        )
        record = {"name": local.name, "local_version": local.version}
        if tracker is not None:
            remote = MLflowArtifactRegistry(tracker).register(
                args.register, args.output / "best.pt", args.output / "artifact.json"
            )
            record["mlflow_version"] = remote.version
        save_history(args.output / "registry-reference.json", record)
    print(RESEARCH_NOTICE)
    print(
        json.dumps(
            {"epochs": len(result.history), "tracking_run_id": result.tracking_run_id}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
