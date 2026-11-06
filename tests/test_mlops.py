"""Regression coverage for provenance, replay contracts and local registries."""

import json
import os
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from neurovision.core.errors import ConfigurationError
from neurovision.datasets.manifest import DatasetCase, DatasetManifest, VolumeEntry
from neurovision.imaging.volume import Modality
from neurovision.mlops.provenance import (
    DataProvenance,
    LifecycleError,
    manifest_fingerprint,
    tensor_dataset_sha256,
)
from neurovision.mlops.registry import LocalArtifactRegistry, MLflowArtifactRegistry
from neurovision.mlops.tracking import MLflowTracker
from neurovision.models.checkpoints import CheckpointManager
from neurovision.models.unet import UNetConfig
from neurovision.training.checkpoints import (
    TrainingError,
    load_checkpoint,
    save_checkpoint,
)
from neurovision.training.config import (
    ExperimentConfig,
    TrainingConfig,
    load_experiment,
)
from neurovision.training.example import synthetic_dataset
from neurovision.training.runner import fit


class LifecycleTests(unittest.TestCase):
    def test_tracking_finalize_failure_keeps_complete_checkpoint(self) -> None:
        class FailingTracker:
            def start(self, metadata: dict[str, Any]) -> str:
                return "tracking-fixture"

            def metrics(self, epoch: int, values: Mapping[str, float]) -> None:
                pass

            def finish(self, status: str, artifacts: tuple[Path, ...]) -> None:
                raise RuntimeError("simulated artifact-store failure")

        cfg = ExperimentConfig(
            UNetConfig(2, features=(4, 8)), training=TrainingConfig(epochs=1)
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(LifecycleError, "artifacts are saved"):
                fit(
                    synthetic_dataset(cfg.model, samples=2, seed=31),
                    synthetic_dataset(cfg.model, samples=2, seed=32),
                    cfg,
                    output=output,
                    data_id="failure-fixture",
                    tracker=FailingTracker(),
                )
            self.assertEqual(len(load_checkpoint(output / "last.pt")["history"]), 1)
            self.assertTrue((output / "artifact.json").is_file())
            recipe = output / "unsupported.json"
            recipe.write_text('{"schema_version":2,"model":{"spatial_dims":2}}')
            with self.assertRaises(ConfigurationError):
                load_experiment(recipe)

    def test_fingerprints_bind_bytes_split_and_tensor_content_without_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "subject.nii").write_bytes(b"synthetic fixture")
            case = DatasetCase(
                "private-study-key", (VolumeEntry(Modality.T1, "subject.nii"),), "train"
            )
            first = manifest_fingerprint(root, DatasetManifest((case,)))
            (root / "subject.nii").write_bytes(b"changed")
            second = manifest_fingerprint(root, DatasetManifest((case,)))
            self.assertNotEqual(first["content_sha256"], second["content_sha256"])
            third = manifest_fingerprint(
                root, DatasetManifest((replace(case, split="test"),))
            )
            self.assertNotEqual(second["content_sha256"], third["content_sha256"])
            self.assertNotIn("subject", json.dumps(first))
            self.assertNotIn("private-study-key", json.dumps(first))
        dataset = synthetic_dataset(UNetConfig(2), samples=1, seed=4)
        first_tensor = tensor_dataset_sha256(dataset)
        self.assertIsNotNone(first_tensor)
        images, _ = dataset[0]
        images.add_(1)
        self.assertNotEqual(first_tensor, tensor_dataset_sha256(dataset))

    def test_training_metadata_registry_integrity_and_legacy_resume(self) -> None:
        cfg = ExperimentConfig(
            UNetConfig(2, features=(4, 8)), training=TrainingConfig(epochs=1)
        )
        train = synthetic_dataset(cfg.model, samples=2, seed=1)
        valid = synthetic_dataset(cfg.model, samples=2, seed=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            with patch.dict(os.environ, {"RESEARCH_TEST_SECRET": "must-not-appear"}):
                fit(
                    train,
                    valid,
                    cfg,
                    output=output,
                    data_id="fixture",
                    provenance=DataProvenance(
                        "synthetic", "1", preprocessing={"generator": "ellipsoids"}
                    ),
                )
            metadata = json.loads((output / "artifact.json").read_text())
            for field in (
                "architecture",
                "hyperparameters",
                "dataset_reference",
                "preprocessing",
                "metrics",
                "commit",
                "dependencies",
            ):
                self.assertIn(field, metadata)
            self.assertNotIn("must-not-appear", json.dumps(metadata))
            registry = LocalArtifactRegistry(root / "registry")
            first = registry.register(
                "research-unet", output / "best.pt", output / "artifact.json"
            )
            again = registry.register(
                "research-unet", output / "best.pt", output / "artifact.json"
            )
            self.assertEqual(first.version, again.version)
            self.assertEqual(registry.resolve(first.name, first.version), first)
            (registry.objects / (first.checkpoint_sha256 + ".pt")).write_bytes(
                b"corrupt"
            )
            with self.assertRaises(LifecycleError):
                registry.resolve(first.name, first.version)

            state = load_checkpoint(output / "last.pt")
            state["run"].pop("mlops")
            save_checkpoint(output / "last.pt", state)
            resumed_cfg = replace(cfg, training=replace(cfg.training, epochs=2))
            fit(
                train, valid, resumed_cfg, output=output, data_id="fixture", resume=True
            )

    def test_same_declared_dataset_with_changed_tensors_rejects_resume(self) -> None:
        cfg = ExperimentConfig(
            UNetConfig(2, features=(4, 8)), training=TrainingConfig(epochs=1)
        )
        train = synthetic_dataset(cfg.model, samples=2, seed=11)
        valid = synthetic_dataset(cfg.model, samples=2, seed=12)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            fit(train, valid, cfg, output=output, data_id="same-reference")
            images, _ = train[0]
            images.add_(0.1)
            with self.assertRaisesRegex(TrainingError, "configuration differs"):
                fit(
                    train,
                    valid,
                    cfg,
                    output=output,
                    data_id="same-reference",
                    resume=True,
                )


class MLflowTests(unittest.TestCase):
    def test_local_mlflow_tracking_and_registry(self) -> None:
        cfg = ExperimentConfig(
            UNetConfig(2, features=(4, 8)), training=TrainingConfig(epochs=1)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracker = MLflowTracker(root / "mlflow")
            result = fit(
                synthetic_dataset(cfg.model, samples=2, seed=21),
                synthetic_dataset(cfg.model, samples=2, seed=22),
                cfg,
                output=root / "run",
                data_id="synthetic-v1",
                tracker=tracker,
                provenance=DataProvenance(
                    "synthetic", "1", preprocessing={"steps": []}
                ),
            )
            assert result.tracking_run_id is not None
            run = tracker.client.get_run(result.tracking_run_id)
            self.assertEqual(run.info.status, "FINISHED")
            self.assertIn("validation_loss", run.data.metrics)
            registry = MLflowArtifactRegistry(tracker)
            record = registry.register(
                "synthetic-model", root / "run/best.pt", root / "run/artifact.json"
            )
            self.assertEqual(registry.resolve(record.name, record.version), record)
            CheckpointManager.load_model(
                root / "run/best.pt", sha256=record.checkpoint_sha256
            )
