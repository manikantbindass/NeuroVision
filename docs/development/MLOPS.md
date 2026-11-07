# MLOps lifecycle and reproducibility

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

Training records are produced alongside the existing checkpoints. MLflow tracking and trained-artifact registries are opt-in. The architecture factory registry continues to select network implementations.

## Run a local experiment

Follow the README's CPU PyTorch setup, then install the optional tracking client and SQLite backend libraries.

```sh
python -m pip install ".[training,imaging,mlops]"
python -m neurovision.mlops.example --config configs/baseline-2d.json --output artifacts/mlops-run --mlflow artifacts/mlflow --register synthetic-unet
```

Use a fresh output directory. The example uses generated ellipsoids, records dataset and configuration identities, logs metrics and selected artifacts to a local MLflow experiment, and publishes explicit model versions to the local and MLflow registries. **Benchmark pending.**

The adapter uses the MLflow client with an explicit local SQLite URI and a local artifact directory. It does not use the process-global active run. Unless already configured by the operator, it disables MLflow usage telemetry and uses short-lived SQLite connections. The example prints the tracking run ID. The `mlops` extra installs MLflow's lightweight client and the SQL libraries needed for the local store. The full MLflow UI/server package is optional and should be installed only in an operator-managed environment. [MLflow client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)

## Training artifacts

| File | Contents |
| --- | --- |
| `best.pt`, `last.pt` | Model, optimizer, schedule and precision state, history, selected epoch, experiment identity and tracking run ID |
| `artifact.json` | Architecture, hyperparameters, dataset reference, preprocessing declaration, observed validation metrics, Git revision, dependency versions and checkpoint SHA-256 |
| `configuration.json` | Versioned normalized recipe and fingerprint with a separate epoch budget |
| `environment.json` | Installed distribution names and versions, excluding installation URLs and environment variables |
| `reproducibility.json` | Artifact and input fingerprints, Git state and explicit missing-provenance warnings |
| `MODEL_CARD.md` | Run-specific model facts, selection rule, limitations and checkpoint identity |
| `history.json` | Existing epoch-by-epoch diagnostics |
| `metrics.json` | Numeric summary produced by the MLOps example |
| `registry-reference.json` | Explicit registry version IDs when registration is requested |

Checkpoint formats and restricted loading remain compatible with previous phases. New checkpoints reject a changed tensor dataset even when the declared `data_id` is unchanged. Resume compares the normalized recipe, content identities, code revision and dependency snapshot. The epoch budget may increase. Older checkpoints keep their original compatibility checks, which lack these added content guarantees. Each tracked resume attempt gets a separate MLflow run.

If tracking fails, the call raises an error. Completed epoch checkpoints remain available. If final artifact upload fails, local artifacts remain saved and the MLflow adapter marks its run failed. Review the last saved epoch before resuming.

## Dataset and preprocessing identity

The trainer fingerprints stored CPU `TensorDataset` tensors in order, including dimensions and dtypes. It does not evaluate arbitrary dataset transforms merely to hash their outputs. For other datasets, provide a reviewed content fingerprint and version through `DataProvenance`. Missing content identity produces a warning.

```python
from neurovision.mlops.provenance import DataProvenance, manifest_fingerprint

identity = manifest_fingerprint(dataset_root, validated_manifest)
provenance = DataProvenance(
    reference="study-local-dataset",
    version="reviewed-release-1",
    fingerprint=identity,
    preprocessing={"recipe_version": 1, "recipe_sha256": reviewed_recipe_hash},
)
# Pass provenance=provenance to the existing fit(...) entry point.
```

Validate the manifest and MRI files with the data engine first. The fingerprint binds canonical manifest content, file bytes, case membership, modality and split. It returns only hashes and counts. Review manifests for subject linkage and dataset license requirements. A hash can detect content changes but cannot prove independence of subjects or correctness of a label.

Tensor fingerprints reflect exact tensor bytes in order. File fingerprints reflect on-disk bytes. Recompressing a NIfTI can change its file fingerprint without changing decoded voxels. Preprocessing provenance is supplied explicitly and must identify the recipe used to produce training inputs.

Git capture records HEAD and whether tracked or untracked changes exist. It omits remotes, diffs, branch names and file paths. A dirty or unavailable checkout produces a warning. Commit the reviewed code before running experiments intended for replay. Package snapshots contain names and versions and cannot recreate every system library, driver or local package edit. Keep environment images or lockfiles appropriate to the experiment.

## Configuration versions

Committed training recipes declare `schema_version: 1`. Existing recipes without that field retain version-1 behavior. Unsupported versions fail validation. The recorded normalized configuration hash excludes the epoch budget so an epoch-boundary resume can extend training. Both the budget and completed epochs remain explicit metadata.

## Artifact registry

`ArtifactRegistry` has `register` and `resolve` methods. `LocalArtifactRegistry` stores content-addressed checkpoints and immutable version rows in SQLite. Registration copies and checks the weights before publishing a visible version. Re-registering the same name, checkpoint and metadata returns the existing version. Resolving an explicit version checks the metadata and checkpoint digests.

`MLflowArtifactRegistry` registers artifacts from a finished local tracked run and resolves explicit versions through MLflow. Content-addressed metadata paths preserve prior run artifacts. A registry entry is a checkpoint artifact reference and must be loaded with NeuroVision's restricted checkpoint loader. It is not an MLflow serving flavor. Apply model-card and scientific review before making an entry available in the inference API.

## DVC-compatible data workflow

The root `dvc.yaml` describes a synthetic training stage with source dependencies, versioned JSON parameters and an artifact-directory output. The example command and DVC's local cache workflow are exercised independently of any patient dataset.

```sh
python -m pip install ".[dvc]"
dvc init
dvc repro
dvc status
```

DVC normally replaces cached stage outputs when reproducing a stage. Use the declared output directory only for the stage's generated artifacts. Track the reviewed `dvc.yaml`, configuration and resulting `dvc.lock` in Git. Runtime weights, caches and data stay excluded from Git. [DVC pipeline structure](https://doc.dvc.org/user-guide/project-structure/dvcyaml-files)

For an authorized research dataset, stage it under the ignored `data/` directory, validate it, then use `dvc add data/research`. Commit only its reviewed pointer and manifest metadata. Configure an access-controlled remote locally with `dvc remote add --local`. Pull or push data only under the dataset's license and privacy agreement. This repository neither configures a remote nor downloads restricted data. Keep secrets in local configuration, excluded by Git.

The synthetic example is suitable for verifying tooling. It provides no evidence of accuracy, calibration, fairness, performance or clinical utility on brain MRI.
