"""Bridge validated API requests to the existing imaging and research engines."""

import io
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal
from uuid import UUID, uuid4

import nibabel as nib
import numpy as np
import torch
from numpy.typing import NDArray

from apps.api.adapters import Artifact, ArtifactStore, Job
from apps.api.contracts import (
    APIError,
    InferenceRequest,
    JobInput,
    MapRequest,
    MetricsRequest,
    PreprocessRequest,
)
from neurovision.core.errors import NeuroVisionError
from neurovision.core.pipeline import RESEARCH_NOTICE, RunContext
from neurovision.evaluation.segmentation import segmentation_metrics
from neurovision.imaging.limits import IngestionLimits
from neurovision.imaging.nifti import read_nifti
from neurovision.imaging.volume import ImagingError, Modality, MRIVolume
from neurovision.inference.engine import InferenceEngine
from neurovision.mapping.atlas import AtlasRegion, LocalLabelAtlas
from neurovision.mapping.engine import NeuroMapEngine
from neurovision.preprocessing.contracts import matrix_values, same_grid
from neurovision.preprocessing.intensity import NormalizeIntensity
from neurovision.preprocessing.pipeline import PreprocessingPipeline
from neurovision.uncertainty.example import SyntheticThresholdModel

Metadata = dict[str, Any]
Role = Literal["image", "labels"]


@dataclass(frozen=True, slots=True)
class ModelEntry:
    engine: InferenceEngine
    classes: tuple[str, ...]
    revision: str
    modality: Modality | None = None
    synthetic: bool = False

    def __post_init__(self) -> None:
        spec = self.engine.model.input_spec
        if (
            spec.spatial_dims != 3
            or spec.in_channels != 1
            or not 2 <= len(self.classes) == spec.num_classes <= 256
            or any(not name.strip() or len(name) > 160 for name in self.classes)
            or not self.revision.strip()
        ):
            raise ValueError(
                "API models require a declared single-channel 3D label schema"
            )


def demo_models() -> dict[str, ModelEntry]:
    return {
        "synthetic-threshold": ModelEntry(
            InferenceEngine(SyntheticThresholdModel(0.5)),
            ("background", "synthetic foreground"),
            "analytic-v1",
            synthetic=True,
        )
    }


@dataclass(frozen=True, slots=True)
class Scan:
    volume: MRIVolume
    role: Role
    metadata: Metadata


@dataclass(frozen=True, slots=True)
class Segmentation:
    volume: MRIVolume
    labels: NDArray[np.uint16]
    confidence: NDArray[np.float32]
    entropy: NDArray[np.float32]
    classes: tuple[str, ...]
    metadata: Metadata


def _json_size(metadata: Metadata) -> int:
    return len(json.dumps(metadata, allow_nan=False).encode("utf-8"))


def nifti_bytes(volume: MRIVolume, array: NDArray[np.generic]) -> bytes:
    image = nib.Nifti2Image(array, volume.affine)
    image.header.set_xyzt_units("mm" if not volume.assumed_mm else "unknown")
    if volume.reference_space == "unknown":
        image.set_sform(volume.affine, code="aligned")
    else:
        code = {
            "scanner": "scanner",
            "aligned": "aligned",
            "talairach": "talairach",
            "mni": "mni",
            "template": "template",
        }.get(volume.reference_space, "aligned")
        image.set_sform(volume.affine, code=code)
    return bytes(image.to_bytes())


class ResearchService:
    def __init__(
        self,
        store: ArtifactStore,
        models: Mapping[str, ModelEntry],
        limits: IngestionLimits,
    ) -> None:
        self.store = store
        self.models = dict(models)
        self.limits = limits

    def upload(
        self,
        owner: str,
        stream: io.BytesIO,
        suffix: str,
        modality: Modality,
        role: Role,
    ) -> Metadata:
        try:
            volume = read_nifti(
                stream,
                filename="upload." + suffix,
                limits=self.limits,
                modality=modality,
                label_map=role == "labels",
            )
        except ImagingError:
            raise APIError(
                422, "invalid_scan", "NIfTI failed decoding or safety checks."
            ) from None
        key = uuid4()
        metadata = {
            "id": str(key),
            "role": role,
            "shape": volume.data.shape,
            "spacing_mm": volume.spacing_mm,
            "affine": matrix_values(volume.affine),
            "reference_space": volume.reference_space,
            "modality": volume.modality,
            "assumed_mm": volume.assumed_mm,
            "notice": RESEARCH_NOTICE,
        }
        self.store.put(
            owner,
            key,
            Artifact(
                "scan",
                Scan(volume, role, metadata),
                volume.data.nbytes + _json_size(metadata),
            ),
        )
        return metadata

    def scan(self, owner: str, key: UUID, role: Role | None = None) -> Scan:
        data = self.store.get(owner, key).payload
        if not isinstance(data, Scan) or (role is not None and data.role != role):
            raise APIError(404, "not_found", "Scan not found for the requested role.")
        return data

    def segmentation(self, owner: str, key: UUID) -> Segmentation:
        data = self.store.get(owner, key).payload
        if not isinstance(data, Segmentation):
            raise APIError(404, "not_found", "Segmentation not found.")
        return data

    def summary(self, owner: str, key: UUID, kind: str | None = None) -> Metadata:
        artifact = self.store.get(owner, key)
        if kind is not None and artifact.kind != kind:
            raise APIError(
                404, "not_found", "Artifact not found for the requested route."
            )
        data = artifact.payload
        if isinstance(data, (Scan, Segmentation)):
            return data.metadata
        if isinstance(data, dict):
            return data
        raise RuntimeError("Unknown artifact payload")

    def catalog(self) -> list[Metadata]:
        return [
            {
                "id": key,
                "architecture": model.engine.model.model_config.architecture,
                "class_names": model.classes,
                "revision": model.revision,
                "modality": model.modality,
                "synthetic": model.synthetic,
            }
            for key, model in sorted(self.models.items())
        ]

    def check_inputs(self, owner: str, body: JobInput) -> None:
        if isinstance(body, (PreprocessRequest, InferenceRequest)):
            self.scan(owner, body.scan_id, "image")
            if isinstance(body, InferenceRequest) and body.model_id not in self.models:
                raise APIError(404, "not_found", "Model not found.")
        elif isinstance(body, MetricsRequest):
            self.segmentation(owner, body.segmentation_id)
            self.scan(owner, body.target_scan_id, "labels")
        elif isinstance(body, MapRequest):
            self.segmentation(owner, body.segmentation_id)
            self.scan(owner, body.atlas_scan_id, "labels")
        else:
            for key in body.artifact_ids:
                if self.store.get(owner, key).kind == "reports":
                    raise APIError(
                        422, "invalid_report_source", "Reports cannot be nested."
                    )

    def run(self, job: Job) -> str:
        try:
            return self._run(job)
        except (NeuroVisionError, ValueError, ArithmeticError):
            raise APIError(
                422, "operation_rejected", "Inputs failed research operation checks."
            ) from None

    def _run(self, job: Job) -> str:
        owner, key, body = job.owner, job.id, job.payload
        self.check_inputs(owner, body)
        base = {"id": str(key), "notice": RESEARCH_NOTICE}
        output: object
        array_bytes = 0
        route = "segmentations" if job.kind == "inference" else job.kind
        if isinstance(body, PreprocessRequest):
            scan = self.scan(owner, body.scan_id, "image")
            steps = (
                ()
                if body.normalization == "none"
                else (NormalizeIntensity(method=body.normalization),)
            )
            result = PreprocessingPipeline(steps).run(
                scan.volume, context=RunContext(str(key), body.seed)
            )
            metadata = {
                **scan.metadata,
                **base,
                "source_scan_id": str(body.scan_id),
                "usable": result.volume is not None,
                "provenance": json.loads(result.to_json()),
            }
            output = metadata
            if result.volume is not None:
                output = Scan(result.volume, "image", metadata)
                array_bytes = result.volume.data.nbytes
        elif isinstance(body, InferenceRequest):
            output = self._infer(owner, body, base)
            metadata = output.metadata
            array_bytes = (
                output.labels.nbytes
                + output.confidence.nbytes
                + output.entropy.nbytes
                + output.volume.data.nbytes
            )
        elif isinstance(body, MetricsRequest):
            metadata = {**base, **self._metrics(owner, body)}
            output = metadata
        elif isinstance(body, MapRequest):
            metadata = {**base, **self._map(owner, body)}
            output = metadata
        else:
            metadata = {
                **base,
                "schema_version": 1,
                "sources": [
                    {
                        "kind": self.store.get(owner, item).kind,
                        "summary": self.summary(owner, item),
                    }
                    for item in body.artifact_ids
                ],
            }
            output = metadata
        self.store.put(
            owner, key, Artifact(job.kind, output, array_bytes + _json_size(metadata))
        )
        return f"/api/v1/{route}/{key}"

    def _infer(
        self, owner: str, body: InferenceRequest, base: Metadata
    ) -> Segmentation:
        volume = self.scan(owner, body.scan_id, "image").volume
        model = self.models[body.model_id]
        if model.modality is not None and volume.modality != model.modality:
            raise APIError(
                422, "modality_mismatch", "Scan modality does not match the model."
            )
        tensor = torch.from_numpy(volume.data.copy())[None, None]
        prediction = model.engine.predict(tensor)
        probs = prediction.probabilities[0]
        confidence = probs.amax(dim=0).numpy()
        entropy = (
            (
                -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(
                    dim=0
                )
                / np.log(len(model.classes))
            )
            .clamp(0, 1)
            .numpy()
        )
        labels = prediction.labels[0].numpy().astype(np.uint16)
        for array in (labels, confidence, entropy):
            array.setflags(write=False)
        metadata = {
            **base,
            "scan_id": str(body.scan_id),
            "shape": volume.data.shape,
            "class_names": model.classes,
            "model_id": body.model_id,
            "model_revision": model.revision,
            "device": str(model.engine.device),
            "precision": model.engine.precision,
            "synthetic": model.synthetic,
            "label_voxels": np.bincount(
                labels.ravel(), minlength=len(model.classes)
            ).tolist(),
            "uncertainty": {
                "mean_confidence": float(confidence.mean(dtype=np.float64)),
                "mean_normalized_entropy": float(entropy.mean(dtype=np.float64)),
                "calibrated": False,
            },
        }
        return Segmentation(
            volume, labels, confidence, entropy, model.classes, metadata
        )

    def _metrics(self, owner: str, body: MetricsRequest) -> Metadata:
        prediction = self.segmentation(owner, body.segmentation_id)
        target = self.scan(owner, body.target_scan_id, "labels").volume
        if not same_grid(prediction.volume, target):
            raise APIError(
                422, "grid_mismatch", "Metrics require matching segmentation grids."
            )
        axes = target.affine[:3, :3] / np.asarray(target.spacing_mm)
        orthogonal = np.allclose(axes.T @ axes, np.eye(3), rtol=0, atol=1e-5)
        spacing = (
            target.spacing_mm
            if (
                orthogonal
                and not target.assumed_mm
                and not prediction.volume.assumed_mm
            )
            else None
        )
        scores = segmentation_metrics(
            prediction.labels.astype(np.int64),
            target.data.astype(np.int64),
            num_classes=len(prediction.classes),
            spacing_mm=spacing,
        )
        return {
            "segmentation_id": str(body.segmentation_id),
            "target_scan_id": str(body.target_scan_id),
            "physical_distances": spacing is not None,
            "classes": [asdict(score) for score in scores],
        }

    def _map(self, owner: str, body: MapRequest) -> Metadata:
        prediction = self.segmentation(owner, body.segmentation_id)
        atlas_volume = self.scan(owner, body.atlas_scan_id, "labels").volume
        adapter = LocalLabelAtlas(
            atlas_volume,
            body.atlas_name,
            body.atlas_version,
            body.atlas_license,
            body.atlas_space,
            tuple(
                AtlasRegion(region.code, region.name, region.hemisphere)
                for region in body.regions
            ),
        )
        result = NeuroMapEngine(adapter, max_voxels=self.limits.max_voxels).map(
            prediction.volume,
            prediction.labels,
            class_names=dict(enumerate(prediction.classes)),
            segmentation_to_atlas_world=np.asarray(
                body.segmentation_to_atlas_world, dtype=np.float64
            ),
            registration_provenance=body.registration_provenance,
            confidence=prediction.confidence,
        )
        return {
            "segmentation_id": str(body.segmentation_id),
            "atlas_scan_id": str(body.atlas_scan_id),
            "atlas": {
                "name": result.atlas_name,
                "version": result.atlas_version,
                "license": result.atlas_license,
                "space": result.atlas_space,
            },
            "registration_provenance": result.registration_provenance,
            "segmentation_to_atlas_world": result.segmentation_to_atlas_world,
            "measurements": [asdict(region) for region in result.measurements],
        }
