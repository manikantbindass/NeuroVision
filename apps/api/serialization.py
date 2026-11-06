"""Explicit array/JSON artifact encoding for trusted shared database storage."""

import io
import json
from typing import Any, cast
from zipfile import ZipFile

import numpy as np

from apps.api.adapters import Artifact, ArtifactKind
from apps.api.contracts import APIError
from apps.api.service import Scan, Segmentation
from neurovision.imaging.volume import Modality, MRIVolume, canonical_volume


def encode(artifact: Artifact) -> bytes:
    arrays: dict[str, Any] = {}
    payload = artifact.payload
    header: dict[str, Any] = {"kind": artifact.kind, "size": artifact.size}
    if isinstance(payload, (Scan, Segmentation)):
        volume = payload.volume
        arrays.update(image=volume.data, affine=volume.affine)
        header.update(
            type="scan" if isinstance(payload, Scan) else "segmentation",
            metadata=payload.metadata,
            modality=volume.modality,
            source_format=volume.source_format,
            reference_space=volume.reference_space,
            assumed_mm=volume.assumed_mm,
        )
        if isinstance(payload, Scan):
            header["role"] = payload.role
        else:
            arrays.update(
                labels=payload.labels,
                confidence=payload.confidence,
                entropy=payload.entropy,
            )
            header["classes"] = payload.classes
    elif isinstance(payload, dict):
        header.update(type="json", metadata=payload)
    else:
        raise APIError(500, "invalid_artifact", "Unsupported stored artifact.")
    arrays["header"] = np.frombuffer(
        json.dumps(header, allow_nan=False).encode(), dtype=np.uint8
    )
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    data = buffer.getvalue()
    if len(data) > 64 * 1024**2:
        raise APIError(
            507, "artifact_too_large", "Encoded artifact exceeds its byte limit."
        )
    return data


def decode(data: bytes) -> Artifact:
    if len(data) > 64 * 1024**2:
        raise APIError(
            500, "invalid_artifact", "Stored artifact exceeds its byte limit."
        )
    with ZipFile(io.BytesIO(data)) as archive:
        if (
            len(archive.infolist()) > 8
            or sum(item.file_size for item in archive.infolist()) > 64 * 1024**2
        ):
            raise APIError(
                500, "invalid_artifact", "Stored artifact exceeds its decoded limit."
            )
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        header = json.loads(arrays["header"].tobytes())
        kind = cast(ArtifactKind, header["kind"])
        payload: object
        if header["type"] == "json":
            payload = header["metadata"]
        else:
            volume: MRIVolume = canonical_volume(
                arrays["image"],
                arrays["affine"],
                source_format=header["source_format"],
                modality=Modality(header["modality"]),
                reference_space=header["reference_space"],
                assumed_mm=header["assumed_mm"],
            )
            if header["type"] == "scan":
                payload = Scan(volume, header["role"], header["metadata"])
            else:
                labels, confidence, entropy = (
                    arrays[key].copy() for key in ("labels", "confidence", "entropy")
                )
                for array in (labels, confidence, entropy):
                    array.setflags(write=False)
                payload = Segmentation(
                    volume,
                    labels,
                    confidence,
                    entropy,
                    tuple(header["classes"]),
                    header["metadata"],
                )
        return Artifact(kind, payload, int(header["size"]))
