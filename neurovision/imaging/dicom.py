"""Import pre-reviewed classic MR series from a trusted local staging directory."""

import struct
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from io import BytesIO
from math import isfinite
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydicom.datadict import dictionary_VR
from pydicom.dataset import Dataset
from pydicom.filereader import read_dataset
from pydicom.tag import BaseTag
from pydicom.uid import (
    UID,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    MRImageStorage,
)

from neurovision.imaging.limits import (
    DEFAULT_LIMITS,
    IngestionLimits,
    bounded_read,
    regular_file_size,
)
from neurovision.imaging.volume import (
    ImagingError,
    Modality,
    MRIVolume,
    canonical_volume,
)

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
_SPACING_ATOL_MM = 0.001
_LABELS = {
    "t1": Modality.T1,
    "t1w": Modality.T1,
    "t1ce": Modality.T1CE,
    "t1gd": Modality.T1CE,
    "t2": Modality.T2,
    "t2w": Modality.T2,
    "flair": Modality.FLAIR,
}
_IDENTIFIERS = (
    "AccessionNumber",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "StudyID",
    "StationName",
    "DeviceSerialNumber",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "AcquisitionDateTime",
    "ContentDate",
    "ProtocolName",
    "StudyDescription",
)


@dataclass(frozen=True, slots=True)
class DicomSeries:
    """Discovery result. Paths stay local and are hidden from the default repr."""

    files: tuple[Path, ...] = field(repr=False)
    shape: tuple[int, int, int]
    modality: Modality


@dataclass(slots=True)
class _Slice:
    path: Path
    group: tuple[str, str]
    instance_uid: str
    frame_uid: str
    rows: int
    columns: int
    orientation: NDArray[np.float64]
    position: NDArray[np.float64]
    spacing: NDArray[np.float64]
    modality: Modality
    pixels: NDArray[np.float32] | None


def discover_dicom_series(
    directory: Path,
    *,
    privacy_reviewed: bool = False,
    limits: IngestionLimits = DEFAULT_LIMITS,
) -> tuple[DicomSeries, ...]:
    """Group local files by study and series and verify geometry."""
    _require_review(privacy_reviewed)
    try:
        paths = []
        for index, path in enumerate(directory.iterdir(), 1):
            if index > limits.max_directory_entries:
                raise ImagingError("DICOM directory exceeds the configured entry limit")
            if path.suffix.lower() in (".dcm", ".dicom") or (
                not path.suffix and (path.is_file() or path.is_symlink())
            ):
                paths.append(path)
    except OSError:
        raise ImagingError("DICOM directory is inaccessible") from None
    groups: dict[tuple[str, str], list[_Slice]] = {}
    for item in _collect(sorted(paths), limits, decode=False):
        groups.setdefault(item.group, []).append(item)
    series = []
    for slices in groups.values():
        ordered, _, modality = _geometry(slices)
        first = ordered[0]
        series.append(
            DicomSeries(
                tuple(s.path for s in ordered),
                (first.columns, first.rows, len(ordered)),
                modality,
            )
        )
    return tuple(series)


def load_dicom_series(
    files: Iterable[Path],
    *,
    privacy_reviewed: bool = False,
    limits: IngestionLimits = DEFAULT_LIMITS,
) -> MRIVolume:
    """Load one pre-reviewed series and recheck its contents at load time."""
    _require_review(privacy_reviewed)
    slices = _collect(files, limits, decode=True)
    ordered, affine, modality = _geometry(slices)
    planes = []
    for item in ordered:
        assert item.pixels is not None
        planes.append(item.pixels.T)
    return canonical_volume(
        np.stack(planes, axis=2),
        affine,
        source_format="dicom",
        modality=modality,
        reference_space="scanner",
    )


def _require_review(reviewed: bool) -> None:
    if reviewed is not True:
        raise ImagingError("DICOM ingestion requires prior trusted privacy review")


def _collect(
    files: Iterable[Path], limits: IngestionLimits, *, decode: bool
) -> list[_Slice]:
    slices: list[_Slice] = []
    instances: set[str] = set()
    remaining = limits.max_input_bytes
    total_voxels = 0
    for path in files:
        if len(slices) >= limits.max_directory_entries:
            raise ImagingError("DICOM input exceeds the configured file count limit")
        limit = min(remaining, limits.max_dicom_file_bytes)
        if regular_file_size(path) > limit:
            raise ImagingError("DICOM input exceeds the configured byte limit")
        try:
            with path.open("rb") as stream:
                payload = bounded_read(stream, limit)
            remaining -= len(payload)
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                item = _parse_slice(payload, path, limits, total_voxels, decode=decode)
        except (
            OSError,
            EOFError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            OverflowError,
            struct.error,
            UserWarning,
            RecursionError,
        ):
            raise ImagingError("Corrupt or unsupported DICOM input") from None
        if item.instance_uid in instances:
            raise ImagingError("DICOM input contains duplicate instances")
        instances.add(item.instance_uid)
        total_voxels += item.rows * item.columns
        slices.append(item)
    if not slices:
        raise ImagingError("No eligible DICOM files found")
    return slices


def _parse_slice(
    payload: bytes,
    path: Path,
    limits: IngestionLimits,
    previous_voxels: int,
    *,
    decode: bool,
) -> _Slice:
    # Inspect the file meta before reading the dataset or invoking any codec.
    if len(payload) < 144 or payload[128:140] != b"DICM\x02\x00\x00\x00UL\x04\x00":
        raise ImagingError("Expected a DICOM Part 10 file with standard file meta")
    meta_length = struct.unpack_from("<I", payload, 140)[0]
    end = 144 + meta_length
    if not 0 < meta_length <= 65536 or end >= len(payload):
        raise ImagingError("Invalid DICOM file meta size")
    meta = read_dataset(BytesIO(payload[144:end]), False, True)
    syntax = meta.get("TransferSyntaxUID")
    if syntax not in (ExplicitVRLittleEndian, ImplicitVRLittleEndian):
        raise ImagingError(
            "Only native little-endian DICOM transfer syntaxes are supported"
        )
    if meta.get("MediaStorageSOPClassUID") != MRImageStorage:
        raise ImagingError("Only classic MR Image Storage is supported")
    stream = BytesIO(payload)
    stream.seek(end)
    ds = read_dataset(
        stream, syntax == ImplicitVRLittleEndian, True, stop_when=_guard_element
    )
    if ds.get("SOPClassUID") != MRImageStorage or ds.get("Modality") != "MR":
        raise ImagingError("Only classic MR Image Storage is supported")
    _check_privacy(ds)
    if ds.get("AnatomicalOrientationType", "BIPED") != "BIPED":
        raise ImagingError("Only biped DICOM patient geometry is supported")
    if "MOSAIC" in ds.get("ImageType", ()):
        raise ImagingError("DICOM mosaic images are unsupported")
    instance = _uid(ds, "SOPInstanceUID")
    if meta.get("MediaStorageSOPInstanceUID") != instance:
        raise ImagingError("DICOM file meta and instance identifiers disagree")
    group = (_uid(ds, "StudyInstanceUID"), _uid(ds, "SeriesInstanceUID"))
    frame = _uid(ds, "FrameOfReferenceUID")
    rows, cols = int(ds.Rows), int(ds.Columns)
    if rows <= 0 or cols <= 0:
        raise ImagingError("Invalid DICOM image dimensions")
    limits.check_voxels(previous_voxels + rows * cols)
    orientation = _vector(ds.ImageOrientationPatient, 6)
    position = _vector(ds.ImagePositionPatient, 3)
    spacing = _vector(ds.PixelSpacing, 2)
    if np.any(spacing <= 0):
        raise ImagingError("Invalid DICOM pixel spacing")
    desc = str(ds.get("SeriesDescription", "")).strip().lower()
    modality = _LABELS.get(desc, Modality.UNKNOWN)
    if modality == Modality.T1 and ds.get("ContrastBolusAgent"):
        modality = Modality.UNKNOWN
    pixels = _pixels(ds, rows, cols, decode=decode)
    return _Slice(
        path,
        group,
        instance,
        frame,
        rows,
        cols,
        orientation,
        position,
        spacing,
        modality,
        pixels,
    )


def _guard_element(tag: BaseTag, vr: str | None, length: int) -> bool:
    if vr is None:
        try:
            vr = dictionary_VR(tag)
        except KeyError:
            pass
    if vr == "SQ" or length == 0xFFFFFFFF:
        raise ImagingError(
            "DICOM sequences and undefined-length elements are unsupported"
        )
    return False


def _check_privacy(ds: Dataset) -> None:
    if (
        ds.get("PatientIdentityRemoved"),
        ds.get("BurnedInAnnotation"),
        ds.get("RecognizableVisualFeatures"),
    ) != ("YES", "NO", "NO"):
        raise ImagingError("DICOM privacy declarations are missing or unsafe")
    if any(ds.get(key) for key in _IDENTIFIERS):
        raise ImagingError("DICOM contains potentially identifying fields")
    for element in ds:
        tag = element.tag
        if (
            tag.is_private
            or 0x5000 <= tag.group <= 0x50FF
            or 0x6000 <= tag.group <= 0x60FF
            or element.VR == "SQ"
            or ((tag.group == 0x0010 or element.VR == "PN") and not element.is_empty)
        ):
            raise ImagingError(
                "DICOM contains unsupported or potentially identifying elements"
            )


def _uid(ds: Dataset, keyword: str) -> str:
    value = ds.get(keyword)
    if not isinstance(value, str) or not UID(value).is_valid:
        raise ImagingError("Missing or invalid DICOM UID")
    return value


def _vector(value: object, length: int) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ImagingError("Invalid DICOM spatial metadata")
    return array


def _pixels(
    ds: Dataset, rows: int, cols: int, *, decode: bool
) -> NDArray[np.float32] | None:
    bits = int(ds.BitsStored)
    signed = int(ds.PixelRepresentation)
    if (
        int(ds.get("NumberOfFrames", 1)) != 1
        or int(ds.SamplesPerPixel) != 1
        or ds.PhotometricInterpretation not in ("MONOCHROME1", "MONOCHROME2")
        or int(ds.BitsAllocated) != 16
        or not 1 <= bits <= 16
        or int(ds.HighBit) != bits - 1
        or signed not in (0, 1)
    ):
        raise ImagingError("Only single-frame 16-bit monochrome DICOM is supported")
    data = ds.PixelData
    if not isinstance(data, bytes) or len(data) != rows * cols * 2:
        raise ImagingError("DICOM pixel payload length does not match its dimensions")
    slope, intercept = (
        float(ds.get("RescaleSlope", 1)),
        float(ds.get("RescaleIntercept", 0)),
    )
    if not isfinite(slope) or slope == 0 or not isfinite(intercept):
        raise ImagingError("DICOM intensity scaling is invalid")
    if not decode:
        return None
    raw = np.frombuffer(data, dtype="<u2").reshape(rows, cols)
    values = (raw & ((1 << bits) - 1)).astype(np.int32)
    if signed:
        sign = 1 << (bits - 1)
        values = (values ^ sign) - sign
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(values * slope + intercept, dtype=np.float32)


def _geometry(
    slices: list[_Slice],
) -> tuple[list[_Slice], NDArray[np.float64], Modality]:
    first = slices[0]
    row, column = first.orientation[:3], first.orientation[3:]
    if not np.allclose(
        [
            float(np.linalg.norm(row)),
            float(np.linalg.norm(column)),
            float(row @ column),
        ],
        [1, 1, 0],
        atol=1e-4,
        rtol=0,
    ):
        raise ImagingError(
            "DICOM orientation must contain orthonormal direction cosines"
        )
    for item in slices:
        if (
            item.group != first.group
            or item.frame_uid != first.frame_uid
            or (item.rows, item.columns) != (first.rows, first.columns)
            or not np.allclose(item.orientation, first.orientation, atol=1e-4, rtol=0)
            or not np.allclose(
                item.spacing, first.spacing, atol=_SPACING_ATOL_MM, rtol=0
            )
        ):
            raise ImagingError("DICOM series contains inconsistent image geometry")
    normal = np.cross(row, column)
    ordered = sorted(slices, key=lambda item: float(item.position @ normal))
    if len(ordered) < 2:
        raise ImagingError(
            "A DICOM series needs at least two slices to establish spacing"
        )
    positions = np.stack([item.position for item in ordered])
    steps = np.diff(positions, axis=0)
    step = (positions[-1] - positions[0]) / (len(ordered) - 1)
    if (
        step @ normal <= _SPACING_ATOL_MM
        or not np.allclose(steps, step, atol=_SPACING_ATOL_MM, rtol=0)
        or np.any(steps @ normal <= _SPACING_ATOL_MM)
    ):
        raise ImagingError("DICOM slice positions are duplicate or irregular")
    affine = np.eye(4)
    affine[:3, 0] = row * first.spacing[1]
    affine[:3, 1] = column * first.spacing[0]
    affine[:3, 2] = step
    affine[:3, 3] = positions[0]
    modality = (
        first.modality
        if all(item.modality == first.modality for item in slices)
        else Modality.UNKNOWN
    )
    return ordered, _LPS_TO_RAS @ affine, modality
