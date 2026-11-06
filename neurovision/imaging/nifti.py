"""Bounded NIfTI-1 and NIfTI-2 single-file ingestion."""

import gzip
from io import BytesIO
from math import prod
from pathlib import Path
from zlib import error as ZlibError

import nibabel as nib
import numpy as np

from neurovision.imaging.limits import (
    DEFAULT_LIMITS,
    ByteReader,
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

_SPACES = {1: "scanner", 2: "aligned", 3: "talairach", 4: "mni", 5: "template"}
_MM_SCALE = {"unknown": 1.0, "mm": 1.0, "meter": 1000.0, "micron": 0.001}


def nifti_suffix(name: str) -> bool:
    return name.lower().endswith((".nii", ".nii.gz"))


def load_nifti(
    path: Path,
    *,
    modality: Modality = Modality.UNKNOWN,
    limits: IngestionLimits = DEFAULT_LIMITS,
    label_map: bool = False,
) -> MRIVolume:
    if regular_file_size(path) > limits.max_input_bytes:
        raise ImagingError("MRI input exceeds the configured byte limit")
    try:
        with path.open("rb") as stream:
            return read_nifti(
                stream,
                filename=path.name,
                modality=modality,
                limits=limits,
                label_map=label_map,
            )
    except OSError:
        raise ImagingError("MRI input is inaccessible") from None


def read_nifti(
    stream: ByteReader,
    *,
    filename: str,
    modality: Modality = Modality.UNKNOWN,
    limits: IngestionLimits = DEFAULT_LIMITS,
    label_map: bool = False,
) -> MRIVolume:
    """Read an upload stream. The filename is used only as a format hint."""
    if not nifti_suffix(filename):
        raise ImagingError("Expected a .nii or .nii.gz input")
    try:
        payload = bounded_read(stream, limits.max_input_bytes)
        if filename.lower().endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(payload)) as decompressor:
                payload = bounded_read(decompressor, limits.max_decoded_bytes)
        elif len(payload) > limits.max_decoded_bytes:
            raise ImagingError(
                "NIfTI payload exceeds the configured decoded-byte limit"
            )
        return _decode(payload, modality, limits, label_map=label_map)
    except (
        OSError,
        EOFError,
        ZlibError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        nib.spatialimages.HeaderDataError,
    ):
        raise ImagingError("Corrupt or unsupported NIfTI input") from None


def _decode(
    payload: bytes, modality: Modality, limits: IngestionLimits, *, label_map: bool
) -> MRIVolume:
    # Read only the fixed header. Extensions may contain identifying metadata.
    header: nib.Nifti1Header
    if len(payload) >= 544 and payload[4:12] == b"n+2\x00\r\n\x1a\n":
        header = nib.Nifti2Header(payload[:540], check=False)
        minimum_offset = 544
    elif len(payload) >= 352 and payload[344:348] == b"n+1\x00":
        header = nib.Nifti1Header(payload[:348], check=False)
        minimum_offset = 352
    else:
        raise ImagingError("Invalid NIfTI single-file header")
    if int(header["sizeof_hdr"]) != minimum_offset - 4:
        raise ImagingError("Invalid NIfTI header size")
    shape = tuple(int(size) for size in header.get_data_shape())
    ndim = int(header["dim"][0])
    if ndim != 3 or len(shape) != 3 or min(shape, default=0) <= 0:
        raise ImagingError("Only nonempty scalar 3D NIfTI volumes are supported")
    count = prod(shape)
    limits.check_voxels(count)
    dtype = header.get_data_dtype()
    if dtype.kind not in "iuf" or dtype.itemsize > 8:
        raise ImagingError("Unsupported NIfTI scalar datatype")
    if int(header["bitpix"]) != dtype.itemsize * 8:
        raise ImagingError("NIfTI datatype and bit depth disagree")
    raw_offset = header["vox_offset"].item()
    offset = int(raw_offset)
    if (
        offset != raw_offset
        or offset < minimum_offset
        or offset + count * dtype.itemsize > len(payload)
    ):
        raise ImagingError("NIfTI voxel offset or payload length is invalid")
    sform, s_code = header.get_sform(coded=True)
    affine, code = (sform, s_code) if s_code else header.get_qform(coded=True)
    if affine is None or code not in _SPACES:
        raise ImagingError("NIfTI requires a coded sform or qform affine")
    units, _ = header.get_xyzt_units()
    with np.errstate(over="ignore", invalid="ignore"):
        affine[:3] *= _MM_SCALE[units]
        raw = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(
            shape, order="F"
        )
        slope, intercept = header.get_slope_inter()
        if slope is not None and (slope != 1 or intercept != 0):
            raw = raw.astype(np.float64) * slope + intercept
        if label_map and (
            not np.isfinite(raw).all()
            or np.any(raw < 0)
            or np.any(raw > 65535)
            or np.any(raw != np.floor(raw))
        ):
            raise ImagingError(
                "NIfTI label map must contain integer labels from 0 through 65535"
            )
        data = raw.astype(np.float32)
    return canonical_volume(
        data,
        affine,
        source_format="nifti",
        modality=modality,
        reference_space=_SPACES[code],
        assumed_mm=units == "unknown",
    )
