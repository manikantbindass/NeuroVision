"""Resource limits shared by readers and future upload handlers."""

import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from neurovision.imaging.volume import ImagingError

_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class IngestionLimits:
    max_input_bytes: int = 128 * _MIB
    max_decoded_bytes: int = 256 * _MIB
    max_voxels: int = 32_000_000
    max_dicom_file_bytes: int = 16 * _MIB
    max_directory_entries: int = 1024

    def __post_init__(self) -> None:
        if (
            min(
                self.max_input_bytes,
                self.max_decoded_bytes,
                self.max_voxels,
                self.max_dicom_file_bytes,
                self.max_directory_entries,
            )
            < 1
        ):
            raise ValueError("Ingestion limits must be positive")

    def check_voxels(self, count: int) -> None:
        if count < 1 or count > self.max_voxels or count * 4 > self.max_decoded_bytes:
            raise ImagingError(
                "MRI volume exceeds configured voxel or decoded-byte limits"
            )


DEFAULT_LIMITS = IngestionLimits()


class ByteReader(Protocol):
    def read(self, size: int, /) -> bytes: ...


def bounded_read(stream: ByteReader, limit: int) -> bytes:
    """Read through EOF while bounding allocation even for short-read streams."""
    data = bytearray()
    while chunk := stream.read(min(_MIB, limit + 1 - len(data))):
        data.extend(chunk)
        if len(data) > limit:
            raise ImagingError("MRI input exceeds the configured byte limit")
    return bytes(data)


def regular_file_size(path: Path) -> int:
    try:
        info = path.lstat()
    except OSError:
        raise ImagingError("MRI input is inaccessible") from None
    if not stat.S_ISREG(info.st_mode):
        raise ImagingError("MRI input must be a regular file")
    return info.st_size
