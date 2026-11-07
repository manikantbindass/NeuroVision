"""Adapter for manually staged BraTS 2021 segmentation volumes."""

import re
from dataclasses import dataclass
from pathlib import Path

from neurovision.datasets.manifest import (
    DatasetCase,
    DatasetError,
    DatasetManifest,
    VolumeEntry,
)
from neurovision.imaging.volume import Modality

BRATS_MODALITIES = (Modality.T1, Modality.T1CE, Modality.T2, Modality.FLAIR)
_CASE_FOLDER = re.compile(r"BraTS2021_[0-9]{5}")


@dataclass(frozen=True, slots=True)
class Brats2021Adapter:
    split: str = "unassigned"

    def build_manifest(self, root: Path) -> DatasetManifest:
        try:
            folders = []
            for index, folder in enumerate(root.iterdir(), 1):
                if index > 10_000:
                    raise DatasetError(
                        "BraTS directory exceeds the supported entry count"
                    )
                if _CASE_FOLDER.fullmatch(folder.name) and folder.is_dir():
                    folders.append(folder)
            cases = [self._case(folder) for folder in sorted(folders)]
        except OSError:
            raise DatasetError("BraTS directory is inaccessible") from None
        if not cases:
            raise DatasetError("No BraTS 2021 case directories were found")
        return DatasetManifest(tuple(cases))

    def _case(self, folder: Path) -> DatasetCase:
        volumes = []
        for modality in BRATS_MODALITIES:
            path = _find_volume(folder, modality.value)
            if path is None:
                raise DatasetError("A BraTS 2021 case is missing a required modality")
            volumes.append(VolumeEntry(modality, path))
        return DatasetCase(
            folder.name, tuple(volumes), self.split, _find_volume(folder, "seg")
        )


def _find_volume(folder: Path, suffix: str) -> str | None:
    names = [f"{folder.name}_{suffix}{ext}" for ext in (".nii", ".nii.gz")]
    present = [name for name in names if (folder / name).is_file()]
    if len(present) > 1:
        raise DatasetError("A BraTS case has multiple files for one modality or label")
    return f"{folder.name}/{present[0]}" if present else None
