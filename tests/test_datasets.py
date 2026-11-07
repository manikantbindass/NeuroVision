import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from neurovision.datasets.brats import BRATS_MODALITIES, Brats2021Adapter
from neurovision.datasets.manifest import (
    DatasetCase,
    DatasetError,
    DatasetManifest,
    DatasetSummary,
    VolumeEntry,
    read_manifest,
    validate_dataset,
)
from neurovision.imaging.volume import Modality


def _save(path: Path, *, offset: float = 0.0, value: float = 1.0) -> None:
    affine = np.eye(4)
    affine[0, 3] = offset
    image = nib.Nifti1Image(np.full((2, 3, 4), value, dtype=np.float64), affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)


class ManifestTests(unittest.TestCase):
    def test_local_brats_adapter_round_trip_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / "BraTS2021_99999"
            case.mkdir()
            for modality in BRATS_MODALITIES:
                _save(case / f"{case.name}_{modality.value}.nii.gz")
            label = case / f"{case.name}_seg.nii.gz"
            _save(label, value=4)
            adapter = Brats2021Adapter(split="train")
            manifest = adapter.build_manifest(root)
            manifest_file = root / "manifest.json"
            manifest_file.write_text(manifest.to_json(), encoding="utf-8")
            self.assertEqual(read_manifest(manifest_file), manifest)
            self.assertEqual(
                validate_dataset(
                    manifest, root, required_modalities=frozenset(BRATS_MODALITIES)
                ),
                DatasetSummary(1, 4, 1),
            )
            _save(label, value=1.000000001)
            with self.assertRaisesRegex(DatasetError, "integer labels"):
                validate_dataset(manifest, root)
            _save(case / f"{case.name}_flair.nii")
            with self.assertRaisesRegex(DatasetError, "multiple files"):
                adapter.build_manifest(root)

    def test_rejects_unsafe_paths_duplicate_files_and_case_leakage(self) -> None:
        def entry(path: str) -> VolumeEntry:
            return VolumeEntry(Modality.T1, path)

        for bad in (
            "../test.nii",
            "/abs/test.nii",
            "D:/test.nii",
            "case\\test.nii",
            "a//test.nii",
        ):
            with (
                self.subTest(path=bad),
                self.assertRaisesRegex(DatasetError, "relative"),
            ):
                DatasetManifest((DatasetCase("case-a", (entry(bad),)),))
        with self.assertRaisesRegex(DatasetError, "IDs"):
            DatasetManifest(
                (
                    DatasetCase("case-a", (entry("a.nii"),), "train"),
                    DatasetCase("CASE-a", (entry("b.nii"),), "test"),
                )
            )
        with self.assertRaisesRegex(DatasetError, "assigned more than once"):
            DatasetManifest(
                (
                    DatasetCase("case-a", (entry("a.nii"),), "train"),
                    DatasetCase("case-b", (entry("A.nii"),), "test"),
                )
            )

    def test_rejects_duplicate_json_keys_and_malformed_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text(
                '{"schema_version":1,"schema_version":1,"cases":[]}', encoding="utf-8"
            )
            with self.assertRaisesRegex(DatasetError, "duplicate JSON keys"):
                read_manifest(path)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "cases": [{"id": "case-a", "volumes": {"t1": 7}}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetError, "paths must be strings"):
                read_manifest(path)

    def test_missing_required_modalities_unaligned_and_corrupt_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _save(root / "t1.nii")
            _save(root / "flair.nii", offset=0.01)
            case = DatasetCase(
                "case-a",
                (
                    VolumeEntry(Modality.T1, "t1.nii"),
                    VolumeEntry(Modality.FLAIR, "flair.nii"),
                ),
            )
            manifest = DatasetManifest((case,))
            with self.assertRaisesRegex(DatasetError, "missing required modalities"):
                validate_dataset(
                    manifest, root, required_modalities=frozenset({Modality.T2})
                )
            with self.assertRaisesRegex(DatasetError, "physical grid"):
                validate_dataset(manifest, root)
            (root / "flair.nii").write_bytes(b"corrupt")
            with self.assertRaisesRegex(DatasetError, "volume failed validation"):
                validate_dataset(manifest, root)
