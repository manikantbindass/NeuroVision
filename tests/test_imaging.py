import gzip
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    UID,
    DeflatedExplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    MRImageStorage,
)

from neurovision.imaging.dicom import discover_dicom_series, load_dicom_series
from neurovision.imaging.limits import IngestionLimits
from neurovision.imaging.nifti import read_nifti
from neurovision.imaging.volume import ImagingError, Modality


class NiftiTests(unittest.TestCase):
    def test_nifti1_and_2_preserve_physical_coordinates_after_flip_and_permutation(
        self,
    ) -> None:
        source = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        affine = np.array([[0, -2, 0, 8], [3, 0, 0, 9], [0, 0, 4, 10], [0, 0, 0, 1.0]])
        for version in (1, 2):
            if version == 1:
                image = nib.Nifti1Image(source, affine)
            else:
                image = nib.Nifti2Image(source, affine)
            image.header.set_xyzt_units("mm")
            for compressed in (False, True):
                with self.subTest(version=version, compressed=compressed):
                    payload = image.to_bytes()
                    volume = read_nifti(
                        BytesIO(gzip.compress(payload) if compressed else payload),
                        filename="image.nii.gz" if compressed else "image.nii",
                    )
                    self.assertEqual(volume.spacing_mm, (2, 3, 4))
                    self.assertEqual(volume.orientation, ("R", "A", "S"))
                    self.assertFalse(volume.data.flags.writeable)
                    self.assertFalse(volume.affine.flags.writeable)
                    self.assertEqual(volume.modality, Modality.UNKNOWN)
                    self.assertEqual(volume.data[0, 1, 2], source[1, 2, 2])
                    np.testing.assert_allclose(
                        volume.affine @ [0, 1, 2, 1], affine @ [1, 2, 2, 1]
                    )

    def test_nifti_qform_unit_conversion_and_intensity_scaling(self) -> None:
        image = nib.Nifti1Image(np.full((2, 2, 2), 3, dtype=np.int16), np.eye(4))
        image.set_sform(None, code=0)
        image.set_qform(np.diag([0.002, 0.003, 0.004, 1]), code=1)
        image.header.set_xyzt_units("meter")
        image.header.set_slope_inter(2, -1)
        volume = read_nifti(
            BytesIO(image.to_bytes()), filename="vol.nii", modality=Modality.T1CE
        )
        np.testing.assert_allclose(volume.spacing_mm, [2, 3, 4], rtol=1e-6)
        np.testing.assert_array_equal(volume.data, np.full((2, 2, 2), 5))
        self.assertEqual(volume.reference_space, "scanner")
        self.assertEqual(volume.modality, Modality.T1CE)

    def test_bounds_and_corruption(self) -> None:
        image = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
        raw = image.to_bytes()
        compressed = gzip.compress(raw)
        for payload, name, limits in (
            (raw, "file.zip", IngestionLimits()),
            (raw, "file.nii", IngestionLimits(max_input_bytes=30)),
            (compressed, "file.nii.gz", IngestionLimits(max_decoded_bytes=400)),
            (compressed[:-2], "file.nii.gz", IngestionLimits()),
            (raw[:-1], "file.nii", IngestionLimits()),
            (raw, "file.nii", IngestionLimits(max_voxels=32)),
            (b"garbage", "file.nii", IngestionLimits()),
        ):
            with (
                self.subTest(name=name, limits=limits),
                self.assertRaises(ImagingError),
            ):
                read_nifti(BytesIO(payload), filename=name, limits=limits)

    def test_rejects_nonfinite_and_nonscalar_volumes(self) -> None:
        for data in (np.full((2, 2, 2), np.nan), np.zeros((2, 2, 2, 1))):
            image = nib.Nifti1Image(data, np.eye(4))
            with self.assertRaises(ImagingError):
                read_nifti(BytesIO(image.to_bytes()), filename="vol.nii")

    def test_forged_nifti2_shape_is_rejected_before_voxel_allocation(self) -> None:
        image = nib.Nifti2Image(np.ones((2, 2, 2)), np.eye(4))
        payload = bytearray(image.to_bytes())
        payload[24:48] = (2**32).to_bytes(8, "little") * 3
        with self.assertRaisesRegex(ImagingError, "voxel"):
            read_nifti(BytesIO(payload), filename="vol.nii")


def _slice(
    path: Path, z: int, *, series: int = 1, implicit: bool = False
) -> FileDataset:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = (
        ImplicitVRLittleEndian if implicit else ExplicitVRLittleEndian
    )
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = UID(f"1.2.826.0.1.3680043.10.543.1.{series}.{z}")
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=bytes(128))
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "1.2.826.0.1.3680043.10.543.2"
    ds.SeriesInstanceUID = f"1.2.826.0.1.3680043.10.543.3.{series}"
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.543.4"
    ds.Modality = "MR"
    ds.SeriesDescription = "FLAIR" if series == 1 else "nonstandard sequence"
    ds.PatientIdentityRemoved = "YES"
    ds.BurnedInAnnotation = "NO"
    ds.RecognizableVisualFeatures = "NO"
    ds.Rows = 2
    ds.Columns = 3
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 1
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [10, 20, z]
    ds.PixelSpacing = [2, 3]
    ds.RescaleSlope = 2
    ds.RescaleIntercept = 5
    ds.PixelData = (np.arange(6, dtype="<i2") + z - 10).tobytes()
    return ds


class DicomTests(unittest.TestCase):
    def test_group_sort_reorient_and_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for series in (1, 2):
                for index, z in enumerate((6, 0, 3)):
                    path = root / f"series{series}-{index}.dcm"
                    _slice(path, z, series=series, implicit=series == 2).save_as(
                        path, enforce_file_format=True
                    )
            groups = discover_dicom_series(root, privacy_reviewed=True)
            self.assertEqual(
                [group.modality for group in groups], [Modality.FLAIR, Modality.UNKNOWN]
            )
            volume = load_dicom_series(groups[0].files, privacy_reviewed=True)
            self.assertEqual(volume.data.shape, (3, 2, 3))
            self.assertEqual(volume.spacing_mm, (3, 2, 3))
            self.assertEqual(volume.data[0, 0, 0], -5)
            self.assertEqual(volume.data[2, 1, 2], -3)
            np.testing.assert_allclose(volume.affine @ [2, 1, 2, 1], [-10, -20, 6, 1])
            implicit = load_dicom_series(groups[1].files, privacy_reviewed=True)
            np.testing.assert_array_equal(implicit.data, volume.data)
            with self.assertRaises(ImagingError):
                load_dicom_series(
                    (*groups[0].files, *groups[1].files), privacy_reviewed=True
                )
            with self.assertRaises(ImagingError):
                discover_dicom_series(
                    root,
                    privacy_reviewed=True,
                    limits=IngestionLimits(max_directory_entries=2),
                )

    def test_requires_review_and_refuses_privacy_flags_identifiers_and_private_tags(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "1.dcm", root / "2.dcm"
            _slice(first, 0).save_as(first, enforce_file_format=True)
            with self.assertRaisesRegex(ImagingError, "privacy review"):
                discover_dicom_series(root)
            with self.assertRaisesRegex(ImagingError, "privacy review"):
                load_dicom_series([first])
            for key, value in (
                ("BurnedInAnnotation", "YES"),
                ("PatientName", "Synthetic^Test"),
                ("RecognizableVisualFeatures", ""),
            ):
                ds = _slice(second, 3)
                setattr(ds, key, value)
                ds.save_as(second, enforce_file_format=True)
                with self.assertRaises(ImagingError) as failure:
                    load_dicom_series([first, second], privacy_reviewed=True)
                self.assertNotIn("Synthetic", str(failure.exception))
            ds = _slice(second, 3)
            ds.add_new((0x0011, 0x0010), "LO", "Synthetic private creator")
            ds.save_as(second, enforce_file_format=True)
            with self.assertRaisesRegex(ImagingError, "identifying"):
                discover_dicom_series(root, privacy_reviewed=True)

    def test_rejects_truncated_pixels_irregular_positions_and_unsupported_syntax(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{z}.dcm" for z in (0, 3, 9)]
            for path in paths:
                _slice(path, int(path.stem)).save_as(path, enforce_file_format=True)
            with self.assertRaisesRegex(ImagingError, "irregular"):
                load_dicom_series(paths, privacy_reviewed=True)
            paths[1].write_bytes(paths[1].read_bytes()[:-2])
            with self.assertRaises(ImagingError):
                load_dicom_series(paths[:2], privacy_reviewed=True)
            ds = _slice(paths[1], 3)
            ds.file_meta.TransferSyntaxUID = DeflatedExplicitVRLittleEndian
            ds.save_as(paths[1], enforce_file_format=True)
            with self.assertRaisesRegex(ImagingError, "transfer syntax"):
                load_dicom_series(paths[:2], privacy_reviewed=True)
