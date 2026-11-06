"""Exercise the FastAPI research workflow with synthetic NIfTI inputs only."""

import logging
import unittest
from collections.abc import Iterator
from time import monotonic, sleep

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from apps.api.adapters import (
    BearerTokens,
    FixedWindow,
    Job,
    JobEvent,
    JobRunner,
    LocalJobs,
)
from apps.api.main import create_app
from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.imaging.limits import IngestionLimits

PREFIX = "/api/v1"
OWNER = {"Authorization": "Bearer " + "a" * 40}
OTHER = {"Authorization": "Bearer " + "b" * 40}
UPLOAD = {
    **OWNER,
    "Content-Type": "application/octet-stream",
    "X-Deidentified-Research": "true",
}


def _auth() -> BearerTokens:
    return BearerTokens({"a" * 40: "one", "b" * 40: "two"})


def _nifti(data: NDArray[np.generic], spacing: float = 2.0) -> bytes:
    image = nib.Nifti1Image(data, np.diag([spacing, spacing, spacing, 1.0]))
    image.header.set_xyzt_units("mm")
    image.set_sform(image.affine, code="scanner")
    return bytes(image.to_bytes())


class BackendTests(unittest.TestCase):
    def _upload(self, client: TestClient, body: bytes, role: str = "image") -> str:
        response = client.post(
            f"{PREFIX}/scans?role={role}&modality=t1", content=body, headers=UPLOAD
        )
        self.assertEqual(response.status_code, 201, response.text)
        return str(response.json()["id"])

    def _wait(self, client: TestClient, job_id: str) -> dict[str, object]:
        deadline = monotonic() + 10
        while monotonic() < deadline:
            response = client.get(f"{PREFIX}/jobs/{job_id}", headers=OWNER)
            self.assertEqual(response.status_code, 200, response.text)
            status: dict[str, object] = response.json()
            if status["status"] in ("failed", "succeeded"):
                return status
            sleep(0.03)
        self.fail("Research job did not finish")

    def _submit(self, client: TestClient, path: str, body: dict[str, object]) -> str:
        response = client.post(f"{PREFIX}/{path}", json=body, headers=OWNER)
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["status"], "queued")
        return str(response.json()["id"])

    def _result(self, client: TestClient, job_id: str) -> dict[str, object]:
        result = self._wait(client, job_id)
        self.assertEqual(result["status"], "succeeded", result)
        response = client.get(str(result["result_url"]), headers=OWNER)
        self.assertEqual(response.status_code, 200, response.text)
        metadata: dict[str, object] = response.json()
        self.assertEqual(metadata["notice"], RESEARCH_NOTICE)
        return metadata

    def test_research_workflow_and_owner_isolation(self) -> None:
        with TestClient(create_app(authenticator=_auth())) as client:
            signal = np.linspace(0, 1, 64, dtype=np.float32).reshape(4, 4, 4)
            scan_id = self._upload(client, _nifti(signal))
            truth = self._upload(
                client, _nifti((signal > 0.5).astype(np.uint16)), "labels"
            )
            atlas = self._upload(
                client, _nifti(np.ones(signal.shape, dtype=np.uint16)), "labels"
            )
            self.assertEqual(
                client.get(f"{PREFIX}/scans", headers=OTHER).json(), {"ids": []}
            )
            self.assertEqual(
                client.get(f"{PREFIX}/scans/{scan_id}", headers=OTHER).status_code, 404
            )
            preprocessing = self._submit(
                client, "preprocess", {"scan_id": scan_id, "normalization": "none"}
            )
            self.assertTrue(self._result(client, preprocessing)["usable"])

            model_response = client.get(f"{PREFIX}/models", headers=OWNER)
            self.assertTrue(model_response.json()["models"][0]["synthetic"])
            prediction = self._submit(
                client,
                "inference",
                {"scan_id": preprocessing, "model_id": "synthetic-threshold"},
            )
            self.assertEqual(self._result(client, prediction)["label_voxels"], [32, 32])
            self.assertEqual(
                client.get(f"{PREFIX}/jobs/{prediction}", headers=OTHER).status_code,
                404,
            )
            self.assertEqual(
                client.get(
                    f"{PREFIX}/segmentations/{prediction}", headers=OTHER
                ).status_code,
                404,
            )
            downloaded = client.get(
                f"{PREFIX}/segmentations/{prediction}/labels.nii", headers=OWNER
            )
            self.assertEqual(downloaded.status_code, 200, downloaded.text)
            label_image = nib.Nifti2Image.from_bytes(downloaded.content)
            np.testing.assert_array_equal(label_image.get_fdata(), signal > 0.5)
            np.testing.assert_allclose(label_image.affine, np.diag([2, 2, 2, 1]))
            uncertain = client.get(f"{PREFIX}/uncertainty/{prediction}", headers=OWNER)
            self.assertFalse(uncertain.json()["calibrated"])
            entropy = client.get(
                f"{PREFIX}/uncertainty/{prediction}/entropy.nii", headers=OWNER
            )
            self.assertEqual(entropy.status_code, 200)
            entropy_map = nib.Nifti2Image.from_bytes(entropy.content).get_fdata()
            self.assertTrue(np.all((entropy_map >= 0) & (entropy_map <= 1)))

            metric_id = self._submit(
                client,
                "metrics",
                {"segmentation_id": prediction, "target_scan_id": truth},
            )
            metric_result = self._result(client, metric_id)
            self.assertEqual(
                metric_result["classes"],
                [
                    {
                        "label": 1,
                        "dice": 1.0,
                        "iou": 1.0,
                        "precision": 1.0,
                        "sensitivity": 1.0,
                        "specificity": 1.0,
                        "hausdorff_mm": 0.0,
                        "hd95_mm": 0.0,
                        "assd_mm": 0.0,
                    }
                ],
            )
            map_id = self._submit(
                client,
                "maps",
                {
                    "segmentation_id": prediction,
                    "atlas_scan_id": atlas,
                    "atlas_name": "Synthetic atlas",
                    "atlas_version": "1",
                    "atlas_license": "MIT",
                    "atlas_space": "phantom",
                    "registration_provenance": "Same synthetic grid",
                    "segmentation_to_atlas_world": np.eye(4).tolist(),
                    "regions": [
                        {"code": 1, "name": "Test region", "hemisphere": "midline"}
                    ],
                },
            )
            result = client.get(
                str(self._wait(client, map_id)["result_url"]), headers=OWNER
            )
            self.assertEqual(result.json()["measurements"][0]["voxels"], 32)
            self.assertAlmostEqual(result.json()["measurements"][0]["volume_mm3"], 256)
            report_id = self._submit(
                client, "reports", {"artifact_ids": [map_id, metric_id]}
            )
            result = client.get(
                str(self._wait(client, report_id)["result_url"]), headers=OWNER
            )
            self.assertEqual(
                [source["kind"] for source in result.json()["sources"]],
                ["maps", "metrics"],
            )
            self.assertEqual(
                client.delete(f"{PREFIX}/jobs/{prediction}", headers=OWNER).status_code,
                204,
            )
            self.assertEqual(
                client.get(
                    f"{PREFIX}/uncertainty/{prediction}", headers=OWNER
                ).status_code,
                404,
            )

    def test_request_boundaries_and_sanitized_errors(self) -> None:
        logger = logging.getLogger("neurovision.api.test")
        app = create_app(
            authenticator=_auth(),
            logger=logger,
            limits=IngestionLimits(
                max_input_bytes=1024, max_decoded_bytes=1024, max_voxels=64
            ),
        )
        with TestClient(app) as client, self.assertLogs(logger, "INFO") as logs:
            self.assertEqual(
                client.get(f"{PREFIX}/health").json()["notice"], RESEARCH_NOTICE
            )
            schema = client.get(f"{PREFIX}/openapi.json")
            self.assertEqual(schema.status_code, 200)
            self.assertIn(f"{PREFIX}/inference", schema.json()["paths"])
            self.assertIn("HTTPBearer", schema.json()["components"]["securitySchemes"])
            self.assertEqual(client.get(f"{PREFIX}/models").status_code, 401)
            invalid = client.post(
                f"{PREFIX}/preprocess",
                headers={**OWNER, "X-Request-ID": "do-not-log"},
                json={"scan_id": "private-input", "extra": "do-not-log"},
            )
            self.assertEqual(invalid.status_code, 422, invalid.text)
            self.assertEqual(
                invalid.json()["error"]["request_id"], invalid.headers["x-request-id"]
            )
            self.assertNotIn("private-input", invalid.text)
            self.assertEqual(invalid.headers["cache-control"], "no-store")
            self.assertEqual(
                client.post(
                    f"{PREFIX}/scans", headers=UPLOAD, content=b"x" * 1025
                ).status_code,
                413,
            )

            def chunks() -> Iterator[bytes]:
                yield b"x" * 600
                yield b"y" * 425

            oversized = client.post(f"{PREFIX}/scans", headers=UPLOAD, content=chunks())
            self.assertEqual(oversized.status_code, 413, oversized.text)
            large_json = client.post(
                f"{PREFIX}/inference",
                headers={**OWNER, "Content-Type": "application/json"},
                content=iter([b" " * 40_000, b" " * 30_000]),
            )
            self.assertEqual(large_json.status_code, 413, large_json.text)
            self.assertEqual(
                client.post(
                    f"{PREFIX}/scans", headers=UPLOAD, content=b"broken"
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    f"{PREFIX}/scans", headers=OWNER, content=b"broken"
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    f"{PREFIX}/scans",
                    headers={**UPLOAD, "Content-Type": "text/plain"},
                    content=b"broken",
                ).status_code,
                415,
            )
            self.assertNotIn("do-not-log", "\n".join(logs.output))
            self.assertNotIn("private-input", "\n".join(logs.output))

    def test_rate_limit_is_owner_scoped(self) -> None:
        with TestClient(
            create_app(authenticator=_auth(), limiter=FixedWindow(1, 3600))
        ) as client:
            self.assertEqual(
                client.get(f"{PREFIX}/models", headers=OWNER).status_code, 200
            )
            limited = client.get(f"{PREFIX}/models", headers=OWNER)
            self.assertEqual(limited.status_code, 429)
            self.assertGreater(int(limited.headers["retry-after"]), 0)
            self.assertEqual(
                client.get(f"{PREFIX}/models", headers=OTHER).status_code, 200
            )

    def test_queue_bound_and_async_failure(self) -> None:
        from threading import Event

        started, proceed = Event(), Event()

        def queue_factory(run: JobRunner, emit: JobEvent) -> LocalJobs:
            def wait_then_run(job: Job) -> str:
                started.set()
                if not proceed.wait(timeout=5):
                    raise RuntimeError("test queue gate timed out")
                return run(job)

            return LocalJobs(wait_then_run, emit, pending=1)

        app = create_app(authenticator=_auth(), queue_factory=queue_factory)
        with TestClient(app) as client:
            scan_id = self._upload(client, _nifti(np.ones((2, 2, 2), dtype=np.float32)))
            job_id = self._submit(client, "preprocess", {"scan_id": scan_id})
            try:
                self.assertTrue(started.wait(timeout=5))
                busy = client.post(
                    f"{PREFIX}/preprocess", headers=OWNER, json={"scan_id": scan_id}
                )
                self.assertEqual(busy.status_code, 503, busy.text)
                self.assertEqual(
                    client.delete(f"{PREFIX}/jobs/{job_id}", headers=OWNER).status_code,
                    409,
                )
                self.assertEqual(
                    client.delete(
                        f"{PREFIX}/scans/{scan_id}", headers=OWNER
                    ).status_code,
                    204,
                )
            finally:
                proceed.set()
            self.assertEqual(self._wait(client, job_id)["status"], "failed")
