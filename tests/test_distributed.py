"""Durable API/worker integration, quota and stale-worker regressions."""

import io
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep
from uuid import UUID, uuid4

import fakeredis
import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import update

from apps.api.adapters import Artifact, BearerTokens
from apps.api.contracts import APIError, InferenceRequest
from apps.api.deployed import RedisRateLimiter
from apps.api.durable import Database, SQLArtifactStore, SQLJobs, jobs, metadata
from apps.api.main import DEFAULT_LIMITS, create_app
from apps.api.service import ResearchService, demo_models
from apps.api.worker import run_lease
from neurovision.imaging.volume import Modality


def create_test_database(directory: str) -> Database:
    url = os.environ.get("NEUROVISION_TEST_DATABASE_URL")
    if url:
        if not url.endswith("/neurovision_test"):
            raise ValueError("Use only a dedicated neurovision_test database.")
        database = Database(url)
        metadata.drop_all(database.engine)
    else:
        database = Database("sqlite:///" + (Path(directory) / "research.db").as_posix())
    database.initialize()
    return database


class DistributedTests(unittest.TestCase):
    def test_separate_api_worker_restart_and_private_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = create_test_database(directory)
            store, queue = SQLArtifactStore(database), SQLJobs(database)
            try:
                token = "research-fixture-" + "a" * 32
                headers = {"Authorization": "Bearer " + token}
                app = create_app(
                    authenticator=BearerTokens({token: "owner"}),
                    store=store,
                    queue_factory=lambda run, event: queue,
                    readiness=database.ready,
                )
                with TestClient(app) as client:
                    self.assertEqual(client.get("/api/v1/ready").status_code, 200)
                    values = np.linspace(0, 1, 64, dtype=np.float32).reshape(4, 4, 4)
                    image = nib.Nifti1Image(values, np.eye(4))
                    image.header.set_xyzt_units("mm")
                    image.set_sform(np.eye(4), code="scanner")
                    uploaded = client.post(
                        "/api/v1/scans?modality=t1",
                        headers={
                            **headers,
                            "X-Deidentified-Research": "true",
                            "Content-Type": "application/octet-stream",
                        },
                        content=image.to_bytes(),
                    )
                    self.assertEqual(uploaded.status_code, 201, uploaded.text)
                    submitted = client.post(
                        "/api/v1/inference",
                        headers=headers,
                        json={
                            "scan_id": uploaded.json()["id"],
                            "model_id": "synthetic-threshold",
                        },
                    )
                    self.assertEqual(submitted.status_code, 202, submitted.text)
                    lease = queue.claim()
                    assert lease is not None
                    run_lease(database, queue, lease, logging.getLogger("worker-test"))
                    status = client.get(
                        f"/api/v1/jobs/{lease.job.id}", headers=headers
                    ).json()
                    self.assertEqual(status["status"], "succeeded", status)
                    result = client.get(status["result_url"], headers=headers)
                    self.assertEqual(result.json()["label_voxels"], [32, 32])
                    second = queue.submit(
                        "owner",
                        "inference",
                        InferenceRequest(
                            scan_id=UUID(uploaded.json()["id"]),
                            model_id="synthetic-threshold",
                        ),
                        "process-fixture",
                    )
                    connection_url = database.engine.url.render_as_string(
                        hide_password=False
                    )
                    environment = {
                        **os.environ,
                        "NEUROVISION_DATABASE_URL": connection_url,
                        "NEUROVISION_WORKER_HEALTH_PORT": "0",
                        "NEUROVISION_WORKER_DEVICE": "cpu",
                    }
                    environment.pop("NEUROVISION_DATABASE_URL_FILE", None)
                    process = subprocess.Popen(
                        [sys.executable, "-m", "apps.api.worker"],
                        env=environment,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW
                        if os.name == "nt"
                        else 0,
                    )
                    try:
                        deadline = monotonic() + 30
                        while queue.get("owner", second.id).status in {
                            "queued",
                            "running",
                        }:
                            if process.poll() is not None or monotonic() > deadline:
                                self.fail(
                                    "Separate worker process did not finish its job"
                                )
                            sleep(0.1)
                        self.assertEqual(
                            queue.get("owner", second.id).status, "succeeded"
                        )
                        removed = client.delete(
                            f"/api/v1/jobs/{second.id}", headers=headers
                        )
                        self.assertEqual(removed.status_code, 204)
                        with self.assertRaises(APIError):
                            store.get("owner", second.id)
                    finally:
                        process.terminate()
                        process.wait(timeout=10)
                    with self.assertRaises(APIError):
                        store.get("another-owner", lease.job.id)
                reopened = Database(
                    database.engine.url.render_as_string(hide_password=False)
                )
                try:
                    result = SQLArtifactStore(reopened).get("owner", lease.job.id)
                    self.assertEqual(result.kind, "inference")
                finally:
                    reopened.engine.dispose()
            finally:
                database.engine.dispose()

    def test_expired_lease_cannot_publish_and_completed_output_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = create_test_database(directory)
            try:
                store, queue = SQLArtifactStore(database), SQLJobs(database)
                service = ResearchService(store, demo_models(), DEFAULT_LIMITS)
                image = nib.Nifti1Image(
                    np.linspace(0, 1, 64, dtype=np.float32).reshape(4, 4, 4), np.eye(4)
                )
                image.header.set_xyzt_units("mm")
                image.set_sform(np.eye(4), code="scanner")
                scan = service.upload(
                    "owner", io.BytesIO(image.to_bytes()), "nii", Modality.T1, "image"
                )
                submitted = queue.submit(
                    "owner",
                    "inference",
                    InferenceRequest(
                        scan_id=UUID(scan["id"]), model_id="synthetic-threshold"
                    ),
                    "request",
                )
                with ThreadPoolExecutor(4) as executor:
                    claimed = list(executor.map(lambda _: queue.claim(), range(4)))
                leases = [lease for lease in claimed if lease is not None]
                self.assertEqual(len(leases), 1)
                old = leases[0]
                with database.engine.begin() as db:
                    db.execute(
                        update(jobs)
                        .where(jobs.c.id == str(submitted.id))
                        .values(lease_until=0)
                    )
                new = queue.claim()
                assert new is not None
                with self.assertRaisesRegex(APIError, "lease"):
                    SQLArtifactStore(database, lease=old).put(
                        "owner", submitted.id, Artifact("inference", {}, 0)
                    )
                self.assertFalse(queue.complete(old))
                leased_service = ResearchService(
                    SQLArtifactStore(database, lease=new), demo_models(), DEFAULT_LIMITS
                )
                leased_service.run(new.job)
                with database.engine.begin() as db:
                    db.execute(
                        update(jobs)
                        .where(jobs.c.id == str(submitted.id))
                        .values(lease_until=0)
                    )
                queue.claim()
                self.assertEqual(queue.get("owner", submitted.id).status, "succeeded")
                self.assertEqual(len(store.keys("owner", "inference")), 1)
            finally:
                database.engine.dispose()

    def test_shared_rate_limits_and_storage_quota(self) -> None:
        redis = fakeredis.FakeRedis()
        first, second = RedisRateLimiter(redis, 1), RedisRateLimiter(redis, 1)
        self.assertIsNone(first.retry_after("owner"))
        self.assertIsNotNone(second.retry_after("owner"))
        self.assertIsNone(first.retry_after("another"))
        with tempfile.TemporaryDirectory() as directory:
            database = create_test_database(directory)
            try:
                store = SQLArtifactStore(database, max_items=1)
                key = uuid4()
                store.put("owner", key, Artifact("reports", {"notice": "fixture"}, 5))
                with self.assertRaises(APIError):
                    store.put("owner", uuid4(), Artifact("reports", {}, 5))
                store.remove("owner", key)
                store.put("owner", uuid4(), Artifact("reports", {}, 5))
            finally:
                database.engine.dispose()
