"""Smoke-test a running Compose stack with synthetic bytes and a local token file."""

import argparse
import json
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", default="http://127.0.0.1:8080")
parser.add_argument("--token-file", type=Path, default=Path(".secrets/api_token"))
args = parser.parse_args()
token = args.token_file.read_text().strip()


def request(
    path: str, body: bytes | None = None, content_type: str | None = None
) -> dict:
    headers = {"Authorization": "Bearer " + token}
    if content_type:
        headers.update(
            {"Content-Type": content_type, "X-Deidentified-Research": "true"}
        )
    query = urllib.request.Request(args.base + path, data=body, headers=headers)
    with urllib.request.urlopen(query, timeout=5) as response:
        return json.load(response)


deadline = time.monotonic() + 180
while True:
    try:
        request("/api/v1/ready")
        break
    except (urllib.error.URLError, TimeoutError):
        if time.monotonic() > deadline:
            raise
        time.sleep(1)

data = bytearray(352 + 64 * 4)
struct.pack_into("<i", data, 0, 348)
struct.pack_into("<8h", data, 40, 3, 4, 4, 4, 1, 1, 1, 1)
struct.pack_into("<hh", data, 70, 16, 32)
struct.pack_into("<4f", data, 76, 1, 1, 1, 1)
struct.pack_into("<f", data, 108, 352)
data[123] = 2
struct.pack_into("<h", data, 254, 1)
for offset in (280, 300, 320):
    struct.pack_into("<f", data, offset, 1)
data[344:348] = b"n+1\0"
for index in range(64):
    struct.pack_into("<f", data, 352 + index * 4, index / 63)
scan = request("/api/v1/scans?modality=t1", bytes(data), "application/octet-stream")
job = request(
    "/api/v1/inference",
    json.dumps({"scan_id": scan["id"], "model_id": "synthetic-threshold"}).encode(),
    "application/json",
)
deadline = time.monotonic() + 120
while True:
    result = request("/api/v1/jobs/" + job["id"])
    if result["status"] in {"failed", "succeeded"}:
        break
    if time.monotonic() > deadline:
        raise RuntimeError("Worker smoke test timed out")
    time.sleep(1)
if result["status"] != "succeeded":
    raise RuntimeError("Synthetic inference failed")
prediction = request(result["result_url"])
if prediction["label_voxels"] != [32, 32]:
    raise RuntimeError("Synthetic inference returned unexpected labels")
print("Compose API, shared database, worker and frontend proxy smoke test passed.")
