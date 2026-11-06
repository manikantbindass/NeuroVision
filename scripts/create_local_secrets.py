"""Create disposable local Compose credentials without displaying their values."""

import argparse
import os
import secrets
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--directory", type=Path, default=Path(".secrets"))
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
os.chmod(args.directory, 0o700)
if any(args.directory.iterdir()):
    parser.error("Use an empty secret directory. Existing credentials were preserved.")
password = secrets.token_hex(32)
files = {
    "api_token": secrets.token_hex(32),
    "postgres_password": password,
    "database_url": f"postgresql+psycopg://neurovision:{password}@postgres:5432/neurovision",
    "redis_url": "redis://redis:6379/0",
}
for name, value in files.items():
    path = args.directory / name
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value + "\n")
    # Compose bind-mounts file secrets without remapping file ownership.
    # A private parent protects the host while containers read the mounted file.
    os.chmod(path, 0o444)
print("Created local development secrets. Restrict access to the directory.")
