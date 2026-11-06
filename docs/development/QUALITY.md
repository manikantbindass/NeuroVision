# Quality gates

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.

The Python suite covers core units, synthetic preprocessing geometry and intensity regressions, model tensor shapes and training, API authorization and exports, and SQL/worker integration. The CI matrix runs on Linux and Windows across Python 3.11 and 3.14. The Linux 3.14 job enforces 75% combined statement and branch coverage across `neurovision` and `apps.api`. Coverage is a regression floor. It measures execution of the synthetic suite and provides no estimate of clinical performance.

Browser checks cover the mobile and desktop guided demo, corrupt upload recovery, and the connected workflow through preprocessing, inference, regional mapping, metrics and report downloads. Separate jobs exercise PostgreSQL and the Compose workflow with synthetic fixtures. GPU execution and live cloud checks require target infrastructure.

Run the Python gates after installing the development and test extras listed in [CI](../../.github/workflows/ci.yml).

```sh
python -m pip install "coverage>=7.10,<8"
python -m coverage run -m unittest discover -s tests -v
python -m coverage report
python -m ruff check neurovision apps tests scripts
python -m ruff check --select S neurovision apps/api
python -m mypy
python scripts/check_research_data.py
```

The security workflow checks Python application dependencies against known advisories and uses the npm lockfile for Node advisory checks. Redacted Gitleaks scans Git history, including the checkout being tested. The local pre-commit hook and CI reject common medical file extensions from the index without printing filenames. Review staged content for renamed files, metadata, screenshots and identifiers that format or secret scanners cannot detect. [Security workflow](../../.github/workflows/security.yml)
