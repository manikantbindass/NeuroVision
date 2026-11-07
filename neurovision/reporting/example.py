"""Generate a complete research report from an in-memory synthetic API workflow."""

import argparse
import io
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

from apps.api.adapters import Job, MemoryStore
from apps.api.contracts import (
    InferenceRequest,
    MetricsRequest,
    PreprocessRequest,
    ReportRequest,
)
from apps.api.main import DEFAULT_LIMITS
from apps.api.service import ResearchService, demo_models, nifti_bytes
from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.imaging.volume import Modality, canonical_volume
from neurovision.reporting.export import report_html, report_pdf
from neurovision.reporting.research import Document, report_json


def synthetic_report() -> Document:
    values = np.linspace(0, 1, 512, dtype=np.float32).reshape((8, 8, 8))
    volume = canonical_volume(
        values,
        np.diag([1.5, 1.5, 2.0, 1.0]),
        source_format="nifti",
        modality=Modality.T1,
        reference_space="scanner",
    )
    service = ResearchService(MemoryStore(), demo_models(), DEFAULT_LIMITS)
    owner = "synthetic-local-demo"
    scan = service.upload(
        owner, io.BytesIO(nifti_bytes(volume, values)), "nii", Modality.T1, "image"
    )
    truth = service.upload(
        owner,
        io.BytesIO(nifti_bytes(volume, (values > 0.5).astype(np.uint16))),
        "nii",
        Modality.T1,
        "labels",
    )
    prep, prediction, metrics, report = (uuid4() for _ in range(4))
    service.run(
        Job(
            owner,
            prep,
            "preprocess",
            PreprocessRequest(scan_id=UUID(scan["id"]), normalization="none"),
            "demo",
        )
    )
    service.run(
        Job(
            owner,
            prediction,
            "inference",
            InferenceRequest(scan_id=prep, model_id="synthetic-threshold"),
            "demo",
        )
    )
    service.run(
        Job(
            owner,
            metrics,
            "metrics",
            MetricsRequest(
                segmentation_id=prediction, target_scan_id=UUID(truth["id"])
            ),
            "demo",
        )
    )
    service.run(
        Job(owner, report, "reports", ReportRequest(artifact_ids=[metrics]), "demo")
    )
    return service.summary(owner, report, "reports")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic research report.", epilog=RESEARCH_NOTICE
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args(argv)
    report = synthetic_report()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "research-report.json").write_bytes(report_json(report))
    (args.output / "research-report.html").write_bytes(report_html(report))
    if args.pdf:
        (args.output / "research-report.pdf").write_bytes(report_pdf(report))
    print(RESEARCH_NOTICE)
    print("Wrote the synthetic AI-Assisted Research Segmentation Report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
