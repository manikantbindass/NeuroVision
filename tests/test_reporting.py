"""Report snapshot integrity, escaping, missing data and PDF export checks."""

import io
import json
import unittest
from typing import Any
from unittest.mock import patch

from neurovision.reporting.export import report_html, report_pdf
from neurovision.reporting.research import (
    TITLE,
    PDFUnavailable,
    ReportingError,
    build_report,
    report_json,
    software_versions,
)


def sample_report() -> dict[str, object]:
    return build_report(
        "report-fixture",
        [
            {
                "kind": "scan",
                "summary": {
                    "id": "scan-fixture",
                    "created_at": "2026-11-06T12:00:00+00:00",
                    "shape": [4, 4, 4],
                    "spacing_mm": [2, 2, 2],
                    "modality": "t1",
                    "reference_space": "scanner",
                    "assumed_mm": False,
                },
            },
            {
                "kind": "inference",
                "summary": {
                    "id": "prediction-fixture",
                    "created_at": "2026-11-06T12:01:00+00:00",
                    "model": {
                        "id": "fixture",
                        "revision": "1",
                        "synthetic": True,
                        "state_sha256": "a" * 64,
                    },
                    "class_names": ["background", "Synthetic region"],
                    "volumetry": [
                        {
                            "name": "Synthetic region",
                            "voxel_count": 32,
                            "volume_mm3": 256,
                        }
                    ],
                    "confidence_statistics": {"mean": 0.8, "threshold": 0.8},
                    "uncertainty": {
                        "mean_normalized_entropy": 0.3,
                        "calibrated": False,
                    },
                    "model_limitations": ["Synthetic data only."],
                },
            },
        ],
        created_at="2026-11-06T12:02:00+00:00",
    )


class ReportingTests(unittest.TestCase):
    def test_immutable_deterministic_snapshot_and_missing_sections(self) -> None:
        source: dict[str, Any] = {
            "kind": "scan",
            "summary": {"id": "one", "shape": [3, 4, 5]},
        }
        with patch(
            "neurovision.reporting.research.software_versions",
            return_value={"python": "test"},
        ):
            first = build_report(
                "report", [source], created_at="2026-01-01T00:00:00+00:00"
            )
            second = build_report(
                "report", [source], created_at="2026-01-01T00:00:00+00:00"
            )
        self.assertEqual(first, second)
        source["summary"]["shape"][0] = 999
        self.assertEqual(first["sources"][0]["summary"]["shape"], [3, 4, 5])
        self.assertIn("Evaluation against reference labels", first["missing_sections"])
        self.assertEqual(json.loads(report_json(first))["title"], TITLE)
        first["sections"]["scan_metadata"][0]["shape"][0] = 2
        with self.assertRaisesRegex(ReportingError, "fingerprint"):
            report_html(first)

    def test_html_escapes_untrusted_text_and_json_rejects_nonfinite_values(
        self,
    ) -> None:
        report = build_report(
            "r",
            [
                {
                    "kind": "scan",
                    "summary": {"id": "s", "name": '<script>alert("scan")</script>'},
                }
            ],
        )
        html = report_html(report).decode()
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)
        self.assertIn("Unavailable", html)
        self.assertIn("Content-Security-Policy", html)
        with self.assertRaises(ReportingError):
            build_report(
                "r", [{"kind": "scan", "summary": {"id": "s", "value": float("nan")}}]
            )
        with self.assertRaises(ReportingError):
            build_report("r", [report])

    def test_pdf_optional_dependency_and_extracted_content(self) -> None:
        from pypdf import PdfReader

        report = sample_report()
        data = report_pdf(report)
        self.assertTrue(data.startswith(b"%PDF"))
        reader = PdfReader(io.BytesIO(data))
        text = " ".join(" ".join(page.extract_text().split()) for page in reader.pages)
        for value in (
            TITLE,
            "Synthetic region",
            "256",
            "Synthetic data only.",
            "Not a medical device",
            "Reproducibility metadata",
        ):
            self.assertIn(value, text)
        assert reader.metadata is not None
        self.assertEqual(reader.metadata.title, TITLE)
        self.assertIn("python", software_versions())
        with patch.dict("sys.modules", {"reportlab": None}):
            with self.assertRaises(PDFUnavailable):
                report_pdf(report)
