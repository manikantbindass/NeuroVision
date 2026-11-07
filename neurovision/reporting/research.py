"""Build bounded report documents from explicit, immutable artifact snapshots."""

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from neurovision.core.errors import PipelineError
from neurovision.core.pipeline import RESEARCH_NOTICE

TITLE = "AI-Assisted Research Segmentation Report"
Document = dict[str, Any]
DEFAULT_LIMITATIONS = (
    "Benchmark pending.",
    "Model confidence and entropy are uncalibrated unless validated independently.",
    "Quantities depend on scan geometry, preprocessing, segmentation and registration.",
    "Atlas associations depend on atlas licensing, label definitions and alignment QC.",
    "A reproducibility fingerprint identifies inputs and model state. Bitwise replay "
    "depends on hardware and software versions.",
)
SECTION_TITLES = (
    ("scan_metadata", "Scan metadata"),
    ("preprocessing", "Preprocessing configuration and provenance"),
    ("qc", "Quality control findings"),
    ("models", "Models and versions"),
    ("segmentation_classes", "Segmentation classes"),
    ("volumetric_measurements", "Volumetric measurements"),
    ("confidence_statistics", "Confidence statistics"),
    ("uncertainty_summary", "Uncertainty summary"),
    ("evaluation_metrics", "Evaluation against reference labels"),
    ("atlas_measurements", "Atlas associations"),
    ("model_limitations", "Model limitations"),
    ("reproducibility", "Reproducibility metadata"),
)


class ReportingError(PipelineError):
    """The report input is inconsistent, unsupported or too large."""


class PDFUnavailable(ReportingError):
    """The optional PDF renderer is unavailable."""


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def software_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in (
        "neurovision-2028",
        "numpy",
        "scipy",
        "nibabel",
        "torch",
        "monai",
        "fastapi",
        "pydantic",
        "reportlab",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    return result


def canonical_json(document: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (ValueError, TypeError, RecursionError):
        raise ReportingError("Report data must be finite JSON values.") from None


def fingerprint(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def build_report(
    report_id: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    created_at: str | None = None,
) -> Document:
    """Snapshot supplied sources. Absent research data stays explicitly unavailable."""
    if not report_id.strip() or not 1 <= len(sources) <= 32:
        raise ReportingError("A report requires an ID and at most 32 sources.")
    encoded = canonical_json({"sources": list(sources)})
    if len(encoded.encode("utf-8")) > 1_000_000:
        raise ReportingError("Report sources exceed the metadata budget.")
    frozen = json.loads(encoded)["sources"]
    ids: set[str] = set()
    for source in frozen:
        if (
            not isinstance(source, dict)
            or source.get("kind")
            not in {"scan", "preprocess", "inference", "metrics", "maps"}
            or not isinstance(source.get("summary"), dict)
            or not isinstance(source["summary"].get("id"), str)
            or not source["summary"]["id"]
            or source["summary"]["id"] in ids
        ):
            raise ReportingError(
                "Report sources need unique artifact IDs and supported kinds."
            )
        ids.add(source["summary"]["id"])

    generated = created_at or timestamp()
    try:
        instant = datetime.fromisoformat(generated)
        if instant.utcoffset() is None:
            raise ValueError
    except ValueError:
        raise ReportingError(
            "Report timestamps require an explicit UTC offset."
        ) from None

    sections: Document = {key: [] for key, _ in SECTION_TITLES}
    recorded_scan_ids = {
        source["summary"]["id"]
        for source in frozen
        if source["kind"] in {"scan", "preprocess"}
    }
    recorded_preprocessing_ids = {
        source["summary"]["id"] for source in frozen if source["kind"] == "preprocess"
    }
    limitations = list(DEFAULT_LIMITATIONS)
    for source in frozen:
        kind, data = source["kind"], source["summary"]
        artifact_id = data["id"]
        row = {"artifact_id": artifact_id}
        if kind == "scan":
            sections["scan_metadata"].append(
                {
                    key: value
                    for key, value in data.items()
                    if key not in {"notice", "volume_sha256"}
                }
            )
        elif kind == "preprocess":
            sections["scan_metadata"].append(
                {
                    key: value
                    for key, value in data.items()
                    if key not in {"notice", "provenance", "volume_sha256"}
                }
            )
            provenance = data.get("provenance", {})
            sections["preprocessing"].append(
                {
                    **row,
                    **{key: value for key, value in provenance.items() if key != "qc"},
                }
            )
            sections["qc"].append(
                {
                    **row,
                    **{
                        key: value
                        for key, value in provenance.get("qc", {}).items()
                        if key != "notice"
                    },
                }
            )
        elif kind == "inference":
            for section, field in (
                ("models", "model"),
                ("segmentation_classes", "class_names"),
                ("volumetric_measurements", "volumetry"),
                ("confidence_statistics", "confidence_statistics"),
                ("uncertainty_summary", "uncertainty"),
            ):
                value = data.get(field)
                sections[section].append(
                    {
                        **row,
                        field: value,
                        "availability": "recorded"
                        if value is not None
                        else "unavailable",
                    }
                )
            snapshot = data.get("input_scan_snapshot")
            if snapshot and data.get("scan_id") not in recorded_scan_ids:
                sections["scan_metadata"].append(
                    {
                        **row,
                        "inference_input": {
                            key: value
                            for key, value in snapshot.items()
                            if key not in {"notice", "provenance", "volume_sha256"}
                        },
                    }
                )
            provenance = data.get("preprocessing_snapshot")
            if provenance and data.get("scan_id") not in recorded_preprocessing_ids:
                sections["preprocessing"].append({**row, **provenance})
                sections["qc"].append({**row, **provenance.get("qc", {})})
            limitations.extend(data.get("model_limitations", []))
        elif kind == "metrics":
            sections["evaluation_metrics"].append(data)
        elif kind == "maps":
            sections["atlas_measurements"].append(data)
        sections["reproducibility"].append(
            {
                **row,
                "kind": kind,
                "metadata_sha256": fingerprint(data),
                "created_at": data.get("created_at"),
                "input_volume_sha256": data.get(
                    "input_volume_sha256", data.get("volume_sha256")
                ),
                "labels_sha256": data.get("labels_sha256"),
                "model_state_sha256": data.get("model", {}).get("state_sha256"),
                "software": data.get(
                    "software", data.get("provenance", {}).get("software")
                ),
            }
        )
    sections["model_limitations"] = list(dict.fromkeys(limitations))
    report: Document = {
        "schema_version": 2,
        "id": report_id,
        "title": TITLE,
        "notice": RESEARCH_NOTICE,
        "created_at": generated,
        "software": software_versions(),
        "sections": sections,
        "sources": frozen,
        "missing_sections": [
            title for key, title in SECTION_TITLES if not sections[key]
        ],
    }
    report["content_sha256"] = fingerprint(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if (
        report.get("schema_version") != 2
        or report.get("title") != TITLE
        or report.get("notice") != RESEARCH_NOTICE
        or not isinstance(report.get("sections"), dict)
        or any(key not in report["sections"] for key, _ in SECTION_TITLES)
    ):
        raise ReportingError("Unsupported research report schema.")
    unsigned = {key: value for key, value in report.items() if key != "content_sha256"}
    if fingerprint(unsigned) != report.get("content_sha256"):
        raise ReportingError("Report snapshot fingerprint does not match.")


def report_json(report: Mapping[str, Any]) -> bytes:
    validate_report(report)
    return (
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
