"""Human-readable HTML and optional PDF views of a fingerprinted snapshot."""

from collections.abc import Iterator, Mapping
from html import escape
from io import BytesIO
from typing import Any

from neurovision.reporting.research import (
    SECTION_TITLES,
    Document,
    PDFUnavailable,
    validate_report,
)


def _label(key: str) -> str:
    return key.replace("_", " ")


def _rows(value: object, prefix: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        if not value:
            yield prefix, "Unavailable"
        for key, item in value.items():
            title = f"{prefix} / {_label(str(key))}" if prefix else _label(str(key))
            yield from _rows(item, title)
    elif isinstance(value, (tuple, list)):
        if not value:
            yield prefix, "No entries recorded"
        elif prefix and all(
            not isinstance(item, (dict, list, tuple)) for item in value
        ):
            yield prefix, ", ".join(str(item) for item in value)
        else:
            for i, item in enumerate(value):
                title = f"{prefix} / item {i + 1}" if prefix else f"Item {i + 1}"
                yield from _rows(item, title)
    else:
        text = "Unavailable" if value is None else str(value)
        yield prefix, text


def _blocks(report: Mapping[str, Any]) -> Iterator[tuple[str, object]]:
    yield (
        "Report record",
        {
            "report_id": report["id"],
            "created_at": report["created_at"],
            "software": report["software"],
            "content_sha256": report["content_sha256"],
        },
    )
    for key, title in SECTION_TITLES:
        yield title, report["sections"][key] or None


def report_html(report: Document) -> bytes:
    validate_report(report)
    sections = []
    for title, value in _blocks(report):
        rows = "".join(
            f'<tr><th scope="row">{escape(key)}</th><td>{escape(item)}</td></tr>'
            for key, item in _rows(value)
        )
        sections.append(
            f"<section><h2>{escape(title)}</h2><table>{rows}</table></section>"
        )
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'\">"
        f"<title>{escape(report['title'])}</title>"
        "<style>"
        "body{font:14px/1.5 system-ui,sans-serif;color:#173344;background:#edf2f5;"
        "max-width:1000px;margin:0 auto;padding:32px}"
        "header,section{background:white;padding:22px;margin-bottom:18px;"
        "border:1px solid #c6d5de;border-radius:8px}"
        "header{border-top:5px solid #176956}"
        "h1{font-size:30px;line-height:1.15}h2{font-size:20px;color:#176956}"
        ".notice{padding:12px;border:1px solid #829ca8;background:#f2f8f7}"
        "table{border-collapse:collapse;width:100%;table-layout:fixed}"
        "th,td{border-bottom:1px solid #d7e2e8;padding:7px;text-align:left;"
        "vertical-align:top;overflow-wrap:anywhere}"
        "th{width:40%;font-weight:500;color:#345567}"
        "footer{font-size:12px} @media(max-width:600px){body{padding:10px}"
        "header,section{padding:12px}h1{font-size:24px}}"
        "@media print{body{background:white;padding:0}section{break-inside:auto}"
        "tr{break-inside:avoid}h2{break-after:avoid}}"
        "</style></head><body><header>"
        "<p>NEUROVISION / RESEARCH RECORD</p>"
        f"<h1>{escape(report['title'])}</h1>"
        f'<p class="notice">{escape(report["notice"])}</p>'
        "<p>Unavailable entries identify information absent from this snapshot.</p>"
        f"</header>{''.join(sections)}"
        f"<footer>{escape(report['notice'])}</footer></body></html>"
    )
    return html.encode("utf-8")


def pdf_available() -> bool:
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return False
    return True


def report_pdf(report: Document) -> bytes:
    validate_report(report)
    try:
        from pathlib import Path

        import reportlab
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            LongTable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            TableStyle,
        )
    except ImportError:
        raise PDFUnavailable("PDF export requires the pdf extra.") from None

    # ReportLab's bundled Vera font avoids machine-specific font paths.
    font = "NeuroVisionVera"
    if font not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(
            TTFont(font, str(Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"))
        )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "NVTitle",
            fontName=font,
            fontSize=20,
            leading=26,
            spaceAfter=16,
            textColor=colors.HexColor("#163e4c"),
        )
    )
    styles.add(
        ParagraphStyle(
            "NVHeading",
            fontName=font,
            fontSize=13,
            leading=18,
            spaceBefore=14,
            spaceAfter=7,
            keepWithNext=True,
            textColor=colors.HexColor("#176956"),
        )
    )
    styles.add(
        ParagraphStyle(
            "NVBody",
            fontName=font,
            fontSize=8.5,
            leading=12,
            spaceAfter=6,
            alignment=TA_LEFT,
            splitLongWords=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "NVKey",
            parent=styles["NVBody"],
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#526873"),
            spaceAfter=2,
            keepWithNext=False,
        )
    )
    flow = [
        Paragraph(escape(report["title"]), styles["NVTitle"]),
        Paragraph(escape(report["notice"]), styles["NVBody"]),
        Paragraph(
            "Unavailable entries identify information absent from this snapshot.",
            styles["NVBody"],
        ),
        Spacer(1, 10),
    ]
    for title, value in _blocks(report):
        flow.append(Paragraph(escape(title), styles["NVHeading"]))
        rows = [
            [
                Paragraph(escape(key), styles["NVKey"]),
                Paragraph(escape(item).replace("\n", "<br/>"), styles["NVBody"]),
            ]
            for key, item in _rows(value)
        ]
        table = LongTable(rows, colWidths=[186, 325], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 0),
                        (-1, -1),
                        [colors.HexColor("#f3f7f8"), colors.white],
                    ),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d6e2e7")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        flow.append(table)

    stream = BytesIO()
    document = SimpleDocTemplate(
        stream,
        pagesize=(8.27 * inch, 11.69 * inch),
        rightMargin=42,
        leftMargin=42,
        topMargin=44,
        bottomMargin=44,
        title=report["title"],
        author="NeuroVision",
    )

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#bed2d9"))
        canvas.line(42, 32, doc.pagesize[0] - 42, 32)
        canvas.setFont(font, 6.5)
        canvas.drawString(42, 21, "NeuroVision | Research and educational use only")
        canvas.drawRightString(doc.pagesize[0] - 42, 21, f"Page {doc.page}")
        canvas.restoreState()

    document.build(flow, onFirstPage=footer, onLaterPages=footer)
    return stream.getvalue()
