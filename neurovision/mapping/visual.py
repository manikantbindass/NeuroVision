"""Write a bounded, self-contained atlas and attribution comparison report."""

from html import escape
from itertools import groupby
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurovision.imaging.volume import MRIVolume
from neurovision.mapping.atlas import MappingError
from neurovision.mapping.engine import NeuroMapResult


def _color(code: int) -> str:
    return f"hsl({(code * 137) % 360}, 65%, 60%)" if code else "#141c2b"


def _panel(
    title: str, pixels: NDArray[np.integer], *, categorical: bool = False
) -> str:
    height, width = pixels.shape
    parts = [
        f'<figure><svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{title}" shape-rendering="crispEdges">'
    ]
    for y, row in enumerate(pixels):
        x = 0
        for value, run in groupby(row):
            length = sum(1 for _ in run)
            code = int(value)
            color = _color(code) if categorical else f"rgb({code},{code},{code})"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{length}" height="1" fill="{color}"/>'
            )
            x += length
    parts.append(f"</svg><figcaption>{title}</figcaption></figure>")
    return "".join(parts)


def write_comparison_html(
    path: Path,
    reference: MRIVolume,
    labels: NDArray[np.integer],
    result: NeuroMapResult,
    *,
    slice_index: int,
    saliency: NDArray[np.floating] | None = None,
    occlusion: NDArray[np.floating] | None = None,
    attribution_note: str = "",
) -> None:
    """Show a voxel-grid slice with independently scaled sensitivity panels."""
    shape = reference.data.shape
    if (
        reference.reference_space != result.segmentation_space
        or labels.shape != shape
        or result.aligned_atlas.shape != shape
        or labels.dtype.kind not in "iu"
        or type(slice_index) is not int
        or not 0 <= slice_index < shape[2]
    ):
        raise MappingError("Comparison panels require a matching 3D grid and slice")
    stride = max(1, (max(shape[:2]) + 95) // 96)
    index = (slice(None, None, stride), slice(None, None, stride), slice_index)

    def display_slice(
        values: NDArray[np.floating], *, signed: bool
    ) -> NDArray[np.int32]:
        if values.shape != shape:
            raise MappingError("Comparison panel data must match the MRI grid")
        sampled = values[index].T[::-1].astype(np.float64)
        if not np.isfinite(sampled).all():
            raise MappingError("Comparison panel data must be finite")
        if signed:
            bound = float(np.abs(sampled).max())
            normalized = 0.5 + 0.5 * (sampled / (bound or 1))
        else:
            low, high = float(sampled.min()), float(sampled.max())
            normalized = (sampled - low) / (high - low or 1)
        return np.rint(normalized * 255).astype(np.int32)

    panels = [
        _panel("MRI", display_slice(reference.data, signed=False)),
        _panel("Segmentation", labels[index].T[::-1], categorical=True),
        _panel("Aligned atlas", result.aligned_atlas[index].T[::-1], categorical=True),
    ]
    if saliency is not None:
        panels.append(
            _panel("Gradient magnitude", display_slice(saliency, signed=False))
        )
    if occlusion is not None:
        panels.append(
            _panel("Signed occlusion drop", display_slice(occlusion, signed=True))
        )

    def swatch(code: int) -> str:
        return f'<span class="swatch" style="background:{_color(code)}"></span>'

    rows = []
    for region in result.measurements:
        associations = ", ".join(
            f"{swatch(entry.region.code)}{escape(entry.region.name)} "
            f"({entry.region.hemisphere}, {entry.voxels:,} voxels, "
            f"{entry.volume_mm3:,.3f} mm³, {entry.fraction_of_segment:.1%})"
            for entry in region.associations
        )
        confidence = (
            "—" if region.mean_confidence is None else f"{region.mean_confidence:.3f}"
        )
        rows.append(
            f"<tr><td>{swatch(region.code)}{escape(region.name)}</td>"
            f"<td>{region.voxels:,}</td><td>{region.volume_mm3:,.3f}</td>"
            f"<td>{region.fraction_of_foreground:.1%}</td><td>{confidence}</td>"
            f"<td>{region.hemisphere}</td><td>{region.atlas_coverage:.1%}</td>"
            f"<td>{associations or '—'}</td></tr>"
        )
    path.write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuroMap research comparison</title>
<style>
body{font:16px system-ui;background:#101727;color:#edf4ff;
max-width:1250px;margin:auto;padding:24px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
figure{margin:0;text-align:center}svg{width:100%;height:200px}
.notice{padding:12px;border:1px solid #99abc6}
.table{overflow:auto}table{border-collapse:collapse;width:100%;text-align:left}
th,td{padding:8px;border-bottom:1px solid #718099}
.swatch{display:inline-block;width:.7em;height:.7em;margin-right:.4em}
</style>
<body><h1>NeuroMap research comparison</h1>"""
        f'<p class="notice">{escape(result.notice)}</p>'
        f"<p>Atlas {escape(result.atlas_name)} {escape(result.atlas_version)} "
        f"· {escape(result.atlas_license)} · {escape(result.atlas_space)}</p>"
        f"<p>Registration provenance: {escape(result.registration_provenance)}.</p>"
        f"<p>Grid slice {slice_index} on axis 2, sampled every {stride} voxel(s) "
        "in-plane. Panels show voxel indices without physical aspect correction. "
        "Signed occlusion uses mid-gray for zero, brighter for positive logit drops, "
        "and darker for negative drops. Gradient magnitude uses its own slice "
        "scale.</p>"
        f"<p>{escape(attribution_note)}</p>"
        f'<div class="panels">{"".join(panels)}</div>'
        '<h2>Full-volume measurements</h2><div class="table"><table>'
        "<thead><tr><th>Segmentation region</th><th>Voxels</th><th>Volume (mm³)</th>"
        "<th>Share of foreground</th><th>Mean confidence</th><th>Hemisphere</th>"
        "<th>Atlas coverage</th><th>Atlas overlaps (% of region)</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        "<p>Atlas overlaps depend on registration and atlas coverage. Confidence "
        "may be uncalibrated. Sensitivity maps describe model responses to input "
        "changes and cannot establish causality or clinical relevance.</p>"
        "</body></html>",
        encoding="utf-8",
    )
