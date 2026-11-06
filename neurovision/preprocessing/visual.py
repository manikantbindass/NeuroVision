"""Portable three-plane QC montages for local review."""

import base64
import struct
import zlib
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from numpy.typing import NDArray

from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.imaging.volume import MRIVolume
from neurovision.preprocessing.contracts import PreprocessingError, same_grid

_PLANES = (
    (2, 0, 1, "Axial", "R", "A"),
    (1, 0, 2, "Coronal", "R", "S"),
    (0, 1, 2, "Sagittal", "A", "S"),
)


def write_qc_montage(
    volume: MRIVolume, path: Path, *, comparison: MRIVolume | None = None
) -> None:
    """Save one or two rows of center slices as a self-contained SVG."""
    if (
        volume.data.ndim != 3
        or not volume.data.size
        or volume.orientation != ("R", "A", "S")
        or len(volume.spacing_mm) != 3
        or any(not np.isfinite(v) or v <= 0 for v in volume.spacing_mm)
        or not np.isfinite(volume.data).all()
    ):
        raise PreprocessingError(
            "QC montage requires finite voxels on a valid canonical grid"
        )
    if comparison is not None and (
        not same_grid(volume, comparison)
        or comparison.orientation != volume.orientation
        or not np.isfinite(comparison.data).all()
    ):
        raise PreprocessingError(
            "QC montage comparison must share the same finite grid"
        )
    scans = (volume,) if comparison is None else (volume, comparison)
    height = 50 + 270 * len(scans)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="840" height="{height}" '
        f'viewBox="0 0 840 {height}" role="img" aria-labelledby="title">',
        '<title id="title">MRI research quality control montage</title>',
        '<rect width="100%" height="100%" fill="#101827"/>',
        '<g font-family="sans-serif" fill="#e5eef7">',
        f'<text x="16" y="23" font-size="12">{escape(RESEARCH_NOTICE)}</text>',
    ]
    sampled = volume.data.ravel()[:: max(1, (volume.data.size + 99_999) // 100_000)]
    low, high = (float(v) for v in np.percentile(sampled, [1, 99]))
    if high <= low:
        low, high = float(volume.data.min()), float(volume.data.max())
    for row, scan in enumerate(scans):
        y = 54 + row * 270
        for col, (axis, horizontal, vertical, name, right, up) in enumerate(_PLANES):
            x = 15 + col * 280
            row_label = (
                ("Primary", "Comparison")[row] if comparison is not None else "MRI"
            )
            svg.append(
                f'<text x="{x}" y="{y}" font-size="14">{row_label} · {name}</text>'
            )
            svg.append(
                f'<text x="{x}" y="{y + 20}" font-size="11">'
                f"Right: {right} · Up: {up} (closest)</text>"
            )
            plane = np.take(scan.data, scan.data.shape[axis] // 2, axis=axis).T[::-1]
            height_px, width_px = plane.shape
            rows = np.linspace(0, height_px - 1, min(height_px, 256), dtype=np.intp)
            cols = np.linspace(0, width_px - 1, min(width_px, 256), dtype=np.intp)
            image = plane[np.ix_(rows, cols)].astype(np.float64)
            scale = high - low if high > low else 1
            gray = np.rint(np.clip((image - low) / scale, 0, 1) * 255).astype(np.uint8)
            ratios = np.asarray(volume.spacing_mm) / max(volume.spacing_mm)
            extent_x, extent_y = (
                width_px * ratios[horizontal],
                height_px * ratios[vertical],
            )
            extent = max(extent_x, extent_y)
            width = max(1, round(220 * extent_x / extent)) if extent else 1
            img_height = max(1, round(220 * extent_y / extent)) if extent else 1
            svg.append(
                f'<image x="{x}" y="{y + 30}" width="{width}" height="{img_height}" '
                'preserveAspectRatio="none" '
                f'href="data:image/png;base64,{_png(gray)}"/>'
            )
    svg.extend(("</g>", "</svg>"))
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _png(data: NDArray[np.uint8]) -> str:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    height, width = data.shape
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00" + row.tobytes() for row in data)
    image = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(image).decode("ascii")
