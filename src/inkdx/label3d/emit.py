"""Emit true-3D ink labels in the community's exact conventions.

(Z, Y, X) uint8 zarr, class codes {0: background, 1: ink, 2: ignore}, with the
`3d_ink_params.json` and `remap.json` sidecars established by villa's
labels_to_zarr.py. Surface-volume mode: z is the layer axis and r = z − z_center.

Class policy per column:
- non-ink valid vertex: 0 (background) for |r| <= bg_distance, else 2
- ink vertex: 1 for r in the estimated band; the REST of the ink column stays
  2 (ignore) by default — we know ink was not *detected* there, but calling it
  background under a stroke would be an overclaim
- invalid vertex: 2 everywhere
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from inkdx import __version__

CLASS_BACKGROUND = 0
CLASS_INK = 1
CLASS_IGNORE = 2


def emit_labels(
    out_path: str | Path,
    *,
    nz: int,
    z_center: float,
    ink_mask: np.ndarray,  # (H, W) bool
    valid: np.ndarray,  # (H, W) bool
    band: tuple[float, float],
    bg_distance: float = 8.0,
    ink_column_rest: str = "ignore",  # or "bg"
    band_source: str = "signal",  # "signal" | "fallback"
    provenance: dict | None = None,
    chunks: tuple[int, int, int] | None = None,
) -> Path:
    import zarr
    from numcodecs import Blosc

    out_path = Path(out_path)
    h, w = ink_mask.shape
    a, b = band

    store = zarr.open(str(out_path), mode="w")
    arr = store.create_dataset(
        "0",
        shape=(nz, h, w),
        chunks=chunks or (min(nz, 65), 256, 256),
        dtype="u1",
        compressor=Blosc(cname="zstd", clevel=3),
        fill_value=CLASS_IGNORE,
    )

    ink = ink_mask.astype(bool) & valid
    bg_vertex = valid & ~ink

    for z in range(nz):
        r = float(z) - z_center
        layer = np.full((h, w), CLASS_IGNORE, dtype=np.uint8)
        if abs(r) <= bg_distance:
            layer[bg_vertex] = CLASS_BACKGROUND
            if ink_column_rest == "bg":
                layer[ink] = CLASS_BACKGROUND
        if a <= r <= b:
            layer[ink] = CLASS_INK
        arr[z] = layer

    params = {
        "labels": {"background": str(CLASS_BACKGROUND), "ink": str(CLASS_INK),
                   "ignore": str(CLASS_IGNORE)},
        "inkdx": {
            "version": __version__,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "band": [a, b],
            "band_source": band_source,
            "bg_distance": bg_distance,
            "ink_column_rest": ink_column_rest,
            **(provenance or {}),
        },
    }
    (out_path / "3d_ink_params.json").write_text(json.dumps(params, indent=2))
    remap = {
        "mappings": {str(CLASS_IGNORE): "ignore",
                     str(CLASS_BACKGROUND): "background",
                     str(CLASS_INK): "ink"},
        "padded_labels": False,
        "expand_labels": False,
    }
    (out_path / "remap.json").write_text(json.dumps(remap, indent=2))
    return out_path
