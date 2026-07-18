"""tifxyz segment surfaces: a 2D quad-grid of 3D points.

Implements a light, dependency-free reader/writer for the tifxyz directory
format (x.tif / y.tif / z.tif / meta.json) as specified by Volume Cartographer
(see villa/lasagna/tifxyz_format.md). Conventions honored:

- invalid points carry the sentinel (-1, -1, -1)
- points with Z <= 0 are invalidated at load time
- meta.json "scale" is stored C++-style as [x_scale, y_scale]; we keep
  (scale_y, scale_x) to match array indexing
- optional mask.tif: channel 0 nonzero = valid

Coordinates are kept at stored resolution. Diagnostics operate on the stored
grid; `scale` maps grid steps to full-resolution surface units.
"""

from __future__ import annotations

import json
import uuid as uuid_mod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

INVALID = -1.0


@dataclass
class Segment:
    """A papyrus surface as a stored-resolution quad-grid of 3D points."""

    x: np.ndarray  # (H, W) float32, volume x per grid vertex
    y: np.ndarray  # (H, W) float32
    z: np.ndarray  # (H, W) float32
    valid: np.ndarray  # (H, W) bool
    scale: tuple[float, float] = (1.0, 1.0)  # (scale_y, scale_x)
    uuid: str = ""
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    def __post_init__(self) -> None:
        shapes = {self.x.shape, self.y.shape, self.z.shape, self.valid.shape}
        if len(shapes) != 1:
            raise ValueError(f"coordinate/mask shapes differ: {shapes}")

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.x.shape

    def xyz(self) -> np.ndarray:
        """Stacked (H, W, 3) coordinates."""
        return np.stack([self.x, self.y, self.z], axis=-1)

    def normals(self) -> np.ndarray:
        """Per-vertex unit normals, (H, W, 3) float32, NaN where undefined.

        Matches the reference vesuvius-library convention: central differences,
        normal = t_row x t_col; defined only at interior vertices whose 4-neighbors
        are all valid. Orientation is geometric (from grid winding) — callers that
        need a physically consistent side must sign-check against the volume.
        """
        h, w = self.grid_shape
        out = np.full((h, w, 3), np.nan, dtype=np.float32)
        if h < 3 or w < 3:
            return out

        v = self.valid
        interior = (
            v[1:-1, 1:-1] & v[1:-1, :-2] & v[1:-1, 2:] & v[:-2, 1:-1] & v[2:, 1:-1]
        )

        def central(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            d_col = a[1:-1, 2:] - a[1:-1, :-2]
            d_row = a[2:, 1:-1] - a[:-2, 1:-1]
            return d_row, d_col

        rx, cx = central(self.x)
        ry, cy = central(self.y)
        rz, cz = central(self.z)

        # n = t_row x t_col
        nx = ry * cz - rz * cy
        ny = rz * cx - rx * cz
        nz = rx * cy - ry * cx
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        with np.errstate(invalid="ignore"):
            norm = np.where(norm > 1e-10, norm, np.nan)
            nx, ny, nz = nx / norm, ny / norm, nz / norm

        for k, n in enumerate((nx, ny, nz)):
            out[1:-1, 1:-1, k] = np.where(interior, n, np.nan).astype(np.float32)
        return out


def _read_coord(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected single-channel 2D image, got {arr.shape}")
    return arr.astype(np.float32)


def read_tifxyz(path: str | Path, *, load_mask: bool = True) -> Segment:
    """Read a tifxyz directory into a Segment (stored resolution)."""
    path = Path(path)
    for req in ("x.tif", "y.tif", "z.tif", "meta.json"):
        if not (path / req).exists():
            raise FileNotFoundError(f"{path} is not a tifxyz directory: missing {req}")

    x = _read_coord(path / "x.tif")
    y = _read_coord(path / "y.tif")
    z = _read_coord(path / "z.tif")
    if not (x.shape == y.shape == z.shape):
        raise ValueError(f"coordinate shapes differ: {x.shape}, {y.shape}, {z.shape}")

    meta = json.loads((path / "meta.json").read_text())
    scale_raw = meta.get("scale", [20.0, 20.0])
    # C++ writes [x_scale, y_scale]; we store (scale_y, scale_x)
    scale = (float(scale_raw[1]), float(scale_raw[0]))

    valid = z > 0  # load-time invalidation rule
    mask_path = path / "mask.tif"
    if load_mask and mask_path.exists():
        mask = tifffile.imread(mask_path)
        if mask.ndim == 3:  # channel 0 is validity
            mask = mask[..., 0] if mask.shape[-1] <= 4 else mask[0]
        if mask.shape != x.shape:
            raise ValueError(f"mask.tif shape {mask.shape} != grid shape {x.shape}")
        valid &= mask != 0

    for a in (x, y, z):
        a[~valid] = INVALID

    return Segment(
        x=x, y=y, z=z, valid=valid, scale=scale,
        uuid=str(meta.get("uuid", path.name)), meta=meta, path=path,
    )


def write_tifxyz(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    scale: tuple[float, float] = (1.0, 1.0),  # (scale_y, scale_x)
    uuid: str | None = None,
    extra_meta: dict | None = None,
) -> Path:
    """Write a tifxyz directory. Invalid vertices get the (-1,-1,-1) sentinel."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    x = x.astype(np.float32).copy()
    y = y.astype(np.float32).copy()
    z = z.astype(np.float32).copy()
    if valid is None:
        valid = z > 0
    else:
        valid = valid.astype(bool)
    for a in (x, y, z):
        a[~valid] = INVALID

    tifffile.imwrite(path / "x.tif", x)
    tifffile.imwrite(path / "y.tif", y)
    tifffile.imwrite(path / "z.tif", z)
    tifffile.imwrite(path / "mask.tif", valid.astype(np.uint8) * 255)

    if valid.any():
        bbox = [
            float(x[valid].min()), float(y[valid].min()), float(z[valid].min()),
            float(x[valid].max()), float(y[valid].max()), float(z[valid].max()),
        ]
    else:
        bbox = [0.0] * 6
    meta = {
        "uuid": uuid or uuid_mod.uuid4().hex[:12],
        "scale": [scale[1], scale[0]],  # C++ order: [x_scale, y_scale]
        "bbox": bbox,
        **(extra_meta or {}),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    return path
