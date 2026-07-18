"""Surface-failure ablations: perturb a segment mesh in controlled ways.

Used to validate attribution: an offset/tilted/wobbled mesh must flip tiles to
SURFACE_SUSPECT while the scan stage stays green.
"""

from __future__ import annotations

import numpy as np

from inkdx.io.segment import Segment


def _clone(seg: Segment, z: np.ndarray, uuid: str) -> Segment:
    return Segment(
        x=seg.x.copy(), y=seg.y.copy(), z=z.astype(np.float32),
        valid=seg.valid.copy(), scale=seg.scale, uuid=uuid, meta=dict(seg.meta),
    )


def _region_mask(seg: Segment, region: tuple[int, int, int, int] | None) -> np.ndarray:
    mask = np.zeros(seg.grid_shape, dtype=bool)
    r0, c0, r1, c1 = region or (0, 0, *seg.grid_shape)
    mask[r0:r1, c0:c1] = True
    return mask


def offset_z(
    seg: Segment, dz: float, *, region: tuple[int, int, int, int] | None = None
) -> Segment:
    """Rigid z-offset (whole mesh or a UV region): systematic drift off-sheet."""
    z = seg.z.copy()
    m = _region_mask(seg, region)
    z[m] += dz
    return _clone(seg, z, f"{seg.uuid}-dz{dz:g}")


def tilt(seg: Segment, dz_per_col: float) -> Segment:
    """Linear ramp along the column axis: the mesh crosses the sheet obliquely."""
    cols = np.arange(seg.grid_shape[1], dtype=np.float32)
    ramp = (cols - cols.mean()) * dz_per_col
    return _clone(seg, seg.z + ramp[None, :], f"{seg.uuid}-tilt{dz_per_col:g}")


def wobble(
    seg: Segment, amplitude: float, period: float, *, seed: int = 0
) -> Segment:
    """Sinusoidal z-wobble with a random phase: oscillating tracking error."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    cols = np.arange(seg.grid_shape[1], dtype=np.float32)
    wave = amplitude * np.sin(2.0 * np.pi * cols / period + phase)
    return _clone(seg, seg.z + wave[None, :], f"{seg.uuid}-wob{amplitude:g}x{period:g}")
