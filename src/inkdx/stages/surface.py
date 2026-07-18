"""Surface stage: is the mesh actually on the papyrus sheet?

Two metric families:

- Profile metrics (need the volume): where is the sheet relative to the mesh —
  peak offset, noise-normalized prominence, multiplicity (neighbor sheets in
  window = sheet-switch risk), and the sub-voxel centroid offset whose spatial
  field separates systematic drift (smooth) from noise (rough).
- Geometry metrics (mesh only): grid tearing (coordinate jumps = splice/switch
  signature), normal coherence (folds/flips), stretch anomaly (over/under-
  stretched parametrization), hole fraction.

Segment-level post-passes (`com_smoothness`, `stretch_anomaly`) operate on the
assembled tile maps because they compare tiles against their neighbors or the
whole segment.
"""

from __future__ import annotations

import numpy as np

from inkdx.grid import Tile, TileGrid
from inkdx.io.segment import Segment
from inkdx.sampling import TileProfiles
from inkdx.stages.profile_features import analyze_profile, noise_sigma

_EPS = 1e-6

SURFACE_PROFILE_METRICS = (
    "peak_offset", "peak_prominence", "peak_multiplicity", "com_offset",
)
SURFACE_GEOMETRY_METRICS = (
    "grid_tearing", "normal_coherence", "step_u", "step_v", "hole_fraction",
)


def compute_surface_profile_metrics(profiles: TileProfiles) -> dict[str, float]:
    """Per-tile surface metrics from normal profiles."""
    out = dict.fromkeys(SURFACE_PROFILE_METRICS, np.nan)
    if profiles.n_points == 0:
        return out

    med = profiles.median_profile()
    feats = analyze_profile(med, profiles.offsets)
    if not feats.ok:
        return out

    out["peak_offset"] = feats.r_star  # signed; |.| is the drift magnitude
    out["peak_multiplicity"] = float(feats.multiplicity)
    out["com_offset"] = feats.com_offset

    sigma = profiles.noise_sigma_raw
    if not np.isfinite(sigma):
        sigma = noise_sigma(profiles.profiles)
    if np.isfinite(sigma):
        # Median-profile noise shrinks ~ sqrt(N); normalize accordingly so the
        # threshold "is there a sheet at all" is comparable across tile support.
        n_eff = max(np.isfinite(profiles.profiles).all(axis=1).sum(), 1)
        sigma_med = max(sigma / np.sqrt(n_eff), _EPS)
        out["peak_prominence"] = feats.prominence / sigma_med
    return out


def compute_surface_geometry_metrics(segment: Segment, tile: Tile) -> dict[str, float]:
    """Per-tile mesh-geometry metrics. No volume access."""
    out = dict.fromkeys(SURFACE_GEOMETRY_METRICS, np.nan)
    rows, cols = tile.rows, tile.cols
    valid = np.asarray(segment.valid[rows, cols])
    out["hole_fraction"] = float(1.0 - valid.mean()) if valid.size else np.nan
    if valid.sum() < 9:
        return out

    # Window-local reads only — never materialize the full grid.
    xyz = np.stack(
        [
            np.asarray(segment.x[rows, cols], dtype=np.float32),
            np.asarray(segment.y[rows, cols], dtype=np.float32),
            np.asarray(segment.z[rows, cols], dtype=np.float32),
        ],
        axis=-1,
    )  # (h, w, 3)

    # Steps between adjacent valid vertices, in volume units.
    du = np.linalg.norm(np.diff(xyz, axis=1), axis=-1)  # (h, w-1) along cols
    dv = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)  # (h-1, w) along rows
    du_ok = valid[:, 1:] & valid[:, :-1]
    dv_ok = valid[1:, :] & valid[:-1, :]
    steps_u = du[du_ok]
    steps_v = dv[dv_ok]
    if steps_u.size:
        out["step_u"] = float(np.median(steps_u))
    if steps_v.size:
        out["step_v"] = float(np.median(steps_v))

    all_steps = np.concatenate([steps_u, steps_v])
    if all_steps.size >= 8:
        med = max(np.median(all_steps), _EPS)
        out["grid_tearing"] = float(all_steps.max() / med)

    # Normal coherence: mean dot product between column-adjacent normals.
    n = segment.normals_window(rows, cols)
    a, b = n[:, :-1, :], n[:, 1:, :]
    dots = (a * b).sum(axis=-1)
    dots = dots[np.isfinite(dots)]
    if dots.size >= 8:
        out["normal_coherence"] = float(dots.mean())
    return out


def com_smoothness(com_map: np.ndarray) -> np.ndarray:
    """Roughness of the com_offset field: |deviation from the 3x3 local mean|.

    Systematic drift gives a smooth field — including a constant gradient, which
    this measure (unlike a plain neighborhood std) does not penalize at interior
    tiles. A noisy peak estimate gives a rough field. NaN where the neighborhood
    has <4 tiles.
    """
    h, w = com_map.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if not np.isfinite(com_map[i, j]):
                continue
            block = com_map[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
            vals = block[np.isfinite(block)]
            if vals.size >= 4:
                out[i, j] = abs(com_map[i, j] - vals.mean())
    return out


def stretch_anomaly(step_u_map: np.ndarray, step_v_map: np.ndarray) -> np.ndarray:
    """|log2| deviation of each tile's step length from the segment median.

    0 = parametrization consistent with the segment; 1 = 2x over/under-stretch.
    """
    out = np.full(step_u_map.shape, np.nan, dtype=np.float32)
    finite_u = step_u_map[np.isfinite(step_u_map)]
    finite_v = step_v_map[np.isfinite(step_v_map)]
    if finite_u.size < 4 or finite_v.size < 4:
        return out
    gu = max(np.median(finite_u), _EPS)
    gv = max(np.median(finite_v), _EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        au = np.abs(np.log2(step_u_map / gu))
        av = np.abs(np.log2(step_v_map / gv))
    return np.fmax(au, av).astype(np.float32)


def hole_localization(segment: Segment, min_hole_area: int = 16) -> list[dict]:
    """Connected components of invalid regions enclosed by valid mesh.

    Lightweight stand-in for persistent-homology localization: reports interior
    holes (not the segment's outer boundary) with bbox and area, largest first.
    """
    from scipy.ndimage import binary_fill_holes, label

    valid = segment.valid
    interior_invalid = binary_fill_holes(valid) & ~valid
    labels, n = label(interior_invalid)
    holes = []
    for k in range(1, n + 1):
        rr, cc = np.nonzero(labels == k)
        if rr.size < min_hole_area:
            continue
        holes.append({
            "area": int(rr.size),
            "uv_bbox": [int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1],
            "centroid": [float(rr.mean()), float(cc.mean())],
        })
    holes.sort(key=lambda h: -h["area"])
    return holes


def geometry_maps(segment: Segment, grid: TileGrid) -> dict[str, np.ndarray]:
    """Assemble all geometry metrics into per-tile maps."""
    maps = {k: grid.new_map() for k in SURFACE_GEOMETRY_METRICS}
    for t in grid.tiles():
        m = compute_surface_geometry_metrics(segment, t)
        for k, v in m.items():
            maps[k][t.i, t.j] = v
    return maps
