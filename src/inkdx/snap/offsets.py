"""Per-tile dense offset estimation for `inkdx snap`.

For every valid vertex: pooled profile → candidate peaks → nearest-peak
selection (anti-wrap-jump) → sub-voxel offset + confidence weight + status.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from inkdx.grid import Tile
from inkdx.io.segment import Segment
from inkdx.sampling import dense_tile_profiles
from inkdx.stages.profile_features import analyze_profiles_dense

# Status codes (uint8) — CAPS verdict-style, stable across releases.
STATUS_INVALID = 0
STATUS_SNAPPED = 1
STATUS_HELD_LOW_CONF = 2
STATUS_HELD_MULTIWRAP = 3
STATUS_HELD_OUTLIER = 4
STATUS_CLAMPED = 5
STATUS_NAMES = {
    0: "INVALID", 1: "SNAPPED", 2: "HELD_LOW_CONF",
    3: "HELD_MULTIWRAP", 4: "HELD_OUTLIER", 5: "CLAMPED",
}


@dataclass
class TileOffsets:
    r_hat: np.ndarray  # (th, tw) float32 — proposed offset along +n, NaN = held
    weight: np.ndarray  # (th, tw) float32 in [0, 1]
    status: np.ndarray  # (th, tw) uint8
    normals: np.ndarray  # (th, tw, 3) float32 — globally-signed unit normals


def _pool_block(block: np.ndarray, pool: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware UV-average of the profile block; returns (pooled, n_eff)."""
    if pool <= 1:
        n_eff = np.isfinite(block[..., 0]).astype(np.float32)
        return block, n_eff
    finite = np.isfinite(block)
    filled = np.where(finite, block, 0.0)
    ksize = (pool, pool, 1)
    s = uniform_filter(filled, size=ksize, mode="constant")
    c = uniform_filter(finite.astype(np.float32), size=ksize, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.where(c > 1e-6, s / c, np.nan)
    n_eff = c[..., 0] * pool * pool  # effective contributing vertices at r=0
    return pooled.astype(np.float32), n_eff


def compute_tile_offsets(
    volume,
    segment: Segment,
    tile: Tile,
    *,
    halfwidth: int,
    sign: int = 1,
    stride: int = 1,
    pool: int = 3,
    snr_lo: float = 3.0,
    snr_hi: float = 6.0,
    nearest_frac: float = 0.6,
) -> TileOffsets:
    d = dense_tile_profiles(volume, segment, tile, halfwidth=halfwidth, stride=stride)
    th, tw, p = d.block.shape

    block = d.block
    normals = d.normals * float(sign)
    if sign < 0:  # mirroring the profile == sampling along the flipped normal
        block = block[..., ::-1]

    r_hat = np.full((th, tw), np.nan, dtype=np.float32)
    weight = np.zeros((th, tw), dtype=np.float32)
    status = np.full((th, tw), STATUS_INVALID, dtype=np.uint8)
    status[d.valid] = STATUS_HELD_LOW_CONF  # until proven otherwise

    pooled, n_eff = _pool_block(block, pool)
    feats = analyze_profiles_dense(pooled, d.offsets, return_smoothed=True)
    ok = np.isfinite(feats["r_star"])
    if not ok.any():
        return TileOffsets(r_hat, weight, status, normals)

    smoothed = feats["smoothed"]
    candidates = feats["candidates"]  # (th, tw, P) bool, interior local maxima
    gap = feats["gap_value"]
    prominence = feats["prominence"]
    multiplicity = feats["multiplicity"]

    # --- nearest-peak selection policy ------------------------------------
    cand_prom = np.where(candidates, smoothed - gap[..., None], -np.inf)
    strongest_prom = cand_prom.max(axis=-1)  # -inf where no candidates
    has_cand = np.isfinite(strongest_prom)

    absr = np.abs(d.offsets)[None, None, :]
    near_rank = np.where(candidates, absr, np.inf)
    i_near = near_rank.argmin(axis=-1)
    ii, jj = np.meshgrid(np.arange(th), np.arange(tw), indexing="ij")
    near_prom = cand_prom[ii, jj, i_near]

    i_strong = cand_prom.argmax(axis=-1)

    near_qualifies = has_cand & (near_prom >= nearest_frac * strongest_prom)
    single = has_cand & (multiplicity <= 1)
    chosen = np.where(near_qualifies, i_near, np.where(single, i_strong, -1))
    held_multi = ok & has_cand & (chosen < 0)
    resolved = ok & has_cand & (chosen >= 0)

    # --- sub-voxel refinement at the chosen peak ---------------------------
    ic = np.clip(chosen, 1, p - 2)
    s_m = smoothed[ii, jj, ic - 1]
    s_0 = smoothed[ii, jj, ic]
    s_p = smoothed[ii, jj, ic + 1]
    denom = s_m - 2.0 * s_0 + s_p
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 0.5 * (s_m - s_p) / denom
    delta = np.clip(np.where(np.abs(denom) > 1e-12, delta, 0.0), -0.5, 0.5)
    step = float(d.offsets[1] - d.offsets[0]) if p > 1 else 1.0
    r_sel = d.offsets[ic] + delta * step

    # com cross-check: bimodal/asymmetric profiles halve the confidence
    com = feats["com_offset"]
    com_penalty = np.where(
        np.isfinite(com) & (np.abs(r_sel - com) > 1.5), 0.5, 1.0
    )

    # --- confidence weight -------------------------------------------------
    sigma = d.noise_sigma_raw if np.isfinite(d.noise_sigma_raw) else np.nan
    if np.isfinite(sigma) and sigma > 1e-6:
        snr = prominence * np.sqrt(np.maximum(n_eff, 1.0)) / sigma
        g_prom = np.clip((snr - snr_lo) / max(snr_hi - snr_lo, 1e-6), 0.0, 1.0)
    else:
        g_prom = np.where(prominence > 0, 1.0, 0.0)  # noiseless: any peak counts
    g_mult = np.where(multiplicity <= 1, 1.0, 0.5)
    w = g_prom * g_mult * com_penalty

    good = resolved & (w > 0)
    r_hat[good] = r_sel[good].astype(np.float32)
    weight[good] = w[good].astype(np.float32)
    status[good] = STATUS_SNAPPED
    status[held_multi] = STATUS_HELD_MULTIWRAP
    return TileOffsets(r_hat, weight, status, normals)
