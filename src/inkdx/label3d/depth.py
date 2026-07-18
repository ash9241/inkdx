"""Signal-based ink depth localization: the Δ(r) estimator.

Where in depth does the ink signature live? Compare the normal-intensity
profiles of 2D-ink-labeled pixels against a locally matched background
annulus: Δ(r) = median_ink I(r) − median_bg I(r). The differencing isolates
*ink* from papyrus condition, scan brightness, and surface quality.

Honesty machinery: block-level bootstrap (spatially correlated pixels must not
inflate N) and a block permutation test. The tool never silently invents a
depth band — NO_DEPTH_SIGNAL with a quantified upper bound is a first-class
result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.signal import savgol_filter

from inkdx.io.segment import Segment
from inkdx.sampling import profiles_for_indices

STATUS_LOCALIZED = "INK_DEPTH_LOCALIZED"
STATUS_NO_SIGNAL = "NO_DEPTH_SIGNAL"
STATUS_INSUFFICIENT = "INSUFFICIENT_LABELS"


@dataclass
class Label3dConfig:
    halfwidth: int = 16
    bg_inner: int = 3  # annulus: dilate(inner)..dilate(outer) around strokes
    bg_outer: int = 15
    min_ink_px: int = 200
    max_pixels: int = 150_000  # per class, seeded subsample for tractability
    block: int = 32  # UV block size for bootstrap/permutation units
    bootstrap: int = 200
    sig_z: float = 4.0
    p_thresh: float = 0.01
    band_frac: float = 0.5  # FWHM of |delta|
    fallback_distance: float = 8.0
    bg_distance: float = 8.0
    tile_px: int = 256
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class DepthResult:
    status: str
    delta: np.ndarray | None  # (P,) ink-minus-background profile
    se: np.ndarray | None  # (P,) bootstrap standard error
    offsets: np.ndarray | None
    band: tuple[float, float] | None  # signal-driven depth band, or None
    r_ink: float | None  # extremum location
    delta_peak: float | None  # signed Δ at the extremum
    p_value: float | None
    n_ink: int = 0
    n_bg: int = 0
    n_blocks: int = 0
    upper_bound: float | None = None  # |Δ| bound when NOT significant


def build_pixel_sets(
    ink_mask: np.ndarray, valid: np.ndarray, cfg: Label3dConfig
) -> tuple[np.ndarray, np.ndarray]:
    """(ink_selector, bg_selector) boolean H×W maps."""
    ink = ink_mask.astype(bool) & valid
    ink = binary_erosion(ink, iterations=1)  # drop boundary-uncertain pixels
    near = binary_dilation(ink_mask.astype(bool), iterations=cfg.bg_inner)
    far = binary_dilation(ink_mask.astype(bool), iterations=cfg.bg_outer)
    bg = far & ~near & valid & ~ink_mask.astype(bool)
    return ink, bg


def _block_profiles(
    profiles: np.ndarray, gr: np.ndarray, gc: np.ndarray, block: int
) -> np.ndarray:
    """Median profile per UV block: the exchangeable units for resampling."""
    ids = (gr // block).astype(np.int64) * 1_000_003 + (gc // block)
    uniq, inv = np.unique(ids, return_inverse=True)
    out = np.full((uniq.size, profiles.shape[1]), np.nan, dtype=np.float32)
    for k in range(uniq.size):
        sel = inv == k
        if sel.sum() >= 4:
            with np.errstate(all="ignore"):
                out[k] = np.nanmedian(profiles[sel], axis=0)
    keep = np.isfinite(out).all(axis=1)
    return out[keep]


def estimate_depth(
    volume,
    segment: Segment,
    ink_mask: np.ndarray,
    cfg: Label3dConfig | None = None,
) -> DepthResult:
    cfg = cfg or Label3dConfig()
    rng = np.random.default_rng(cfg.seed)

    valid = np.asarray(segment.valid)
    ink_sel, bg_sel = build_pixel_sets(ink_mask, valid, cfg)
    n_ink_all = int(ink_sel.sum())
    if n_ink_all < cfg.min_ink_px:
        return DepthResult(STATUS_INSUFFICIENT, None, None, None, None, None,
                           None, None, n_ink=n_ink_all)

    def pick(sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rr, cc = np.nonzero(sel)
        if rr.size > cfg.max_pixels:
            idx = rng.choice(rr.size, size=cfg.max_pixels, replace=False)
            rr, cc = rr[idx], cc[idx]
        return rr, cc

    ink_r, ink_c = pick(ink_sel)
    bg_r, bg_c = pick(bg_sel)

    ink_prof, offsets = profiles_for_indices(
        volume, segment, ink_r, ink_c, halfwidth=cfg.halfwidth, tile_px=cfg.tile_px
    )
    bg_prof, _ = profiles_for_indices(
        volume, segment, bg_r, bg_c, halfwidth=cfg.halfwidth, tile_px=cfg.tile_px
    )

    ink_blocks = _block_profiles(ink_prof, ink_r, ink_c, cfg.block)
    bg_blocks = _block_profiles(bg_prof, bg_r, bg_c, cfg.block)
    if ink_blocks.shape[0] < 4 or bg_blocks.shape[0] < 4:
        return DepthResult(STATUS_INSUFFICIENT, None, None, offsets, None, None,
                           None, None, n_ink=ink_r.size, n_bg=bg_r.size,
                           n_blocks=int(ink_blocks.shape[0]))

    delta = (np.median(ink_blocks, axis=0) - np.median(bg_blocks, axis=0)).astype(np.float32)

    # --- block bootstrap SE -------------------------------------------------
    boots = np.empty((cfg.bootstrap, delta.size), dtype=np.float32)
    for b in range(cfg.bootstrap):
        bi = rng.integers(0, ink_blocks.shape[0], ink_blocks.shape[0])
        bb = rng.integers(0, bg_blocks.shape[0], bg_blocks.shape[0])
        boots[b] = np.median(ink_blocks[bi], axis=0) - np.median(bg_blocks[bb], axis=0)
    se = boots.std(axis=0).astype(np.float32)

    # --- null test: background-vs-background splits -------------------------
    # A label permutation over the pooled blocks fails here BY CONSTRUCTION:
    # with a strong signal the pool is a mixture, and the median's robustness
    # means any imbalanced permutation carries the full ink signature — the
    # test loses all power exactly when the effect is large. The valid null
    # is "no-signal variability at the same block count": split the BACKGROUND
    # blocks into two random halves and ask how large max|Δ| gets between two
    # groups that both contain no ink.
    observed = float(np.max(np.abs(delta)))
    n_half = bg_blocks.shape[0] // 2
    exceed = 0
    for _ in range(cfg.bootstrap):
        perm = rng.permutation(bg_blocks.shape[0])
        d = (np.median(bg_blocks[perm[:n_half]], axis=0)
             - np.median(bg_blocks[perm[n_half:]], axis=0))
        if float(np.max(np.abs(d))) >= observed:
            exceed += 1
    p_value = (exceed + 1) / (cfg.bootstrap + 1)

    i_peak = int(np.argmax(np.abs(delta)))
    r_ink = float(offsets[i_peak])
    delta_peak = float(delta[i_peak])
    significant = (
        np.abs(delta_peak) >= cfg.sig_z * max(float(se[i_peak]), 1e-9)
        and p_value < cfg.p_thresh
    )

    if not significant:
        return DepthResult(
            STATUS_NO_SIGNAL, delta, se, offsets, None, r_ink, delta_peak,
            p_value, n_ink=ink_r.size, n_bg=bg_r.size,
            n_blocks=int(ink_blocks.shape[0]),
            upper_bound=float(cfg.sig_z * se.max()),
        )

    # --- band: FWHM of |smoothed Δ| around the extremum ---------------------
    window = min(7, delta.size if delta.size % 2 else delta.size - 1)
    sm = savgol_filter(delta.astype(np.float64), window, 2) if window >= 5 else delta
    sgn = np.sign(sm[i_peak])
    prof = sgn * sm  # positive at the ink extremum regardless of contrast sign
    half = cfg.band_frac * prof[i_peak]
    lo = i_peak
    while lo > 0 and prof[lo - 1] >= half:
        lo -= 1
    hi = i_peak
    while hi < prof.size - 1 and prof[hi + 1] >= half:
        hi += 1
    band = (float(offsets[lo]), float(offsets[hi]))

    return DepthResult(
        STATUS_LOCALIZED, delta, se, offsets, band, r_ink, delta_peak,
        p_value, n_ink=ink_r.size, n_bg=bg_r.size,
        n_blocks=int(ink_blocks.shape[0]),
    )
