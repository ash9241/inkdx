"""Regularization of the per-vertex offset field.

Operates on full H×W float32 planes. Order: robust outlier rejection →
normalized convolution (confidence-weighted smoothing that never extrapolates
into blind regions) → per-iteration step clamp → tangential gradient limiting
(topology preservation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from inkdx.snap.offsets import STATUS_HELD_OUTLIER, STATUS_SNAPPED


@dataclass
class RegularizeConfig:
    smooth: float = 3.0  # normalized-convolution sigma (grid steps)
    outlier_abs: float = 2.0  # voxels
    outlier_nmad: float = 3.0
    max_step: float = 2.0  # per-iteration clamp (voxels)
    grad_max: float = 0.5  # voxels per grid step
    min_weight_mass: float = 0.05


def _nc(field: np.ndarray, weight: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalized convolution: G∗(w·f)/G∗w. Returns (smoothed, weight mass)."""
    wf = np.where(np.isfinite(field), field * weight, 0.0)
    w = np.where(np.isfinite(field), weight, 0.0)
    num = gaussian_filter(wf, sigma=sigma, mode="constant")
    den = gaussian_filter(w, sigma=sigma, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = np.where(den > 1e-12, num / den, np.nan)
    return sm.astype(np.float32), den.astype(np.float32)


def limit_gradient(field: np.ndarray, g_max: float, *, passes: int = 2) -> np.ndarray:
    """Enforce |gradient| <= g_max via the Lipschitz envelope.

    Shrinks magnitudes toward zero where needed (conservative: under-correct,
    never overshoot), keeping the correction field g_max-Lipschitz along both
    grid axes — bounded tangential gradient keeps the snapped grid injective
    (no fold-over). Two-pass min-filter per axis is exact in 1D; alternating
    axes a couple of times settles the 2D field.
    """
    mag = np.abs(np.nan_to_num(field, nan=0.0)).astype(np.float32)
    for _ in range(passes):
        for axis in (0, 1):
            m = np.moveaxis(mag, axis, 0)
            for i in range(1, m.shape[0]):  # forward
                np.minimum(m[i], m[i - 1] + g_max, out=m[i])
            for i in range(m.shape[0] - 2, -1, -1):  # backward
                np.minimum(m[i], m[i + 1] + g_max, out=m[i])
            mag = np.moveaxis(m, 0, axis)
    result = np.sign(np.nan_to_num(field, nan=0.0)) * mag
    result[~np.isfinite(field)] = np.nan
    return result.astype(np.float32)


def regularize_offsets(
    r_hat: np.ndarray,
    weight: np.ndarray,
    status: np.ndarray,
    cfg: RegularizeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline. Returns (r_smooth with NaN where held, updated status)."""
    status = status.copy()

    # 1) robust outlier rejection vs a lightly-smoothed reference field
    ref, _ = _nc(r_hat, weight, sigma=max(cfg.smooth * 0.7, 1.0))
    resid = r_hat - ref
    absresid = np.abs(resid)
    scale, _ = _nc(absresid, weight, sigma=cfg.smooth * 2)
    thresh = np.maximum(cfg.outlier_abs, cfg.outlier_nmad * 1.4826 * scale)
    outlier = np.isfinite(resid) & (absresid > thresh)
    weight = np.where(outlier, 0.0, weight)
    status[outlier & (status == STATUS_SNAPPED)] = STATUS_HELD_OUTLIER

    # 2) normalized convolution with the cleaned weights
    r_smooth, mass = _nc(r_hat, weight, sigma=cfg.smooth)

    # hold where there's no confident data nearby — never extrapolate blind
    blind = mass < cfg.min_weight_mass
    r_smooth[blind] = np.nan

    # 3) per-iteration step clamp
    r_smooth = np.clip(r_smooth, -cfg.max_step, cfg.max_step)

    # 4) tangential gradient limiting
    r_smooth = limit_gradient(r_smooth, cfg.grad_max)

    return r_smooth, status
