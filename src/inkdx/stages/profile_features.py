"""Shared analysis of normal-intensity profiles.

Both the scan stage (is there signal?) and the surface stage (is the mesh on
the sheet?) read the same features off a tile's median profile: peak position,
prominence, width, flanking gaps, and multiplicity. Keeping this in one place
guarantees the two stages can't disagree about where the sheet is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter


@dataclass
class ProfileFeatures:
    r_star: float  # peak offset from mesh (voxels along the normal)
    peak_value: float  # smoothed intensity at the peak
    gap_value: float  # mean of the flanking minima (inter-sheet gap level)
    prominence: float  # peak_value - gap_value (raw, un-normalized)
    fwhm: float  # full width at half prominence (voxels); NaN if not resolvable
    multiplicity: int  # peaks with prominence >= 0.5x the main peak's
    com_offset: float  # intensity-weighted centroid offset in the central window

    @property
    def ok(self) -> bool:
        return np.isfinite(self.r_star)


_NAN_FEATURES = ProfileFeatures(
    r_star=np.nan, peak_value=np.nan, gap_value=np.nan,
    prominence=np.nan, fwhm=np.nan, multiplicity=0, com_offset=np.nan,
)


def noise_sigma(profiles: np.ndarray) -> float:
    """Robust noise estimate from second differences along r.

    For i.i.d. noise, Var(I(r-1) - 2 I(r) + I(r+1)) = 6 sigma^2. The MAD makes
    the estimate robust to the smooth sheet peak (a small fraction of samples
    with nonzero curvature).
    """
    if profiles.size == 0:
        return np.nan
    d2 = profiles[:, 2:] - 2.0 * profiles[:, 1:-1] + profiles[:, :-2]
    d2 = d2[np.isfinite(d2)]
    if d2.size < 16:
        return np.nan
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(1.4826 * mad / np.sqrt(6.0))


def _smooth(values: np.ndarray) -> np.ndarray:
    n = values.size
    window = min(7, n if n % 2 else n - 1)
    if window < 5:
        return values
    return savgol_filter(values, window_length=window, polyorder=2)


def analyze_profiles_dense(
    block: np.ndarray, offsets: np.ndarray, *, return_smoothed: bool = False
) -> dict[str, np.ndarray]:
    """Vectorized per-vertex profile features over a (th, tw, P) block.

    Returns per-vertex arrays: r_star, peak_value, gap_value, prominence,
    multiplicity, com_offset, and sub-voxel r_star_subvox (parabolic
    refinement). Semantics match `analyze_profile` per vertex — the CI
    agreement test pins the two implementations together. Vertices with any
    interior NaN along r get NaN features (same give-up rule as the scalar
    path).
    """
    th, tw, p = block.shape
    nan_map = np.full((th, tw), np.nan, dtype=np.float32)
    out = {
        "r_star": nan_map.copy(), "peak_value": nan_map.copy(),
        "gap_value": nan_map.copy(), "prominence": nan_map.copy(),
        "multiplicity": np.zeros((th, tw), dtype=np.int16),
        "com_offset": nan_map.copy(), "r_star_subvox": nan_map.copy(),
    }
    if p < 9:
        return out

    finite = np.isfinite(block)
    # Same rule as analyze_profile: only fully-finite center windows qualify;
    # for the dense path we require the WHOLE profile finite (dense callers use
    # windows that fit the volume; edge vertices fall back to held).
    ok = finite.all(axis=-1)
    if not ok.any():
        return out

    # Zero-fill non-finite entries: savgol's edge polyfit is batched across the
    # whole array and chokes on any NaN. Only fully-finite (`ok`) rows are ever
    # written to the outputs, so the fill value never leaks.
    vals = np.where(finite, block, 0.0).astype(np.float64)
    n = p if p % 2 else p - 1
    window = min(7, n)
    if window >= 5:
        smoothed = savgol_filter(vals, window_length=window, polyorder=2, axis=-1)
    else:
        smoothed = vals

    i_star = np.argmax(smoothed, axis=-1)
    ii, jj = np.meshgrid(np.arange(th), np.arange(tw), indexing="ij")
    peak_value = smoothed[ii, jj, i_star]
    r_star = offsets[i_star].astype(np.float64)
    gap_value = np.percentile(smoothed, 10, axis=-1)
    prominence = peak_value - gap_value

    # Local maxima by array comparison; multiplicity = peaks with >= 0.5x the
    # main prominence (matches find_peaks(prominence=0.5*prom) closely enough
    # for gating; the agreement test bounds the difference).
    interior = smoothed[..., 1:-1]
    is_max = (interior >= smoothed[..., :-2]) & (interior >= smoothed[..., 2:])
    peak_prom = interior - gap_value[..., None]
    big = is_max & (peak_prom >= 0.5 * np.maximum(prominence[..., None], 1e-12))
    multiplicity = np.maximum(big.sum(axis=-1), (prominence > 0).astype(int))

    # Parabolic sub-voxel refinement around i_star (clipped to interior).
    ic = np.clip(i_star, 1, p - 2)
    s_m = smoothed[ii, jj, ic - 1]
    s_0 = smoothed[ii, jj, ic]
    s_p = smoothed[ii, jj, ic + 1]
    denom = s_m - 2.0 * s_0 + s_p
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 0.5 * (s_m - s_p) / denom
    delta = np.clip(np.where(np.abs(denom) > 1e-12, delta, 0.0), -0.5, 0.5)
    step = float(offsets[1] - offsets[0]) if p > 1 else 1.0
    r_subvox = offsets[ic] + delta * step

    # Intensity-weighted centroid in a central window (same as scalar path,
    # with the fwhm-free default window of +/- 4).
    win = np.abs(offsets[None, None, :] - r_star[..., None]) <= 4.0
    w = np.clip(vals - gap_value[..., None], 0.0, None) * win
    wsum = w.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        com = (offsets[None, None, :] * w).sum(axis=-1) / wsum

    for key, arr in (
        ("r_star", r_star), ("peak_value", peak_value), ("gap_value", gap_value),
        ("prominence", prominence), ("com_offset", com), ("r_star_subvox", r_subvox),
    ):
        out[key][ok] = arr[ok].astype(np.float32)
    out["multiplicity"][ok] = multiplicity[ok].astype(np.int16)
    out["com_offset"][ok & (wsum <= 0)] = np.nan

    if return_smoothed:
        out["smoothed"] = smoothed.astype(np.float32)
        # Interior local maxima with prominence >= 0.3x main: the candidate set
        # for snap's nearest-peak selection policy.
        candidates = np.zeros_like(block, dtype=bool)
        cand = is_max & (peak_prom >= 0.3 * np.maximum(prominence[..., None], 1e-12))
        candidates[..., 1:-1] = cand & ok[..., None]
        out["candidates"] = candidates
    return out


def analyze_profile(median_profile: np.ndarray, offsets: np.ndarray) -> ProfileFeatures:
    """Extract sheet-peak features from a tile's median profile."""
    finite = np.isfinite(median_profile)
    if finite.sum() < 9:
        return _NAN_FEATURES

    # Work on the widest finite center window.
    idx = np.nonzero(finite)[0]
    lo, hi = idx.min(), idx.max() + 1
    if not finite[lo:hi].all():  # interior NaNs: give up rather than interpolate
        return _NAN_FEATURES
    vals = median_profile[lo:hi].astype(np.float64)
    offs = offsets[lo:hi].astype(np.float64)

    smoothed = _smooth(vals)
    i_star = int(np.argmax(smoothed))
    r_star = float(offs[i_star])
    peak_value = float(smoothed[i_star])

    # Gap level: 10th percentile of the smoothed profile. Robust to an
    # off-center peak squashing one flank (a surface failure must not read as
    # low contrast — that would cross-talk into the scan stage).
    gap_value = float(np.percentile(smoothed, 10))
    prominence = peak_value - gap_value

    # FWHM at half prominence around the main peak.
    half = gap_value + 0.5 * prominence
    above = smoothed >= half
    li = i_star
    while li > 0 and above[li - 1]:
        li -= 1
    ri = i_star
    while ri < smoothed.size - 1 and above[ri + 1]:
        ri += 1
    if li == 0 or ri == smoothed.size - 1:
        fwhm = np.nan  # peak not resolved inside the window
    else:
        fwhm = float(offs[ri] - offs[li]) or np.nan

    # Multiplicity: peaks comparable to the main one (neighbor sheets in window).
    if prominence > 0:
        peaks, _ = find_peaks(smoothed, prominence=0.5 * prominence)
        multiplicity = max(int(peaks.size), 1)
    else:
        multiplicity = 0

    # Intensity-weighted centroid in a central window (sub-voxel drift).
    win = np.abs(offs - r_star) <= max(4.0, 1.5 * (fwhm if np.isfinite(fwhm) else 4.0))
    w = np.clip(vals[win] - gap_value, 0.0, None)
    com_offset = float((offs[win] * w).sum() / w.sum()) if w.sum() > 0 else np.nan

    return ProfileFeatures(
        r_star=r_star, peak_value=peak_value, gap_value=gap_value,
        prominence=prominence, fwhm=fwhm, multiplicity=multiplicity,
        com_offset=com_offset,
    )
