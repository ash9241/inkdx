"""Scan stage: is there usable signal in the CT around this tile?

All metrics come from the tile's normal profiles alone — no ground truth, no
model. They quantify the physics: noise floor, sheet/gap contrast, dynamic
range, and haze (blur of the sheet boundary).

`haze_index` is the profile-peak FWHM over the expected sheet thickness: blur —
from scanning, phase retrieval, or compression scatter — widens the apparent
sheet. A mesh crossing the sheet obliquely widens it too; the surface stage's
spatial pattern disambiguates (segment-wide widening = scan, patchy = mesh).
"""

from __future__ import annotations

import numpy as np

from inkdx.sampling import TileProfiles
from inkdx.stages.profile_features import analyze_profile, noise_sigma

_EPS = 1e-6

SCAN_METRICS = (
    "noise_sigma", "snr", "cnr", "dynamic_range",
    "saturation_frac", "haze_index", "median_intensity",
)


def compute_scan_metrics(
    profiles: TileProfiles,
    *,
    vmax: float = 255.0,
    expected_thickness: float = 12.0,  # voxels; ~papyrus sheet at 7.9 um
) -> dict[str, float]:
    """Per-tile scan metrics. Returns NaNs for empty/degenerate tiles."""
    out = dict.fromkeys(SCAN_METRICS, np.nan)
    if profiles.n_points == 0:
        return out

    samples = profiles.profiles[np.isfinite(profiles.profiles)]
    if samples.size:
        out["dynamic_range"] = float(
            np.percentile(samples, 95) - np.percentile(samples, 5)
        )
        out["saturation_frac"] = float(
            np.mean((samples <= 0.0) | (samples >= vmax))
        )
        out["median_intensity"] = float(np.median(samples))

    # Prefer the raw-slab estimate: trilinear interpolation attenuates noise,
    # so a profile-based estimate is biased low. Fall back when no slab stats
    # exist (e.g. pre-extracted layer stacks).
    sigma = profiles.noise_sigma_raw
    if not np.isfinite(sigma):
        sigma = noise_sigma(profiles.profiles)
    out["noise_sigma"] = sigma

    med = profiles.median_profile()
    feats = analyze_profile(med, profiles.offsets)
    if not feats.ok:
        return out

    if np.isfinite(sigma):
        out["snr"] = feats.peak_value / max(sigma, _EPS)
        out["cnr"] = feats.prominence / max(sigma, _EPS)

    if np.isfinite(feats.fwhm):
        out["haze_index"] = float(feats.fwhm / expected_thickness)
    return out


def intensity_drift(median_intensity_map: np.ndarray) -> np.ndarray:
    """Segment-level post-pass: robust z-score of each tile's median intensity.

    Flags ring artifacts, beam hardening, and stitching seams as spatial
    outliers relative to the whole segment.
    """
    finite = median_intensity_map[np.isfinite(median_intensity_map)]
    if finite.size < 4:
        return np.full_like(median_intensity_map, np.nan)
    center = np.median(finite)
    mad = np.median(np.abs(finite - center))
    scale = max(1.4826 * mad, _EPS)
    return ((median_intensity_map - center) / scale).astype(np.float32)
