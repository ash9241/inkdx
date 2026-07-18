"""Scan-failure ablations: degrade volume data in controlled, seeded ways.

Used to validate attribution: a noised/blurred volume must flip tiles to
SCAN_SUSPECT and leave surface/model verdicts alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


def add_noise(volume: np.ndarray, sigma: float, *, seed: int = 0) -> np.ndarray:
    """Additive Gaussian noise, clipped to the dtype range."""
    rng = np.random.default_rng(seed)
    noisy = volume.astype(np.float32) + rng.normal(0.0, sigma, volume.shape)
    info = np.iinfo(volume.dtype) if np.issubdtype(volume.dtype, np.integer) else None
    if info is not None:
        return np.clip(noisy, info.min, info.max).astype(volume.dtype)
    return noisy.astype(volume.dtype)


def blur(volume: np.ndarray, sigma: float) -> np.ndarray:
    """Isotropic Gaussian blur (simulates focus/phase-retrieval degradation)."""
    out = gaussian_filter(volume.astype(np.float32), sigma=sigma)
    info = np.iinfo(volume.dtype) if np.issubdtype(volume.dtype, np.integer) else None
    if info is not None:
        return np.clip(out, info.min, info.max).astype(volume.dtype)
    return out.astype(volume.dtype)


def ablate_layer_stack(
    layers_dir: str | Path,
    out_dir: str | Path,
    *,
    noise_sigma: float = 0.0,
    blur_sigma: float = 0.0,
    region: tuple[int, int, int, int] | None = None,  # (row0, col0, row1, col1)
    seed: int = 0,
) -> Path:
    """Write a degraded copy of a layer-stack segment (optionally windowed).

    With `region`, only that UV window is written (a smaller stack) — sized
    ablations keep GPU re-inference cheap. Blur is applied per full 3D window
    so the z-axis blurs too.
    """
    from inkdx.io.volume import LayerStackVolume

    src = LayerStackVolume(layers_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r0, c0, r1, c1 = region or (0, 0, src.shape[1], src.shape[2])
    window = np.asarray(src[0:src.shape[0], r0:r1, c0:c1])

    if blur_sigma > 0:
        window = blur(window, blur_sigma)
    if noise_sigma > 0:
        window = add_noise(window, noise_sigma, seed=seed)

    for k in range(window.shape[0]):
        tifffile.imwrite(out_dir / f"{k:02}.tif", window[k])
    return out_dir
