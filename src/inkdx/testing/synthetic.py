"""Synthetic phantom: parametric papyrus sheets in a small CT-like volume.

The phantom is the unit-test ground truth for every metric: sheet position,
thickness, contrast, noise, and spacing are all known analytically, so metric
implementations can be validated against exact expected values (e.g. an
injected mesh offset of k voxels must be recovered as peak_offset ≈ k).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from inkdx.io.segment import Segment


@dataclass
class PhantomParams:
    shape: tuple[int, int, int] = (64, 96, 128)  # (z, y, x) voxels
    sheet_z0: float = 32.0  # mean sheet height (keep > amp + margin: z>0 rule)
    undulation_amp: float = 3.0  # sinusoidal undulation amplitude (voxels)
    undulation_period: float = 64.0  # along x (voxels)
    sheet_sigma: float = 2.5  # Gaussian half-thickness of the sheet (voxels)
    sheet_contrast: float = 120.0  # peak intensity above background
    background: float = 30.0
    noise_sigma: float = 3.0  # additive Gaussian noise
    second_sheet_dz: float | None = None  # optional neighbor sheet offset (voxels)
    seed: int = 0
    # Synthetic ink: intensity delta over r in ink_band (relative to the sheet
    # surface, along +z) inside the labeled UV regions. Signed contrast — real
    # ink may be radiodense or radiolucent.
    ink_band: tuple[float, float] | None = None  # (a, b) in voxels, e.g. (-1, 3)
    ink_contrast: float = -40.0
    ink_uv_regions: tuple[tuple[int, int, int, int], ...] = ()  # (y0, x0, y1, x1)


@dataclass
class Phantom:
    volume: np.ndarray  # (z, y, x) uint8
    segment: Segment  # perfect mesh on the (first) sheet
    params: PhantomParams
    sheet_z: np.ndarray = field(repr=False, default=None)  # (y, x) true sheet height
    ink_mask: np.ndarray = field(repr=False, default=None)  # (y, x) bool 2D ink label


def sheet_height(params: PhantomParams) -> np.ndarray:
    """True sheet height z_s(y, x)."""
    _, ny, nx = params.shape
    xx = np.arange(nx, dtype=np.float32)
    zline = params.sheet_z0 + params.undulation_amp * np.sin(
        2.0 * np.pi * xx / params.undulation_period
    )
    return np.broadcast_to(zline, (ny, nx)).astype(np.float32).copy()


def make_phantom(params: PhantomParams | None = None) -> Phantom:
    """Build volume + perfect segment for the given parameters."""
    p = params or PhantomParams()
    nz, ny, nx = p.shape
    if not p.sheet_z0 - p.undulation_amp > 0:
        raise ValueError("sheet must satisfy z > 0 everywhere (tifxyz validity rule)")

    z_s = sheet_height(p)  # (y, x)
    zz = np.arange(nz, dtype=np.float32)[:, None, None]  # (z, 1, 1)

    d2 = (zz - z_s[None, :, :]) ** 2
    vol = p.background + p.sheet_contrast * np.exp(-d2 / (2.0 * p.sheet_sigma**2))
    if p.second_sheet_dz is not None:
        d2b = (zz - (z_s[None, :, :] + p.second_sheet_dz)) ** 2
        vol += p.sheet_contrast * np.exp(-d2b / (2.0 * p.sheet_sigma**2))

    ink_mask = np.zeros((ny, nx), dtype=bool)
    if p.ink_band is not None and p.ink_uv_regions:
        for y0, x0, y1, x1 in p.ink_uv_regions:
            ink_mask[y0:y1, x0:x1] = True
        a, b = p.ink_band
        r = zz - z_s[None, :, :]  # offset from sheet surface along +z
        in_band = (r >= a) & (r <= b) & ink_mask[None, :, :]
        vol = vol + p.ink_contrast * in_band.astype(np.float32)

    rng = np.random.default_rng(p.seed)
    vol += rng.normal(0.0, p.noise_sigma, size=vol.shape)
    vol = np.clip(vol, 0, 255).astype(np.uint8)

    yy, xx = np.meshgrid(
        np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32), indexing="ij"
    )
    segment = Segment(
        x=xx, y=yy, z=z_s.copy(),
        valid=np.ones((ny, nx), dtype=bool),
        scale=(1.0, 1.0),
        uuid="phantom",
    )
    return Phantom(volume=vol, segment=segment, params=p, sheet_z=z_s, ink_mask=ink_mask)
