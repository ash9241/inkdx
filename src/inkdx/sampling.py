"""NormalProfileSampler: intensity profiles along mesh normals.

For each tile, sample up to `samples_per_tile` valid grid vertices and read the
volume intensity I(r) for r in [-halfwidth, +halfwidth] voxels along the vertex
normal. Scan- and surface-stage metrics both consume these profiles, so the
volume I/O happens exactly once.

The volume only needs numpy-style slicing over (z, y, x) — a numpy array, a
zarr array, or any lazy wrapper with `.shape` and `__getitem__` works. Each
tile reads one bounding slab and interpolates trilinearly inside it, which maps
naturally onto chunked/remote stores.

Normal orientation: tifxyz grid normals have geometric (winding-dependent)
sign. Per tile, normals are flipped to agree with the tile's dominant normal
direction, so profiles within a tile share an orientation; the global sign
remains a convention and signed metrics (peak_offset) are documented as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from inkdx.grid import Tile, TileGrid
from inkdx.io.segment import Segment


@dataclass
class TileProfiles:
    """Profiles for one tile: (n_points, 2*halfwidth+1), NaN outside the volume."""

    profiles: np.ndarray  # (N, P) float32
    grid_rc: np.ndarray  # (N, 2) int32 — sampled (row, col) in the stored grid
    offsets: np.ndarray  # (P,) float32 — r values, -halfwidth..+halfwidth
    # Noise sigma estimated from RAW slab voxels (before trilinear interpolation,
    # which attenuates noise and would bias any profile-based estimate low).
    noise_sigma_raw: float = np.nan

    @property
    def n_points(self) -> int:
        return self.profiles.shape[0]

    def median_profile(self) -> np.ndarray:
        """Median profile across points; NaN where fewer than 3 points contribute."""
        with np.errstate(all="ignore"):
            med = np.nanmedian(self.profiles, axis=0)
        support = np.isfinite(self.profiles).sum(axis=0)
        med[support < 3] = np.nan
        return med.astype(np.float32)


def _slab_noise_sigma(slab: np.ndarray) -> float:
    """Robust noise sigma from raw voxel second differences along z.

    Var(second difference) = 6 sigma^2 for i.i.d. noise; the MAD keeps the
    estimate robust to the sheet's smooth curvature (a minority of voxels).
    """
    if slab.shape[0] < 4:
        return np.nan
    d2 = slab[2:] - 2.0 * slab[1:-1] + slab[:-2]
    d2 = d2[np.isfinite(d2)]
    if d2.size < 64:
        return np.nan
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(1.4826 * mad / np.sqrt(6.0))


class NormalProfileSampler:
    def __init__(
        self,
        volume,  # (z, y, x) sliceable with .shape
        segment: Segment,
        *,
        halfwidth: int = 32,
        samples_per_tile: int = 256,
        seed: int = 0,
    ) -> None:
        self.volume = volume
        self.segment = segment
        self.halfwidth = int(halfwidth)
        self.samples_per_tile = int(samples_per_tile)
        self.seed = int(seed)
        self.offsets = np.arange(-self.halfwidth, self.halfwidth + 1, dtype=np.float32)
        self._normals = segment.normals()  # (H, W, 3), NaN where undefined

    def sample_tile(self, tile: Tile) -> TileProfiles:
        seg = self.segment
        rows, cols = tile.rows, tile.cols

        valid = seg.valid[rows, cols] & np.isfinite(self._normals[rows, cols, 0])
        rr, cc = np.nonzero(valid)
        n_pts = min(self.samples_per_tile, rr.size)
        empty = TileProfiles(
            profiles=np.empty((0, self.offsets.size), dtype=np.float32),
            grid_rc=np.empty((0, 2), dtype=np.int32),
            offsets=self.offsets,
        )
        if n_pts == 0:
            return empty

        # Deterministic per-tile subsample.
        rng = np.random.default_rng((self.seed, tile.i, tile.j))
        pick = rng.choice(rr.size, size=n_pts, replace=False)
        rr, cc = rr[pick], cc[pick]
        gr = rr + rows.start  # stored-grid coordinates
        gc = cc + cols.start

        pos = np.stack([seg.x[gr, gc], seg.y[gr, gc], seg.z[gr, gc]], axis=1)  # (N,3) xyz
        nrm = self._normals[gr, gc]  # (N, 3) xyz

        # Orient normals consistently within the tile.
        mean_n = np.nanmean(nrm, axis=0)
        mean_n /= max(np.linalg.norm(mean_n), 1e-12)
        flip = (nrm @ mean_n) < 0
        nrm = np.where(flip[:, None], -nrm, nrm)

        # Sample coordinates: (N, P, 3) in xyz, then to (z, y, x) order.
        coords = pos[:, None, :] + self.offsets[None, :, None] * nrm[:, None, :]
        czyx = coords[..., ::-1]

        # One bounding slab per tile, clipped to the volume.
        vol_shape = np.asarray(self.volume.shape[-3:], dtype=np.int64)
        lo = np.floor(np.nanmin(czyx, axis=(0, 1))).astype(np.int64) - 1
        hi = np.ceil(np.nanmax(czyx, axis=(0, 1))).astype(np.int64) + 2
        lo = np.clip(lo, 0, vol_shape)
        hi = np.clip(hi, 0, vol_shape)
        if (hi - lo).min() <= 0:
            return empty
        slab = np.asarray(
            self.volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32
        )
        sigma_raw = _slab_noise_sigma(slab)

        local = (czyx - lo[None, None, :]).reshape(-1, 3).T  # (3, N*P)
        vals = map_coordinates(slab, local, order=1, mode="constant", cval=np.nan)
        profiles = vals.reshape(n_pts, self.offsets.size).astype(np.float32)

        # Points sampled outside the slab (clipped) are NaN via cval; also mask
        # anything that left the volume bounds entirely.
        oob = (
            (czyx < 0).any(axis=-1)
            | (czyx > (vol_shape - 1)[None, None, :]).any(axis=-1)
        )
        profiles[oob] = np.nan

        return TileProfiles(
            profiles=profiles,
            grid_rc=np.stack([gr, gc], axis=1).astype(np.int32),
            offsets=self.offsets,
            noise_sigma_raw=sigma_raw,
        )

    def sample_grid(self, grid: TileGrid):
        """Yield (tile, TileProfiles) for every tile."""
        for tile in grid.tiles():
            yield tile, self.sample_tile(tile)
