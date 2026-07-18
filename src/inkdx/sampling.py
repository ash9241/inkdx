"""Intensity profiles along mesh normals.

Three entry points share one core (`profiles_at`):

- `NormalProfileSampler.sample_tile` — subsampled per-tile profiles for
  diagnostics (unchanged v0.1 behavior).
- `dense_tile_profiles` — a profile for EVERY valid vertex of a tile (optional
  stride), returned as a `(th, tw, P)` block that preserves UV structure so
  spatial pooling stays a plain array op. Used by `inkdx snap`.
- `profiles_for_indices` — profiles for an explicit vertex list, chunked into
  per-tile slab reads internally. Used by `inkdx label3d`.

The volume only needs numpy-style slicing over (z, y, x) — a numpy array, a
zarr array, or any lazy wrapper with `.shape` and `__getitem__` works. Each
call reads one bounding slab and interpolates trilinearly inside it.

Normal orientation: tifxyz grid normals have geometric (winding-dependent)
sign. Per tile, normals are flipped to agree with the tile's dominant normal
direction; the global sign remains a convention (snap's orient pass handles
cross-tile consistency where it matters).
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


def profiles_at(
    volume,
    pos_xyz: np.ndarray,
    normals_xyz: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Sample I(r) along each normal: one bounding slab, trilinear inside.

    pos_xyz, normals_xyz: (N, 3) in xyz order. offsets: (P,) r values.
    Returns ((N, P) float32 profiles with NaN outside the volume,
    raw-slab noise sigma).
    """
    n_pts = pos_xyz.shape[0]
    n_off = offsets.size
    if n_pts == 0:
        return np.empty((0, n_off), dtype=np.float32), np.nan

    coords = pos_xyz[:, None, :] + offsets[None, :, None] * normals_xyz[:, None, :]
    czyx = coords[..., ::-1]

    vol_shape = np.asarray(volume.shape[-3:], dtype=np.int64)
    lo = np.floor(np.nanmin(czyx, axis=(0, 1))).astype(np.int64) - 1
    hi = np.ceil(np.nanmax(czyx, axis=(0, 1))).astype(np.int64) + 2
    lo = np.clip(lo, 0, vol_shape)
    hi = np.clip(hi, 0, vol_shape)
    if (hi - lo).min() <= 0:
        return np.full((n_pts, n_off), np.nan, dtype=np.float32), np.nan

    slab = np.asarray(volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
    sigma_raw = _slab_noise_sigma(slab)

    local = (czyx - lo[None, None, :]).reshape(-1, 3).T
    vals = map_coordinates(slab, local, order=1, mode="constant", cval=np.nan)
    profiles = vals.reshape(n_pts, n_off).astype(np.float32)

    oob = (
        (czyx < 0).any(axis=-1)
        | (czyx > (vol_shape - 1)[None, None, :]).any(axis=-1)
    )
    profiles[oob] = np.nan
    return profiles, sigma_raw


def _oriented_tile_normals(normals: np.ndarray) -> np.ndarray:
    """Flip normals to agree with the (nan-mean) dominant direction."""
    flat = normals.reshape(-1, 3)
    finite = np.isfinite(flat[:, 0])
    if not finite.any():
        return normals
    mean_n = flat[finite].mean(axis=0)
    norm = np.linalg.norm(mean_n)
    if norm < 1e-12:
        return normals
    mean_n /= norm
    flip = (flat @ mean_n) < 0
    flat = np.where(flip[:, None], -flat, flat)
    return flat.reshape(normals.shape)


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

    def sample_tile(self, tile: Tile) -> TileProfiles:
        seg = self.segment
        rows, cols = tile.rows, tile.cols

        # Per-tile normals (1-vertex halo): memory scales with the tile, never
        # the grid — gigapixel identity meshes must not materialize anything.
        tile_normals = seg.normals_window(rows, cols)
        valid = np.asarray(seg.valid[rows, cols]) & np.isfinite(tile_normals[..., 0])
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

        pos = np.stack(
            [np.asarray(seg.x[gr, gc]), np.asarray(seg.y[gr, gc]), np.asarray(seg.z[gr, gc])],
            axis=1,
        )  # (N, 3) xyz
        nrm = _oriented_tile_normals(tile_normals[rr, cc])

        profiles, sigma_raw = profiles_at(self.volume, pos, nrm, self.offsets)
        if profiles.shape[0] == 0:
            return empty

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


@dataclass
class DenseTileProfiles:
    """Profiles for every (strided) valid vertex of a tile, UV structure kept.

    `block[i, j]` is the profile of vertex (rows.start + i*stride,
    cols.start + j*stride); NaN rows where the vertex is invalid or its
    normal undefined.
    """

    block: np.ndarray  # (th, tw, P) float32
    valid: np.ndarray  # (th, tw) bool
    normals: np.ndarray  # (th, tw, 3) float32, tile-consistent orientation
    offsets: np.ndarray  # (P,)
    stride: int
    noise_sigma_raw: float = np.nan


def dense_tile_profiles(
    volume,
    segment: Segment,
    tile: Tile,
    *,
    halfwidth: int,
    stride: int = 1,
) -> DenseTileProfiles:
    """Profile for every valid vertex of the tile (at `stride`)."""
    offsets = np.arange(-halfwidth, halfwidth + 1, dtype=np.float32)
    rows, cols = tile.rows, tile.cols

    normals = segment.normals_window(rows, cols)[::stride, ::stride]
    valid = (
        np.asarray(segment.valid[rows, cols])[::stride, ::stride]
        & np.isfinite(normals[..., 0])
    )
    th, tw = valid.shape
    block = np.full((th, tw, offsets.size), np.nan, dtype=np.float32)

    rr, cc = np.nonzero(valid)
    if rr.size == 0:
        return DenseTileProfiles(block, valid, _oriented_tile_normals(normals),
                                 offsets, stride)

    gr = rows.start + rr * stride
    gc = cols.start + cc * stride
    pos = np.stack(
        [
            np.asarray(segment.x[gr, gc], dtype=np.float32),
            np.asarray(segment.y[gr, gc], dtype=np.float32),
            np.asarray(segment.z[gr, gc], dtype=np.float32),
        ],
        axis=1,
    )
    oriented = _oriented_tile_normals(normals)
    profiles, sigma_raw = profiles_at(volume, pos, oriented[rr, cc], offsets)
    block[rr, cc] = profiles
    return DenseTileProfiles(block, valid, oriented, offsets, stride, sigma_raw)


def profiles_for_indices(
    volume,
    segment: Segment,
    gr: np.ndarray,
    gc: np.ndarray,
    *,
    halfwidth: int,
    tile_px: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Profiles for explicit stored-grid vertices (gr, gc), any order.

    Chunks the work by tile so each slab read stays bounded even when the
    vertex list spans a large curved region. Returns ((N, P) profiles in the
    input order, (P,) offsets). Vertices that are invalid or have undefined
    normals get all-NaN profiles.
    """
    offsets = np.arange(-halfwidth, halfwidth + 1, dtype=np.float32)
    gr = np.asarray(gr, dtype=np.int64)
    gc = np.asarray(gc, dtype=np.int64)
    out = np.full((gr.size, offsets.size), np.nan, dtype=np.float32)
    if gr.size == 0:
        return out, offsets

    h, w = segment.grid_shape
    grid = TileGrid((h, w), tile_px=tile_px)
    tile_ids = (gr // tile_px) * grid.shape[1] + (gc // tile_px)
    for tid in np.unique(tile_ids):
        sel = np.nonzero(tile_ids == tid)[0]
        i, j = int(tid) // grid.shape[1], int(tid) % grid.shape[1]
        tile = grid.tile(i, j)
        rows, cols = tile.rows, tile.cols

        normals = segment.normals_window(rows, cols)
        rr = gr[sel] - rows.start
        cc = gc[sel] - cols.start
        valid = (
            np.asarray(segment.valid[rows, cols])[rr, cc]
            & np.isfinite(normals[rr, cc, 0])
        )
        if not valid.any():
            continue
        vsel = sel[valid]
        rrv, ccv = rr[valid], cc[valid]
        pos = np.stack(
            [
                np.asarray(segment.x[gr[vsel], gc[vsel]], dtype=np.float32),
                np.asarray(segment.y[gr[vsel], gc[vsel]], dtype=np.float32),
                np.asarray(segment.z[gr[vsel], gc[vsel]], dtype=np.float32),
            ],
            axis=1,
        )
        oriented = _oriented_tile_normals(normals)
        profiles, _ = profiles_at(volume, pos, oriented[rrv, ccv], offsets)
        out[vsel] = profiles
    return out, offsets
