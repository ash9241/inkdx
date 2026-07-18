"""TileGrid: the shared per-tile coordinate system all diagnostics report on.

Every stage produces per-tile scalar maps on this grid; the verdict layer and
reports consume only these maps. A new metric is one function returning a tile
map — nothing else needs to change.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    i: int  # tile row
    j: int  # tile col
    rows: slice  # grid-row extent (stored resolution)
    cols: slice  # grid-col extent


@dataclass(frozen=True)
class TileGrid:
    """Partition of a segment's UV grid (stored resolution) into square tiles."""

    grid_shape: tuple[int, int]  # (H, W) of the segment grid
    tile_px: int = 256

    def __post_init__(self) -> None:
        if self.tile_px < 1:
            raise ValueError(f"tile_px must be >= 1, got {self.tile_px}")

    @property
    def shape(self) -> tuple[int, int]:
        """(tile rows, tile cols)."""
        h, w = self.grid_shape
        return (-(-h // self.tile_px), -(-w // self.tile_px))

    @property
    def n_tiles(self) -> int:
        th, tw = self.shape
        return th * tw

    def tile(self, i: int, j: int) -> Tile:
        th, tw = self.shape
        if not (0 <= i < th and 0 <= j < tw):
            raise IndexError(f"tile ({i}, {j}) outside grid {self.shape}")
        h, w = self.grid_shape
        r0, c0 = i * self.tile_px, j * self.tile_px
        rows = slice(r0, min(r0 + self.tile_px, h))
        cols = slice(c0, min(c0 + self.tile_px, w))
        return Tile(i, j, rows, cols)

    def tiles(self) -> Iterator[Tile]:
        th, tw = self.shape
        for i in range(th):
            for j in range(tw):
                yield self.tile(i, j)

    def new_map(self, fill: float = np.nan, dtype=np.float32) -> np.ndarray:
        """Allocate a per-tile map."""
        return np.full(self.shape, fill, dtype=dtype)

    def uv_bbox(self, i: int, j: int) -> tuple[int, int, int, int]:
        """(row0, col0, row1, col1) of a tile in stored-grid coordinates."""
        t = self.tile(i, j)
        return (t.rows.start, t.cols.start, t.rows.stop, t.cols.stop)

    def reduce(self, per_vertex: np.ndarray, fn=np.nanmean) -> np.ndarray:
        """Reduce an (H, W[, ...]) per-vertex array to a per-tile map with fn."""
        if per_vertex.shape[:2] != self.grid_shape:
            raise ValueError(
                f"per-vertex shape {per_vertex.shape[:2]} != grid {self.grid_shape}"
            )
        out = self.new_map()
        with np.errstate(invalid="ignore"):
            for t in self.tiles():
                block = per_vertex[t.rows, t.cols]
                if block.size:
                    out[t.i, t.j] = fn(block)
        return out
