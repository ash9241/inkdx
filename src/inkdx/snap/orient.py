"""Global normal-sign consistency across tiles.

Per-tile normal orientation (dominant-direction flipping) can leave adjacent
tiles with opposite signs. The pointwise snap update r*n is invariant under a
joint flip, but cross-tile smoothing of the scalar offset field is not: the
same physical displacement would read +2 in one tile and -2 in its neighbor.
This pass assigns one sign per tile so adjacent dominant normals agree.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from inkdx.grid import TileGrid
from inkdx.io.segment import Segment


def dominant_normals(segment: Segment, grid: TileGrid) -> np.ndarray:
    """(th, tw, 3) mean unit normal per tile (NaN where no valid normals)."""
    th, tw = grid.shape
    out = np.full((th, tw, 3), np.nan, dtype=np.float32)
    for t in grid.tiles():
        n = segment.normals_window(t.rows, t.cols)
        flat = n.reshape(-1, 3)
        finite = np.isfinite(flat[:, 0])
        if finite.sum() < 4:
            continue
        m = flat[finite].mean(axis=0)
        norm = np.linalg.norm(m)
        if norm > 1e-12:
            out[t.i, t.j] = m / norm
    return out


def tile_signs(
    segment: Segment, grid: TileGrid, *, warn_dot: float = 0.2
) -> tuple[np.ndarray, list[str]]:
    """BFS sign assignment so adjacent tiles' dominant normals agree.

    Returns ((th, tw) int8 signs {+1, -1, 0=no data}, warnings). Each connected
    component is oriented independently, seeded from its largest-support tile.
    """
    dom = dominant_normals(segment, grid)
    th, tw = grid.shape
    has = np.isfinite(dom[..., 0])
    signs = np.zeros((th, tw), dtype=np.int8)
    warnings: list[str] = []

    def neighbors(i: int, j: int):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < th and 0 <= nj < tw and has[ni, nj]:
                yield ni, nj

    unvisited = {(i, j) for i, j in zip(*np.nonzero(has), strict=True)}
    while unvisited:
        seed = max(
            unvisited,
            key=lambda ij: np.asarray(segment.valid[grid.tile(*ij).rows,
                                                    grid.tile(*ij).cols]).sum(),
        )
        signs[seed] = 1
        unvisited.discard(seed)
        queue = deque([seed])
        while queue:
            i, j = queue.popleft()
            for ni, nj in neighbors(i, j):
                if (ni, nj) not in unvisited:
                    continue
                # neighbor sign s_n must satisfy (dom_n * s_n).(dom_i * s_i) > 0
                dot = float(np.dot(dom[i, j] * signs[i, j], dom[ni, nj]))
                signs[ni, nj] = np.int8(1) if dot >= 0 else np.int8(-1)
                if abs(dot) < warn_dot:
                    warnings.append(
                        f"tile ({i},{j})->({ni},{nj}): adjacency dot {dot:+.2f} "
                        "< threshold — twisted segment? orientation may be unreliable"
                    )
                unvisited.discard((ni, nj))
                queue.append((ni, nj))
    return signs, warnings
