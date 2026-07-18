import numpy as np
import pytest

from inkdx.grid import TileGrid


def test_tiling_covers_grid_exactly():
    g = TileGrid(grid_shape=(500, 300), tile_px=128)
    assert g.shape == (4, 3)

    seen = np.zeros((500, 300), dtype=int)
    for t in g.tiles():
        seen[t.rows, t.cols] += 1
    assert (seen == 1).all()


def test_edge_tiles_are_clipped():
    g = TileGrid(grid_shape=(500, 300), tile_px=128)
    r0, c0, r1, c1 = g.uv_bbox(3, 2)
    assert (r0, c0, r1, c1) == (384, 256, 500, 300)


def test_reduce_per_vertex():
    g = TileGrid(grid_shape=(64, 64), tile_px=32)
    per_vertex = np.zeros((64, 64), dtype=np.float32)
    per_vertex[:32, :32] = 5.0
    m = g.reduce(per_vertex)
    assert m.shape == (2, 2)
    assert m[0, 0] == 5.0 and m[1, 1] == 0.0


def test_reduce_handles_nan():
    g = TileGrid(grid_shape=(32, 32), tile_px=32)
    per_vertex = np.full((32, 32), np.nan, dtype=np.float32)
    per_vertex[0, 0] = 2.0
    assert g.reduce(per_vertex)[0, 0] == 2.0


def test_bad_inputs():
    with pytest.raises(ValueError):
        TileGrid(grid_shape=(10, 10), tile_px=0)
    g = TileGrid(grid_shape=(10, 10), tile_px=4)
    with pytest.raises(IndexError):
        g.tile(9, 0)
    with pytest.raises(ValueError):
        g.reduce(np.zeros((5, 5)))
