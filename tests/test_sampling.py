import numpy as np
import pytest

from inkdx.grid import TileGrid
from inkdx.sampling import NormalProfileSampler
from inkdx.testing.synthetic import PhantomParams, make_phantom


@pytest.fixture(scope="module")
def flat_phantom():
    return make_phantom(PhantomParams(undulation_amp=0.0, noise_sigma=0.0))


def _peak_offset(profiles) -> float:
    med = profiles.median_profile()
    return float(profiles.offsets[np.nanargmax(med)])


def test_perfect_mesh_peaks_at_zero(flat_phantom):
    ph = flat_phantom
    sampler = NormalProfileSampler(ph.volume, ph.segment, halfwidth=16)
    grid = TileGrid(ph.segment.grid_shape, tile_px=48)
    for tile in grid.tiles():
        profiles = sampler.sample_tile(tile)
        assert profiles.n_points > 0
        assert abs(_peak_offset(profiles)) <= 1.0


def test_offset_mesh_recovers_injected_shift(flat_phantom):
    ph = flat_phantom
    seg = ph.segment
    for k in (4.0, 8.0):
        shifted = type(seg)(
            x=seg.x, y=seg.y, z=seg.z + k, valid=seg.valid,
            scale=seg.scale, uuid=f"shift{k}",
        )
        sampler = NormalProfileSampler(ph.volume, shifted, halfwidth=16)
        tile = TileGrid(seg.grid_shape, tile_px=48).tile(0, 0)
        # sheet is now at -k or +k along the (sign-conventional) normal
        assert abs(abs(_peak_offset(sampler.sample_tile(tile))) - k) <= 0.5


def test_undulating_mesh_still_centered():
    ph = make_phantom(PhantomParams(undulation_amp=4.0, noise_sigma=0.0))
    sampler = NormalProfileSampler(ph.volume, ph.segment, halfwidth=12)
    tile = TileGrid(ph.segment.grid_shape, tile_px=64).tile(0, 0)
    assert abs(_peak_offset(sampler.sample_tile(tile))) <= 1.0


def test_determinism(flat_phantom):
    ph = flat_phantom
    tile = TileGrid(ph.segment.grid_shape, tile_px=48).tile(0, 1)
    a = NormalProfileSampler(ph.volume, ph.segment, seed=3).sample_tile(tile)
    b = NormalProfileSampler(ph.volume, ph.segment, seed=3).sample_tile(tile)
    c = NormalProfileSampler(ph.volume, ph.segment, seed=4).sample_tile(tile)
    np.testing.assert_array_equal(a.grid_rc, b.grid_rc)
    assert not np.array_equal(a.grid_rc, c.grid_rc)


def test_all_invalid_tile_is_empty(flat_phantom):
    ph = flat_phantom
    seg = ph.segment
    dead = type(seg)(
        x=seg.x, y=seg.y, z=seg.z,
        valid=np.zeros_like(seg.valid), scale=seg.scale, uuid="dead",
    )
    sampler = NormalProfileSampler(ph.volume, dead)
    tile = TileGrid(seg.grid_shape, tile_px=48).tile(0, 0)
    assert sampler.sample_tile(tile).n_points == 0


def test_profiles_nan_outside_volume(flat_phantom):
    ph = flat_phantom
    # halfwidth exceeding distance to volume edge -> NaN tails, finite center
    sampler = NormalProfileSampler(ph.volume, ph.segment, halfwidth=60)
    tile = TileGrid(ph.segment.grid_shape, tile_px=48).tile(0, 0)
    profiles = sampler.sample_tile(tile)
    assert np.isnan(profiles.profiles).any()
    center = profiles.profiles[:, profiles.offsets.size // 2]
    assert np.isfinite(center).all()
