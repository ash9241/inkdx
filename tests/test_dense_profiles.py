import numpy as np
import pytest

from inkdx.grid import TileGrid
from inkdx.sampling import dense_tile_profiles, profiles_for_indices
from inkdx.stages.profile_features import analyze_profile, analyze_profiles_dense
from inkdx.testing.synthetic import PhantomParams, make_phantom


@pytest.fixture(scope="module")
def phantom():
    return make_phantom(PhantomParams(undulation_amp=2.0, noise_sigma=3.0))


def test_dense_block_covers_all_valid_vertices(phantom):
    ph = phantom
    tile = TileGrid(ph.segment.grid_shape, tile_px=48).tile(0, 1)
    d = dense_tile_profiles(ph.volume, ph.segment, tile, halfwidth=12)
    assert d.block.shape == (48, 48, 25)
    # interior vertices all valid; profiles finite in the center
    assert d.valid[2:-2, 2:-2].all()
    center = d.block[..., 12]
    assert np.isfinite(center[d.valid]).all()


def test_dense_stride(phantom):
    ph = phantom
    tile = TileGrid(ph.segment.grid_shape, tile_px=48).tile(0, 0)
    d = dense_tile_profiles(ph.volume, ph.segment, tile, halfwidth=8, stride=4)
    assert d.block.shape == (12, 12, 17)
    assert d.stride == 4


def test_dense_features_agree_with_scalar(phantom):
    """The CI contract: vectorized features match the scalar path per vertex."""
    ph = phantom
    tile = TileGrid(ph.segment.grid_shape, tile_px=48).tile(0, 0)
    d = dense_tile_profiles(ph.volume, ph.segment, tile, halfwidth=12)
    feats = analyze_profiles_dense(d.block, d.offsets)

    rng = np.random.default_rng(0)
    rr, cc = np.nonzero(d.valid & np.isfinite(feats["r_star"]))
    pick = rng.choice(rr.size, size=min(40, rr.size), replace=False)
    for i, j in zip(rr[pick], cc[pick], strict=True):
        scalar = analyze_profile(d.block[i, j].astype(np.float64), d.offsets)
        assert feats["r_star"][i, j] == pytest.approx(scalar.r_star, abs=1e-6)
        assert feats["peak_value"][i, j] == pytest.approx(scalar.peak_value, rel=1e-5)
        assert feats["gap_value"][i, j] == pytest.approx(scalar.gap_value, rel=1e-4)
        assert feats["prominence"][i, j] == pytest.approx(scalar.prominence, rel=1e-4)
        # multiplicity may differ by at most the plateau-edge convention
        assert abs(int(feats["multiplicity"][i, j]) - scalar.multiplicity) <= 1


def test_subvoxel_refinement_on_analytic_gaussian():
    offsets = np.arange(-10, 11, dtype=np.float32)
    true_centers = [-2.3, 0.0, 1.75]
    block = np.stack(
        [[120.0 * np.exp(-((offsets - c) ** 2) / (2 * 2.5**2)) + 30.0
          for c in true_centers]]
    ).astype(np.float32)  # (1, 3, P)
    feats = analyze_profiles_dense(block, offsets)
    for j, c in enumerate(true_centers):
        assert feats["r_star_subvox"][0, j] == pytest.approx(c, abs=0.15)


def test_profiles_for_indices_matches_dense(phantom):
    ph = phantom
    tile = TileGrid(ph.segment.grid_shape, tile_px=48).tile(0, 0)
    d = dense_tile_profiles(ph.volume, ph.segment, tile, halfwidth=10)

    rr, cc = np.nonzero(d.valid)
    pick = np.random.default_rng(1).choice(rr.size, size=25, replace=False)
    gr, gc = rr[pick], cc[pick]  # tile 0,0 => grid coords == tile coords
    profiles, offsets = profiles_for_indices(
        ph.volume, ph.segment, gr, gc, halfwidth=10, tile_px=48
    )
    np.testing.assert_allclose(profiles, d.block[gr, gc], atol=1e-5)


def test_profiles_for_indices_spanning_tiles_and_invalid(phantom):
    ph = phantom
    seg = ph.segment
    seg2 = type(seg)(x=seg.x, y=seg.y, z=seg.z, valid=seg.valid.copy(),
                     scale=seg.scale, uuid="t")
    seg2.valid[10, 10] = False
    gr = np.array([10, 50, 90, 5])
    gc = np.array([10, 60, 120, 5])
    profiles, _ = profiles_for_indices(ph.volume, seg2, gr, gc, halfwidth=8, tile_px=48)
    assert np.isnan(profiles[0]).all()  # invalidated vertex
    assert np.isfinite(profiles[1:, 8]).all()  # others fine at r=0


def test_phantom_ink_band():
    p = PhantomParams(
        undulation_amp=0.0, noise_sigma=0.0,
        ink_band=(-1.0, 3.0), ink_contrast=-40.0,
        ink_uv_regions=((20, 30, 60, 90),),
    )
    ph = make_phantom(p)
    assert ph.ink_mask[30, 50] and not ph.ink_mask[5, 5]
    z0 = int(p.sheet_z0)
    # inside band, ink region darker than non-ink region at same depth
    ink_val = float(ph.volume[z0 + 1, 40, 60])
    clean_val = float(ph.volume[z0 + 1, 80, 60])
    assert clean_val - ink_val >= 25.0
    # outside band (below sheet), equal
    assert abs(float(ph.volume[z0 - 6, 40, 60]) - float(ph.volume[z0 - 6, 80, 60])) <= 1.0
