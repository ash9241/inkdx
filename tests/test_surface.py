import numpy as np
import pytest

from inkdx.grid import TileGrid
from inkdx.io.segment import Segment
from inkdx.sampling import NormalProfileSampler
from inkdx.stages.surface import (
    com_smoothness,
    compute_surface_geometry_metrics,
    compute_surface_profile_metrics,
    geometry_maps,
    hole_localization,
    stretch_anomaly,
)
from inkdx.testing.synthetic import PhantomParams, make_phantom


@pytest.fixture(scope="module")
def flat_phantom():
    return make_phantom(PhantomParams(undulation_amp=0.0, noise_sigma=3.0))


def shifted(seg: Segment, dz: float) -> Segment:
    return Segment(x=seg.x, y=seg.y, z=seg.z + dz, valid=seg.valid.copy(),
                   scale=seg.scale, uuid=f"dz{dz}")


def profile_metrics(ph, seg, halfwidth=16):
    sampler = NormalProfileSampler(ph.volume, seg, halfwidth=halfwidth)
    tile = TileGrid(seg.grid_shape, tile_px=64).tile(0, 0)
    return compute_surface_profile_metrics(sampler.sample_tile(tile))


def test_on_sheet_mesh(flat_phantom):
    m = profile_metrics(flat_phantom, flat_phantom.segment)
    assert abs(m["peak_offset"]) <= 1.0
    assert m["peak_prominence"] > 50.0  # sheet obviously there
    assert m["peak_multiplicity"] == 1.0
    assert abs(m["com_offset"]) <= 1.0


def test_offset_mesh_detected(flat_phantom):
    m = profile_metrics(flat_phantom, shifted(flat_phantom.segment, 8.0))
    assert abs(abs(m["peak_offset"]) - 8.0) <= 0.5


def test_second_sheet_raises_multiplicity():
    ph = make_phantom(
        PhantomParams(undulation_amp=0.0, noise_sigma=2.0, second_sheet_dz=10.0)
    )
    m = profile_metrics(ph, ph.segment, halfwidth=16)
    assert m["peak_multiplicity"] >= 2.0


def test_far_off_mesh_has_low_prominence(flat_phantom):
    # Mesh 25 voxels off with a +/-10 window: no sheet in sight.
    on = profile_metrics(flat_phantom, flat_phantom.segment, halfwidth=10)
    off = profile_metrics(flat_phantom, shifted(flat_phantom.segment, 25.0), halfwidth=10)
    assert off["peak_prominence"] < 0.1 * on["peak_prominence"]


def test_spliced_grid_tearing(flat_phantom):
    seg = flat_phantom.segment
    torn = shifted(seg, 0.0)
    torn.z[:, 48:] += 10.0  # splice: half the grid jumps 10 voxels
    grid = TileGrid(seg.grid_shape, tile_px=32)
    maps = geometry_maps(torn, grid)
    tearing = maps["grid_tearing"]
    assert np.nanmax(tearing[:, 1]) > 8.0  # tiles containing the splice
    assert np.nanmax(tearing[:, 0]) < 3.0  # clean tiles


def test_normal_coherence_drops_on_crumpled_mesh(flat_phantom):
    seg = flat_phantom.segment
    rng = np.random.default_rng(0)
    crumpled = shifted(seg, 0.0)
    crumpled.z += rng.normal(0.0, 2.0, size=seg.z.shape).astype(np.float32)
    tile = TileGrid(seg.grid_shape, tile_px=64).tile(0, 0)
    smooth = compute_surface_geometry_metrics(seg, tile)["normal_coherence"]
    rough = compute_surface_geometry_metrics(crumpled, tile)["normal_coherence"]
    assert smooth > 0.99
    assert rough < 0.7


def test_hole_fraction_and_localization(flat_phantom):
    seg = flat_phantom.segment
    holey = shifted(seg, 0.0)
    holey.valid[30:40, 30:45] = False  # interior hole
    tile = TileGrid(seg.grid_shape, tile_px=64).tile(0, 0)
    m = compute_surface_geometry_metrics(holey, tile)
    assert m["hole_fraction"] > 0.03

    holes = hole_localization(holey)
    assert len(holes) == 1
    assert holes[0]["area"] == 150
    assert holes[0]["uv_bbox"] == [30, 30, 40, 45]


def test_stretch_anomaly_flags_stretched_region(flat_phantom):
    # Stretch a minority region (last quarter) so the segment median stays a
    # sane reference — the realistic failure shape.
    seg = flat_phantom.segment
    stretched = shifted(seg, 0.0)
    stretched.x = seg.x.copy()
    stretched.x[:, 96:] = seg.x[:, 96:] * 2.0 - seg.x[:, 96:97]  # 2x column spacing
    grid = TileGrid(seg.grid_shape, tile_px=32)
    maps = geometry_maps(stretched, grid)
    anomaly = stretch_anomaly(maps["step_u"], maps["step_v"])
    assert np.nanmax(anomaly[:, 3]) > 0.8  # ~2x stretch = |log2| ~ 1
    assert np.nanmax(anomaly[:, 0]) < 0.3


def test_com_smoothness_separates_drift_from_noise():
    smooth_field = np.tile(np.linspace(0, 5, 8), (8, 1)).astype(np.float32)
    rng = np.random.default_rng(1)
    rough_field = rng.normal(0, 2, (8, 8)).astype(np.float32)
    assert np.nanmean(com_smoothness(smooth_field)) < 0.5
    assert np.nanmean(com_smoothness(rough_field)) > 1.0
