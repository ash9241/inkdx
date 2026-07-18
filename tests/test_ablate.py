import numpy as np
import tifffile

from inkdx.ablate.mesh import offset_z, tilt, wobble
from inkdx.ablate.noise import ablate_layer_stack, add_noise, blur
from inkdx.grid import TileGrid
from inkdx.sampling import NormalProfileSampler
from inkdx.stages.scan import compute_scan_metrics
from inkdx.stages.surface import compute_surface_profile_metrics
from inkdx.testing.synthetic import PhantomParams, make_phantom


def test_noise_ablation_degrades_cnr_only():
    ph = make_phantom(PhantomParams(undulation_amp=0.0, noise_sigma=2.0))
    noisy = add_noise(ph.volume, 15.0, seed=1)

    tile = TileGrid(ph.segment.grid_shape, tile_px=64).tile(0, 0)
    clean_m = compute_scan_metrics(
        NormalProfileSampler(ph.volume, ph.segment, halfwidth=16).sample_tile(tile)
    )
    noisy_p = NormalProfileSampler(noisy, ph.segment, halfwidth=16).sample_tile(tile)
    noisy_m = compute_scan_metrics(noisy_p)
    assert noisy_m["cnr"] < 0.4 * clean_m["cnr"]
    # surface still findable: peak where the mesh is
    surf = compute_surface_profile_metrics(noisy_p)
    assert abs(surf["peak_offset"]) <= 1.0


def test_blur_ablation_raises_haze():
    ph = make_phantom(PhantomParams(undulation_amp=0.0, noise_sigma=2.0))
    blurred = blur(ph.volume, 5.0)
    tile = TileGrid(ph.segment.grid_shape, tile_px=64).tile(0, 0)
    m0 = compute_scan_metrics(
        NormalProfileSampler(ph.volume, ph.segment, halfwidth=16).sample_tile(tile)
    )
    m1 = compute_scan_metrics(
        NormalProfileSampler(blurred, ph.segment, halfwidth=16).sample_tile(tile)
    )
    assert m1["haze_index"] > 1.4 * m0["haze_index"]


def test_mesh_ablations_move_peak():
    ph = make_phantom(PhantomParams(undulation_amp=0.0, noise_sigma=2.0))
    tile = TileGrid(ph.segment.grid_shape, tile_px=64).tile(0, 0)

    def peak(seg):
        p = NormalProfileSampler(ph.volume, seg, halfwidth=16).sample_tile(tile)
        return compute_surface_profile_metrics(p)["peak_offset"]

    assert abs(abs(peak(offset_z(ph.segment, 6.0))) - 6.0) <= 0.5
    assert abs(peak(ph.segment)) <= 0.5

    tilted = tilt(ph.segment, 0.15)  # +/- 9.6 voxels across 128 cols
    p = NormalProfileSampler(ph.volume, tilted, halfwidth=16).sample_tile(
        TileGrid(ph.segment.grid_shape, tile_px=32).tile(0, 0)
    )
    m = compute_surface_profile_metrics(p)
    assert abs(m["peak_offset"]) >= 2.0  # left edge of the ramp is off-sheet

    wob = wobble(ph.segment, 5.0, 32.0, seed=2)
    assert np.abs(wob.z - ph.segment.z).max() >= 4.0


def test_regional_offset_only_affects_region():
    ph = make_phantom(PhantomParams(undulation_amp=0.0, noise_sigma=2.0))
    seg = offset_z(ph.segment, 8.0, region=(0, 64, 96, 128))
    grid = TileGrid(ph.segment.grid_shape, tile_px=64)

    def peak(t):
        p = NormalProfileSampler(ph.volume, seg, halfwidth=16).sample_tile(t)
        return compute_surface_profile_metrics(p)["peak_offset"]

    assert abs(peak(grid.tile(0, 0))) <= 0.5  # untouched half
    assert abs(abs(peak(grid.tile(0, 1))) - 8.0) <= 0.5  # ablated half


def test_ablate_layer_stack_roundtrip(tmp_path):
    ph = make_phantom(PhantomParams(noise_sigma=2.0))
    src = tmp_path / "layers"
    src.mkdir()
    for k in range(ph.volume.shape[0]):
        tifffile.imwrite(src / f"{k:02}.tif", ph.volume[k])

    out = ablate_layer_stack(src, tmp_path / "noisy", noise_sigma=10.0,
                             region=(10, 20, 60, 100), seed=3)
    stack = [tifffile.imread(p) for p in sorted(out.glob("*.tif"))]
    assert len(stack) == ph.volume.shape[0]
    assert stack[0].shape == (50, 80)
    clean_window = ph.volume[:, 10:60, 20:100].astype(np.float32)
    assert np.abs(np.stack(stack).astype(np.float32) - clean_window).mean() > 5.0
