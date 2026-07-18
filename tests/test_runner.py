import numpy as np
import tifffile

from inkdx.io.volume import LayerStackVolume, identity_segment
from inkdx.runner import ALL_METRICS, DiagnosticsConfig, run_diagnostics
from inkdx.testing.synthetic import PhantomParams, make_phantom


def test_run_diagnostics_on_phantom():
    ph = make_phantom(PhantomParams(noise_sigma=3.0))
    cfg = DiagnosticsConfig(tile_px=48, halfwidth=16, samples_per_tile=64)
    maps = run_diagnostics(ph.volume, ph.segment, cfg)

    assert set(maps) == set(ALL_METRICS)
    shape = maps["cnr"].shape
    assert shape == (2, 3)  # 96x128 grid, 48px tiles
    for key in ("cnr", "peak_offset", "peak_prominence", "grid_tearing", "hole_fraction"):
        assert np.isfinite(maps[key]).all(), key
    assert np.abs(maps["peak_offset"]).max() <= 1.0
    assert (maps["n_points"] == 64).all()


def test_run_diagnostics_parallel_matches_sequential():
    ph = make_phantom(PhantomParams(noise_sigma=3.0))
    seq = run_diagnostics(ph.volume, ph.segment, DiagnosticsConfig(tile_px=48, halfwidth=12))
    par = run_diagnostics(
        ph.volume, ph.segment, DiagnosticsConfig(tile_px=48, halfwidth=12, processes=2)
    )
    for k in ALL_METRICS:
        np.testing.assert_array_equal(seq[k], par[k], err_msg=k)


def test_layer_stack_volume(tmp_path):
    ph = make_phantom(PhantomParams(noise_sigma=2.0))
    layers = tmp_path / "layers"
    layers.mkdir()
    for k in range(ph.volume.shape[0]):
        tifffile.imwrite(layers / f"{k:02}.tif", ph.volume[k])

    stack = LayerStackVolume(layers)
    assert stack.shape == ph.volume.shape
    np.testing.assert_array_equal(stack[3:7, 10:20, 30:40], ph.volume[3:7, 10:20, 30:40])
    np.testing.assert_array_equal(stack[5, :, :], ph.volume[5])


def test_identity_segment_profiles_read_down_the_stack(tmp_path):
    # Surface-volume semantics: profile at (r, c) is the stack column there.
    ph = make_phantom(PhantomParams(undulation_amp=0.0, noise_sigma=0.0))
    layers = tmp_path / "layers"
    layers.mkdir()
    for k in range(ph.volume.shape[0]):
        tifffile.imwrite(layers / f"{k:02}.tif", ph.volume[k])

    stack = LayerStackVolume(layers)
    z_c = float(ph.params.sheet_z0)  # mesh exactly on the sheet layer
    seg = identity_segment(stack.shape[1], stack.shape[2], z_center=z_c)

    maps = run_diagnostics(stack, seg, DiagnosticsConfig(tile_px=48, halfwidth=12))
    assert np.abs(maps["peak_offset"]).max() <= 1.0
    assert np.nanmin(maps["peak_prominence"]) > 10.0
