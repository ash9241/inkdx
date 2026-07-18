import json

import numpy as np
import pytest
import tifffile
import zarr
from typer.testing import CliRunner

from inkdx.cli import app
from inkdx.label3d.depth import (
    STATUS_INSUFFICIENT,
    STATUS_LOCALIZED,
    STATUS_NO_SIGNAL,
    Label3dConfig,
    estimate_depth,
)
from inkdx.label3d.emit import emit_labels
from inkdx.testing.synthetic import PhantomParams, make_phantom

runner = CliRunner()

INK_REGIONS = ((16, 16, 80, 112),)
CFG = Label3dConfig(halfwidth=14, min_ink_px=100, bootstrap=100, block=16,
                    tile_px=48, max_pixels=20_000)


def ink_phantom(contrast: float, band=(-1.0, 3.0), noise=2.0):
    return make_phantom(PhantomParams(
        undulation_amp=0.0, noise_sigma=noise,
        ink_band=band, ink_contrast=contrast, ink_uv_regions=INK_REGIONS,
    ))


def band_matches(band, true_band, tol=1.5) -> bool:
    """Sign-agnostic: the estimator's band lives in profile convention, which
    may be the mirror of the phantom's +z convention."""
    a, b = band
    ta, tb = true_band
    direct = abs(a - ta) <= tol and abs(b - tb) <= tol
    mirror = abs(-b - ta) <= tol and abs(-a - tb) <= tol
    return direct or mirror


@pytest.mark.parametrize("contrast", [-40.0, 40.0])
def test_recovers_ink_band_both_signs(contrast):
    ph = ink_phantom(contrast)
    r = estimate_depth(ph.volume, ph.segment, ph.ink_mask, CFG)
    assert r.status == STATUS_LOCALIZED, (r.status, r.p_value)
    assert band_matches(r.band, (-1.0, 3.0)), r.band
    assert np.sign(r.delta_peak) == np.sign(contrast)


def test_no_signal_when_contrast_vanishes():
    # sweep down: significance must fail before the band estimate goes wrong
    ph = ink_phantom(-1.0)  # far below noise 2.0 at block level
    r = estimate_depth(ph.volume, ph.segment, ph.ink_mask, CFG)
    assert r.status == STATUS_NO_SIGNAL
    assert r.upper_bound is not None and r.upper_bound < 10.0


def test_insufficient_labels():
    ph = ink_phantom(-40.0)
    tiny = np.zeros_like(ph.ink_mask)
    tiny[40:43, 40:43] = True
    r = estimate_depth(ph.volume, ph.segment, tiny, CFG)
    assert r.status == STATUS_INSUFFICIENT


def test_emit_codes_and_sidecars(tmp_path):
    ph = ink_phantom(-40.0)
    out = tmp_path / "labels.zarr"
    emit_labels(out, nz=64, z_center=32.0, ink_mask=ph.ink_mask,
                valid=np.ones_like(ph.ink_mask), band=(-1.0, 3.0),
                bg_distance=8.0)
    arr = zarr.open(str(out), mode="r")["0"]
    assert arr.shape == (64, 96, 128)
    assert arr.dtype == np.uint8

    layer_in_band = arr[33]  # r = +1, inside (-1, 3)
    assert (layer_in_band[ph.ink_mask] == 1).all()
    assert (layer_in_band[~ph.ink_mask] == 0).all()  # bg within +/-8

    layer_below = arr[38]  # r = +6: outside band, inside bg_distance
    assert (layer_below[ph.ink_mask] == 2).all()  # ignore under strokes
    assert (layer_below[~ph.ink_mask] == 0).all()

    layer_far = arr[50]  # r = +18: outside everything
    assert (layer_far == 2).all()

    params = json.loads((out / "3d_ink_params.json").read_text())
    assert params["labels"]["ink"] == "1"
    remap = json.loads((out / "remap.json").read_text())
    assert remap["mappings"]["1"] == "ink"


def test_label3d_cli_end_to_end(tmp_path):
    ph = ink_phantom(-40.0)
    layers = tmp_path / "layers"
    layers.mkdir()
    for k in range(ph.volume.shape[0]):
        tifffile.imwrite(layers / f"{k:02}.tif", ph.volume[k])
    tifffile.imwrite(tmp_path / "ink.tif", (ph.ink_mask * 255).astype(np.uint8))

    res = runner.invoke(app, [
        "label3d", "--volume", str(layers), "--labels", str(tmp_path / "ink.tif"),
        "--out", str(tmp_path / "l3d.zarr"), "--halfwidth", "14",
        "--min-ink-px", "100", "--bootstrap", "100", "--tile", "48",
    ])
    assert res.exit_code == 0, res.output
    assert "INK_DEPTH_LOCALIZED" in res.output

    rep = json.loads((tmp_path / "l3d_report.json").read_text()) if (
        tmp_path / "l3d_report.json").exists() else json.loads(
        (tmp_path / "l3d_report" / "label3d_report.json").read_text())
    assert rep["status"] == "INK_DEPTH_LOCALIZED"
    # band_used is in z-layer convention: must match the true +z ink band
    assert rep["band_used"][0] == pytest.approx(-1.0, abs=1.5)
    assert rep["band_used"][1] == pytest.approx(3.0, abs=1.5)
    assert (tmp_path / "l3d_report" / "delta_r.png").exists()
    assert zarr.open(str(tmp_path / "l3d.zarr"), mode="r")["0"].shape[0] == 64
