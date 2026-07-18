import numpy as np
import pytest

from inkdx.grid import TileGrid
from inkdx.sampling import NormalProfileSampler
from inkdx.stages.scan import compute_scan_metrics, intensity_drift
from inkdx.testing.synthetic import PhantomParams, make_phantom


def scan_on_phantom(params: PhantomParams, halfwidth: int = 20) -> dict:
    ph = make_phantom(params)
    sampler = NormalProfileSampler(ph.volume, ph.segment, halfwidth=halfwidth)
    tile = TileGrid(ph.segment.grid_shape, tile_px=96).tile(0, 0)
    return compute_scan_metrics(sampler.sample_tile(tile))


def test_noise_sigma_recovers_truth():
    for true_sigma in (3.0, 6.0):
        m = scan_on_phantom(PhantomParams(noise_sigma=true_sigma))
        assert m["noise_sigma"] == pytest.approx(true_sigma, rel=0.35)


def test_cnr_and_snr_decrease_with_noise():
    sigmas = (2.0, 5.0, 10.0, 20.0)
    results = [scan_on_phantom(PhantomParams(noise_sigma=s)) for s in sigmas]
    cnrs = [m["cnr"] for m in results]
    snrs = [m["snr"] for m in results]
    assert all(np.isfinite(cnrs))
    assert cnrs == sorted(cnrs, reverse=True)
    assert snrs == sorted(snrs, reverse=True)


def test_cnr_increases_with_contrast():
    lo = scan_on_phantom(PhantomParams(sheet_contrast=40.0, noise_sigma=4.0))
    hi = scan_on_phantom(PhantomParams(sheet_contrast=160.0, noise_sigma=4.0))
    assert hi["cnr"] > 2.0 * lo["cnr"]


def test_haze_increases_with_blur():
    # Blurrier sheets widen the apparent profile peak.
    widths = (2.5, 5.0, 8.0)
    hazes = [
        scan_on_phantom(PhantomParams(sheet_sigma=w, noise_sigma=3.0))["haze_index"]
        for w in widths
    ]
    assert all(np.isfinite(hazes))
    assert hazes == sorted(hazes)
    assert hazes[-1] > 2.0 * hazes[0]  # FWHM scales with sheet sigma


def test_saturation_detected():
    clean = scan_on_phantom(PhantomParams(sheet_contrast=120.0, noise_sigma=3.0))
    clipped = scan_on_phantom(PhantomParams(sheet_contrast=400.0, noise_sigma=3.0))
    assert clipped["saturation_frac"] > 10.0 * max(clean["saturation_frac"], 1e-4)


def test_dynamic_range_sane():
    m = scan_on_phantom(PhantomParams(noise_sigma=0.5))
    p = PhantomParams()
    assert 0.5 * p.sheet_contrast < m["dynamic_range"] <= 255.0


def test_empty_tile_gives_nans():
    ph = make_phantom(PhantomParams())
    seg = ph.segment
    seg.valid[:] = False
    sampler = NormalProfileSampler(ph.volume, seg)
    tile = TileGrid(seg.grid_shape, tile_px=96).tile(0, 0)
    m = compute_scan_metrics(sampler.sample_tile(tile))
    assert all(np.isnan(v) for v in m.values())


def test_intensity_drift_flags_outlier_tile():
    m = np.full((4, 4), 100.0, dtype=np.float32)
    m[2, 2] = 160.0
    drift = intensity_drift(m)
    assert abs(drift[2, 2]) > 5.0
    assert abs(drift[0, 0]) < 1.0
