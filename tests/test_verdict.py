"""Attribution matrix on the phantom: each ablation must flip tiles to its own
stage's verdict — the miniature of the w00 killer demo."""

import numpy as np
import pytest

from inkdx.ablate.mesh import offset_z
from inkdx.ablate.noise import add_noise, blur
from inkdx.calibration import CalibrationPack
from inkdx.runner import DiagnosticsConfig, run_diagnostics
from inkdx.testing.synthetic import PhantomParams, make_phantom
from inkdx.verdict import (
    VERDICT_ID,
    assign_verdicts,
    stage_scores,
    verdict_fractions,
)

CFG = DiagnosticsConfig(tile_px=32, halfwidth=16, samples_per_tile=96)


@pytest.fixture(scope="module")
def healthy():
    ph = make_phantom(PhantomParams(undulation_amp=0.0, noise_sigma=3.0))
    maps = run_diagnostics(ph.volume, ph.segment, CFG)
    # synthetic "model saw text" metrics so the model gate participates
    rng = np.random.default_rng(0)
    shape = maps["cnr"].shape
    maps["indecision_mass"] = rng.uniform(0.01, 0.05, shape).astype(np.float32)
    maps["prob_separation"] = rng.uniform(0.7, 0.9, shape).astype(np.float32)
    maps["entropy"] = rng.uniform(0.1, 0.3, shape).astype(np.float32)
    maps["ink_frac"] = rng.uniform(0.1, 0.3, shape).astype(np.float32)
    pack = CalibrationPack.fit(maps, name="phantom-healthy")
    return ph, maps, pack


def with_model(maps, healthy_maps):
    """Carry the healthy synthetic model metrics into an ablated run."""
    out = dict(maps)
    for k in ("indecision_mass", "prob_separation", "entropy", "ink_frac"):
        out[k] = healthy_maps[k]
    return out


def test_healthy_is_ink_ok(healthy):
    _, maps, pack = healthy
    v = assign_verdicts(maps, pack)
    frac = verdict_fractions(v["verdict"])
    assert frac["INK_OK"] > 0.95
    assert np.nanmedian(v["score_scan"]) > 0.4


def test_noise_ablation_attributes_to_scan(healthy):
    ph, hmaps, pack = healthy
    noisy = add_noise(ph.volume, 25.0, seed=1)
    maps = with_model(run_diagnostics(noisy, ph.segment, CFG), hmaps)
    v = assign_verdicts(maps, pack)
    frac = verdict_fractions(v["verdict"])
    assert frac["SCAN_SUSPECT"] > 0.8, frac
    assert frac["SURFACE_SUSPECT"] < 0.1, frac


def test_blur_ablation_attributes_to_scan(healthy):
    ph, hmaps, pack = healthy
    blurred = blur(ph.volume, 5.0)
    maps = with_model(run_diagnostics(blurred, ph.segment, CFG), hmaps)
    v = assign_verdicts(maps, pack)
    frac = verdict_fractions(v["verdict"])
    assert frac["SCAN_SUSPECT"] > 0.5, frac


def test_mesh_offset_attributes_to_surface(healthy):
    ph, hmaps, pack = healthy
    seg = offset_z(ph.segment, 8.0)
    maps = with_model(run_diagnostics(ph.volume, seg, CFG), hmaps)
    v = assign_verdicts(maps, pack)
    frac = verdict_fractions(v["verdict"])
    assert frac["SURFACE_SUSPECT"] > 0.8, frac
    assert frac["SCAN_SUSPECT"] < 0.1, frac


def test_confused_model_attributes_to_model(healthy):
    ph, hmaps, pack = healthy
    maps = dict(run_diagnostics(ph.volume, ph.segment, CFG))
    rng = np.random.default_rng(2)
    shape = maps["cnr"].shape
    maps["indecision_mass"] = rng.uniform(0.7, 0.9, shape).astype(np.float32)
    maps["prob_separation"] = rng.uniform(0.05, 0.15, shape).astype(np.float32)
    maps["entropy"] = rng.uniform(0.9, 1.0, shape).astype(np.float32)
    maps["ink_frac"] = hmaps["ink_frac"]
    v = assign_verdicts(maps, pack)
    frac = verdict_fractions(v["verdict"])
    assert frac["MODEL_SUSPECT"] > 0.8, frac
    assert frac["SCAN_SUSPECT"] + frac["SURFACE_SUSPECT"] < 0.1, frac


def test_blank_but_healthy_is_no_ink_evidence(healthy):
    ph, hmaps, pack = healthy
    maps = dict(run_diagnostics(ph.volume, ph.segment, CFG))
    shape = maps["cnr"].shape
    maps["indecision_mass"] = np.full(shape, 0.005, dtype=np.float32)
    maps["prob_separation"] = hmaps["prob_separation"]
    maps["entropy"] = np.full(shape, 0.05, dtype=np.float32)
    maps["ink_frac"] = np.full(shape, 0.001, dtype=np.float32)
    v = assign_verdicts(maps, pack)
    frac = verdict_fractions(v["verdict"])
    assert frac["NO_INK_EVIDENCE"] > 0.9, frac


def test_holes_are_no_data(healthy):
    ph, hmaps, pack = healthy
    seg = offset_z(ph.segment, 0.0)
    seg.valid[:, :32] = False  # kill the first tile column
    maps = with_model(run_diagnostics(ph.volume, seg, CFG), hmaps)
    v = assign_verdicts(maps, pack)
    assert (v["verdict"][:, 0] == VERDICT_ID["NO_DATA"]).all()


def test_pack_roundtrip(tmp_path, healthy):
    _, maps, pack = healthy
    p = pack.save(tmp_path / "pack.json")
    loaded = CalibrationPack.load(p)
    assert loaded.stats == pack.stats
    s1 = stage_scores(maps, pack)
    s2 = stage_scores(maps, loaded)
    for k in s1:
        np.testing.assert_array_equal(s1[k], s2[k])
