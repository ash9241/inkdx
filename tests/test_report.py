import json

import pytest

from inkdx.ablate.mesh import offset_z
from inkdx.calibration import CalibrationPack
from inkdx.report.json_report import write_report
from inkdx.report.schema import GridInfo
from inkdx.runner import DiagnosticsConfig, run_diagnostics
from inkdx.testing.synthetic import PhantomParams, make_phantom
from inkdx.verdict import assign_verdicts

CFG = DiagnosticsConfig(tile_px=32, halfwidth=16, samples_per_tile=96)


@pytest.fixture(scope="module")
def surface_failure_report(tmp_path_factory):
    ph = make_phantom(PhantomParams(undulation_amp=0.0, noise_sigma=3.0))
    healthy = run_diagnostics(ph.volume, ph.segment, CFG)
    pack = CalibrationPack.fit(healthy, name="phantom")

    # regional surface failure: right half of the mesh off-sheet
    seg = offset_z(ph.segment, 9.0, region=(0, 64, 96, 128))
    maps = run_diagnostics(ph.volume, seg, CFG)
    verdicts = assign_verdicts(maps, pack)

    out = tmp_path_factory.mktemp("report")
    path = write_report(
        out,
        maps=maps,
        verdicts=verdicts,
        pack=pack,
        grid_info=GridInfo(tile_px=32, n_tiles=maps["cnr"].shape,
                           samples_per_tile=96, halfwidth=16),
        inputs={"volume": {"kind": "phantom"}},
    )
    return json.loads(path.read_text()), out


def test_report_summary(surface_failure_report):
    report, _ = surface_failure_report
    assert report["schema_version"] == 1
    assert report["summary"]["dominant_failure"] == "SURFACE_SUSPECT"
    assert "surface" in report["summary"]["headline"]
    frac = report["summary"]["verdict_fractions"]
    assert 0.3 < frac["SURFACE_SUSPECT"] < 0.7  # right half of tiles
    assert frac["SCAN_SUSPECT"] < 0.1


def test_report_regions_locate_failure(surface_failure_report):
    report, _ = surface_failure_report
    regions = report["summary"]["regions"]
    assert regions, "expected at least one region"
    top = regions[0]
    assert top["verdict"] == "SURFACE_SUSPECT"
    assert top["uv_bbox"][1] >= 2  # failure is in the right tile-columns
    assert "peak_offset" in top["explanation"]


def test_report_sidecars_written(surface_failure_report):
    report, out = surface_failure_report
    import tifffile

    for key in ("verdict", "cnr", "peak_offset"):
        p = out / report["maps"][key]
        assert p.exists()
        assert tifffile.imread(p).shape == tuple(report["grid"]["n_tiles"])


def test_tile_table_shapes(surface_failure_report):
    report, _ = surface_failure_report
    n = report["grid"]["n_tiles"][0] * report["grid"]["n_tiles"][1]
    assert len(report["tiles"]["verdict"]) == n
    assert len(report["tiles"]["metrics"]["cnr"]) == n
    assert set(report["tiles"]["scores"]) >= {"scan", "surface"}


def test_nan_tiles_serialize_as_null(tmp_path):
    """NO_DATA tiles have NaN scores/metrics — they must serialize as null."""
    import numpy as np

    from inkdx.verdict import VERDICT_ID

    maps = {"cnr": np.array([[np.nan, 20.0]], dtype=np.float32),
            "hole_fraction": np.array([[1.0, 0.0]], dtype=np.float32)}
    pack = CalibrationPack(name="t", stats={"cnr": {"median": 20.0, "mad": 2.0}})
    verdicts = {
        "verdict": np.array([[VERDICT_ID["NO_DATA"], VERDICT_ID["INK_OK"]]], dtype=np.int8),
        "confidence": np.array([[np.nan, 0.9]], dtype=np.float32),
        "score_scan": np.array([[np.nan, 0.5]], dtype=np.float32),
    }
    path = write_report(tmp_path, maps=maps, verdicts=verdicts, pack=pack,
                        grid_info=GridInfo(tile_px=256, n_tiles=(1, 2),
                                           samples_per_tile=1, halfwidth=1))
    rep = json.loads(path.read_text())
    assert rep["tiles"]["metrics"]["cnr"] == [None, 20.0]
    assert rep["tiles"]["scores"]["scan"][0] is None
