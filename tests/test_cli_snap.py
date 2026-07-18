import json

import numpy as np
import tifffile
from typer.testing import CliRunner

from inkdx.cli import app
from inkdx.io.segment import read_tifxyz, write_tifxyz
from inkdx.testing.synthetic import PhantomParams, make_phantom

runner = CliRunner()


def make_layers(tmp_path, ph):
    layers = tmp_path / "layers"
    layers.mkdir()
    for k in range(ph.volume.shape[0]):
        tifffile.imwrite(layers / f"{k:02}.tif", ph.volume[k])
    return layers


def test_snap_cli_mesh_mode(tmp_path):
    ph = make_phantom(PhantomParams(undulation_amp=0.0, noise_sigma=2.0))
    layers = make_layers(tmp_path, ph)
    seg = ph.segment
    # write an offset mesh as tifxyz, then snap it back via the CLI
    write_tifxyz(tmp_path / "seg", seg.x, seg.y, seg.z + 4.0,
                 valid=seg.valid, scale=seg.scale, uuid="offset4")

    res = runner.invoke(app, [
        "snap", "--volume", str(layers), "--segment", str(tmp_path / "seg"),
        "--out", str(tmp_path / "snapped"),
        "--halfwidth", "14", "--tile", "48", "--iterations", "4",
        "--max-step", "3", "--smooth", "2", "--no-receipt",
    ])
    assert res.exit_code == 0, res.output

    snapped = read_tifxyz(tmp_path / "snapped")
    err = np.abs(snapped.z[snapped.valid] - ph.sheet_z[snapped.valid])
    assert float(np.median(err)) < 0.6

    # provenance convention established
    assert snapped.meta["parent_uuid"] == "offset4"
    assert snapped.meta["snap"]["tool"] == "inkdx"
    assert snapped.meta["snap"]["iterations_run"] >= 1

    # QA maps + report written
    rep_dir = tmp_path / "snapped_report"
    rep = json.loads((rep_dir / "snap_report.json").read_text())
    assert rep["final_status_fracs"]["SNAPPED"] > 0.5
    assert (rep_dir / "maps" / "snap_offset.tif").exists()


def test_snap_cli_identity_mode_with_receipt(tmp_path):
    ph = make_phantom(PhantomParams(undulation_amp=0.0, noise_sigma=2.0))
    layers = make_layers(tmp_path, ph)

    res = runner.invoke(app, [
        "snap", "--volume", str(layers), "--out", str(tmp_path / "rep"),
        "--halfwidth", "14", "--tile", "48", "--iterations", "2",
        "--max-step", "3", "--smooth", "2",
    ])
    assert res.exit_code == 0, res.output
    rep = json.loads((tmp_path / "rep" / "snap_report.json").read_text())
    assert "before_after" in rep
    po = rep["before_after"]["peak_offset"]
    # identity mesh at stack center: sheet is at z0=32 == center, so
    # before is already near 0; after must not be worse
    assert abs(po["after"]["median"]) <= abs(po["before"]["median"]) + 0.5
    assert (tmp_path / "rep" / "maps" / "before_peak_offset.tif").exists()
