"""inkdx command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from inkdx import __version__

app = typer.Typer(
    name="inkdx",
    help="Ink-failure diagnostics for the Vesuvius Challenge: "
    "attribute missing ink to scan, surface, or model.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """inkdx: scan / surface / model failure attribution for scroll segments."""


@app.command()
def version() -> None:
    """Print the inkdx version."""
    typer.echo(__version__)


@app.command()
def run(
    volume: str = typer.Option(..., help="surface volume: layer-TIFF dir or OME-Zarr"),
    out: Path = typer.Option(..., help="output directory (report.json, maps/)"),
    segment: str | None = typer.Option(
        None, help="tifxyz directory; omit for the identity mesh (surface volumes)"
    ),
    prediction: str | None = typer.Option(
        None, help="ink prediction map (tif/npy) for the model stage"
    ),
    calibration: str | None = typer.Option(
        None, help="calibration pack JSON; omit to self-calibrate on this run"
    ),
    tile: int = typer.Option(256),
    halfwidth: int | None = typer.Option(None, help="profile half-width (voxels)"),
    samples: int = typer.Option(256, help="sampled points per tile"),
    processes: int = typer.Option(0, help="worker processes (0 = sequential)"),
    seed: int = typer.Option(0),
    expected_thickness: float = typer.Option(12.0, help="sheet thickness (voxels)"),
) -> None:
    """Run diagnostics and write a report."""
    import numpy as np

    from inkdx.calibration import CalibrationPack
    from inkdx.grid import TileGrid
    from inkdx.io.segment import read_tifxyz
    from inkdx.io.volume import identity_segment, open_surface_volume
    from inkdx.report.json_report import write_report
    from inkdx.report.schema import GridInfo
    from inkdx.runner import DiagnosticsConfig, run_diagnostics
    from inkdx.stages.model import model_maps
    from inkdx.verdict import assign_verdicts

    vol = open_surface_volume(volume)
    nz, h, w = vol.shape

    if segment:
        seg = read_tifxyz(segment)
        if seg.grid_shape != (h, w):
            raise typer.BadParameter(
                f"segment grid {seg.grid_shape} != volume plane {(h, w)}; "
                "full-resolution mesh required"
            )
    else:
        seg = identity_segment(h, w, z_center=(nz - 1) / 2.0)

    hw = halfwidth if halfwidth is not None else max(4, (nz - 1) // 2)
    cfg = DiagnosticsConfig(
        tile_px=tile, halfwidth=hw, samples_per_tile=samples,
        seed=seed, processes=processes, expected_thickness=expected_thickness,
    )
    typer.echo(f"volume {vol.shape}; grid {seg.grid_shape}; tile {tile}; halfwidth {hw}")
    maps = run_diagnostics(vol, seg, cfg, progress=True)

    if prediction:
        import tifffile

        pred = (
            np.load(prediction, mmap_mode="r")
            if prediction.endswith(".npy")
            else tifffile.memmap(prediction, mode="r")
        )
        grid = TileGrid(seg.grid_shape, tile_px=tile)
        maps.update(model_maps(pred, grid, valid=seg.valid))

    if calibration:
        pack = CalibrationPack.load(calibration)
    else:
        pack = CalibrationPack.fit(maps, name="self-calibrated")
        typer.echo("WARNING: self-calibrated on this very run — verdicts are "
                   "relative to this segment's own median; supply --calibration "
                   "from a known-good control for real attribution.")

    verdicts = assign_verdicts(maps, pack)
    path = write_report(
        out,
        maps=maps,
        verdicts=verdicts,
        pack=pack,
        grid_info=GridInfo(
            tile_px=tile, n_tiles=maps["cnr"].shape,
            samples_per_tile=samples, halfwidth=hw,
        ),
        inputs={
            "volume": {"path": str(volume), "shape": list(vol.shape)},
            "segment": {"path": str(segment) if segment else "identity"},
            "prediction": {"path": str(prediction) if prediction else None},
        },
    )
    typer.echo(f"wrote {path}")


@app.command()
def calibrate(
    from_run: Path = typer.Option(..., help="a run output dir (reads maps/*.tif)"),
    name: str = typer.Option(..., help="pack name, e.g. w00_pherc_paris4"),
    out: Path = typer.Option(..., help="output pack JSON"),
    only_ink_ok: bool = typer.Option(
        True, help="fit only on tiles the run judged INK_OK/NO_INK_EVIDENCE"
    ),
) -> None:
    """Fit a calibration pack from a known-good control run."""
    import numpy as np
    import tifffile

    from inkdx.calibration import ORIENTATION, CalibrationPack
    from inkdx.verdict import VERDICT_ID

    maps_dir = from_run / "maps"
    maps = {
        p.stem: tifffile.imread(p)
        for p in sorted(maps_dir.glob("*.tif"))
        if p.stem in ORIENTATION
    }
    if not maps:
        raise typer.BadParameter(f"no metric maps in {maps_dir}")

    select = None
    verdict_path = maps_dir / "verdict.tif"
    if only_ink_ok and verdict_path.exists():
        v = tifffile.imread(verdict_path)
        select = np.isin(v, [VERDICT_ID["INK_OK"], VERDICT_ID["NO_INK_EVIDENCE"]])
        typer.echo(f"fitting on {int(select.sum())}/{select.size} healthy tiles")

    pack = CalibrationPack.fit(maps, name=name, select=select)
    pack.save(out)
    typer.echo(f"wrote {out} ({len(pack.stats)} metrics)")
