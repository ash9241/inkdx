"""Assemble per-tile diagnostic maps across a whole segment.

Sequential by default; pass `processes` to fan tiles out over a fork-based
pool (memmap/zarr-backed volumes fork cheaply). Post-passes that need the
whole map (intensity drift, com smoothness, stretch anomaly) run at the end.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from inkdx.grid import TileGrid
from inkdx.io.segment import Segment
from inkdx.sampling import NormalProfileSampler
from inkdx.stages.scan import SCAN_METRICS, compute_scan_metrics, intensity_drift
from inkdx.stages.surface import (
    SURFACE_GEOMETRY_METRICS,
    SURFACE_PROFILE_METRICS,
    com_smoothness,
    compute_surface_geometry_metrics,
    compute_surface_profile_metrics,
    stretch_anomaly,
)

_POST_METRICS = ("intensity_drift", "com_smoothness", "stretch_anomaly", "n_points")
ALL_METRICS = SCAN_METRICS + SURFACE_PROFILE_METRICS + SURFACE_GEOMETRY_METRICS + _POST_METRICS


@dataclass
class DiagnosticsConfig:
    tile_px: int = 256
    halfwidth: int = 32
    samples_per_tile: int = 256
    seed: int = 0
    vmax: float = 255.0
    expected_thickness: float = 12.0
    processes: int = 0  # 0 = sequential
    extra: dict = field(default_factory=dict)


_worker_state: dict = {}


def _init_worker(volume, segment, cfg) -> None:
    _worker_state["sampler"] = NormalProfileSampler(
        volume, segment,
        halfwidth=cfg.halfwidth,
        samples_per_tile=cfg.samples_per_tile,
        seed=cfg.seed,
    )
    _worker_state["segment"] = segment
    _worker_state["cfg"] = cfg
    _worker_state["grid"] = TileGrid(segment.grid_shape, tile_px=cfg.tile_px)


def _tile_metrics(ij: tuple[int, int]) -> tuple[tuple[int, int], dict[str, float]]:
    sampler = _worker_state["sampler"]
    segment = _worker_state["segment"]
    cfg = _worker_state["cfg"]
    tile = _worker_state["grid"].tile(*ij)

    profiles = sampler.sample_tile(tile)
    m: dict[str, float] = {"n_points": float(profiles.n_points)}
    m.update(
        compute_scan_metrics(
            profiles, vmax=cfg.vmax, expected_thickness=cfg.expected_thickness
        )
    )
    m.update(compute_surface_profile_metrics(profiles))
    m.update(compute_surface_geometry_metrics(segment, tile))
    return ij, m


def run_diagnostics(
    volume,
    segment: Segment,
    config: DiagnosticsConfig | None = None,
    *,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Compute every scan + surface metric as a per-tile map."""
    cfg = config or DiagnosticsConfig()
    grid = TileGrid(segment.grid_shape, tile_px=cfg.tile_px)
    maps = {k: grid.new_map() for k in ALL_METRICS}

    indices = [(t.i, t.j) for t in grid.tiles()]
    if cfg.processes and os.name == "posix":
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(
            cfg.processes, initializer=_init_worker, initargs=(volume, segment, cfg)
        ) as pool:
            results = pool.imap_unordered(_tile_metrics, indices, chunksize=8)
            for n_done, (ij, m) in enumerate(results, 1):
                for k, v in m.items():
                    maps[k][ij] = v
                if progress and n_done % 250 == 0:
                    print(f"  {n_done}/{len(indices)} tiles", flush=True)
    else:
        _init_worker(volume, segment, cfg)
        for n_done, ij in enumerate(indices, 1):
            _, m = _tile_metrics(ij)
            for k, v in m.items():
                maps[k][ij] = v
            if progress and n_done % 250 == 0:
                print(f"  {n_done}/{len(indices)} tiles", flush=True)

    maps["intensity_drift"] = intensity_drift(maps["median_intensity"])
    maps["com_smoothness"] = com_smoothness(maps["com_offset"])
    maps["stretch_anomaly"] = stretch_anomaly(maps["step_u"], maps["step_v"])
    return maps
