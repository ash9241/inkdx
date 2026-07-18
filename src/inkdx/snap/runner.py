"""The `inkdx snap` iteration loop.

Per iteration: global tile-sign orientation → per-tile dense offsets (fork
pool) → field regularization → apply displacement along the signed normals.
Held vertices never move; a divergence guard rolls back to the best iterate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from inkdx.grid import TileGrid
from inkdx.io.segment import Segment
from inkdx.snap.offsets import (
    STATUS_CLAMPED,
    STATUS_INVALID,
    STATUS_NAMES,
    compute_tile_offsets,
)
from inkdx.snap.orient import tile_signs
from inkdx.snap.regularize import RegularizeConfig, regularize_offsets


@dataclass
class SnapConfig:
    halfwidth: int = 16
    tile_px: int = 256
    iterations: int = 3
    damping: float = 1.0
    max_offset: float = 8.0  # cumulative budget (voxels)
    max_step: float = 2.0
    stride: int = 1
    pool: int = 3
    smooth: float = 3.0
    snr_lo: float = 3.0
    snr_hi: float = 6.0
    nearest_frac: float = 0.6
    converge_median: float = 0.1
    processes: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class SnapResult:
    segment: Segment  # snapped copy
    offset_total: np.ndarray  # (H, W) float32 — cumulative applied offset
    status: np.ndarray  # (H, W) uint8 — final per-vertex status
    weight: np.ndarray  # (H, W) float32 — last-iteration confidence
    iterations: list[dict]  # per-iteration stats
    converged: bool
    warnings: list[str]


_ws: dict = {}


def _init_worker(volume, segment, cfg, signs) -> None:
    _ws.update(volume=volume, segment=segment, cfg=cfg, signs=signs,
               grid=TileGrid(segment.grid_shape, tile_px=cfg.tile_px))


def _tile_job(ij: tuple[int, int]):
    cfg = _ws["cfg"]
    tile = _ws["grid"].tile(*ij)
    sign = int(_ws["signs"][ij]) or 1
    to = compute_tile_offsets(
        _ws["volume"], _ws["segment"], tile,
        halfwidth=cfg.halfwidth, sign=sign, stride=cfg.stride, pool=cfg.pool,
        snr_lo=cfg.snr_lo, snr_hi=cfg.snr_hi, nearest_frac=cfg.nearest_frac,
    )
    return ij, to.r_hat, to.weight, to.status, to.normals


def _clone_segment(seg: Segment, x, y, z) -> Segment:
    return Segment(x=x, y=y, z=z, valid=np.asarray(seg.valid).copy(),
                   scale=seg.scale, uuid=seg.uuid, meta=dict(seg.meta))


def run_snap(volume, segment: Segment, config: SnapConfig | None = None) -> SnapResult:
    cfg = config or SnapConfig()
    if cfg.stride != 1:
        raise NotImplementedError("stride > 1 lands with the strided-fill path")

    h, w = segment.grid_shape
    cur = _clone_segment(
        segment,
        np.asarray(segment.x, dtype=np.float32).copy(),
        np.asarray(segment.y, dtype=np.float32).copy(),
        np.asarray(segment.z, dtype=np.float32).copy(),
    )
    grid = TileGrid((h, w), tile_px=cfg.tile_px)

    offset_total = np.zeros((h, w), dtype=np.float32)
    status_final = np.full((h, w), STATUS_INVALID, dtype=np.uint8)
    weight_last = np.zeros((h, w), dtype=np.float32)
    all_warnings: list[str] = []
    iter_stats: list[dict] = []
    best = None  # (median_step, x, y, z, offset_total)
    converged = False

    reg_cfg = RegularizeConfig(smooth=cfg.smooth, max_step=cfg.max_step)

    for k in range(cfg.iterations):
        signs, warnings = tile_signs(cur, grid)
        all_warnings.extend(warnings)

        r_hat = np.full((h, w), np.nan, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        status = np.full((h, w), STATUS_INVALID, dtype=np.uint8)
        nrm = np.zeros((h, w, 3), dtype=np.float32)

        indices = [(t.i, t.j) for t in grid.tiles()]
        if cfg.processes and os.name == "posix":
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            with ctx.Pool(cfg.processes, initializer=_init_worker,
                          initargs=(volume, cur, cfg, signs)) as pool:
                results = pool.imap_unordered(_tile_job, indices, chunksize=4)
                results = list(results)
        else:
            _init_worker(volume, cur, cfg, signs)
            results = [_tile_job(ij) for ij in indices]

        for ij, rh, wt, st, nm in results:
            t = grid.tile(*ij)
            r_hat[t.rows, t.cols] = rh
            weight[t.rows, t.cols] = wt
            status[t.rows, t.cols] = st
            nrm[t.rows, t.cols] = nm

        r_smooth, status = regularize_offsets(r_hat, weight, status, reg_cfg)

        # cumulative budget: clamp the total, not just the step
        proposed_total = offset_total + np.nan_to_num(r_smooth, nan=0.0)
        clamped_total = np.clip(proposed_total, -cfg.max_offset, cfg.max_offset)
        step_arr = clamped_total - offset_total
        at_budget = np.isfinite(r_smooth) & (np.abs(proposed_total) > cfg.max_offset)
        status[at_budget] = STATUS_CLAMPED

        # A vertex can only move along a defined normal: NC smoothing spreads
        # offsets onto border vertices whose own normals are NaN — without this
        # guard the update writes NaN into the coordinates.
        movable = (
            np.isfinite(r_smooth)
            & np.asarray(cur.valid)
            & np.isfinite(nrm).all(axis=-1)
        )
        step_arr = np.where(movable, step_arr, 0.0) * cfg.damping

        moved = np.abs(step_arr) > 1e-6
        median_step = float(np.median(np.abs(step_arr[moved]))) if moved.any() else 0.0
        stats = {
            "iteration": k + 1,
            "median_step": median_step,
            "p95_step": float(np.percentile(np.abs(step_arr[moved]), 95)) if moved.any() else 0.0,
            "updated_frac": float(moved.mean()),
            "status_fracs": {
                STATUS_NAMES[s]: float((status == s).mean())
                for s in STATUS_NAMES
            },
        }
        iter_stats.append(stats)

        if best is None or median_step <= best[0]:
            best = (median_step, cur.x.copy(), cur.y.copy(), cur.z.copy(),
                    offset_total.copy())
        elif median_step > 1.2 * iter_stats[-2]["median_step"]:
            all_warnings.append(
                f"divergence at iteration {k + 1} "
                f"(median step {median_step:.2f}) — rolled back to best iterate"
            )
            _, bx, by, bz, bo = best
            cur = _clone_segment(cur, bx, by, bz)
            offset_total = bo
            break

        cur.x[moved] += (step_arr * nrm[..., 0])[moved]
        cur.y[moved] += (step_arr * nrm[..., 1])[moved]
        cur.z[moved] += (step_arr * nrm[..., 2])[moved]
        offset_total = clamped_total if moved.any() else offset_total
        status_final = status
        weight_last = weight

        if median_step < cfg.converge_median:
            converged = True
            break

    return SnapResult(
        segment=cur,
        offset_total=offset_total,
        status=status_final,
        weight=weight_last,
        iterations=iter_stats,
        converged=converged,
        warnings=all_warnings,
    )
