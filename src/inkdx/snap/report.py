"""The snap receipt: proof of what moved, what held, and what improved.

Emits per-vertex QA maps (offset/confidence/status), snap_report.json with
per-iteration convergence stats, and — when diagnostics are enabled — a
before/after comparison from the existing run_diagnostics pipeline, so the
receipt literally shows SURFACE_SUSPECT tiles turning healthy.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from inkdx import __version__
from inkdx.io.segment import Segment
from inkdx.runner import DiagnosticsConfig, run_diagnostics
from inkdx.snap.offsets import STATUS_NAMES
from inkdx.snap.runner import SnapConfig, SnapResult

RECEIPT_METRICS = ("peak_offset", "peak_prominence", "cnr")


def _dist(m: np.ndarray) -> dict[str, float]:
    v = m[np.isfinite(m)]
    if v.size == 0:
        return {"median": None, "p5": None, "p95": None}
    return {
        "median": float(np.median(v)),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
    }


def write_snap_receipt(
    out_dir: str | Path,
    *,
    result: SnapResult,
    cfg: SnapConfig,
    volume=None,
    segment_before: Segment | None = None,
    diagnostics: bool = True,
    inputs: dict | None = None,
) -> Path:
    """Write QA maps + snap_report.json (+ before/after diagnostics)."""
    import tifffile

    out_dir = Path(out_dir)
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(maps_dir / "snap_offset.tif", result.offset_total)
    tifffile.imwrite(maps_dir / "snap_confidence.tif", result.weight)
    tifffile.imwrite(maps_dir / "snap_status.tif", result.status)

    report: dict = {
        "inkdx_version": __version__,
        "schema_version": 1,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": inputs or {},
        "config": {
            k: v for k, v in cfg.__dict__.items() if not isinstance(v, dict)
        },
        "converged": result.converged,
        "iterations": result.iterations,
        "warnings": result.warnings,
        "status_legend": {str(k): v for k, v in STATUS_NAMES.items()},
        "offset_applied": _dist(np.where(result.offset_total != 0,
                                         result.offset_total, np.nan)),
        "final_status_fracs": {
            name: float((result.status == code).mean())
            for code, name in STATUS_NAMES.items()
        },
    }

    if diagnostics and volume is not None and segment_before is not None:
        diag_cfg = DiagnosticsConfig(
            tile_px=cfg.tile_px, halfwidth=cfg.halfwidth,
            samples_per_tile=128, seed=0, processes=cfg.processes,
        )
        before = run_diagnostics(volume, segment_before, diag_cfg)
        after = run_diagnostics(volume, result.segment, diag_cfg)
        report["before_after"] = {
            m: {"before": _dist(before[m]), "after": _dist(after[m])}
            for m in RECEIPT_METRICS
        }
        for m in RECEIPT_METRICS:
            tifffile.imwrite(maps_dir / f"before_{m}.tif", before[m])
            tifffile.imwrite(maps_dir / f"after_{m}.tif", after[m])

    path = out_dir / "snap_report.json"
    path.write_text(json.dumps(report, indent=2))
    return path
