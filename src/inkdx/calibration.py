"""Calibration packs: what "healthy" looks like, per metric.

A pack stores robust location/scale (median, MAD) for every metric, fitted on
a control segment where ink recovery is known-good. Diagnostics then express
each tile's metrics as *oriented z-scores* — negative = worse — relative to
the pack. Relative mode is the default operating model: bring your own
control; absolute thresholds are advisory and travel with scan metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_EPS = 1e-6

# +1: higher is healthier. -1: higher is worse. "abs": magnitude is what's bad.
ORIENTATION: dict[str, int | str] = {
    "cnr": +1, "snr": +1, "dynamic_range": +1,
    "noise_sigma": -1, "haze_index": -1, "saturation_frac": -1,
    "intensity_drift": "abs",
    "peak_offset": "abs", "peak_prominence": +1, "peak_multiplicity": -1,
    "com_offset": "abs", "com_smoothness": -1,
    "grid_tearing": -1, "normal_coherence": +1, "stretch_anomaly": -1,
    "hole_fraction": -1,
    "mean_prob": +1, "p95_prob": +1, "ink_frac": +1,
    "entropy": -1, "indecision_mass": -1, "prob_separation": +1,
    "confusion_index": -1, "pred_coverage": +1,
}


@dataclass
class CalibrationPack:
    name: str
    stats: dict[str, dict[str, float]]  # metric -> {median, mad}
    version: int = 1
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        maps: dict[str, np.ndarray],
        *,
        name: str,
        select: np.ndarray | None = None,
        meta: dict | None = None,
    ) -> CalibrationPack:
        """Fit healthy distributions from a control run's tile maps.

        `select` restricts fitting to known-healthy tiles (bool tile map).
        """
        stats: dict[str, dict[str, float]] = {}
        for k, m in maps.items():
            if k not in ORIENTATION:
                continue
            vals = m[select] if select is not None else m
            if ORIENTATION[k] == "abs":
                vals = np.abs(vals)
            vals = vals[np.isfinite(vals)]
            if vals.size < 8:
                continue
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            stats[k] = {"median": med, "mad": mad}
        return cls(name=name, stats=stats, meta=meta or {})

    def z(self, metric: str, values: np.ndarray) -> np.ndarray:
        """Oriented robust z-scores: negative = worse than the healthy control."""
        if metric not in self.stats:
            return np.full_like(np.asarray(values, dtype=np.float32), np.nan)
        orient = ORIENTATION[metric]
        v = np.asarray(values, dtype=np.float32)
        if orient == "abs":
            v = np.abs(v)
        s = self.stats[metric]
        # Scale floor at 5% of the median: an ultra-homogeneous control (tiny
        # MAD) must not turn ordinary variation into huge z-scores.
        scale = max(1.4826 * s["mad"], 0.05 * abs(s["median"]), _EPS)
        z = (v - s["median"]) / scale
        if orient == -1 or orient == "abs":
            z = -z
        return z.astype(np.float32)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        payload = {
            "inkdx_calibration_version": self.version,
            "name": self.name,
            "meta": self.meta,
            "stats": self.stats,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path) -> CalibrationPack:
        d = json.loads(Path(path).read_text())
        return cls(
            name=d["name"], stats=d["stats"],
            version=d.get("inkdx_calibration_version", 1), meta=d.get("meta", {}),
        )
