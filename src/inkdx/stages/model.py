"""Model stage: does the ink model see and commit to signal?

Metrics over an ink-probability map (any dense prediction: TIFF/npy/zarr,
values normalized to [0, 1]). A model that sees text is *bimodal* — confident
ink strokes on confident background. A model that sees nothing is confidently
blank. A model that is confused is mid-gray everywhere: high entropy, high
indecision mass, low separation. That distinction — blank vs confused — is
what separates NO_INK_EVIDENCE from MODEL_SUSPECT downstream.

Needs only the prediction map; the model itself is not loaded.
"""

from __future__ import annotations

import numpy as np

from inkdx.grid import TileGrid

MODEL_METRICS = (
    "mean_prob", "p95_prob", "ink_frac", "entropy",
    "indecision_mass", "prob_separation", "confusion_index", "pred_coverage",
)

_EPS = 1e-6


def compute_model_metrics(
    prob_tile: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, float]:
    """Per-tile model metrics from a [0,1] probability tile."""
    out = dict.fromkeys(MODEL_METRICS, np.nan)
    p = prob_tile.astype(np.float32).ravel()
    if valid is not None:
        p = p[valid.ravel()]
    p = p[np.isfinite(p)]
    if p.size < 16:
        return out

    out["mean_prob"] = float(p.mean())
    out["p95_prob"] = float(np.percentile(p, 95))
    out["ink_frac"] = float((p > 0.5).mean())
    out["indecision_mass"] = float(((p >= 0.35) & (p <= 0.65)).mean())
    out["prob_separation"] = float(np.percentile(p, 90) - np.percentile(p, 10))

    q = np.clip(p, _EPS, 1.0 - _EPS)
    ent = -(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q))
    out["entropy"] = float(ent.mean())

    # Confusion = fat middle AND no bimodality. Text tiles legitimately carry
    # indecision at stroke boundaries, but they also separate strongly — this
    # product stays low for them, and for confidently-blank tiles, and rises
    # only for mid-gray mush. Lives on an absolute [0,1] scale by construction.
    out["confusion_index"] = out["indecision_mass"] * (1.0 - out["prob_separation"])
    out["pred_coverage"] = 1.0  # overwritten by map-level driver where known
    return out


def model_maps(
    prob: np.ndarray,
    grid: TileGrid,
    *,
    valid: np.ndarray | None = None,
    vmax: float | None = None,
) -> dict[str, np.ndarray]:
    """Assemble model metrics into per-tile maps.

    `prob` is any 2D array-like sliceable per tile (numpy, memmap, zarr) whose
    plane matches the grid. Integer inputs are normalized by `vmax` (inferred
    from the dtype when omitted).
    """
    if prob.shape != grid.grid_shape:
        raise ValueError(f"prediction plane {prob.shape} != grid {grid.grid_shape}")
    if vmax is None:
        vmax = float(np.iinfo(prob.dtype).max) if np.issubdtype(prob.dtype, np.integer) else 1.0

    maps = {k: grid.new_map() for k in MODEL_METRICS}
    for t in grid.tiles():
        tile_p = np.asarray(prob[t.rows, t.cols], dtype=np.float32) / vmax
        v = None if valid is None else np.asarray(valid[t.rows, t.cols])
        m = compute_model_metrics(tile_p, v)
        cov = 1.0 if v is None else float(np.asarray(v).mean())
        m["pred_coverage"] = cov if np.isfinite(m["mean_prob"]) else 0.0
        for k, val in m.items():
            maps[k][t.i, t.j] = val
    return maps
