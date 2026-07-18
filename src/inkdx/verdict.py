"""Attribution: per-tile stage scores and causally-gated verdicts.

Stage score = sigmoid(median of the worst-2 oriented z-scores of that stage's
core metrics) — a healthy tile scores ~0.5, a clearly broken one ~0. Gates
fire in causal order (data -> scan -> surface -> model): downstream metrics
are meaningless when an upstream stage is broken, so the first failing stage
claims the tile. A tile whose whole chain is healthy but which shows no ink is
NO_INK_EVIDENCE — trustworthy *because* everything upstream checked out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from inkdx.calibration import CalibrationPack

VERDICTS = (
    "NO_DATA",          # 0
    "SCAN_SUSPECT",     # 1
    "SURFACE_SUSPECT",  # 2
    "MODEL_SUSPECT",    # 3
    "NO_INK_EVIDENCE",  # 4
    "INK_OK",           # 5
)
VERDICT_ID = {name: i for i, name in enumerate(VERDICTS)}

STAGE_CORES: dict[str, tuple[str, ...]] = {
    "scan": ("cnr", "haze_index", "noise_sigma"),
    "surface": ("peak_offset", "peak_prominence", "peak_multiplicity", "grid_tearing"),
}

# The model stage scores on an ABSOLUTE scale, not z-vs-control: probability
# maps live on a normalized [0,1] domain, and a control fitted on mostly-blank
# tiles makes every text tile an "indecision outlier" (stroke boundaries carry
# legitimate mid-probabilities — a base-rate trap found on the first real w00
# run). confusion_index (= indecision * (1 - separation)) is ~0 for both text
# and blank tiles and rises only for mid-gray mush; this value maps it to a
# score of 0. Scale calibrated on real data: healthy w00 tiles sit at p95 ~=
# 0.05, an undertrained (2k-iter) model at median ~= 0.19, a fully confused
# phantom at ~0.7 — 0.125 puts the default gate (tau=0.2) at CI > 0.1, 2x the
# real healthy p95.
CONFUSION_FULL_SCALE = 0.125


@dataclass
class VerdictConfig:
    tau_scan: float = 0.2
    tau_surface: float = 0.2
    tau_model: float = 0.2
    blank_ink_frac: float = 0.02
    no_data_hole_fraction: float = 0.5
    extra: dict = field(default_factory=dict)


def stage_scores(
    maps: dict[str, np.ndarray], pack: CalibrationPack
) -> dict[str, np.ndarray]:
    """Per-stage [0,1] health scores from oriented z-scores (worst-2 median)."""
    out: dict[str, np.ndarray] = {}
    for stage, cores in STAGE_CORES.items():
        zs = [pack.z(k, maps[k]) for k in cores if k in maps and k in pack.stats]
        if not zs:
            continue
        stack = np.stack(zs)  # (n_metrics, th, tw)
        # median of the two most negative z-scores per tile
        low2 = np.sort(stack, axis=0)[:2]
        with np.errstate(invalid="ignore"):
            agg = np.nanmedian(low2, axis=0)
        out[stage] = (1.0 / (1.0 + np.exp(-np.clip(agg, -50, 50)))).astype(np.float32)
        out[stage][~np.isfinite(agg)] = np.nan

    ci = maps.get("confusion_index")
    if ci is None and "indecision_mass" in maps and "prob_separation" in maps:
        ci = maps["indecision_mass"] * (1.0 - maps["prob_separation"])
    if ci is not None:
        score = 1.0 - np.clip(ci / CONFUSION_FULL_SCALE, 0.0, 1.0)
        score = score.astype(np.float32)
        score[~np.isfinite(ci)] = np.nan
        out["model"] = score
    return out


def assign_verdicts(
    maps: dict[str, np.ndarray],
    pack: CalibrationPack,
    config: VerdictConfig | None = None,
) -> dict[str, np.ndarray]:
    """Causally-gated per-tile verdicts.

    Returns {"verdict": int8 map (VERDICT_ID), "confidence": float map,
    "score_scan"/"score_surface"/"score_model": the stage scores}.
    Missing stages (e.g. no prediction supplied) skip their gate.
    """
    cfg = config or VerdictConfig()
    scores = stage_scores(maps, pack)
    shape = next(iter(maps.values())).shape
    verdict = np.full(shape, VERDICT_ID["INK_OK"], dtype=np.int8)
    confidence = np.zeros(shape, dtype=np.float32)

    hole = maps.get("hole_fraction")
    n_points = maps.get("n_points")
    no_data = np.zeros(shape, dtype=bool)
    if hole is not None:
        no_data |= np.nan_to_num(hole, nan=1.0) > cfg.no_data_hole_fraction
    if n_points is not None:
        no_data |= np.nan_to_num(n_points, nan=0.0) == 0

    gates: list[tuple[str, np.ndarray, float]] = []
    for stage, tau in (
        ("scan", cfg.tau_scan), ("surface", cfg.tau_surface), ("model", cfg.tau_model)
    ):
        if stage in scores:
            gates.append((stage, scores[stage], tau))

    undecided = ~no_data
    for stage, score, tau in gates:
        bad = undecided & np.isfinite(score) & (score < tau)
        verdict[bad] = VERDICT_ID[f"{stage.upper()}_SUSPECT"]
        confidence[bad] = np.clip((tau - score[bad]) / max(tau, 1e-6), 0.0, 1.0)
        undecided &= ~bad

    ink_frac = maps.get("ink_frac")
    if ink_frac is not None:
        blank = undecided & np.isfinite(ink_frac) & (ink_frac < cfg.blank_ink_frac)
        verdict[blank] = VERDICT_ID["NO_INK_EVIDENCE"]
        confidence[blank] = 1.0 - np.clip(
            ink_frac[blank] / max(cfg.blank_ink_frac, 1e-6), 0.0, 1.0
        )
        undecided &= ~blank

    # healthy tiles: confidence = margin of the weakest gate above threshold
    if gates:
        margins = [
            np.nan_to_num((s - t) / max(1.0 - t, 1e-6), nan=1.0)
            for _, s, t in gates
        ]
        confidence[undecided] = np.clip(np.min(margins, axis=0), 0.0, 1.0)[undecided]

    verdict[no_data] = VERDICT_ID["NO_DATA"]
    confidence[no_data] = 1.0

    out = {"verdict": verdict, "confidence": confidence}
    for stage, score in scores.items():
        out[f"score_{stage}"] = score
    return out


def verdict_fractions(verdict_map: np.ndarray) -> dict[str, float]:
    total = verdict_map.size
    return {
        name: float((verdict_map == VERDICT_ID[name]).sum() / total)
        for name in VERDICTS
    }
