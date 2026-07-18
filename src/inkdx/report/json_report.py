"""Build and write report.json + sidecar metric maps."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from inkdx import __version__
from inkdx.calibration import CalibrationPack
from inkdx.report.schema import (
    SCHEMA_VERSION,
    GridInfo,
    RegionInfo,
    Report,
    Summary,
    TileTable,
)
from inkdx.verdict import VERDICT_ID, VERDICTS, verdict_fractions

_SUSPECT = ("SCAN_SUSPECT", "SURFACE_SUSPECT", "MODEL_SUSPECT")

# Which metrics best explain each failure stage, for region explanations.
_STAGE_EVIDENCE = {
    "SCAN_SUSPECT": ("cnr", "haze_index", "noise_sigma"),
    "SURFACE_SUSPECT": ("peak_offset", "peak_prominence", "peak_multiplicity", "grid_tearing"),
    "MODEL_SUSPECT": ("confusion_index", "indecision_mass", "prob_separation", "entropy"),
}


def extract_regions(
    verdicts: dict[str, np.ndarray],
    maps: dict[str, np.ndarray],
    pack: CalibrationPack,
    *,
    tile_px: int,
    max_regions: int = 20,
    min_tiles: int = 2,
) -> list[RegionInfo]:
    """Connected components of suspect tiles, largest first, with evidence."""
    from scipy.ndimage import label

    vmap = verdicts["verdict"]
    conf = verdicts["confidence"]
    regions: list[RegionInfo] = []
    rid = 0
    for name in _SUSPECT:
        mask = vmap == VERDICT_ID[name]
        if not mask.any():
            continue
        labels, n = label(mask)
        for k in range(1, n + 1):
            rr, cc = np.nonzero(labels == k)
            if rr.size < min_tiles:
                continue
            evidence = []
            for metric in _STAGE_EVIDENCE[name]:
                if metric in maps and metric in pack.stats:
                    z = pack.z(metric, maps[metric][rr, cc])
                    z = z[np.isfinite(z)]
                    if z.size:
                        evidence.append((metric, float(np.median(z))))
            evidence.sort(key=lambda kv: kv[1])
            expl = ", ".join(f"{m} z={z:+.1f}" for m, z in evidence[:3]) or "no evidence metrics"
            regions.append(RegionInfo(
                id=rid,
                verdict=name,
                uv_bbox=(int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1),
                grid_bbox=(
                    int(rr.min()) * tile_px, int(cc.min()) * tile_px,
                    (int(rr.max()) + 1) * tile_px, (int(cc.max()) + 1) * tile_px,
                ),
                n_tiles=int(rr.size),
                confidence=float(np.median(conf[rr, cc])),
                explanation=expl,
            ))
            rid += 1
    regions.sort(key=lambda r: -r.n_tiles)
    return regions[:max_regions]


def make_headline(fractions: dict[str, float], regions: list[RegionInfo]) -> tuple[str | None, str]:
    suspects = {k: v for k, v in fractions.items() if k in _SUSPECT and v > 0}
    dominant = max(suspects, key=suspects.get) if suspects else None
    ok = fractions.get("INK_OK", 0.0)
    blank = fractions.get("NO_INK_EVIDENCE", 0.0)
    if dominant is None:
        headline = (
            f"Pipeline healthy: {ok:.0%} of tiles INK_OK, {blank:.0%} confidently blank."
        )
    else:
        stage = dominant.split("_")[0].lower()
        top = next((r for r in regions if r.verdict == dominant), None)
        where = f"; largest region {top.n_tiles} tiles ({top.explanation})" if top else ""
        headline = (
            f"Dominant failure: {stage} stage on {suspects[dominant]:.0%} of tiles{where}."
        )
    return dominant, headline


def write_report(
    out_dir: str | Path,
    *,
    maps: dict[str, np.ndarray],
    verdicts: dict[str, np.ndarray],
    pack: CalibrationPack,
    grid_info: GridInfo,
    inputs: dict[str, dict] | None = None,
) -> Path:
    """Write report.json and maps/*.tif; returns the report path."""
    import tifffile

    out_dir = Path(out_dir)
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    sidecars: dict[str, str] = {}
    everything = {**maps, "verdict": verdicts["verdict"].astype(np.float32),
                  "confidence": verdicts["confidence"]}
    for k, m in everything.items():
        p = maps_dir / f"{k}.tif"
        tifffile.imwrite(p, np.asarray(m, dtype=np.float32))
        sidecars[k] = str(p.relative_to(out_dir))

    fractions = verdict_fractions(verdicts["verdict"])
    regions = extract_regions(verdicts, maps, pack, tile_px=grid_info.tile_px)
    dominant, headline = make_headline(fractions, regions)

    def col(m: np.ndarray) -> list[float]:
        return [float(x) if np.isfinite(x) else None for x in np.asarray(m).ravel()]

    report = Report(
        inkdx_version=__version__,
        schema_version=SCHEMA_VERSION,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        inputs=inputs or {},
        calibration={"name": pack.name, "version": pack.version, "meta": pack.meta},
        grid=grid_info,
        summary=Summary(
            verdict_fractions=fractions,
            dominant_failure=dominant,
            headline=headline,
            regions=regions,
        ),
        tiles=TileTable(
            verdict=[int(v) for v in verdicts["verdict"].ravel()],
            confidence=col(verdicts["confidence"]),
            scores={
                k.removeprefix("score_"): col(v)
                for k, v in verdicts.items() if k.startswith("score_")
            },
            metrics={k: col(m) for k, m in maps.items()},
        ),
        maps=sidecars,
    )
    path = out_dir / "report.json"
    path.write_text(report.model_dump_json(indent=2))
    # verdict name legend for humans reading the raw json
    legend = {str(i): name for i, name in enumerate(VERDICTS)}
    meta_path = out_dir / "verdict_legend.json"
    meta_path.write_text(json.dumps(legend, indent=2))
    return path
