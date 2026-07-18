# inkdx

**When ink doesn't show up, inkdx tells you why: bad scan, bad surface, or bad model.**

`inkdx` is a diagnostics toolkit for the [Vesuvius Challenge](https://scrollprize.org)
virtual-unwrapping pipeline, addressing
[2026 open problem #9](https://scrollprize.org/2026_open_problems) (ink signal
detection & diagnostics — *"better diagnostics are more important than better
models"*). Given a scroll segment and optionally an ink prediction, it
attributes ink-recovery failure to a pipeline stage, per 256-px tile:

| Stage | Question | Example metrics |
|---|---|---|
| **Scan** | Is there usable CT signal here? | noise σ (raw-voxel), CNR, FWHM haze |
| **Surface** | Is the mesh actually on the papyrus sheet? | profile-peak offset & prominence, sheet-switch multiplicity, tearing, holes |
| **Model** | Does the ink model see and commit to signal? | bimodal separation vs mid-gray confusion |

Verdicts gate causally (data → scan → surface → model), so the first broken
stage claims the tile — and a tile whose whole chain is healthy but blank is
`NO_INK_EVIDENCE`: *trustworthy* blankness, because everything upstream
checked out.

## The w00 baseline (PHerc. Paris 4)

A full 1.6-gigapixel segment — 25,326 tiles — diagnosed in **614 s on 8 CPU
cores** (no GPU):

![verdict overlay](docs/images/w00_verdict_overlay.jpg)

Green = ink found with a healthy chain; blue = healthy chain, honestly blank;
amber = scan-quality suspects (low CNR / haze — interior regions, not edge
artifacts); red = mesh not on a confident sheet (note the fringe exactly at
the scalloped segment boundary); purple = model confusion (near zero here).

![text closeup](docs/images/w00_verdict_closeup.jpg)

## Install

```bash
git clone https://github.com/ash9241/inkdx && cd inkdx
uv sync                      # or: pip install -e .
```

Core is CPU-only and light (numpy/scipy/tifffile/zarr — no torch). The
optional `remote` extra adds streaming from dl.ash2txt.org via the `vesuvius`
library.

## Quickstart

```bash
# a surface volume (pre-extracted segment): layer TIFF dir or OME-Zarr
inkdx run --volume w00.zarr --prediction ink_pred.tif \
          --calibration calibration/w00_pherc_paris4.json \
          --processes 8 --out out/
# → out/report.json  out/report.html  out/maps/*.tif

# fit your own calibration pack from a segment you trust
inkdx calibrate --from-run out/ --name my_control --out my_pack.json
```

`report.json` is machine-readable (schema in [docs/schema.md](docs/schema.md))
with per-tile metrics, stage scores, verdicts, and located suspect regions
with z-score evidence — built to be consumed by other tools.
`report.html` is a single self-contained file: verdict overlay on the
prediction, per-stage heatmaps, healthy-band histograms, region drill-downs.

## Validation

Every metric is unit-tested against a synthetic phantom with analytic ground
truth (injected mesh offsets recovered to ±0.5 voxel, noise σ recovered from
raw voxels, monotonicity under ablation strength). The verdict logic is tested
as an **attribution matrix**: controlled scan / surface / model failures must
each land in their own verdict class. See
[docs/attribution.md](docs/attribution.md) for the causal-gating design and
the real-segment ablation results, and [docs/metrics.md](docs/metrics.md) for
every formula.

## Status

v0.1.0 (July 2026). Built and tested on PHerc. Paris 4 (w00 segment);
calibration packs for other scrolls are community-fittable via
`inkdx calibrate`. Feedback and issues welcome — especially reports from
segments where ink is *missing* and you want to know why.

## License

MIT
