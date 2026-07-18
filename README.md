# inkdx

**When ink doesn't show up, inkdx tells you why: bad scan, bad surface, or bad model.**

`inkdx` is a diagnostics toolkit for the [Vesuvius Challenge](https://scrollprize.org)
virtual-unwrapping pipeline. Given a scroll volume, a segment surface (tifxyz), and
optionally an ink prediction, it attributes ink-recovery failure to a pipeline stage:

| Stage | Question it answers |
|---|---|
| **Scan** | Is there usable signal in the CT here? (noise, contrast, haze) |
| **Surface** | Is the mesh actually on the papyrus sheet? (drift, holes, sheet switches) |
| **Model** | Does the ink model see and commit to signal? (confidence, coverage, input shift) |

Per-tile metric maps are combined through causally-ordered gating into verdicts —
`SCAN_SUSPECT`, `SURFACE_SUSPECT`, `MODEL_SUSPECT`, `NO_INK_EVIDENCE`, `INK_OK`,
`NO_DATA` — and rendered as a machine-readable JSON report plus a self-contained
HTML report.

This addresses [2026 open problem #9](https://scrollprize.org/2026_open_problems)
(ink signal detection & diagnostics): *"better diagnostics are more important than
better models."*

## Status

Under active development (July 2026). Interfaces may change until v0.1.0.

## Install

```bash
uv pip install -e .          # core (CPU-only, light deps)
uv pip install -e '.[remote]'  # + streaming from dl.ash2txt.org via the vesuvius library
```

## Quickstart

```bash
inkdx run \
  --volume /path/to/layers_dir_or_zarr \
  --segment /path/to/segment_tifxyz \
  --prediction /path/to/ink_prediction.tif \
  --out out/
# → out/report.json, out/report.html, out/maps/*.tif
```

## License

MIT
