# inkdx label3d — signal-based true-3D ink labels

Answers villa wishlist issue #192: ink labels *"in true 3d rather than a
single image projected across multiple layers."* Existing tooling projects 2D
ink labels as a **symmetric ±8 band** around the mesh because nobody knew
where in depth the ink signal sits. `label3d` measures it — from the raw CT,
with no model in the loop — and emits labels in the measured band.

## Method

- **Ink set**: 2D-ink-labeled pixels (eroded 1 px). **Background set**: a
  locally matched annulus around the strokes (dilate 3..15 px). The local
  matching isolates *ink* from papyrus condition, scan brightness, and
  surface quality.
- **Δ(r)** = median ink profile − median background profile, along the
  surface normal.
- **Significance**: block-level bootstrap SE (spatially correlated pixels
  must not inflate N) and a **background-vs-background split null** — is
  ink-vs-bg larger than bg-vs-bg variability at the same block count?
  (A label permutation over pooled blocks silently loses all power for
  strong effects: the median's robustness carries the full ink signature
  through any imbalanced permutation. Found the hard way; documented in the
  test suite.)
- **Band**: FWHM of |Δ| around its extremum, converted to z-layer convention
  at emission. Sign-agnostic — ink may be radiodense or radiolucent.

**The tool never silently invents a depth band.** Statuses:
`INK_DEPTH_LOCALIZED` (measured band), `NO_DEPTH_SIGNAL` (symmetric fallback,
flagged, with a quantified upper bound on |Δ|), `INSUFFICIENT_LABELS`.

## The w00 measurement (PHerc. Paris 4)

150k ink px vs 150k matched background px, 19,542 spatial blocks:
**Δ peaks at +21.5 intensity units at r = 7 vox; band [2, 11] vox**
(z-layers ~21–30 of 65); p ≈ 0.005. The signal is **one-sided** — a
symmetric ±8 label column is roughly half empty. Deterministic under the
seeded pipeline; figure: `delta_r.png` in the run report.

## Output format

Exactly the community conventions: `(Z, Y, X)` uint8 zarr,
`{0: background, 1: ink, 2: ignore}`, with `3d_ink_params.json` and
`remap.json` sidecars. Policy: background within ±`bg_distance` of the mesh
on non-ink columns; ink over the measured band; the rest of an ink column
stays **ignore** (calling it background under a stroke would be an
overclaim; `--ink-column-rest bg` if you disagree).

## Usage

```bash
inkdx label3d --volume surface_volume.zarr \
              --labels segment_inklabels.tif \
              --out labels3d.zarr --report report/
```

Key options: `--bootstrap` (default 200), `--band-frac` (band = this fraction
of peak |Δ|, default 0.5), `--bg-distance`, `--fallback-distance`,
`--min-ink-px`.

## Validation

Phantom with synthetic ink at a known depth band: band recovered to ±1.5 vox
for both contrast signs; as contrast → 0 the status flips to
`NO_DEPTH_SIGNAL` **before** the band estimate degrades (the significance
threshold is calibrated to fail safe). Emission codes/precedence and sidecar
schemas round-trip in CI.

## Limitations (v0.2.0)

- One global band per segment (per-tile refinement where labels are dense is
  planned; the per-tile machinery exists in the estimator).
- Surface-volume mode only; raw-volume emission along real tifxyz normals is
  planned.
- Δ(r) measures signal under *your* 2D labels — label mistakes propagate.
  Pair with `inkdx run`/`snap` to check the surface first.
