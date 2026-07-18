# report.json schema (v1)

Top level:

| field | content |
|---|---|
| `inkdx_version`, `schema_version`, `created` | provenance |
| `inputs` | volume / segment / prediction paths and shapes as supplied |
| `calibration` | pack name, version, and its metadata (scan parameters etc.) |
| `grid` | `tile_px`, `n_tiles` (rows, cols), `samples_per_tile`, `halfwidth` |
| `summary` | verdict fractions, `dominant_failure`, one-line `headline`, `regions` |
| `tiles` | columnar per-tile data, row-major over the tile grid |
| `maps` | metric name → sidecar float32 TIFF path (tile resolution) |

`summary.regions[]` — connected components of suspect tiles, largest first:

```jsonc
{
  "id": 0,
  "verdict": "SURFACE_SUSPECT",
  "uv_bbox": [r0, c0, r1, c1],      // tile-grid coords, half-open
  "grid_bbox": [r0, c0, r1, c1],    // stored-grid (vertex) coords
  "n_tiles": 180,
  "confidence": 0.43,               // median gate margin in the region
  "explanation": "peak_prominence z=-2.8, peak_offset z=-1.3, ..."
}
```

`tiles` — `verdict` (int, see `verdict_legend.json`: 0 NO_DATA, 1 SCAN_SUSPECT,
2 SURFACE_SUSPECT, 3 MODEL_SUSPECT, 4 NO_INK_EVIDENCE, 5 INK_OK),
`confidence`, `scores.{scan,surface,model}`, and `metrics.<name>` — all lists
of length `n_tiles[0] * n_tiles[1]`; `null` = not computable (e.g. NO_DATA).

Sidecar maps are plain float32 TIFFs at tile resolution — trivially loadable
(`tifffile.imread`) for overlays in napari/VC3D or downstream tooling.

Breaking changes bump `schema_version`; additive fields do not.
