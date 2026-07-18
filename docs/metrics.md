# Metrics reference

Every metric is computed per tile (default 256×256 vertices of the segment's
UV grid) and reported as a map. Sampling substrate: up to `samples` valid
vertices per tile; at each, the intensity profile I(r) is read along the mesh
normal for r ∈ [−halfwidth, +halfwidth] voxels (trilinear, one bounding-slab
read per tile). For surface volumes (pre-extracted layer stacks / OME-Zarr),
the mesh is the identity grid and profiles read straight down the stack.

Ground-truth ink labels are **never required**. Metrics marked ⚠ have caveats
documented inline.

## Scan stage — is there usable CT signal here?

| metric | definition | direction |
|---|---|---|
| `noise_sigma` | Robust noise σ from **raw** slab voxels: second differences along z, `1.4826·MAD/√6`. Computed pre-interpolation — trilinear sampling attenuates noise and would bias a profile-based estimate low. | higher = worse |
| `snr` | profile peak value / `noise_sigma` | higher = better |
| `cnr` | (peak − gap) / `noise_sigma`, where peak = max of the Savitzky-Golay-smoothed median profile and gap = its 10th percentile (flank-robust: an off-center peak must not read as low contrast). | higher = better |
| `dynamic_range` | p95 − p5 of all profile samples in the tile | higher = better |
| `saturation_frac` | fraction of samples at dtype extremes | higher = worse |
| `haze_index` ⚠ | profile-peak FWHM / `expected_thickness`. Monotone under blur. ⚠ A mesh crossing the sheet obliquely also widens the peak; the spatial pattern disambiguates (segment-wide = scan, patchy = surface). | higher = worse |
| `intensity_drift` | robust z of tile median intensity vs whole segment (ring artifacts, beam hardening, seams) | \|·\| = worse |

## Surface stage — is the mesh actually on the papyrus sheet?

Profile family (volume needed):

| metric | definition | direction |
|---|---|---|
| `peak_offset` | argmax of the smoothed median profile — how far the sheet's bright body sits from the mesh, in voxels along the normal. Signed; magnitude is the drift. | \|·\| = worse |
| `peak_prominence` | (peak − gap) / (σ of the **median** profile = `noise_sigma/√n_eff`) — "is there a sheet at all", comparable across tile support. | higher = better |
| `peak_multiplicity` | count of peaks with ≥50% of the main peak's prominence within the window — neighbor sheet in window = sheet-switch risk | higher = worse |
| `com_offset` | intensity-weighted centroid offset in the central window (sub-voxel drift) | \|·\| = worse |
| `com_smoothness` | \|com_offset − 3×3 neighborhood mean\| — separates systematic drift (smooth field, low value even under a gradient) from noisy peak estimates | higher = worse |

Geometry family (mesh only, no volume access):

| metric | definition | direction |
|---|---|---|
| `grid_tearing` | max adjacent-vertex step / median step — coordinate jumps are the splice / sheet-switch signature | higher = worse |
| `normal_coherence` | mean dot product of adjacent vertex normals (folds/flips dip negative) | higher = better |
| `stretch_anomaly` | \|log2(tile median step / segment median step)\|, worst of row/col directions — parametrization over/under-stretch | higher = worse |
| `hole_fraction` | invalid-vertex fraction per tile; interior holes are additionally localized (connected components with bbox/centroid) | higher = worse |

## Model stage — does the ink model see and commit to signal?

Computed from any dense ink-probability map (TIFF/npy/zarr); the model itself
is never loaded. A model that sees text is **bimodal** (confident strokes on
confident background); a confused model is mid-gray; a model that sees nothing
is confidently blank. Blank ≠ confused — that distinction drives
`NO_INK_EVIDENCE` vs `MODEL_SUSPECT`.

| metric | definition | direction |
|---|---|---|
| `mean_prob`, `p95_prob`, `ink_frac` | basic stats; `ink_frac` = fraction p > 0.5 | context-dependent |
| `entropy` | mean binary entropy of p | higher = worse |
| `indecision_mass` ⚠ | fraction of p ∈ [0.35, 0.65]. ⚠ Text tiles legitimately carry indecision at stroke boundaries — never gate on this alone. | higher = worse |
| `prob_separation` | p90(p) − p10(p) — bimodality without density fitting | higher = better |
| `confusion_index` | `indecision_mass × (1 − prob_separation)`. ~0 for both text and blank tiles; rises only for mid-gray mush. Lives on an absolute [0,1] scale by construction — this is the model gate. | higher = worse |
| `pred_coverage` | valid fraction of the tile's prediction | lower = worse |

## Calibration

`inkdx calibrate` fits robust healthy distributions (median, MAD) per metric
from a control run, restricted to tiles that run judged healthy. Diagnostics
z-score each tile against the pack (oriented so negative = worse), with a
scale floor of 5% of the median so ultra-homogeneous controls don't turn
ordinary variation into huge z-scores. **Relative mode is the operating
model**: bring your own control segment; the shipped
`calibration/w00_pherc_paris4.json` travels with its scan metadata (energy,
resolution, voxel size) and is advisory outside that context. The exception is
the model stage, which scores `confusion_index` on its absolute scale —
probability maps are already normalized, and a blank-dominated control makes
every text tile an outlier (a base-rate trap we hit on the first real
segment).
