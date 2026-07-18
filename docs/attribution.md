# Attribution: from metrics to verdicts

## Stage scores

Scan and surface metrics are converted to oriented z-scores against the
calibration pack (negative = worse). Per tile, each stage's score is

    score = sigmoid( median of the worst-2 core z-scores )

so a healthy tile scores ≈0.5 and a clearly broken one ≈0. Core metrics:

- **scan**: `cnr`, `haze_index`, `noise_sigma`
- **surface**: `peak_offset`, `peak_prominence`, `peak_multiplicity`, `grid_tearing`

The **model** stage scores on an absolute scale instead:

    score_model = 1 − clip(confusion_index / 0.3, 0, 1)

Rationale: probability maps live on a normalized [0,1] domain, and a z-score
against a mostly-blank control flags every text tile as an indecision outlier
(stroke boundaries carry legitimate mid-probabilities). `confusion_index =
indecision_mass × (1 − prob_separation)` is ≈0 for text *and* blank tiles and
rises only for mid-gray mush; 0.3 ≈ fully confused.

## Causal gating

Downstream metrics are meaningless when an upstream stage is broken (noise
destroys peak prominence; an off-sheet mesh feeds the model garbage), so gates
fire in pipeline order and the first failing stage claims the tile:

    hole_fraction > 0.5 or no samples  →  NO_DATA
    score_scan    < τ                  →  SCAN_SUSPECT
    score_surface < τ                  →  SURFACE_SUSPECT
    score_model   < τ                  →  MODEL_SUSPECT
    ink_frac < τ_blank                 →  NO_INK_EVIDENCE
    else                               →  INK_OK

Defaults: τ = 0.2 (≈ z −1.4), τ_blank = 0.02. Every verdict carries a
confidence (margin from its threshold) and the full score vector stays in the
JSON so users can re-gate with their own thresholds.

`NO_INK_EVIDENCE` is the scientifically important verdict: *the chain is
healthy and the model confidently sees nothing.* It is only trustworthy
because the three upstream gates passed — which is the whole point of ordering
them.

Cross-talk safeguards baked into the metrics (each found by a failing test or
a real run, then pinned):

- `peak_prominence` is noise-normalized → noise ablations don't cascade into
  SURFACE verdicts.
- the profile gap level is a percentile, not flanking minima → surface
  failures don't read as low contrast (SCAN).
- the model gate is absolute → text tiles don't read as MODEL_SUSPECT under a
  blank-dominated calibration.

## Validation

Two levels, both in CI:

1. **Phantom attribution matrix** (`tests/test_verdict.py`): a synthetic sheet
   volume with analytic ground truth. Noise / blur / mesh-offset / confused-
   probability / blank / hole ablations must each land ≥80% of affected tiles
   in exactly their own verdict class, with off-diagonal leakage <10%.
2. **Real-data ablations**: controlled failures induced on real data with
   known-recovered ink, re-inferenced (A100) and re-diagnosed:

   | induced failure | attributed to | result |
   |---|---|---|
   | — (control, text window of w00) | healthy | **82%** healthy (10% scan-, 7% surface-suspect: real marginal regions) |
   | Gaussian noise σ=25 on the volume | SCAN | **100%** SCAN_SUSPECT |
   | mesh offset +8 vox (raw Scroll 1 crop + ridge-tracked mesh) | SURFACE | **94%** SURFACE_SUSPECT (its control: 94% healthy) |
   | undertrained 2k-iter checkpoint on clean data | MODEL | **81%** MODEL_SUSPECT |

   ![attribution matrix](images/attribution_matrix.png)

   **Documented limitation:** on *resampled surface volumes* the sheet is
   blurred to ~27 voxels FWHM and tile-level peak offsets naturally vary by
   ±10, so a 12-layer z-roll stays within healthy variation there (a 12-layer
   roll flipped only ~4% of tiles). The surface stage operates at full
   precision against the raw scroll volume (clean-mesh peak offsets recovered
   at MAD 0.0 voxels on Scroll 1) — which is where mesh QC actually happens.
   Three of the calibration/gating rules above (percentile gap, strict-select
   fitting, "center" orientation for peak_offset) were found by exactly these
   real-data runs, not by the phantom.
