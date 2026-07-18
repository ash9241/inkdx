# inkdx snap — raw-signal label/mesh snapping

Moves a tifxyz mesh onto the true papyrus sheet using the **raw CT signal**.
No trained model, no GPU, no predicted distance fields — addressing open
problem #3's ask verbatim: *"moving approximate labels onto the true surface
using the raw CT signal."* (villa's `snap_surf` registers meshes against an
external reference mesh inside the lasagna pipeline; `pred_dt` snapping needs
a trained surface model's output volume. Nothing else snaps to raw CT.)

## How it works

Per iteration:

1. **Global normal orientation** — one sign per tile (BFS over the tile
   adjacency graph) so the scalar offset field is smoothable across tiles.
2. **Dense per-vertex offsets** — every valid vertex gets an intensity profile
   along its normal (spatially pooled over a small UV window for SNR); the
   sheet peak is located with sub-voxel parabolic refinement.
3. **Anti-wrap peak selection** — among candidate peaks, prefer the one
   *nearest the current surface* (≥60% of the strongest candidate's
   prominence); when a strong far peak and a weak near peak conflict, the
   vertex is **held** (`HELD_MULTIWRAP`). Snap follows *its own* sheet, never
   the brightest one. A second-sheet trap test gates every release.
4. **Confidence gating** — soft SNR ramp on noise-normalized prominence
   (below 3σ → held), multiplicity and centroid-consistency penalties.
5. **Field regularization** — robust outlier rejection, confidence-weighted
   normalized convolution (never extrapolates into regions with no confident
   data), per-iteration step clamp, exact Lipschitz gradient limiting
   (fold-free grid), cumulative offset budget.
6. **Update** — held vertices never move: positions stay bit-identical.

A divergence guard rolls back to the best iterate.

## Usage

```bash
# mesh mode: snap a tifxyz, write a snapped tifxyz + receipt
inkdx snap --volume scroll.zarr --segment seg_tifxyz/ --out seg_snapped/ \
           --iterations 6 --processes 8

# identity mode (surface volumes): the offset map is the deliverable
inkdx snap --volume surface_volume.zarr --out snap_report/ --processes 8
```

Key options: `--max-offset` (cumulative budget, default 8 vox),
`--max-step` (per-iteration, default 2 — set `--iterations ≥
max_offset/max_step` for large corrections), `--pool` (UV pooling, default 3),
`--smooth` (regularization σ, default 3).

## The receipt

Every run writes `snap_report.json` + QA maps (`snap_offset`,
`snap_confidence`, `snap_status`) and, by default, a **before/after
diagnostics comparison** (peak offset, prominence, CNR distributions).
Example, w00 window (PHerc. Paris 4 surface volume): median peak offset
**−8.0 → 0.0 vox**, prominence **+39%**, CNR **+39%**, 92% snapped / 8% held
with stated reasons.

## Provenance convention

Snapped tifxyz outputs carry, in `meta.json`:

```json
"parent_uuid": "<input segment uuid>",
"snap": {"tool": "inkdx", "version": "0.2.0", "params": {...},
         "iterations_run": 3, "converged": true}
```

No provenance convention existed for tifxyz; we propose this one.

## Validation

Phantom (analytic ground truth, in CI): rigid 1/3/6-vox offsets recovered to
<0.5 vox median; tilt and wobble <0.6; second-sheet trap (zero midplane
crossings); blank regions 100% held, positions bit-identical; idempotence;
fold-freedom; determinism across process counts. Real data: w00 receipt
above; Scroll 1 raw crop with a ridge-tracked mesh — oscillatory perturbation
(the realistic tracking-error mode) recovered to **0.83 vox median**; clean
mesh moves 0.24 vox (idempotent).

## Known limitations (v0.2.0)

- **Large rigid offsets on raw fibrous CT** (≥4 vox) currently stall partway
  (w00-style surface volumes are unaffected — see the receipt). Under
  investigation; use case guidance: snap corrects tracking error and drift,
  not gross misplacement.
- **Dense per-vertex passes are window/segment scale** (≲10⁸ vertices);
  the strided path for gigapixel grids is planned.
- The held/snapped boundary keeps a real kink (held vertices don't move —
  by contract).
