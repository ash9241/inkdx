# Progress Prize submission — draft answers (July 2026)

Copy-paste material for the Google Form. Adjust to actual field names.

**Project name:** inkdx — ink-failure diagnostics for the virtual-unwrapping pipeline

**One-line summary:** When ink doesn't show up, inkdx tells you why: bad scan,
bad surface, or bad model — per-tile failure attribution with machine-readable
reports.

**Which open problem it addresses:** 2026 open problem #9 (ink signal
detection & diagnostics) — the only open problem whose listed current approach
is "none yet". Also feeds #3 (label quality) and #8 (cross-scroll
generalization: distinguishing "no ink present" from "ink present but
unrecovered" is a stated need).

**Links:**
- Repo (MIT): https://github.com/ash9241/inkdx (tag v0.1.0)
- Discord release thread:
  https://discord.com/channels/1079907749569237093/1162822236521115720/threads/1528098732728652017
- Example report + images: repo README and docs/

**What it does:** Given a segment (surface volume or tifxyz mesh + volume) and
optionally an ink prediction, computes per-256px-tile metrics for three
pipeline stages (scan: raw-voxel noise σ, CNR, FWHM haze; surface:
profile-peak offset/prominence, sheet-switch multiplicity, tearing, holes;
model: bimodal separation vs mid-gray confusion) and gates them causally into
verdicts: SCAN/SURFACE/MODEL_SUSPECT, NO_INK_EVIDENCE (chain healthy, honestly
blank), INK_OK, NO_DATA. Outputs machine-readable report.json (per-tile
metrics, located suspect regions with z-score evidence) + a self-contained
report.html.

**Validation:** (1) synthetic phantom with analytic ground truth (injected
mesh offsets recovered ±0.5 vox; noise σ recovered; monotonicity tests) — 64
tests in CI; (2) real-data attribution matrix: induced failures on data with
known-recovered ink, re-inferenced — noise → 100% SCAN, 8-vox mesh offset
(raw Scroll 1 crop, ridge-tracked mesh) → 94% SURFACE, undertrained 2k-iter
checkpoint → 81% MODEL. Limitations documented (resampled surface volumes blur
sheets → reduced small-offset sensitivity there).

**Scale/cost:** full 1.6-gigapixel w00 segment (25,326 tiles) diagnosed in
614 s on 8 CPU cores — no GPU required for diagnostics.

**Reproducibility:** MIT, `uv sync` + one CLI command; fresh-clone rehearsal
in CI; Dockerfile; seeded sampling (bit-identical parallel/sequential);
shipped calibration pack for PHerc. Paris 4 with scan metadata;
`inkdx calibrate` fits packs for any scroll.

**Team:** Aishwarya Das (aishwarya@diraclabs.com), Discord: anshu231.
Built with heavy use of agentic tooling (Claude Code) — happy to discuss the
workflow.
