"""The label3d receipt: the Δ(r) figure and the JSON record."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from inkdx import __version__
from inkdx.label3d.depth import STATUS_LOCALIZED, DepthResult, Label3dConfig


def write_label3d_report(
    out_dir: str | Path,
    *,
    result: DepthResult,
    cfg: Label3dConfig,
    inputs: dict | None = None,
    band_used: tuple[float, float] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.delta is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        r = result.offsets
        ax.axhline(0, color="#888", lw=0.8)
        ax.fill_between(r, result.delta - cfg.sig_z * result.se,
                        result.delta + cfg.sig_z * result.se,
                        alpha=0.25, color="#4878a8",
                        label=f"±{cfg.sig_z:g}·SE (block bootstrap)")
        ax.plot(r, result.delta, color="#1a3a5c", lw=2, label="Δ(r) = ink − background")
        if result.status == STATUS_LOCALIZED and result.band:
            ax.axvspan(*result.band, color="#1acc4d", alpha=0.2,
                       label=f"ink band [{result.band[0]:g}, {result.band[1]:g}] vox")
        elif band_used:
            ax.axvspan(*band_used, color="#ffa500", alpha=0.15,
                       label="fallback band (no depth signal)")
        ax.set_xlabel("r — voxels along the surface normal (0 = mesh)")
        ax.set_ylabel("intensity difference")
        ax.set_title(
            f"Ink depth signature — {result.status} "
            f"(p={result.p_value:.3f}, {result.n_ink} ink px, "
            f"{result.n_blocks} blocks)"
            if result.p_value is not None else f"Ink depth — {result.status}"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "delta_r.png", bbox_inches="tight")
        plt.close(fig)

    payload = {
        "inkdx_version": __version__,
        "schema_version": 1,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": inputs or {},
        "config": {k: v for k, v in cfg.__dict__.items() if not isinstance(v, dict)},
        "status": result.status,
        "band": list(result.band) if result.band else None,
        "band_used": list(band_used) if band_used else None,
        "r_ink": result.r_ink,
        "delta_peak": result.delta_peak,
        "p_value": result.p_value,
        "n_ink_px": result.n_ink,
        "n_bg_px": result.n_bg,
        "n_blocks": result.n_blocks,
        "upper_bound_abs_delta": result.upper_bound,
        "delta": result.delta.tolist() if result.delta is not None else None,
        "se": result.se.tolist() if result.se is not None else None,
        "offsets": result.offsets.tolist() if result.offsets is not None else None,
    }
    path = out_dir / "label3d_report.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
