"""Self-contained single-file HTML report.

Everything (images, styles, ~60 lines of JS) is inlined so the file can be
dragged into Discord/Slack or opened from disk. Layout:

  header card   — headline, verdict fraction bar, provenance
  hero view     — base image (prediction) with toggleable verdict/score overlays
  stage panels  — metric heatmaps + histograms vs the calibration pack
  regions table — top suspect regions with evidence strings
  methodology   — metric one-liners + reproduce command
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

VERDICT_COLORS = {
    0: (0.15, 0.15, 0.15, 0.0),   # NO_DATA
    1: (1.00, 0.65, 0.00, 0.55),  # SCAN_SUSPECT
    2: (0.90, 0.10, 0.10, 0.55),  # SURFACE_SUSPECT
    3: (0.60, 0.20, 0.80, 0.55),  # MODEL_SUSPECT
    4: (0.20, 0.45, 0.95, 0.30),  # NO_INK_EVIDENCE
    5: (0.10, 0.80, 0.30, 0.30),  # INK_OK
}
VERDICT_CSS = {0: "#606060", 1: "#ffa500", 2: "#e61a1a", 3: "#9933cc",
               4: "#3373f2", 5: "#1acc4d"}

_STAGE_PANELS = {
    "scan": ("cnr", "noise_sigma", "haze_index"),
    "surface": ("peak_offset", "peak_prominence", "peak_multiplicity"),
    "model": ("confusion_index", "prob_separation", "ink_frac"),
}


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _verdict_rgba(verdict: np.ndarray) -> np.ndarray:
    out = np.zeros((*verdict.shape, 4), dtype=np.float32)
    for k, c in VERDICT_COLORS.items():
        out[verdict == k] = c
    return out


def _heatmap_b64(m: np.ndarray, title: str) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    finite = m[np.isfinite(m)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [2, 98])
        im = ax.imshow(m, cmap="viridis", vmin=vmin, vmax=max(vmax, vmin + 1e-6))
        fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    return _png_b64(fig)


def _hist_b64(m: np.ndarray, stats: dict | None, title: str) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    finite = m[np.isfinite(m)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        ax.hist(finite, bins=60, range=(lo, max(hi, lo + 1e-6)),
                color="#4878a8", alpha=0.85)
        if stats:
            med, mad = stats["median"], stats["mad"]
            ax.axvline(med, color="#1acc4d", lw=2, label="healthy median")
            ax.axvspan(med - 1.4826 * mad, med + 1.4826 * mad,
                       color="#1acc4d", alpha=0.15, label="healthy ±1σ (robust)")
            ax.legend(fontsize=7, loc="upper right")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_yticks([])
    return _png_b64(fig)


def _hero_b64(base_image: np.ndarray | None, verdict: np.ndarray,
              scores: dict[str, np.ndarray]) -> dict[str, str]:
    """Base image + upsampled overlay layers, each a separate base64 PNG."""
    import matplotlib.pyplot as plt
    from PIL import Image

    layers: dict[str, str] = {}

    if base_image is not None:
        h, w = base_image.shape[:2]
    else:
        h, w = (v * 8 for v in verdict.shape)

    def to_png(arr_rgba: np.ndarray) -> str:
        img = Image.fromarray((arr_rgba * 255).astype(np.uint8), "RGBA")
        img = img.resize((w, h), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    if base_image is not None:
        img = Image.fromarray(base_image).convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        layers["base"] = base64.b64encode(buf.getvalue()).decode()

    layers["verdicts"] = to_png(_verdict_rgba(verdict))

    cmap = plt.get_cmap("RdYlGn")
    for name, score in scores.items():
        rgba = cmap(np.nan_to_num(score, nan=0.5))
        rgba[..., 3] = np.where(np.isfinite(score), 0.55, 0.0)
        layers[f"score: {name}"] = to_png(rgba.astype(np.float32))
    return layers


_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>inkdx report — {{ title }}</title>
<style>
 body { font-family: -apple-system, "Segoe UI", sans-serif; margin: 0; background:#14161a; color:#e6e6e6; }
 .wrap { max-width: 1240px; margin: 0 auto; padding: 24px; }
 .card { background:#1d2026; border-radius: 10px; padding: 18px 22px; margin-bottom: 18px; }
 h1 { font-size: 21px; margin: 0 0 6px; } h2 { font-size: 16px; margin: 18px 0 8px; }
 .headline { font-size: 15px; color: #ffd479; }
 .fracbar { display:flex; height: 22px; border-radius: 6px; overflow:hidden; margin: 10px 0; }
 .fracbar div { height:100%; }
 .legend span { display:inline-block; margin-right:14px; font-size:12px; }
 .dot { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }
 .hero { position: relative; } .hero img { width: 100%; display:block; border-radius: 8px; }
 .hero img.overlay { position:absolute; left:0; top:0; }
 .toggles label { margin-right: 16px; font-size: 13px; }
 .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 12px; }
 .grid img { width: 100%; border-radius: 6px; }
 table { border-collapse: collapse; width: 100%; font-size: 13px; }
 th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #333; }
 .prov { font-size: 12px; color: #999; } code { background:#000; padding:1px 5px; border-radius:4px; font-size:12px;}
</style></head><body><div class="wrap">

<div class="card">
 <h1>inkdx report — {{ title }}</h1>
 <div class="headline">{{ headline }}</div>
 <div class="fracbar">{% for name, frac, color in fractions %}<div style="width:{{ '%.2f' % (frac*100) }}%;background:{{ color }}" title="{{ name }} {{ '%.1f' % (frac*100) }}%"></div>{% endfor %}</div>
 <div class="legend">{% for name, frac, color in fractions %}<span><span class="dot" style="background:{{ color }}"></span>{{ name }} {{ '%.1f' % (frac*100) }}%</span>{% endfor %}</div>
 <div class="prov">inkdx {{ version }} · schema v{{ schema }} · {{ created }} · calibration: {{ pack_name }}</div>
</div>

<div class="card">
 <h2>Verdict map</h2>
 <div class="toggles">
  {% for name in layer_names %}<label><input type="checkbox" data-layer="{{ name }}" {{ 'checked' if name == 'verdicts' }}> {{ name }}</label>{% endfor %}
 </div>
 <div class="hero">
  {% if has_base %}<img src="data:image/png;base64,{{ layers['base'] }}">{% endif %}
  {% for name in layer_names %}<img class="overlay" data-layer="{{ name }}" src="data:image/png;base64,{{ layers[name] }}" style="{{ 'display:none' if name != 'verdicts' }}">{% endfor %}
 </div>
</div>

{% for stage, panels in stage_panels.items() %}
<div class="card"><h2>{{ stage }} stage</h2><div class="grid">
 {% for img in panels %}<div><img src="data:image/png;base64,{{ img }}"></div>{% endfor %}
</div></div>
{% endfor %}

<div class="card"><h2>Top suspect regions</h2>
<table><tr><th>#</th><th>verdict</th><th>tiles</th><th>grid bbox (r0,c0,r1,c1)</th><th>confidence</th><th>evidence (oriented z vs healthy)</th></tr>
{% for r in regions %}<tr><td>{{ r.id }}</td><td style="color:{{ verdict_css[r.verdict] }}">{{ r.verdict }}</td><td>{{ r.n_tiles }}</td><td><code>{{ r.grid_bbox|join(', ') }}</code></td><td>{{ '%.2f' % r.confidence }}</td><td>{{ r.explanation }}</td></tr>{% endfor %}
</table></div>

<div class="card"><h2>Methodology</h2>
<p style="font-size:13px">Per 256-px tile of the segment's UV grid, inkdx samples intensity profiles along mesh normals and computes stage metrics:
<b>scan</b> — is there usable CT signal (noise σ from raw voxels, contrast-to-noise between sheet peak and gap, FWHM-based haze);
<b>surface</b> — is the mesh on the sheet (profile-peak offset &amp; prominence, neighbor-sheet multiplicity, grid tearing, holes);
<b>model</b> — does the prediction commit (bimodal separation vs mid-gray confusion).
Gates fire causally (data → scan → surface → model); a tile whose whole chain is healthy but blank is <b>NO_INK_EVIDENCE</b> — trustworthy <i>because</i> upstream checked out.
Metrics are z-scored against a known-good control (calibration pack <code>{{ pack_name }}</code>).</p>
<p class="prov">reproduce: <code>{{ repro }}</code></p>
</div>

</div><script>
document.querySelectorAll('.toggles input').forEach(cb => cb.addEventListener('change', () => {
  document.querySelectorAll('img.overlay').forEach(img => {
    if (img.dataset.layer === cb.dataset.layer) img.style.display = cb.checked ? '' : 'none';
  });
}));
</script></body></html>
"""


def write_html_report(
    run_dir: str | Path,
    *,
    base_image: np.ndarray | None = None,
    title: str | None = None,
    repro: str = "inkdx run --volume <surface_volume> --out <dir>",
) -> Path:
    """Render report.html from a run directory (report.json + maps/*.tif)."""
    import tifffile
    from jinja2 import Template

    run_dir = Path(run_dir)
    rep = json.loads((run_dir / "report.json").read_text())
    maps = {p.stem: tifffile.imread(p) for p in sorted((run_dir / "maps").glob("*.tif"))}
    verdict = maps["verdict"].astype(int)

    scores = {k[len("score_"):] if k.startswith("score_") else k: maps[k]
              for k in maps if k.startswith("score_")}
    layers = _hero_b64(base_image, verdict, scores)
    layer_names = [k for k in layers if k != "base"]

    pack_stats = {}  # healthy bands drawn only when the pack is alongside
    pack_path = run_dir / "calibration.json"
    if pack_path.exists():
        pack_stats = json.loads(pack_path.read_text()).get("stats", {})

    stage_panels = {}
    for stage, metrics in _STAGE_PANELS.items():
        panels = []
        for m in metrics:
            if m not in maps:
                continue
            panels.append(_heatmap_b64(maps[m], m))
            panels.append(_hist_b64(maps[m], pack_stats.get(m), f"{m} distribution"))
        if panels:
            stage_panels[stage] = panels

    order = ["INK_OK", "NO_INK_EVIDENCE", "SCAN_SUSPECT", "SURFACE_SUSPECT",
             "MODEL_SUSPECT", "NO_DATA"]
    css_by_name = {"NO_DATA": VERDICT_CSS[0], "SCAN_SUSPECT": VERDICT_CSS[1],
                   "SURFACE_SUSPECT": VERDICT_CSS[2], "MODEL_SUSPECT": VERDICT_CSS[3],
                   "NO_INK_EVIDENCE": VERDICT_CSS[4], "INK_OK": VERDICT_CSS[5]}
    fracs = rep["summary"]["verdict_fractions"]
    fractions = [(n, fracs.get(n, 0.0), css_by_name[n]) for n in order]

    html = Template(_TEMPLATE).render(
        title=title or rep["inputs"].get("volume", {}).get("path", "segment"),
        headline=rep["summary"]["headline"],
        fractions=fractions,
        version=rep["inkdx_version"], schema=rep["schema_version"],
        created=rep["created"], pack_name=rep["calibration"]["name"],
        layers=layers, layer_names=layer_names, has_base="base" in layers,
        stage_panels=stage_panels,
        regions=[type("R", (), r) for r in rep["summary"]["regions"]],
        verdict_css=css_by_name,
        repro=repro,
    )
    out = run_dir / "report.html"
    out.write_text(html)
    return out
