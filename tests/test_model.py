import numpy as np

from inkdx.grid import TileGrid
from inkdx.stages.model import compute_model_metrics, model_maps


def rng():
    return np.random.default_rng(0)


def confident_text_tile(n=128):
    """Bimodal: confident strokes on confident background."""
    p = np.full((n, n), 0.03, dtype=np.float32)
    p[40:60, :] = 0.95  # a "stroke"
    p += rng().normal(0, 0.01, p.shape).astype(np.float32)
    return np.clip(p, 0, 1)


def confused_tile(n=128):
    """Mid-gray mush."""
    return np.clip(
        rng().normal(0.5, 0.06, (n, n)), 0, 1
    ).astype(np.float32)


def blank_tile(n=128):
    """Confidently empty."""
    return np.clip(
        rng().normal(0.02, 0.01, (n, n)), 0, 1
    ).astype(np.float32)


def test_text_vs_confused_vs_blank():
    text = compute_model_metrics(confident_text_tile())
    confused = compute_model_metrics(confused_tile())
    blank = compute_model_metrics(blank_tile())

    # confused: high entropy + indecision, low separation
    assert confused["entropy"] > 0.8
    assert confused["indecision_mass"] > 0.7
    assert confused["prob_separation"] < 0.3

    # text: bimodal — big separation, low indecision
    assert text["prob_separation"] > 0.8
    assert text["indecision_mass"] < 0.05
    assert 0.05 < text["ink_frac"] < 0.5

    # blank: confident nothing — low entropy AND low ink_frac
    assert blank["entropy"] < 0.2
    assert blank["ink_frac"] < 0.01
    assert blank["indecision_mass"] < 0.01


def test_model_maps_and_uint8_normalization():
    prob8 = np.zeros((256, 128), dtype=np.uint8)
    prob8[:128] = (confident_text_tile(128) * 255).astype(np.uint8)
    prob8[128:] = (confused_tile(128) * 255).astype(np.uint8)

    grid = TileGrid((256, 128), tile_px=128)
    maps = model_maps(prob8, grid)
    assert maps["prob_separation"].shape == (2, 1)
    assert maps["prob_separation"][0, 0] > 0.7  # text half
    assert maps["indecision_mass"][1, 0] > 0.6  # confused half


def test_validity_masking_and_coverage():
    n = 128
    p = confident_text_tile(n)
    valid = np.zeros((n, n), dtype=bool)
    valid[:16, :] = True  # only a sliver valid, above the stroke rows
    grid = TileGrid((n, n), tile_px=128)
    maps = model_maps(p, grid, valid=valid)
    assert maps["pred_coverage"][0, 0] == 0.125
    m_all = compute_model_metrics(p)
    assert maps["ink_frac"][0, 0] != m_all["ink_frac"]


def test_empty_tile_nan():
    p = confident_text_tile(64)
    valid = np.zeros((64, 64), dtype=bool)
    m = compute_model_metrics(p, valid)
    assert np.isnan(m["entropy"])
