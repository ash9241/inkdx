"""snap end-to-end on the phantom: recovery, traps, holds, topology."""

import numpy as np
import pytest

from inkdx.ablate.mesh import offset_z, tilt, wobble
from inkdx.grid import TileGrid
from inkdx.io.segment import Segment
from inkdx.snap.offsets import STATUS_NAMES
from inkdx.snap.orient import tile_signs
from inkdx.snap.regularize import limit_gradient
from inkdx.snap.runner import SnapConfig, SnapResult, run_snap
from inkdx.testing.synthetic import PhantomParams, make_phantom

CFG = SnapConfig(halfwidth=14, tile_px=48, iterations=4, pool=3, smooth=2.0,
                 max_offset=10.0, max_step=3.0)


@pytest.fixture(scope="module")
def phantom():
    return make_phantom(PhantomParams(undulation_amp=2.0, noise_sigma=3.0))


def median_z_error(result: SnapResult, true_z: np.ndarray) -> float:
    err = np.abs(result.segment.z - true_z)
    return float(np.median(err[result.segment.valid]))


@pytest.mark.parametrize("k", [1.0, 3.0, 6.0])
def test_recovers_rigid_offset(phantom, k):
    ph = phantom
    r = run_snap(ph.volume, offset_z(ph.segment, k), CFG)
    assert median_z_error(r, ph.sheet_z) < 0.5, r.iterations


def test_recovers_tilt_and_wobble(phantom):
    ph = phantom
    r_tilt = run_snap(ph.volume, tilt(ph.segment, 0.08), CFG)
    assert median_z_error(r_tilt, ph.sheet_z) < 0.6

    r_wob = run_snap(ph.volume, wobble(ph.segment, 4.0, 48.0, seed=3), CFG)
    assert median_z_error(r_wob, ph.sheet_z) < 0.6


def test_second_sheet_trap():
    """RELEASE GATE: mesh offset toward a neighbor sheet must come home."""
    p = PhantomParams(undulation_amp=0.0, noise_sigma=2.0, second_sheet_dz=12.0)
    ph = make_phantom(p)
    # push the mesh 5 voxels toward the second sheet (which is at +12)
    r = run_snap(ph.volume, offset_z(ph.segment, 5.0), CFG)
    err = r.segment.z - ph.sheet_z
    # recovered to own sheet...
    assert float(np.median(np.abs(err[r.segment.valid]))) < 0.75
    # ...and NOT ONE vertex crossed the midplane toward the neighbor
    assert float(err.max()) < 6.0


def test_blank_region_held_bit_identical():
    p_blank = PhantomParams(undulation_amp=0.0, sheet_contrast=0.0, noise_sigma=2.0)
    blank = make_phantom(p_blank)  # no sheet at all
    moved = offset_z(blank.segment, 4.0)
    z_before = moved.z.copy()
    r = run_snap(blank.volume, moved, CFG)
    frac_held = 1.0 - r.iterations[-1]["updated_frac"]
    assert frac_held > 0.95
    np.testing.assert_array_equal(
        r.segment.z[r.status != 1], z_before[r.status != 1]
    )


def test_idempotence_on_perfect_mesh(phantom):
    ph = phantom
    r = run_snap(ph.volume, offset_z(ph.segment, 0.0), CFG)
    assert median_z_error(r, ph.sheet_z) < 0.25
    assert float(np.abs(r.offset_total).max()) < 1.0


def test_topology_preserved(phantom):
    ph = phantom
    r = run_snap(ph.volume, wobble(ph.segment, 5.0, 32.0, seed=1), CFG)
    # Interior only: held border vertices (undefined normals) keep their
    # wobbled positions by contract, so the held/snapped boundary carries a
    # real kink — the snapped interior must be fold-free and smooth.
    z = r.segment.z[2:-2, 2:-2]
    gu = np.abs(np.diff(z, axis=0))[:, :-1]
    gv = np.abs(np.diff(z, axis=1))[:-1, :]
    assert float(np.percentile(np.maximum(gu, gv), 99.5)) < 2.0


def test_determinism_across_process_counts(phantom):
    ph = phantom
    seg = offset_z(ph.segment, 3.0)
    a = run_snap(ph.volume, seg, CFG)
    b = run_snap(ph.volume, offset_z(ph.segment, 3.0),
                 SnapConfig(**{**CFG.__dict__, "processes": 2}))
    np.testing.assert_allclose(a.segment.z, b.segment.z, atol=1e-5)
    np.testing.assert_array_equal(a.status, b.status)


def test_orientation_pass_on_flipped_windings(phantom):
    """Reversing column order flips the geometric normal winding; the sign
    field must still be internally consistent (all one component, no zeros
    in valid tiles)."""
    ph = phantom
    seg = ph.segment
    flipped = Segment(x=seg.x[:, ::-1].copy(), y=seg.y[:, ::-1].copy(),
                      z=seg.z[:, ::-1].copy(), valid=seg.valid[:, ::-1].copy(),
                      scale=seg.scale, uuid="flipped")
    grid = TileGrid(flipped.grid_shape, tile_px=32)
    signs, warnings = tile_signs(flipped, grid)
    assert (np.abs(signs) == 1).all()
    assert not warnings


def test_gradient_limiter_bound():
    field = np.zeros((40, 40), dtype=np.float32)
    field[:, 20:] = 5.0  # a hard step: gradient 5 at the boundary
    limited = limit_gradient(field, g_max=0.5)
    g = np.abs(np.diff(limited, axis=1))
    assert float(g.max()) <= 0.65  # bound holds within tolerance
    assert float(limited.max()) > 0.0  # didn't zero everything


def test_status_names_stable():
    assert STATUS_NAMES[1] == "SNAPPED"
    assert STATUS_NAMES[3] == "HELD_MULTIWRAP"
