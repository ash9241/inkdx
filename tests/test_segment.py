import numpy as np
import pytest

from inkdx.io.segment import INVALID, read_tifxyz, write_tifxyz
from inkdx.testing.synthetic import PhantomParams, make_phantom


@pytest.fixture
def phantom():
    return make_phantom(PhantomParams())


def test_tifxyz_roundtrip(tmp_path, phantom):
    seg = phantom.segment
    write_tifxyz(tmp_path / "seg", seg.x, seg.y, seg.z, valid=seg.valid,
                 scale=seg.scale, uuid="roundtrip")
    loaded = read_tifxyz(tmp_path / "seg")

    assert loaded.grid_shape == seg.grid_shape
    assert loaded.uuid == "roundtrip"
    assert loaded.scale == seg.scale
    np.testing.assert_allclose(loaded.z[loaded.valid], seg.z[seg.valid], atol=1e-5)
    assert loaded.valid.all()


def test_invalid_sentinel_and_z_rule(tmp_path, phantom):
    seg = phantom.segment
    valid = seg.valid.copy()
    valid[:5, :] = False  # explicit mask invalidation
    z = seg.z.copy()
    z[10, 10] = 0.0  # z <= 0 must be invalidated at load

    write_tifxyz(tmp_path / "seg", seg.x, seg.y, z, valid=valid, scale=(1.0, 1.0))
    loaded = read_tifxyz(tmp_path / "seg")

    assert not loaded.valid[:5, :].any()
    assert not loaded.valid[10, 10]
    assert loaded.x[0, 0] == INVALID and loaded.z[10, 10] == INVALID
    assert loaded.valid[20, 20]


def test_scale_cpp_order(tmp_path, phantom):
    seg = phantom.segment
    write_tifxyz(tmp_path / "seg", seg.x, seg.y, seg.z, scale=(4.0, 20.0))  # (y, x)
    loaded = read_tifxyz(tmp_path / "seg")
    assert loaded.scale == (4.0, 20.0)
    # C++ order on disk: [x_scale, y_scale]
    assert loaded.meta["scale"] == [20.0, 4.0]


def test_normals_flat_sheet():
    p = PhantomParams(undulation_amp=0.0, noise_sigma=0.0)
    seg = make_phantom(p).segment
    n = seg.normals()

    interior = np.isfinite(n[..., 0])
    assert interior[2:-2, 2:-2].all()
    assert not np.isfinite(n[0, 0, 0])  # border undefined

    norms = np.linalg.norm(n[interior], axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    # flat sheet in z: normal is +/- z
    assert np.abs(n[..., 2][interior]).min() > 0.999


def test_normals_undefined_next_to_holes(phantom):
    seg = phantom.segment
    seg.valid[40, 40] = False
    n = seg.normals()
    assert not np.isfinite(n[40, 41, 0])  # neighbor of a hole
    assert np.isfinite(n[40, 43, 0])
