import numpy as np

from inkdx.testing.synthetic import PhantomParams, make_phantom, sheet_height


def test_sheet_is_bright_against_background():
    p = PhantomParams(noise_sigma=0.0)
    ph = make_phantom(p)
    z_s = np.round(sheet_height(p)).astype(int)
    yy, xx = np.meshgrid(np.arange(p.shape[1]), np.arange(p.shape[2]), indexing="ij")
    on_sheet = ph.volume[z_s, yy, xx].astype(float)
    far = ph.volume[-1].astype(float)  # top slice, far from sheet
    assert on_sheet.mean() > p.background + 0.8 * p.sheet_contrast
    assert abs(far.mean() - p.background) < 2.0


def test_second_sheet():
    p = PhantomParams(noise_sigma=0.0, undulation_amp=0.0, second_sheet_dz=12.0)
    ph = make_phantom(p)
    z0 = int(p.sheet_z0)
    assert ph.volume[z0 + 12, 10, 10] > p.background + 0.8 * p.sheet_contrast


def test_deterministic_under_seed():
    a = make_phantom(PhantomParams(seed=7)).volume
    b = make_phantom(PhantomParams(seed=7)).volume
    c = make_phantom(PhantomParams(seed=8)).volume
    assert (a == b).all()
    assert (a != c).any()


def test_segment_matches_true_sheet():
    p = PhantomParams()
    ph = make_phantom(p)
    np.testing.assert_allclose(ph.segment.z, ph.sheet_z, atol=1e-6)
    assert ph.segment.valid.all()
