"""Sub-pixel limb fit (circle / rotated ellipse) and centre-to-limb calibration."""

import numpy as np
import pytest

from chase.calib import (
    clv_factor,
    fit_limb,
    limb_disk_center,
    limb_shifts,
    load_clv_table,
    mu_map,
    recenter,
)


def _disc(shape=(420, 400), cx=201.3, cy=208.7, a=150.0, b=146.0, ang=30.0, ld=0.6, seed=0):
    """Limb-darkened elliptical disc with noise."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    u = (c * (xx - cx) + s * (yy - cy)) / a
    v = (-s * (xx - cx) + c * (yy - cy)) / b
    rho = np.hypot(u, v)
    mu = np.sqrt(np.clip(1 - rho ** 2, 0, 1))
    im = np.where(rho < 1, 1000.0 * (1 - ld * (1 - mu)), 5.0)
    from scipy.ndimage import gaussian_filter
    im = gaussian_filter(im, 1.5)
    return im + np.random.default_rng(seed).normal(0, 2.0, shape)


def test_fit_limb_circle_subpixel():
    im = _disc(a=150.0, b=150.0)
    f = fit_limb(im, model="circle")
    assert abs(f["cx"] - 201.3) < 0.1 and abs(f["cy"] - 208.7) < 0.1
    assert abs(f["radius_px"] - 150.0) < 1.5
    assert f["n_kept"] > 600


def test_fit_limb_ellipse_recovers_axes_and_angle():
    im = _disc(a=150.0, b=144.0, ang=30.0)
    f = fit_limb(im, model="ellipse")
    assert abs(f["cx"] - 201.3) < 0.1 and abs(f["cy"] - 208.7) < 0.1
    assert abs((f["a"] - f["b"]) - 6.0) < 0.3
    assert abs(f["angle_deg"] - 30.0) < 3.0
    assert f["rms_px"] < 0.3


def test_ellipse_centre_beats_circle_on_partial_elliptical_limb():
    # an elliptical disc cut by the detector edge: the circle centre is biased
    im = _disc(shape=(380, 400), cx=200.0, cy=225.0, a=140.0, b=150.0, ang=90.0)
    fc = fit_limb(im, model="circle")
    fe = fit_limb(im, model="ellipse")
    err_c = np.hypot(fc["cx"] - 200.0, fc["cy"] - 225.0)
    err_e = np.hypot(fe["cx"] - 200.0, fe["cy"] - 225.0)
    assert err_e < 0.2 and err_c > 3 * err_e


def test_fit_limb_rejects_prominence_rays():
    im = _disc(a=150.0, b=150.0)
    im[200:215, 345:380] = 800.0             # bright blob across the east limb
    f = fit_limb(im, model="circle")
    assert abs(f["cx"] - 201.3) < 0.15 and f["n_kept"] < f["n_pts"]


def test_fit_limb_bad_model():
    with pytest.raises(ValueError):
        fit_limb(_disc(), model="square")


def test_limb_shifts_register_sequence():
    offs = [(0, 0), (3, -2), (-4, 5)]
    ims = [_disc(cx=201.3 + dx, cy=208.7 + dy, seed=k) for k, (dx, dy) in enumerate(offs)]
    sh = limb_shifts(ims, model="ellipse")
    assert sh.dtype.kind == "i"
    assert [tuple(s) for s in sh] == [(0, 0), (-3, 2), (4, -5)]
    reg = recenter(ims[1][None], tuple(sh[1]))[0]
    assert abs(limb_disk_center(reg)[0] - 201.3) < 0.2


def test_mu_map_circle_and_ellipse():
    m = mu_map((101, 101), (50, 50), 40.0)
    assert m[50, 50] == pytest.approx(1.0)
    assert m[50, 90] != m[50, 90]             # NaN on the limb
    assert m[50, 70] == pytest.approx(np.sqrt(1 - 0.25))
    me = mu_map((101, 101), (50, 50), (40.0, 20.0, 90.0))
    assert me[50, 60] == pytest.approx(np.sqrt(1 - 0.25))


def _fake_clv(tmp_path):
    import astropy.io.fits as fits
    wav = np.linspace(6561.0, 6564.6, 37)
    mu = np.linspace(1.0, 0.02, 50)
    prof = 1 - 0.8 * np.exp(-((wav - 6562.8) / 0.5) ** 2)
    # line core darkens less than the wings: ratio = mu**(0.5 * prof)
    I = np.array([prof * m ** (0.5 * prof) for m in mu]).T        # (nw, nmu) as in CDS
    h = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(wav, name="WAV"), fits.ImageHDU(mu, name="MU"),
                      fits.ImageHDU(I, name="INT"), fits.ImageHDU(I, name="INT_SMOOTH")])
    p = tmp_path / "6563_clv.fits"
    h.writeto(p)
    return str(p), wav, prof


def test_load_clv_table_and_factor(tmp_path):
    pytest.importorskip("ISPy")
    p, wav, prof = _fake_clv(tmp_path)
    t = load_clv_table(p)
    assert t["ratio"].shape == (50, 37) and t["mu"][0] < t["mu"][-1]
    assert np.allclose(t["ratio"][-1], 1.0)
    f = clv_factor(wav, 0.5, t)
    assert np.allclose(f, 0.5 ** (0.5 * prof), rtol=2e-3)
    # outside the table: continuum limb darkening
    f2 = clv_factor(np.array([6559.5, 6562.8]), np.array([1.0, 0.3]), t)
    assert f2.shape == (2, 2) and np.allclose(f2[0], 1.0, atol=2e-3)
    assert 0.3 < f2[1, 0] < 0.8


def test_limb_darkening_neckel_labs():
    pytest.importorskip("ISPy")
    from chase.calib import limb_darkening
    from ISPy.spec.calib import limbdarkening
    w = np.array([6563.0, 6569.0])
    for m in (1.0, 0.6, 0.2):
        assert np.allclose(limb_darkening(w, m), limbdarkening(w, mu=m), rtol=1e-10)
    assert limb_darkening(w, np.array([1.0, 0.5])).shape == (2, 2)


def test_atlas_calibrate_off_centre_box_recovers_dc_factor():
    pytest.importorskip("ISPy")
    from ISPy.spec import atlas as isp_atlas
    from chase.calib import atlas_calibrate, limb_darkening

    wav = np.arange(6567.5, 6569.7, 0.048)
    wf, sf, _ = isp_atlas.atlas().get(wav[0] - 0.5, wav[-1] + 0.5, cgs=True, perHz=True)
    ifact_true = 2.5e-8
    mu = 0.45
    prof = np.interp(wav, wf, sf * limb_darkening(wf, mu)) / ifact_true
    cube = np.repeat(prof[:, None, None], 4, axis=1).repeat(4, axis=2)
    i_dc, _ = atlas_calibrate(wav, cube)                        # wrong: no CLV
    i_mu, _ = atlas_calibrate(wav, cube, mu=mu)
    i_map, _ = atlas_calibrate(wav, cube, mu=np.full((4, 4), mu))
    assert abs(i_mu / ifact_true - 1) < 2e-3 and abs(i_map / i_mu - 1) < 1e-6
    assert i_dc / ifact_true > 1.25                              # off by I(1)/I(mu)


def test_download_clv_table_checks_hash(tmp_path):
    from chase.calib import download_clv_table
    p = tmp_path / "6563_clv.fits"
    p.write_bytes(b"not the CDS file")
    with pytest.raises(ValueError):
        download_clv_table("6563_clv.fits", str(tmp_path))       # existing file, wrong hash, no network
    assert download_clv_table("6563_clv.fits", str(tmp_path), verify=False) == str(p)
