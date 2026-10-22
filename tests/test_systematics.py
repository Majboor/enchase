"""Stray-light, throughput, atlas-sensitivity and bug-fix checks (v2.1.1)."""

import warnings

import numpy as np
import pytest

from chase.analysis import (
    contrast_bias,
    decimal_year,
    fit_limb_ellipse,
    offlimb_profile,
    psf_stray_light,
    throughput_trend,
)
from chase.calib import activity_mask, resample_wavelength
from chase.io import robust_shifts


def _disc(H=320, W=320, cx=161.3, cy=158.7, ax=100.0, ay=103.0):
    yy, xx = np.mgrid[:H, :W]
    rho = np.hypot((xx - cx) / ax, (yy - cy) / ay)
    mu = np.sqrt(np.clip(1 - rho ** 2, 0, 1))
    return np.where(rho < 1, 1000.0 * (0.4 + 0.6 * mu), 0.0)


def _scatter(D, eps, a_px, beta):
    """(1-eps) D + eps (D * K), Moffat K, by zero-padded FFT convolution."""
    N = 2 * max(D.shape)
    pad = np.zeros((N, N))
    pad[:D.shape[0], :D.shape[1]] = D
    k = np.fft.fftfreq(N) * N
    K = (1 + (np.hypot(k[:, None], k[None, :]) / a_px) ** 2) ** (-beta)
    S = np.fft.irfft2(np.fft.rfft2(pad) * np.fft.rfft2(K / K.sum()), s=(N, N))[:D.shape[0], :D.shape[1]]
    return (1 - eps) * D + eps * S, eps * S


def test_fit_limb_ellipse_recovers_axes():
    img = _disc()
    L = fit_limb_ellipse(img, threshold=0.2)
    # the 0.2 level sits just inside the true edge on a limb-darkened disc
    assert abs(L["x0"] - 161.3) < 0.3 and abs(L["y0"] - 158.7) < 0.3
    assert abs(L["ax"] - 100.0) < 1.0 and abs(L["ay"] - 103.0) < 1.0
    assert L["ay"] - L["ax"] == pytest.approx(3.0, abs=0.3)


def test_offlimb_profile_and_prominence_rejection():
    img, _ = _scatter(_disc(), eps=0.08, a_px=10.0, beta=1.5)
    L = fit_limb_ellipse(img)
    core = img.copy()
    core[150:170, 255:300] += 500.0       # a bright prominence off the west limb
    p = offlimb_profile(img, limb=L, core_image=core)
    f = p["fraction"]
    assert f["1.02"] > f["1.05"] > f["1.10"] > f["1.20"] > 0
    assert 1 <= p["sectors_rejected"] <= 6
    assert p["sectors_kept"] + p["sectors_rejected"] == 72


def test_psf_stray_light_recovers_disc_centre_fraction():
    D = _disc()
    img, S = _scatter(D, eps=0.08, a_px=12.0, beta=1.5)
    L = fit_limb_ellipse(img)
    p = offlimb_profile(img, limb=L)
    true = float(S[158, 161] / p["I_dc"])
    r = psf_stray_light(img, p, kinds=("moffat",), binning=4, rfit=(1.03, 1.6))
    assert r["best"] == "moffat"
    assert abs(r["models"]["moffat"]["s_dc"] - true) < 0.35 * true
    assert r["s_dc_range"][0] == r["s_dc_range"][1]


def test_contrast_bias():
    assert contrast_bias(0.2, 0.0) == pytest.approx(0.2)
    assert contrast_bias(0.2, 0.1) == pytest.approx(0.28)
    assert contrast_bias(1.0, 0.1) == pytest.approx(1.0)


def test_decimal_year():
    assert decimal_year("2024-01-01T00:00:00") == pytest.approx(2024.0)
    assert decimal_year("2023-07-02T12:00:00") == pytest.approx(2023.5, abs=1e-3)
    assert decimal_year(2025.25) == 2025.25


def test_throughput_trend_recovers_rate():
    t = [2023.24, 2023.99, 2024.75, 2024.95, 2026.51]
    true = -11.7                                     # % per year
    b = -np.log(1 + true / 100)
    ifact = 2.3e-8 * np.exp(b * (np.array(t) - t[0]))
    r = throughput_trend(t, ifact)
    assert r["throughput_pct_per_yr"] == pytest.approx(true, abs=1e-6)
    assert r["resid_rms_pct"] < 1e-8 and r["throughput_pct_per_yr_se"] < 1e-6
    lo, hi = r["jackknife_pct_per_yr"]
    assert lo == pytest.approx(true, abs=1e-6) and hi == pytest.approx(true, abs=1e-6)
    two = throughput_trend(["2023-03-29T02:12:31", "2026-07-04T05:00:00"], [2.302e-8, 3.301e-8])
    assert two["total_change_pct"] == pytest.approx(100 * (2.302 / 3.301 - 1))
    assert "slope_se" not in two


def test_activity_mask_rejects_spot_and_plage_keeps_dark_fibril():
    cont = np.ones((40, 40))
    core = np.ones((40, 40))
    cont[:8, :8] = 0.85                  # penumbra
    core[30:, 30:] = 1.3                 # plage: bright Hα core
    core[20, :] = 0.7                    # dark fibril: kept
    m = activity_mask(cont, core)
    assert not m[:8, :8].any() and not m[30:, 30:].any()
    assert m[20, 15] and m[15, 15]


def test_resample_wavelength_applies_small_real_drift():
    # 0.04 Å drift on a 0.0485 Å grid at 6560 Å (the real CHASE case) is below
    # numpy's default allclose rtol (~0.066 Å); v2.1.0 returned the cube unchanged.
    base = np.arange(118) * 0.0485 + 6559.42
    spec = 1 - 0.6 * np.exp(-0.5 * ((base - 6562.8) / 0.3) ** 2)
    cube = np.repeat(spec[:, None, None], 4, axis=1).reshape(118, 2, 2).astype(np.float32)
    out = resample_wavelength(cube, base + 0.04, base)
    assert np.abs(out - cube).max() > 1e-2
    expect = 1 - 0.6 * np.exp(-0.5 * ((base - 0.04 - 6562.8) / 0.3) ** 2)
    assert np.abs(out[10:-10, 0, 0] - expect[10:-10]).max() < 1e-3


def test_robust_shifts_never_moves_reference():
    # Even scans carry a real -8 px offset that the reference (0, 0) does not:
    # without ref_idx the reference is "repaired" to -8, shifting the series.
    sh = [(0.0, 0.0)] + [(-8.0 if i % 2 == 0 else 1.0, 0.0) for i in range(1, 14)]
    old = robust_shifts(sh, threshold=4.0)
    new = robust_shifts(sh, threshold=4.0, ref_idx=0)
    assert old[0][0] == -8.0
    assert new[0] == (0.0, 0.0)
    assert new[1:] == old[1:]


def _ispy_cube(wav, ifact=2.0e-8, n=20):
    from ISPy.spec import atlas as isp_atlas

    w, s, _ = isp_atlas.atlas().get(wav[0] - 0.5, wav[-1] + 0.5, cgs=True, perHz=True)
    prof = np.interp(wav, w, s)
    return np.repeat((prof / ifact)[:, None, None], n * n, axis=1).reshape(len(wav), n, n)


def test_wing_channels_and_halpha_warning():
    pytest.importorskip("ISPy")
    from chase.calib import atlas_calibrate, wing_channels

    fe = 6567.50 + 0.0485 * np.arange(46)
    ha = 6559.40 + 0.0485 * np.arange(118)
    n = [len(wing_channels(fe, f)) for f in (0.70, 0.85, 0.90, 0.95)]
    assert n[0] >= n[1] >= 38 and n[2] < 10 and n[3] == 0
    assert len(wing_channels(ha, 0.85)) == 0 and len(wing_channels(ha, 0.75)) > 20
    with pytest.warns(UserWarning, match="no channel"):
        atlas_calibrate(ha, _ispy_cube(ha, n=4))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        atlas_calibrate(fe, _ispy_cube(fe, n=4))


def test_atlas_sensitivity_plateau_and_series():
    pytest.importorskip("ISPy")
    from chase.calib import atlas_factor_series, atlas_sensitivity

    fe = 6567.50 + 0.0485 * np.arange(46)
    cube = _ispy_cube(fe)
    r = atlas_sensitivity(fe, cube, thresholds=(0.75, 0.85))
    assert r["reference"]["ifact"] == pytest.approx(2.0e-8, rel=0.01)
    assert 10 < r["reference"]["K_per_pct"] < 25
    assert len(r["fits"]) == 4
    for f in r["fits"]:
        assert abs(f["delta_pct"]) < 0.5 and f["kept_frac"] == pytest.approx(1.0)
    s = atlas_factor_series([(fe, cube), (fe, cube * 1.002)])
    assert s["ifact_sd_pct"] == pytest.approx(100 * np.std([1, 1 / 1.002], ddof=1), rel=0.2)
