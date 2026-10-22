"""Validation metrics and the active-pixel mask, checked against ground truth."""

import numpy as np

from chase.analysis import (
    frame_rms,
    line_centroid,
    residual_pointing,
    residual_wavelength,
    stray_light_estimate,
)
from chase.calib import find_quiet_patch, quiet_sun_mask


def _granulation(H=96, W=96, seed=0, sigma=4):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    return 1.0 + 0.05 * gaussian_filter(rng.standard_normal((H, W)), sigma) / 0.05


def test_residual_pointing_measures_known_shift():
    from scipy.ndimage import shift as nd_shift

    ref = _granulation()
    frames = np.stack([ref, nd_shift(ref, (0.5, -0.25), order=3, mode="nearest")])
    r = residual_pointing(frames, pixel_scale=1.04, local=False)
    assert abs(r["global_rms_px"] - np.hypot(0.5, 0.25)) < 0.08
    assert abs(r["global_rms_arcsec"] - 1.04 * r["global_rms_px"]) < 1e-9


def test_residual_pointing_zero_for_identical_frames():
    ref = _granulation()
    r = residual_pointing(np.stack([ref, ref, ref]))
    assert r["global_rms_px"] < 1e-6 and r["local_median_arcsec"] < 0.05


def test_line_centroid_subchannel():
    wav = 6569.0 + 0.049 * np.arange(-10, 11)
    for true in (6569.0, 6569.013, 6569.031):
        spec = 1 - 0.6 * np.exp(-0.5 * ((wav - true) / 0.08) ** 2)
        assert abs(line_centroid(wav, spec) - true) < 0.002


def test_residual_wavelength_drift_and_kms():
    wav = 6569.0 + 0.049 * np.arange(-10, 11)
    spectra, grids = [], []
    for d in (0.0, 0.02, -0.02, 0.04):
        spectra.append(1 - 0.6 * np.exp(-0.5 * ((wav - 6569.2 - d) / 0.08) ** 2))
        grids.append(wav)
    r = residual_wavelength(spectra, grids, rest_wavelength=6569.2)
    assert 20 < r["std_mA"] < 30
    assert abs(r["std_kms"] - r["std_mA"] * 1e-3 / 6569.2 * 299792.458) < 1e-9


def test_frame_rms_percent_and_cgs():
    f = np.stack([np.full((10, 10), 100.0), np.full((10, 10), 102.0)])
    r = frame_rms(f, ifact=2.0)
    assert abs(r["rms_dn"] - 2.0) < 1e-12 and abs(r["rms_pct"] - 2 / 101 * 100) < 1e-9
    assert r["rms_cgs"] == 4.0


def test_quiet_sun_mask_rejects_penumbra_and_plage():
    cube = np.ones((5, 40, 40))
    cube[2] *= 0.3                       # line core
    cube[:, :10, :10] *= 0.85            # penumbra: dark continuum
    cube[2, 30:, 30:] *= 1.4             # plage: bright core, normal continuum
    m = quiet_sun_mask(cube)
    assert not m[:10, :10].any() and not m[30:, 30:].any() and m[20, 20]


def test_find_quiet_patch_avoids_spot():
    cont = _granulation(300, 300)
    core = _granulation(300, 300, seed=1)
    cont[120:200, 120:200] *= 0.3        # a spot on the reference pixel
    cy, cx = find_quiet_patch(cont, core, (160, 160), size=60, search_radius=100, step=20)
    assert not (90 < cy < 230 and 90 < cx < 230)


def test_stray_light_estimate_recovers_halo():
    H = W = 400
    yy, xx = np.ogrid[:H, :W]
    rr = np.hypot(yy - 200, xx - 200)
    img = np.where(rr < 150, 1000.0, 0.0) + 10.0 * np.exp(-(rr - 150).clip(0) / 30)
    r = stray_light_estimate(img, center=(200, 200), radius=150)
    assert abs(r["fraction"]["1.02"] - 10 * np.exp(-3 / 30) / 1010) < 2e-3
    assert r["fraction"]["1.20"] < r["fraction"]["1.02"]


def test_atlas_calibrate_mask_removes_spot_bias():
    import pytest

    pytest.importorskip("ISPy")
    from ISPy.spec import atlas as isp_atlas

    from chase.calib import atlas_calibrate

    wav = 6567.52 + 0.0485 * np.arange(46)
    w, s, c = isp_atlas.atlas().get(wav[0] - 0.5, wav[-1] + 0.5, cgs=True, perHz=True)
    prof = np.interp(wav, w, s)
    true_ifact = 2.0e-8
    cube = np.repeat((prof / true_ifact)[:, None, None], 40 * 40, axis=1).reshape(46, 40, 40)
    clean, _, q = atlas_calibrate(wav, cube, return_quality=True)
    spotted = cube.copy()
    spotted[:, :16, :16] *= 0.8          # 16 % of the patch is penumbra at 0.8 I_QS
    biased, _ = atlas_calibrate(wav, spotted)
    masked, _ = atlas_calibrate(wav, spotted, mask_active=True)
    assert abs(clean / true_ifact - 1) < 0.01 and q["rms_wings_pct"] < 1.0
    assert abs(biased / true_ifact - 1) > 0.02
    assert abs(masked / true_ifact - 1) < 0.01
