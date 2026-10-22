"""Header recentring, differential rotation, chained references, flow reference."""

import numpy as np
import pytest

import chase._compat  # noqa: F401
from chase.calib.align import flare_mask, optical_flow_align, select_flow_reference
from chase.calib.pointing import (
    header_disc_center,
    header_shifts,
    rotation_shift,
    solar_b0,
    tie_to_header,
    track_shifts,
    windowed_shift,
)


def _smooth_noise(rng, shape, sigma=2.0):
    from scipy.ndimage import gaussian_filter

    f = gaussian_filter(rng.standard_normal(shape), sigma)
    return f / f.std()


# ---------------------------------------------------------------- disc centre
def test_header_disc_center_axes_match_limb_fit():
    """CRPIX1 is the column (x, NAXIS1) and CRPIX2 the row (y, NAXIS2).

    A non-square frame with an off-centre disc: the limb fit and the header
    agree only with CRPIX1 -> x. This is the check for the suspected axis
    swap (there is none in the code path for 2023-12 and later data).
    """
    from chase.analysis.systematics import fit_limb_ellipse

    H, W, cy, cx, R = 150, 190, 68.0, 101.0, 55.0
    yy, xx = np.mgrid[:H, :W]
    img = np.where(np.hypot(yy - cy, xx - cx) < R, 1000.0, 5.0).astype(np.float32)
    hdr = {"CRPIX1": cx + 1, "CRPIX2": cy + 1, "NAXIS1": W, "NAXIS2": H}
    hx, hy = header_disc_center(hdr)
    f = fit_limb_ellipse(img)
    assert abs(hx - f["x0"]) < 0.6 and abs(hy - f["y0"]) < 0.6
    # the swapped reading would be ~33 px off on this frame
    assert np.hypot(hy - f["x0"], hx - f["y0"]) > 20


def test_header_disc_center_fallback_to_array_centre():
    assert header_disc_center({"NAXIS1": 100, "NAXIS2": 80}) == (50.0, 40.0)


# ---------------------------------------------------------------- rotation
def test_solar_b0_extremes_and_zeros():
    assert solar_b0("2024-03-07T00:00:00") == pytest.approx(-7.25, abs=0.1)
    assert solar_b0("2024-09-08T00:00:00") == pytest.approx(7.25, abs=0.1)
    assert abs(solar_b0("2024-06-06T00:00:00")) < 0.3
    assert abs(solar_b0("2024-12-07T00:00:00")) < 0.3


def test_rotation_shift_disc_centre_one_day():
    R = 900.0
    dy, dx = rotation_shift(500.0, 600.0, (600.0, 500.0), R, 86400.0, b0=0.0)
    assert dx == pytest.approx(R * np.sin(np.deg2rad(14.713 - 0.9856)), rel=1e-6)
    assert abs(dy) < 1e-6
    # westward (+x), slower at high latitude, zero off the limb
    _, dx_hi = rotation_shift(500.0 + 0.6 * R, 600.0, (600.0, 500.0), R, 86400.0)
    assert 0 < dx_hi < dx
    assert rotation_shift(500.0, 600.0 + 1.1 * R, (600.0, 500.0), R, 86400.0) == (0.0, 0.0)


def test_rotation_shift_north_angle_rotates_vector():
    R = 900.0
    dy0, dx0 = rotation_shift(500.0, 600.0, (600.0, 500.0), R, 3600.0, north_angle=0.0)
    dy9, dx9 = rotation_shift(500.0, 600.0, (600.0, 500.0), R, 3600.0, north_angle=90.0)
    # north rotated 90 deg counter-clockwise: west now points to +y
    assert dx9 == pytest.approx(0.0, abs=1e-6) and dy9 == pytest.approx(dx0, rel=1e-6)


def test_header_shifts_sign_convention():
    centers = [(1000.0, 1100.0), (1003.0, 1098.0), (1005.5, 1101.0)]
    times = ["2024-12-14T12:00:00", "2024-12-14T12:01:10", "2024-12-14T12:02:20"]
    sh = header_shifts(centers, times, roi_center=(1100.0, 1000.0), radius=930.0, rotation=False)
    assert sh[0] == (0.0, 0.0)
    assert sh[1] == pytest.approx((2.0, -3.0)) and sh[2] == pytest.approx((-1.0, -5.5))
    # rotation at disc centre adds a westward motion -> more negative sx
    rot = header_shifts(centers, times, roi_center=(1100.0, 1000.0), radius=930.0)
    assert rot[2][1] < sh[2][1]


# ---------------------------------------------------------------- chained tracking
def _evolving_series(n=30, size=72, a=0.80, seed=1):
    """A field that decorrelates over time (AR(1) in the image) and drifts."""
    rng = np.random.default_rng(seed)
    base = _smooth_noise(rng, (size + 40, size + 40))
    fields = [base]
    for _ in range(n - 1):
        fields.append(a * fields[-1] + np.sqrt(1 - a * a) * _smooth_noise(rng, base.shape))
    drift = np.cumsum(rng.normal(0, 1.0, (n, 2)), axis=0)
    drift[0] = 0
    wins = []
    for i in range(n):
        dy, dx = np.round(drift[i]).astype(int)
        wins.append(fields[i][20 + dy:20 + dy + size, 20 + dx:20 + dx + size].astype(np.float32))
    true = np.round(drift)           # window moves +d -> content -d -> registration +d
    return wins, true


def test_track_shifts_first_fails_on_evolving_field_chained_does_not():
    wins, true = _evolving_series(a=0.9)
    first, _ = track_shifts(wins, reference="first", shift_fn=windowed_shift)
    prev, _ = track_shifts(wins, reference="previous", shift_fn=windowed_shift)
    run, steps = track_shifts(wins, reference="running", reference_window=3,
                              shift_fn=windowed_shift)
    err = lambda s: np.hypot(*(np.asarray(s) - true).T)
    assert err(first)[-8:].max() > 10          # lost once the field has evolved
    assert err(prev).max() < 2.5               # chained: small, but it random-walks
    assert err(run).max() < 2.5
    # tying the chain to a 1 px header keeps it from wandering
    header = true + np.random.default_rng(0).normal(0, 1.0, true.shape)
    header[0] = 0
    tied = tie_to_header(steps, header)
    assert err(tied).max() < err(run).max() and err(tied).max() < err(header).max()


def test_track_shifts_first_matches_v21_phase_shift():
    from chase.calib.spatial import phase_shift

    wins, _ = _evolving_series(n=5)
    first, steps = track_shifts(wins, reference="first")
    assert first == steps
    assert first[3] == pytest.approx(phase_shift(wins[0], wins[3]))


def test_tie_to_header_bounds_random_walk_and_rejects_outlier():
    rng = np.random.default_rng(3)
    n = 60
    true = np.cumsum(rng.normal(0, 1.0, (n, 2)), axis=0)
    true[0] = 0
    steps = np.vstack([[0, 0], np.diff(true, axis=0) + rng.normal(0, 0.25, (n - 1, 2))])
    steps[30] += (15.0, -12.0)                   # a failed correlation
    header = true + rng.normal(0, 1.0, (n, 2))
    header[0] = 0
    chained = np.cumsum(steps, axis=0)
    tied = np.asarray(tie_to_header(steps, header))
    e = lambda s: np.sqrt(np.mean(np.sum((np.asarray(s) - true) ** 2, axis=1)))
    assert e(chained) > 10                        # the outlier propagates
    assert e(tied) < 0.75 * e(header)             # better than either input
    assert tied[0].tolist() == [0.0, 0.0]


# ---------------------------------------------------------------- loader modes
@pytest.fixture
def haonly_drift_dataset(tmp_path):
    """Hα-only scans of a low-contrast field that moves with the header CRPIX."""
    import astropy.io.fits as fits

    rng = np.random.default_rng(7)
    H = W = 96
    nch = 12
    wav = np.linspace(6559.4, 6565.1, nch)
    spec = 1000.0 * (1.0 - 0.6 * np.exp(-0.5 * ((wav - 6562.8) / 0.4) ** 2))
    scene = 1.0 + 0.02 * _smooth_noise(rng, (H + 40, W + 40), 3.0)
    drift = [(0.0, 0.0), (1.4, -2.3), (-0.6, 3.2), (2.5, 0.7), (-3.1, -1.6)]
    d = tmp_path / "ha"
    d.mkdir()
    from scipy.ndimage import shift as ndshift

    for i, (dy, dx) in enumerate(drift):
        img = ndshift(scene, (dy, dx), order=3, mode="nearest")[20:20 + H, 20:20 + W]
        cube = (spec[:, None, None] * img[None]).astype(np.float32)
        cube += rng.normal(0, 2.0, cube.shape).astype(np.float32)   # noise >> structure in xcorr
        hdu = fits.ImageHDU(data=cube)
        hdu.header["CRVAL3"], hdu.header["CDELT3"] = wav[0], wav[1] - wav[0]
        hdu.header["CRPIX1"], hdu.header["CRPIX2"] = 48.0 + 1 + dx, 48.0 + 1 + dy
        hdu.header["R_SUN"] = 4000.0          # large disc: tiny rotation
        hdu.header["DATE_OBS"] = f"2024-12-14T12:0{i}:00"
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(d / f"RSM2024121400{i}_HA.fits")
    return {"dir": str(d), "drift": drift, "patch": [28, 68, 28, 68]}


def _crop_error(seq, drift):
    """Residual misregistration of the loaded crops (px), from the injected drift."""
    sh = np.asarray(seq["shifts"], dtype=float)
    return np.hypot(*(sh + np.asarray(drift)).T)


def test_loader_header_mode_follows_crpix(haonly_drift_dataset):
    from chase.io import load_flare_sequence

    ds = haonly_drift_dataset
    seq = load_flare_sequence(ds["dir"], patch=ds["patch"], pad=8, verbose=False,
                              align_mode="header", rotation=False)
    assert seq["pointing"]["align_mode"] == "header"
    assert _crop_error(seq, ds["drift"]).max() <= 0.71      # integer rounding only
    sub = load_flare_sequence(ds["dir"], patch=ds["patch"], pad=8, verbose=False,
                              align_mode="header", rotation=False, subpixel=True)
    assert _crop_error(sub, ds["drift"]).max() < 1e-6
    # sub-pixel crops really register the field better than integer crops
    c0 = sub["ha_cubes"][0, 0]
    e_sub = np.mean([np.abs(sub["ha_cubes"][i, 0] - c0).mean() for i in range(1, 5)])
    e_int = np.mean([np.abs(seq["ha_cubes"][i, 0] - seq["ha_cubes"][0, 0]).mean() for i in range(1, 5)])
    assert e_sub < e_int


def test_loader_header_xcorr_and_running_modes_run(haonly_drift_dataset):
    from chase.io import load_flare_sequence

    ds = haonly_drift_dataset
    for kw in (dict(align_mode="header+xcorr"),
               dict(align_mode="header+xcorr", reference="running"),
               dict(align_mode="xcorr", reference="previous", drift_control="header",
                    shift_method="windowed")):
        seq = load_flare_sequence(ds["dir"], patch=ds["patch"], pad=8, verbose=False,
                                  rotation=False, **kw)
        assert seq["ha_cubes"].shape[0] == 5
        assert _crop_error(seq, ds["drift"]).max() < 1.6, kw


def test_loader_default_is_unchanged(synthetic_dataset):
    from chase.io import load_flare_sequence

    a = load_flare_sequence(synthetic_dataset["dir"], patch=synthetic_dataset["patch"],
                            pad=10, verbose=False)
    b = load_flare_sequence(synthetic_dataset["dir"], patch=synthetic_dataset["patch"],
                            pad=10, verbose=False, align_mode="xcorr", reference="first")
    assert a["shifts"] == b["shifts"] and all(isinstance(v, int) for s in a["shifts"] for v in s)
    np.testing.assert_array_equal(a["ha_cubes"], b["ha_cubes"])


def test_loader_rejects_bad_mode(synthetic_dataset):
    from chase.io import load_flare_sequence

    with pytest.raises(ValueError):
        load_flare_sequence(synthetic_dataset["dir"], patch=synthetic_dataset["patch"],
                            verbose=False, align_mode="hough")


def test_config_and_cli_expose_modes():
    from chase.cli import build_parser, config_from_args

    args = build_parser().parse_args(["data", "--align-mode", "header", "--reference", "running",
                                      "--drift-control", "header", "--subpixel", "--no-rotation",
                                      "--flow-reference", "stable+masked", "--disc-center", "auto"])
    cfg = config_from_args(args)
    assert (cfg.align_mode, cfg.reference, cfg.drift_control, cfg.subpixel, cfg.rotation,
            cfg.flow_reference, cfg.disc_center) == ("header", "running", "header", True, False,
                                                     "stable+masked", "auto")
    from chase.config import Config
    d = Config()
    assert (d.align_mode, d.reference, d.flow_reference, d.subpixel) == ("xcorr", "first", "default", False)


# ---------------------------------------------------------------- flow reference
def _flare_cubes(seed=0, n=6, size=80, shift=(1.3, -1.7)):
    """Wing channels with photospheric structure; the last one brightens in a kernel."""
    from scipy.ndimage import shift as ndshift

    rng = np.random.default_rng(seed)
    scene = 1.0 + 0.15 * _smooth_noise(rng, (size, size), 2.5)
    yy, xx = np.mgrid[:size, :size]
    kernel = np.exp(-((yy - 30) ** 2 + (xx - 46) ** 2) / (2 * 6.0 ** 2))
    nch = 5
    wav = np.array([6559.4, 6559.6, 6562.8, 6564.9, 6565.1])
    cubes = np.empty((n, nch, size, size), np.float32)
    for i in range(n):
        s = (shift[0] * i / (n - 1), shift[1] * i / (n - 1))
        for c in range(nch):
            img = scene.copy()
            if c in (3, 4):
                img = img * (1 + 1.5 * kernel * i / (n - 1))   # red wing brightens
            cubes[i, c] = ndshift(img, s, order=3, mode="nearest")
    return cubes, wav, shift


def test_select_flow_reference_avoids_brightening_channel():
    cubes, wav, _ = _flare_cubes()
    sel = select_flow_reference(cubes, wav)
    assert sel["channels"][0] in (0, 1)
    assert sel["score_last_channel"] > 5 * min(sel["scores"].values())
    sel2 = select_flow_reference(cubes, wav, exclude=[0])
    assert sel2["channels"] == [1]


def test_flare_mask_marks_kernel_only():
    cubes, _, _ = _flare_cubes(shift=(0.0, 0.0))
    m = flare_mask(cubes[:, -1], threshold=0.1)
    assert not m[0].any()
    assert m[-1][30, 46] and not m[-1][5, 5]
    assert m[-1].mean() < 0.25


def test_masked_flow_is_less_biased_by_brightening():
    cubes, _, shift = _flare_cubes()
    ref = cubes[:, -1]
    plain = optical_flow_align(cubes, reference=ref, method="farneback")
    masked = optical_flow_align(cubes, reference=ref, method="farneback",
                                mask=flare_mask(ref, threshold=0.1))
    # residual misregistration on a channel without the flare (channel 0)
    e_plain = np.abs(plain[-1, 0, 10:-10, 10:-10] - plain[0, 0, 10:-10, 10:-10]).mean()
    e_mask = np.abs(masked[-1, 0, 10:-10, 10:-10] - masked[0, 0, 10:-10, 10:-10]).mean()
    assert e_mask < e_plain
    # and the mask=None path is the v2.1 path
    again = optical_flow_align(cubes, reference=ref, method="farneback", mask=None)
    np.testing.assert_array_equal(again, plain)
