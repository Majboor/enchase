"""Quantitative validation of each calibration step.

Every function returns plain numbers (dicts of floats) so a run can be written
to JSON and quoted in a paper table:

* :func:`residual_pointing` — leftover frame-to-frame registration error after
  tracking/optical flow, globally (phase correlation) and locally (dense flow),
  in pixels and arcseconds.
* :func:`line_centroid` / :func:`residual_wavelength` — scan-to-scan scatter of
  a line-core position, in mÅ and km/s, before and after wavelength alignment.
* :func:`frame_rms` — scan-to-scan RMS of an image series, in percent of the
  mean and (given an atlas factor) in erg s⁻¹ cm⁻² sr⁻¹ Hz⁻¹.
* :func:`stray_light_estimate` — an upper limit on the scattered-light level
  from the off-limb continuum.

The atlas-fit goodness of fit is :func:`chase.calib.intensity.atlas_fit_quality`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "residual_pointing",
    "line_centroid",
    "residual_wavelength",
    "frame_rms",
    "stray_light_estimate",
]

_C_KMS = 299792.458


def _u8(img):
    lo, hi = np.nanpercentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)


def residual_pointing(
    frames: np.ndarray,
    ref_idx: int = 0,
    pixel_scale: float = 1.04,
    upsample_factor: int = 20,
    local: bool = True,
    border: int = 10,
) -> dict:
    """Residual registration error of an image series.

    Parameters
    ----------
    frames : ndarray (nframes, H, W)
        One flare-free channel per scan (e.g. the Fe I far wing), already
        registered by the step being validated.
    ref_idx : int, optional
        Frame everything was aligned to.
    pixel_scale : float, optional
        Arcsec per pixel (1.04 for 2×2-binned CHASE data).
    upsample_factor : int, optional
        Cross-correlation sub-pixel precision (1/upsample_factor px). The
        Frames are mean-subtracted and Hann-windowed, and the unnormalised
        cross-correlation is used: phase normalisation and the wrap-around
        edges otherwise bias sub-pixel shifts of low-contrast continuum images.
    local : bool, optional
        Also measure the residual dense (Farnebäck) flow magnitude, which
        captures rubber-sheet distortions that a global shift misses.
    border : int, optional
        Pixels trimmed from each edge before the local measurement.

    Returns
    -------
    dict with ``global_rms_px``, ``global_rms_arcsec``, ``global_max_arcsec``,
    ``shifts_px`` (per frame), and, if ``local``, ``local_median_arcsec`` and
    ``local_p95_arcsec`` (median over frames of the per-frame median / 95th
    percentile flow magnitude).
    """
    from skimage.registration import phase_cross_correlation

    frames = np.asarray(frames, dtype=np.float32)
    ref = frames[ref_idx]
    win = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1])).astype(np.float32)

    def _prep(img):
        return (img - np.nanmean(img)) * win

    shifts = []
    for i in range(len(frames)):
        if i == ref_idx:
            continue
        s, _, _ = phase_cross_correlation(_prep(ref), _prep(frames[i]),
                                          upsample_factor=upsample_factor, normalization=None)
        shifts.append((float(s[0]), float(s[1])))
    sh = np.asarray(shifts) if shifts else np.zeros((0, 2))
    mag = np.hypot(sh[:, 0], sh[:, 1]) if len(sh) else np.zeros(0)
    out = {
        "global_rms_px": float(np.sqrt(np.mean(mag ** 2))) if len(mag) else 0.0,
        "global_max_px": float(mag.max()) if len(mag) else 0.0,
        "shifts_px": sh.tolist(),
    }
    out["global_rms_arcsec"] = out["global_rms_px"] * pixel_scale
    out["global_max_arcsec"] = out["global_max_px"] * pixel_scale
    if local:
        import cv2

        r8 = _u8(ref)
        med, p95 = [], []
        b = border
        for i in range(len(frames)):
            if i == ref_idx:
                continue
            flow = cv2.calcOpticalFlowFarneback(r8, _u8(frames[i]), None, 0.5, 3, 25, 3, 5, 1.2, 0)
            m = np.hypot(flow[..., 0], flow[..., 1])[b:-b or None, b:-b or None]
            med.append(np.median(m))
            p95.append(np.percentile(m, 95))
        out["local_median_arcsec"] = float(np.median(med) * pixel_scale) if med else 0.0
        out["local_p95_arcsec"] = float(np.median(p95) * pixel_scale) if p95 else 0.0
    return out


def line_centroid(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    guess: Optional[float] = None,
    half_window: float = 0.25,
) -> float:
    """Line-core wavelength from a Gaussian fit to the core.

    A Gaussian absorption plus constant is fitted to the channels within
    ``half_window`` Å of the intensity minimum (the minimum is searched within
    ±0.3 Å of ``guess`` when given). This is unbiased for a symmetric core at
    any sub-channel position, unlike a few-point parabola. Falls back to a
    3-point parabola if the fit fails.
    """
    from scipy.optimize import curve_fit

    wav = np.asarray(wavelengths, dtype=float)
    spec = np.asarray(spectrum, dtype=float)
    if guess is not None:
        win = np.where(np.abs(wav - guess) < 0.3)[0]
        k = int(win[np.argmin(spec[win])])
    else:
        k = int(np.argmin(spec))
    sel = np.abs(wav - wav[k]) <= half_window
    x, y = wav[sel] - wav[k], spec[sel]

    def g(x, a, mu, sig, c):
        return c - a * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    try:
        p, _ = curve_fit(g, x, y, p0=[y.max() - y.min(), 0.0, 0.1, y.max()], maxfev=5000)
        if abs(p[1]) < half_window:
            return float(wav[k] + p[1])
    except (RuntimeError, ValueError):
        pass
    lo, hi = max(0, k - 1), min(len(wav), k + 2)
    a, b, _ = np.polyfit(wav[lo:hi] - wav[k], spec[lo:hi], 2)
    return float(wav[k] - b / (2 * a)) if a > 0 else float(wav[k])


def residual_wavelength(
    spectra: Sequence[np.ndarray],
    wavelengths,
    rest_wavelength: float = 6569.214,
    guess: Optional[float] = None,
) -> dict:
    """Scan-to-scan scatter of a line-core position.

    Parameters
    ----------
    spectra : sequence of ndarray (nchannels,)
        One spatially averaged (quiet-Sun) profile per scan.
    wavelengths : ndarray (nchannels,) or sequence of such
        The wavelength grid of each spectrum (a single grid for all, or one
        per scan, e.g. each file's own ``CRVAL3``/``CDELT3``).
    rest_wavelength : float, optional
        Line rest wavelength for the km/s conversion (Fe I 6569.214 Å).

    Returns
    -------
    dict with ``centroids_A`` (per scan), ``std_mA``, ``std_kms``,
    ``ptp_mA`` (peak to peak) and ``mean_A``.
    """
    spectra = [np.asarray(s, dtype=float) for s in spectra]
    if np.ndim(wavelengths[0]) == 0:
        grids = [np.asarray(wavelengths, dtype=float)] * len(spectra)
    else:
        grids = [np.asarray(w, dtype=float) for w in wavelengths]
    g = rest_wavelength if guess is None else guess
    c = np.array([line_centroid(w, s, guess=g) for w, s in zip(grids, spectra)])
    std = float(np.std(c, ddof=1)) if len(c) > 1 else 0.0
    return {
        "centroids_A": c.tolist(),
        "mean_A": float(c.mean()),
        "std_mA": std * 1e3,
        "ptp_mA": float(np.ptp(c) * 1e3),
        "std_kms": std / rest_wavelength * _C_KMS,
    }


def frame_rms(
    frames: np.ndarray,
    ifact: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Scan-to-scan RMS of an image series.

    For each consecutive pair the RMS of the difference image is taken over
    ``mask`` (default: all pixels); the median over pairs is reported.

    Returns
    -------
    dict with ``rms_pct`` (percent of the series mean), ``rms_dn`` and, given
    ``ifact``, ``rms_cgs`` in erg s⁻¹ cm⁻² sr⁻¹ Hz⁻¹; plus
    ``mean_intensity_dn``.
    """
    f = np.asarray(frames, dtype=float)
    if mask is None:
        mask = np.ones(f.shape[1:], dtype=bool)
    d = np.diff(f, axis=0)[:, mask]
    rms = np.sqrt(np.mean(d ** 2, axis=1))
    mean = float(np.mean(f[:, mask]))
    out = {
        "rms_dn": float(np.median(rms)),
        "rms_pct": float(100 * np.median(rms) / mean),
        "mean_intensity_dn": mean,
        "rms_cgs": None,
    }
    if ifact is not None:
        out["rms_cgs"] = float(np.median(rms) * ifact)
    return out


def stray_light_estimate(
    image: np.ndarray,
    center: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
    radii: Sequence[float] = (1.02, 1.05, 1.10, 1.20),
    width: float = 0.01,
) -> dict:
    """Upper limit on the stray-light level from the off-limb continuum.

    In the photospheric continuum the true off-limb signal (K/F corona) is
    ~10⁻⁶ of the disc, so any intensity measured just outside the limb is
    instrumental scattered light (plus detector offset). Its ratio to the
    disc-centre intensity is therefore an upper limit on the stray-light
    fraction near the limb, and a conservative estimate of the halo added to
    on-disc pixels.

    Parameters
    ----------
    image : ndarray (H, W)
        A full-disc continuum image (e.g. the Fe I or Hα far-wing channel).
    center : (cx, cy) or None
        Disc centre in pixels; ``None`` → Hough limb fit.
    radius : float or None
        Disc radius in pixels; ``None`` → Hough limb fit.
    radii : sequence of float, optional
        Radii (in units of the disc radius) at which to sample.
    width : float, optional
        Half-width of each annulus, in units of the disc radius.

    Returns
    -------
    dict with ``disc_center``, ``fraction`` (``{"1.05": …}`` off-limb mean over
    disc-centre mean), ``center_px``, ``radius_px``.
    """
    img = np.asarray(image, dtype=float)
    if center is None or radius is None:
        from ..calib.spatial import hough_disk_center

        cx, cy, r = hough_disk_center(img)
        center = (cx, cy) if center is None else center
        radius = r if radius is None else radius
    cx, cy = center
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    rr = np.hypot(yy - cy, xx - cx) / radius
    dc = float(np.nanmedian(img[rr < 0.1]))
    frac = {}
    for r in radii:
        sel = np.abs(rr - r) < width
        frac[f"{r:.2f}"] = float(np.nanmedian(img[sel]) / dc) if sel.any() else None
    return {"disc_center": dc, "fraction": frac,
            "center_px": [float(cx), float(cy)], "radius_px": float(radius)}
