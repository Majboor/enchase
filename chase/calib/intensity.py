"""Intensity calibration: normalisation, scaling, and absolute (atlas) calib.

Three levels, cheapest first:

* :func:`normalize_intensity` — divide by the quiet-Sun mean (relative units).
* :func:`integral_scaling` — match the integral of two spectra (Davis's
  ``satprocess`` method; corrects frame-to-frame throughput drift).
* :func:`atlas_calibrate` — absolute DN→radiance calibration against the FTS
  solar atlas via **ISPy**, fitting the **line wings** (not the core, which is
  non-LTE/smeared) at disk centre (``mu=1.0``).  ISPy is an *optional* dependency
  — the rest of the package works without it.

:func:`planck_inversion` converts an absolute radiance to a brightness
temperature, used by the Fe I photospheric temperature maps.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["normalize_intensity", "integral_scaling", "atlas_calibrate", "atlas_fit_quality",
           "quiet_sun_mask", "find_quiet_patch", "planck_inversion"]

# CGS physical constants
_H = 6.62607015e-27   # erg s
_C = 2.99792458e10    # cm/s
_K = 1.380649e-16     # erg/K


def normalize_intensity(cube: np.ndarray, ref_cube: np.ndarray, clip: Tuple[float, float] = (0, 2)) -> np.ndarray:
    """Normalise a cube by the mean of a quiet-Sun reference region."""
    out = cube / np.mean(ref_cube)
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out


def integral_scaling(ref_spectrum: np.ndarray, target_spectrum: np.ndarray) -> float:
    """Integral-ratio intensity scale factor (Davis ``calculate_scaling``).

    Returns ``∫ref / ∫target`` over the overlap where both are non-zero, or 1.0
    if there is no usable overlap.  Multiply a cube by this to match throughput.

    Ported from ``satprocess`` (BSD-3, Finlay Davis).
    """
    from scipy.integrate import trapezoid

    ref = np.asarray(ref_spectrum, dtype=float)
    tgt = np.asarray(target_spectrum, dtype=float)
    mask = (ref != 0) & (tgt != 0)
    if not np.any(mask):
        return 1.0
    denom = trapezoid(tgt[mask])
    if denom == 0:
        return 1.0
    return float(trapezoid(ref[mask]) / denom)


def quiet_sun_mask(
    disk_center_cube: np.ndarray,
    continuum_tol: float = 0.05,
    core_tol: float = 0.10,
    core_channel: Optional[int] = None,
) -> np.ndarray:
    """Boolean mask of quiet-Sun pixels in a calibration patch.

    The FTS atlas is a quiet-Sun spectrum, so pixels covered by sunspots,
    pores, penumbrae, plage or network bright points bias the fit. A pixel is
    kept only if its continuum (last channel) lies within ``continuum_tol`` of
    the patch median **and** its line-core intensity lies within ``core_tol``
    of the core median. Dark umbrae/penumbrae (continuum 0.75–0.92 of the quiet
    Sun) fail the first test; plage and bright points fail the second.

    Parameters
    ----------
    disk_center_cube : ndarray (nchannels, H, W)
    continuum_tol : float, optional
        Allowed fractional continuum deviation from the median (default 5 %).
    core_tol : float, optional
        Allowed fractional line-core deviation from the median (default 10 %).
    core_channel : int or None, optional
        Index of the line core; ``None`` → the channel with the lowest mean.

    Returns
    -------
    mask : ndarray (H, W) of bool
        ``True`` where the pixel is quiet Sun.
    """
    cube = np.asarray(disk_center_cube, dtype=float)
    cont = cube[-1]
    if core_channel is None:
        core_channel = int(np.argmin(cube.mean(axis=(1, 2))))
    core = cube[core_channel]
    c_med, k_med = np.nanmedian(cont), np.nanmedian(core)
    return (np.abs(cont / c_med - 1.0) <= continuum_tol) & (np.abs(core / k_med - 1.0) <= core_tol)


def find_quiet_patch(
    continuum: np.ndarray,
    core: np.ndarray,
    center: Tuple[int, int],
    size: int = 100,
    search_radius: int = 300,
    step: int = 25,
) -> Tuple[int, int]:
    """Centre of the quietest ``size``×``size`` box near disc centre.

    Candidate boxes are placed on a grid of spacing ``step`` within
    ``search_radius`` pixels of ``center`` (≈ ``mu > 0.95`` for the default on
    a 2×2-binned CHASE disc). The box with the smallest combined fractional
    continuum and line-core standard deviation, plus the offset of its mean
    level from the image median, is returned, which steers the calibration
    patch away from active-region material.

    Returns
    -------
    (cy, cx) : centre of the chosen box.
    """
    cont = np.asarray(continuum, dtype=float)
    core = np.asarray(core, dtype=float)
    H, W = cont.shape
    half = size // 2
    cy0, cx0 = center
    c_ref, k_ref = np.nanmedian(cont), np.nanmedian(core)
    best, best_score = (cy0, cx0), np.inf
    for dy in range(-search_radius, search_radius + 1, step):
        for dx in range(-search_radius, search_radius + 1, step):
            if dy * dy + dx * dx > search_radius * search_radius:
                continue
            cy, cx = cy0 + dy, cx0 + dx
            if cy - half < 0 or cx - half < 0 or cy + half > H or cx + half > W:
                continue
            c = cont[cy - half:cy + half, cx - half:cx + half]
            k = core[cy - half:cy + half, cx - half:cx + half]
            # structure inside the box + offset of its level from the disc median
            score = (np.nanstd(c) / np.nanmean(c) + np.nanstd(k) / np.nanmean(k)
                     + abs(np.nanmean(c) / c_ref - 1) + abs(np.nanmean(k) / k_ref - 1))
            if score < best_score:
                best, best_score = (cy, cx), score
    return best


def atlas_fit_quality(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    ifact: float,
    woff: float,
    wing_continuum_frac: float = 0.85,
    sigma: Optional[np.ndarray] = None,
) -> dict:
    """Goodness of fit of an atlas calibration.

    Compares ``ifact × spectrum`` with the FTS atlas shifted by ``woff`` on the
    observed grid.

    Parameters
    ----------
    wavelengths, spectrum : ndarray (nchannels,)
        Observed wavelength grid and mean (DN) profile that was calibrated.
    ifact, woff : float
        Result of :func:`atlas_calibrate`.
    wing_continuum_frac : float, optional
        Same wing threshold as used in the fit.
    sigma : ndarray (nchannels,) or None, optional
        1σ uncertainty of ``spectrum`` in DN (e.g. the spatial standard
        deviation over the patch divided by √N). Needed for a reduced χ².

    Returns
    -------
    dict with ``rms_wings_pct`` and ``rms_all_pct`` (RMS residual in % of the
    atlas continuum), ``max_abs_wings_pct``, ``n_wing``, and ``chi2_red``
    (wing channels; ``None`` without ``sigma``).
    """
    from ISPy.spec import atlas as isp_atlas

    fts = isp_atlas.atlas()
    wave_fts, spec_fts, cont_fts = fts.get(
        wavelengths[0] - 0.5, wavelengths[-1] + 0.5, cgs=True, perHz=True
    )
    atlas = np.interp(wavelengths, wave_fts - woff, spec_fts)
    model = np.asarray(spectrum, dtype=float) * ifact
    cont = float(cont_fts[0])
    resid = (model - atlas) / cont
    wing = np.interp(wavelengths, wave_fts, spec_fts) > wing_continuum_frac * cont
    out = {
        "rms_wings_pct": float(100 * np.sqrt(np.mean(resid[wing] ** 2))) if wing.any() else None,
        "rms_all_pct": float(100 * np.sqrt(np.mean(resid ** 2))),
        "max_abs_wings_pct": float(100 * np.max(np.abs(resid[wing]))) if wing.any() else None,
        "n_wing": int(wing.sum()),
        "chi2_red": None,
    }
    if sigma is not None and wing.sum() > 2:
        s = np.asarray(sigma, dtype=float) * ifact
        out["chi2_red"] = float(np.sum(((model - atlas)[wing] / s[wing]) ** 2) / (wing.sum() - 2))
    return out


def atlas_calibrate(
    wavelengths: np.ndarray,
    disk_center_cube: np.ndarray,
    wing_continuum_frac: float = 0.85,
    mask_active: bool = False,
    mask: Optional[np.ndarray] = None,
    return_quality: bool = False,
    seed: Optional[int] = 0,
):
    """Absolute DN→radiance calibration against the FTS atlas (ISPy).

    Fits the spatially averaged disk-centre quiet-Sun profile to the FTS solar
    atlas at ``mu=1.0`` / ``calib_at_dc=True``. ``wing_continuum_frac`` is a
    **spectral** selection: the wavelength channels where the *atlas* profile
    exceeds that fraction of its continuum (the line wings) get 20× weight in
    ISPy's least-squares fit; it does not select pixels. To keep active pixels
    out of the average, use ``mask_active`` or pass an explicit ``mask``.

    Parameters
    ----------
    wavelengths : ndarray (nchannels,)
        Wavelength axis of the disk-centre cube, in Å.
    disk_center_cube : ndarray (nchannels, H, W)
        Disk-centre Fe I cube (e.g. the 100×100 CRPIX patch, single frame).
    wing_continuum_frac : float, optional
        Threshold (fraction of continuum) above which a channel counts as wing.
    mask_active : bool, optional
        Average only the pixels kept by :func:`quiet_sun_mask` (default False,
        the v2.0 behaviour).
    mask : ndarray (H, W) of bool, optional
        Explicit pixel mask; overrides ``mask_active``.
    return_quality : bool, optional
        Also return the :func:`atlas_fit_quality` dict (plus ``n_pixels``).
    seed : int or None, optional
        Seed for ISPy's differential-evolution optimiser, so repeated calls give
        identical results (default 0; ``None`` → unseeded).

    Returns
    -------
    ifact : float
        Multiply DN by this to get erg/(Hz·s·sr·cm²).
    woff : float
        Wavelength offset in Å.
    quality : dict
        Only with ``return_quality=True``.

    Raises
    ------
    ImportError
        If ISPy is not installed (``pip install 'enchase[atlas]'``).
    """
    try:
        from ISPy.spec import atlas as isp_atlas
        from ISPy.spec.calib import get_calibration as isp_get_calibration
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError(
            "atlas_calibrate requires ISPy. Install with "
            "`pip install 'enchase[atlas]'` or `pip install ISPy`."
        ) from e
    from scipy.interpolate import interp1d

    cube = np.asarray(disk_center_cube, dtype=float)
    if mask is None and mask_active:
        mask = quiet_sun_mask(cube)
    if mask is not None and np.any(mask):
        pix = cube[:, mask]                       # (nchannels, npix)
    else:
        pix = cube.reshape(cube.shape[0], -1)
    fe_avg = pix.mean(axis=1)

    fts = isp_atlas.atlas()
    wave_fts, spec_fts, cont_fts = fts.get(
        wavelengths[0] - 0.5, wavelengths[-1] + 0.5, cgs=True, perHz=True
    )

    f_atlas = interp1d(wave_fts, spec_fts, kind="cubic", bounds_error=False, fill_value="extrapolate")
    atlas_at_obs = f_atlas(wavelengths)
    wing_idx = np.where(atlas_at_obs > wing_continuum_frac * cont_fts[0])[0]

    # ISPy's differential evolution draws from NumPy's global RNG.
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        calibration = isp_get_calibration(
            wavelengths, fe_avg, wave_fts, spec_fts,
            mu=1.0, calib_at_dc=True, wave_idx=wing_idx,
        )
    finally:
        np.random.set_state(state)
    ifact, woff = float(calibration[0]), float(calibration[1])
    if not return_quality:
        return ifact, woff
    sigma = pix.std(axis=1) / np.sqrt(pix.shape[1])
    quality = atlas_fit_quality(wavelengths, fe_avg, ifact, woff, wing_continuum_frac, sigma=sigma)
    quality["n_pixels"] = int(pix.shape[1])
    return ifact, woff, quality


def planck_inversion(I_nu, wavelength_ang: float = 6569.0):
    """Invert the Planck function ``B_nu(T)`` to brightness temperature (CGS).

    ``T = hν / (k · ln(1 + 2hν³/(c²·I_ν)))``

    Parameters
    ----------
    I_nu : float or ndarray
        Spectral radiance in erg/(Hz·s·sr·cm²).
    wavelength_ang : float, optional
        Wavelength in Å (default 6569, the Fe I line).

    Returns
    -------
    T : float or ndarray
        Temperature in Kelvin.
    """
    lam_cm = wavelength_ang * 1e-8
    nu = _C / lam_cm
    coeff = 2.0 * _H * nu ** 3 / _C ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        T = _H * nu / (_K * np.log(coeff / I_nu + 1.0))
    return T
