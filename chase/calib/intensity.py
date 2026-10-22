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

import warnings
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["normalize_intensity", "integral_scaling", "atlas_calibrate", "atlas_fit_quality",
           "quiet_sun_mask", "activity_mask", "find_quiet_patch", "wing_channels",
           "atlas_sensitivity", "atlas_factor_series", "planck_inversion"]

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


def activity_mask(
    continuum: np.ndarray,
    core: np.ndarray,
    continuum_tol: float = 0.03,
    core_excess: float = 0.10,
) -> np.ndarray:
    """Quiet-Sun mask from a continuum image and a separate chromospheric core.

    A stricter, two-line variant of :func:`quiet_sun_mask` for a calibration
    patch: a pixel is kept only if its continuum lies within ``continuum_tol``
    of the patch median (rejects umbrae, penumbrae and pores, whose continuum
    is 0.75–0.92 of the quiet Sun) **and** its Hα line-core intensity is no
    more than ``core_excess`` above the patch median (rejects plage and
    network, which are bright in the Hα core but nearly invisible in the
    Fe I window). Dark Hα features (fibrils, filaments) are kept, since
    they do not change the photospheric Fe I profile.

    Parameters
    ----------
    continuum : ndarray (H, W)
        Continuum image of the patch (e.g. the last Fe I channel).
    core : ndarray (H, W)
        Hα line-core image of the same patch (channel nearest 6562.8 Å).
    continuum_tol : float, optional
        Allowed fractional continuum deviation from the median (default 3 %).
    core_excess : float, optional
        Allowed fractional Hα-core excess over the median (default 10 %).

    Returns
    -------
    mask : ndarray (H, W) of bool
        ``True`` where the pixel is quiet Sun. Pass it to
        :func:`atlas_calibrate` as ``mask=``.
    """
    cont = np.asarray(continuum, dtype=float)
    k = np.asarray(core, dtype=float)
    return ((np.abs(cont / np.nanmedian(cont) - 1.0) <= continuum_tol)
            & (k / np.nanmedian(k) - 1.0 <= core_excess))


def wing_channels(wavelengths: np.ndarray, wing_continuum_frac: float = 0.85) -> np.ndarray:
    """Indices of the channels :func:`atlas_calibrate` treats as line wings.

    A channel is a wing channel when the FTS atlas, interpolated onto the
    observed grid, exceeds ``wing_continuum_frac`` of the atlas continuum.
    This is a selection in wavelength, not in pixels.

    For the CHASE Fe I window (6567.5–6569.7 Å, in the far red wing of Hα)
    every channel outside the Fe I line is at 0.88–0.92 of the continuum, so
    thresholds 0.70–0.85 select the same 41–43 of 46 channels and leave out
    only the Fe I core. No channel of the Hα window (6559.4–6565.1 Å)
    exceeds 0.85; use about 0.75 for Hα.

    Parameters
    ----------
    wavelengths : ndarray (nchannels,)
        Observed wavelength grid in Å.
    wing_continuum_frac : float, optional
        Threshold as a fraction of the atlas continuum (default 0.85).

    Returns
    -------
    ndarray of int
        Channel indices (may be empty).
    """
    from ISPy.spec import atlas as isp_atlas
    from scipy.interpolate import interp1d

    wav = np.asarray(wavelengths, dtype=float)
    w, sp, c = isp_atlas.atlas().get(wav[0] - 0.5, wav[-1] + 0.5, cgs=True, perHz=True)
    a = interp1d(w, sp, kind="cubic", bounds_error=False, fill_value="extrapolate")(wav)
    return np.where(a > wing_continuum_frac * c[0])[0]


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
    if sigma is not None and wing.sum() > 2 and np.all(np.asarray(sigma)[wing] > 0):
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
        Threshold (fraction of continuum) above which a channel counts as wing
        (see :func:`wing_channels`). The default 0.85 suits Fe I; it selects no
        Hα channel (a ``UserWarning`` is raised), so use 0.75 for Hα.
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
    if len(wing_idx) == 0:
        warnings.warn(
            f"atlas_calibrate: no channel of the atlas exceeds wing_continuum_frac="
            f"{wing_continuum_frac} of the continuum in {wavelengths[0]:.2f}-"
            f"{wavelengths[-1]:.2f} Å, so all channels (line core included) get equal "
            "weight. For Hα use wing_continuum_frac=0.75.",
            UserWarning, stacklevel=2,
        )

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


def atlas_sensitivity(
    wavelengths: np.ndarray,
    disk_center_cube: np.ndarray,
    thresholds: Sequence[float] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    masks: Optional[Dict[str, Optional[np.ndarray]]] = None,
    reference_threshold: float = 0.85,
    continuum_channel: int = -1,
    seed: Optional[int] = 0,
) -> dict:
    """Sensitivity of the atlas calibration to the wing threshold and pixel mask.

    Runs :func:`atlas_calibrate` for every combination of ``thresholds`` and
    ``masks`` on one disc-centre patch and reports each result relative to
    the reference fit (``reference_threshold``, no mask). To test the box
    position or size, call it on different patches and compare the reference
    values (correct for limb darkening when the box is off centre).

    Parameters
    ----------
    wavelengths : ndarray (nchannels,)
    disk_center_cube : ndarray (nchannels, H, W)
    thresholds : sequence of float, optional
        Values of ``wing_continuum_frac`` to test.
    masks : dict of str -> (H, W) bool array or None, optional
        Pixel masks to test, by name. Default: ``{"none": None,
        "quiet_sun_mask": quiet_sun_mask(cube)}``.
    reference_threshold : float, optional
        Threshold of the reference fit (default 0.85, the pipeline default).
    continuum_channel : int, optional
        Channel used for the brightness-temperature change (default: last,
        the Fe I continuum).
    seed : int or None, optional
        Optimiser seed passed to :func:`atlas_calibrate`.

    Returns
    -------
    dict with ``reference`` (``ifact``, ``woff``, ``n_wing``, ``T_K``,
    ``K_per_pct``: brightness-temperature change per 1 % in ``ifact``) and
    ``fits``: one dict per (threshold, mask) with ``threshold``, ``mask``,
    ``n_wing``, ``kept_frac``, ``ifact``, ``woff``, ``delta_pct`` (change of
    ``ifact`` in %), ``delta_T_K`` and ``rms_wings_pct``.
    """
    cube = np.asarray(disk_center_cube, dtype=float)
    wav = np.asarray(wavelengths, dtype=float)
    if masks is None:
        masks = {"none": None, "quiet_sun_mask": quiet_sun_mask(cube)}
    lam_c = float(wav[continuum_channel])
    i_c = float(cube[continuum_channel].mean())

    ref_ifact, ref_woff = atlas_calibrate(wav, cube, wing_continuum_frac=reference_threshold, seed=seed)
    t_ref = float(planck_inversion(i_c * ref_ifact, lam_c))
    k_per_pct = float(planck_inversion(i_c * ref_ifact * 1.01, lam_c)) - t_ref
    out = {"reference": {"threshold": float(reference_threshold), "ifact": ref_ifact, "woff": ref_woff,
                         "n_wing": int(len(wing_channels(wav, reference_threshold))),
                         "T_K": t_ref, "K_per_pct": k_per_pct},
           "fits": []}
    for thr in thresholds:
        n_wing = int(len(wing_channels(wav, thr)))
        for name, m in masks.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                ifact, woff, q = atlas_calibrate(wav, cube, wing_continuum_frac=thr, mask=m,
                                                 return_quality=True, seed=seed)
            i_m = float(cube[continuum_channel][m].mean()) if m is not None and np.any(m) else i_c
            out["fits"].append({
                "threshold": float(thr), "mask": name, "n_wing": n_wing,
                "kept_frac": float(np.mean(m)) if m is not None else 1.0,
                "ifact": ifact, "woff": woff,
                "delta_pct": 100.0 * (ifact / ref_ifact - 1.0),
                "delta_T_K": float(planck_inversion(i_m * ifact, lam_c)) - t_ref,
                "rms_wings_pct": q["rms_wings_pct"],
            })
    return out


def atlas_factor_series(
    scans: Iterable[Tuple[np.ndarray, np.ndarray]],
    wing_continuum_frac: float = 0.85,
    mask_active: bool = False,
    seed: Optional[int] = 0,
) -> dict:
    """Atlas calibration factor for every scan of a sequence.

    Fits each scan independently with :func:`atlas_calibrate` and summarises
    the scatter. The scan-to-scan spread is the repeatability of the
    calibration; comparing the mean between observing dates (at the same
    exposure time and binning) measures the throughput change of the
    instrument (see :func:`chase.analysis.systematics.throughput_trend`).

    Parameters
    ----------
    scans : iterable of (wavelengths, disk_center_cube)
        One wavelength grid and disc-centre patch per scan.
    wing_continuum_frac : float, optional
        Wing threshold (0.85 for Fe I, 0.75 for Hα).
    mask_active : bool, optional
        Passed to :func:`atlas_calibrate`.
    seed : int or None, optional
        Optimiser seed.

    Returns
    -------
    dict with per-scan ``ifact``, ``woff`` and ``rms_wings_pct`` lists, and
    ``ifact_mean``, ``ifact_sd_pct`` (standard deviation over scans, % of the
    mean), ``ifact_ptp_pct``, ``woff_mean`` and ``woff_sd``.
    """
    ifs, wos, rms = [], [], []
    for wav, cube in scans:
        i, w, q = atlas_calibrate(wav, cube, wing_continuum_frac=wing_continuum_frac,
                                  mask_active=mask_active, return_quality=True, seed=seed)
        ifs.append(i)
        wos.append(w)
        rms.append(q["rms_wings_pct"])
    a, b = np.asarray(ifs), np.asarray(wos)
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    return {
        "ifact": a.tolist(), "woff": b.tolist(), "rms_wings_pct": rms,
        "ifact_mean": float(a.mean()),
        "ifact_sd_pct": 100.0 * sd / float(a.mean()),
        "ifact_ptp_pct": 100.0 * float(np.ptp(a)) / float(a.mean()),
        "woff_mean": float(b.mean()),
        "woff_sd": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
    }


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
