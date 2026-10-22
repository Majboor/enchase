"""Intensity calibration: normalisation, scaling, and absolute (atlas) calib.

Three levels, cheapest first:

* :func:`normalize_intensity` — divide by the quiet-Sun mean (relative units).
* :func:`integral_scaling` — match the integral of two spectra (Davis's
  ``satprocess`` method; corrects frame-to-frame throughput drift).
* :func:`atlas_calibrate` — absolute DN→radiance calibration against the FTS
  solar atlas via **ISPy**, fitting the **line wings** (not the core, which is
  non-LTE/smeared) at disk centre (``mu=1.0``).  ISPy is an *optional* dependency
  — the rest of the package works without it.

:func:`atlas_calibrate` takes ``mu`` (a number or a per-pixel map) to calibrate
on a box away from disc centre. The atlas is then multiplied by the
centre-to-limb variation (CLV) I(λ, μ)/I(λ, 1)
(:func:`clv_factor`): Neckel & Labs (1994) continuum limb darkening
(:func:`limb_darkening`, the table shipped with ISPy), or, inside the Hα line,
the observed SST/CRISP profiles of Pietrow et al. (2023, A&A 671, A130), which
are public at CDS (:func:`download_clv_table`, :func:`load_clv_table`).
:func:`clv_reference` gives the quiet-Sun reference profile at any ``mu``.

:func:`planck_inversion` converts an absolute radiance to a brightness
temperature, used by the Fe I photospheric temperature maps.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["normalize_intensity", "integral_scaling", "atlas_calibrate", "atlas_fit_quality",
           "quiet_sun_mask", "activity_mask", "find_quiet_patch", "wing_channels",
           "atlas_sensitivity", "atlas_factor_series", "planck_inversion",
           "limb_darkening", "download_clv_table", "load_clv_table", "clv_factor",
           "clv_reference", "mu_map"]

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
    mu=1.0,
    clv_table: Optional[dict] = None,
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
    mu, clv_table : optional
        As in :func:`atlas_calibrate`; the atlas is compared at this μ and the
        residuals are in % of the continuum at this μ.

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
    wing = np.interp(wavelengths, wave_fts, spec_fts) > wing_continuum_frac * float(cont_fts[0])
    cont = float(cont_fts[0])
    mu_arr = np.asarray(mu, dtype=float)
    if not (mu_arr.ndim == 0 and float(mu_arr) == 1.0):
        scale = _mean_clv(wave_fts, mu_arr.ravel(), clv_table)
        spec_fts = spec_fts * scale
        cont = cont * float(limb_darkening(np.array([wave_fts[0]]), mu_arr.ravel()).mean())
    atlas = np.interp(wavelengths, wave_fts - woff, spec_fts)
    model = np.asarray(spectrum, dtype=float) * ifact
    resid = (model - atlas) / cont
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
    mu=1.0,
    clv_table: Optional[dict] = None,
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
    mu : float or ndarray (H, W), optional
        Heliocentric μ of the box (default 1.0, disc centre, the v2.1
        behaviour). For a box off disc centre pass its μ, or a per-pixel μ map
        (:func:`mu_map`); the atlas is then multiplied by the mean
        centre-to-limb factor of the averaged pixels (:func:`clv_factor`), so
        ``ifact`` stays the DN→radiance factor of the disc-centre calibration.
    clv_table : dict or None, optional
        Line CLV table from :func:`load_clv_table` (e.g. Pietrow et al. 2023
        for Hα). ``None`` → continuum limb darkening only (Neckel & Labs 1994),
        which is right for the far wings and Fe I continuum but not for the
        Hα core.

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
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.ndim == 0 and float(mu_arr) == 1.0:
        mu_pix = None
    elif mu_arr.ndim == 0:
        mu_pix = mu_arr.reshape(1)
    else:
        mu_pix = mu_arr[mask] if (mask is not None and np.any(mask)) else mu_arr.ravel()
    if mu_pix is not None:
        spec_fts = spec_fts * _mean_clv(wave_fts, mu_pix, clv_table)
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
    quality = atlas_fit_quality(wavelengths, fe_avg, ifact, woff, wing_continuum_frac, sigma=sigma,
                                mu=mu if mu_pix is not None else 1.0, clv_table=clv_table)
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

# ---------------------------------------------------------------- CLV ------
#: Pietrow et al. (2023, A&A 671, A130) CLV tables at CDS (J/A+A/671/A130),
#: with the SHA-256 of the files as downloaded on 2026-10-22.
CLV_CDS_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/671/A130/fits/"
CLV_SHA256 = {
    "6563_clv.fits": "704c32a3d949554c4b85b4d3014448bed8035aa0bf5d67e18ff4ed2bfe34c3a4",
    "6301_cont.fits": "04379cf38d0a74bcdb03e99eecd221ffa4713e6e37467ae16a69aca8cda64621",
    "6173_cont.fits": "ad2305522f2a2c6032fb34bfc46dd5426c0e9836b73df23db0d2dbe272d28b66",
}


def _neckel_labs_table():
    """Neckel & Labs (1994) coefficients A0..A5 from ISPy's data file (cached)."""
    global _NL_CACHE
    if _NL_CACHE is None:
        try:
            import ISPy.spec.calib as _ispy_calib
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError("limb_darkening requires ISPy (`pip install 'enchase[atlas]'`).") from e
        import os
        import astropy.io.fits as fits

        path = os.path.join(os.path.dirname(_ispy_calib.__file__), "..", "data",
                            "limbdarkening_Neckel_Labs_1994.fits")
        t = fits.getdata(path)
        wt = np.asarray(t["wavelength"], dtype=float) * 10.0        # nm -> Å
        A = np.array([np.asarray(t[f"A{i}"], dtype=float) for i in range(6)])
        o = np.argsort(wt)
        _NL_CACHE = (wt[o], A[:, o])
    return _NL_CACHE


_NL_CACHE = None


def limb_darkening(wavelengths, mu) -> np.ndarray:
    """Continuum limb darkening I(μ)/I(1) of Neckel & Labs (1994, Sol. Phys. 153, 91).

    Uses the fifth-order polynomial coefficients shipped with ISPy
    (``ISPy.spec.calib.limbdarkening``), interpolated in wavelength.

    Parameters
    ----------
    wavelengths : float or ndarray (nw,)
        Wavelengths in Å.
    mu : float or ndarray (nmu,)
        Cosine of the heliocentric angle.

    Returns
    -------
    ndarray (nmu, nw), or (nw,) for a scalar ``mu``.
    """
    wt, A = _neckel_labs_table()
    w = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    m = np.asarray(mu, dtype=float)
    coef = np.array([np.interp(w, wt, a) for a in A])            # (6, nw)
    pw = np.atleast_1d(m)[:, None] ** np.arange(6)[None, :]      # (nmu, 6)
    out = pw @ coef
    return out[0] if m.ndim == 0 else out


def download_clv_table(name: str = "6563_clv.fits", dest_dir: str = ".", verify: bool = True) -> str:
    """Download a Pietrow et al. (2023) CLV table from CDS and check its SHA-256.

    Parameters
    ----------
    name : str, optional
        File name in ``J/A+A/671/A130/fits`` (default ``"6563_clv.fits"``, Hα).
    dest_dir : str, optional
        Where to save it. An existing file is checked, not downloaded again.
    verify : bool, optional
        Compare the SHA-256 with :data:`CLV_SHA256` (raises on mismatch).

    Returns
    -------
    str
        Local path.
    """
    import hashlib
    import os
    import urllib.request

    dest_dir = os.path.expanduser(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, name)
    if not os.path.exists(path):
        req = urllib.request.Request(CLV_CDS_URL + name, headers={"User-Agent": "enchase"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    if verify and name in CLV_SHA256:
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if h != CLV_SHA256[name]:
            raise ValueError(f"{path}: SHA-256 {h} does not match the CDS file ({CLV_SHA256[name]})")
    return path


def load_clv_table(path: str, smooth: bool = True) -> dict:
    """Read a Pietrow et al. (2023) ``*_clv.fits`` table.

    Parameters
    ----------
    path : str
        File from :func:`download_clv_table`.
    smooth : bool, optional
        Use the ``INT_SMOOTH`` extension (default) rather than ``INT``. In
        ``INT`` the line-core ratio scatters by a few percent from one μ step
        to the next (e.g. 1.04 at μ = 0.90 for Hα), which the smoothed
        profiles do not show.

    Returns
    -------
    dict
        ``wav`` (nw,), ``mu`` (nmu, increasing) and ``ratio`` (nmu, nw), the
        intensity divided by the μ=1 profile at the same wavelength.
    """
    import astropy.io.fits as fits

    with fits.open(path) as h:
        wav = np.asarray(h["WAV"].data, dtype=float)
        mu = np.asarray(h["MU"].data, dtype=float)
        I = np.asarray(h["INT_SMOOTH" if smooth else "INT"].data, dtype=float)
    if I.shape == (len(wav), len(mu)):
        I = I.T                                  # -> (nmu, nw)
    if I.shape != (len(mu), len(wav)):
        raise ValueError(f"{path}: INT has shape {I.shape}, expected ({len(mu)}, {len(wav)})")
    o = np.argsort(mu)
    mu, I = mu[o], I[o]
    i1 = int(np.argmin(np.abs(mu - 1.0)))
    if abs(mu[i1] - 1.0) > 1e-6:
        raise ValueError(f"{path}: no mu=1 profile to normalise by")
    return {"wav": wav, "mu": mu, "ratio": I / I[i1], "source": str(path)}


def clv_factor(wavelengths, mu, table: Optional[dict] = None, wavelength_shift: float = 0.0) -> np.ndarray:
    """Centre-to-limb factor I(λ, μ)/I(λ, 1) for the quiet Sun.

    Outside the wavelength range of ``table`` (or for ``table=None``) the
    continuum limb darkening of :func:`limb_darkening` is used. Inside it the
    observed line CLV is interpolated linearly in λ and μ. ``μ`` below the
    table's lowest value (0.02 for Pietrow et al. 2023) is clipped to it.

    Parameters
    ----------
    wavelengths : ndarray (nw,)
        Wavelengths in Å.
    mu : float or ndarray (nmu,)
    table : dict or None, optional
        From :func:`load_clv_table`.
    wavelength_shift : float, optional
        Added to the table wavelengths (Å) to put them on the frame of
        ``wavelengths``.

    Returns
    -------
    ndarray (nmu, nw), or (nw,) for a scalar ``mu``.
    """
    w = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    m = np.asarray(mu, dtype=float)
    ma = np.clip(np.atleast_1d(m), 0.0, 1.0)
    out = limb_darkening(w, ma)
    if table is not None:
        tw = table["wav"] + wavelength_shift
        tm, R = table["mu"], table["ratio"]
        inside = (w >= tw[0]) & (w <= tw[-1])
        if inside.any():
            mc = np.clip(ma, tm[0], tm[-1])
            j = np.clip(np.searchsorted(tm, mc) - 1, 0, len(tm) - 2)
            f = ((mc - tm[j]) / (tm[j + 1] - tm[j]))[:, None]
            Rm = (1 - f) * R[j] + f * R[j + 1]                      # (nmu, ntw)
            out[:, inside] = np.array([np.interp(w[inside], tw, r) for r in Rm])
    return out[0] if m.ndim == 0 else out


def _mean_clv(wave, mu_pix, table, nbin: int = 200):
    """Mean CLV factor over many pixels (binned in μ for speed)."""
    mu_pix = np.asarray(mu_pix, dtype=float).ravel()
    mu_pix = mu_pix[np.isfinite(mu_pix)]
    if mu_pix.size == 0:
        raise ValueError("mu has no finite values")
    if mu_pix.size <= nbin:
        return clv_factor(wave, mu_pix, table).mean(axis=0)
    edges = np.linspace(mu_pix.min(), mu_pix.max() + 1e-9, nbin + 1)
    cnt, _ = np.histogram(mu_pix, edges)
    cen = 0.5 * (edges[1:] + edges[:-1])
    ok = cnt > 0
    return (clv_factor(wave, cen[ok], table) * cnt[ok, None]).sum(0) / cnt.sum()


def clv_reference(wavelengths, mu=1.0, table: Optional[dict] = None, woff: float = 0.0) -> np.ndarray:
    """Quiet-Sun reference profile at μ: FTS atlas × :func:`clv_factor`.

    Parameters
    ----------
    wavelengths : ndarray (nw,)
        Observed wavelength grid (Å).
    mu : float or ndarray
        μ of the pixel(s); an array is averaged as in :func:`atlas_calibrate`.
    table : dict or None, optional
        Line CLV table (:func:`load_clv_table`).
    woff : float, optional
        Wavelength offset from :func:`atlas_calibrate`.

    Returns
    -------
    ndarray (nw,)
        Radiance in erg s⁻¹ cm⁻² sr⁻¹ Hz⁻¹ on ``wavelengths``. Divide a
        calibrated profile by it for a contrast relative to the quiet Sun at
        the same μ.
    """
    from ISPy.spec import atlas as isp_atlas

    w = np.asarray(wavelengths, dtype=float)
    wf, sf, _ = isp_atlas.atlas().get(w[0] - 0.5, w[-1] + 0.5, cgs=True, perHz=True)
    m = np.asarray(mu, dtype=float)
    sf = sf * (clv_factor(wf, float(m), table) if m.ndim == 0 else _mean_clv(wf, m, table))
    return np.interp(w, wf - woff, sf)


def mu_map(shape, center, radius) -> np.ndarray:
    """μ = cos θ for every pixel of a full-disc frame.

    Parameters
    ----------
    shape : (H, W)
    center : (cx, cy)
        Disc centre in px (e.g. from :func:`chase.calib.spatial.fit_limb`).
    radius : float or (a, b, angle_deg)
        Disc radius in px, or the ellipse of :func:`fit_limb`; with an
        ellipse the radius is measured in the normalised ellipse frame.

    Returns
    -------
    ndarray (H, W)
        μ on the disc, NaN off the disc.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    dx, dy = xx - center[0], yy - center[1]
    if np.ndim(radius) == 0:
        rho = np.hypot(dx, dy) / float(radius)
    else:
        a, b, ang = radius
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        rho = np.hypot((c * dx + s * dy) / a, (-s * dx + c * dy) / b)
    with np.errstate(invalid="ignore"):
        return np.where(rho < 1, np.sqrt(np.clip(1 - rho ** 2, 0, 1)), np.nan)



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
