"""High-level flare-sequence loader.

:func:`load_flare_sequence` is the workhorse the pipeline and TUI call: point it
at a folder of CHASE cubes, give it a patch (or full FOV), and it returns
aligned-ready float32 stacks plus the metadata downstream stages need.  It is a
refactor of the proven ``v50`` loader and preserves every physics rule:

* **shift-then-crop** with a padded buffer (see :func:`chase.calib.track_and_crop`);
* tracking on a **flare-free** channel (FE last wavelength, else HA continuum);
* **per-frame wavelength resampling** onto frame 0's grid;
* a **separate quiet-Sun background patch** for the contrast profile;
* an optional **separate align-patch** (small box for shift estimation, large box
  for the saved data);
* optional **header recentring** (``align_mode="header"``), chained
  **scan-to-scan / running references** for long series, and **sub-pixel**
  crops (see :mod:`chase.calib.pointing`).

Power users who want only one of these steps can call the modular functions in
:mod:`chase.calib` directly — this loader simply orchestrates them with the
streaming, memory-aware FITS reads that big full-disk cubes require.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import _compat  # noqa: F401
from ..calib.pointing import (
    header_disc_center,
    header_shifts,
    limb_disc_centers,
    track_shifts,
    tie_to_header,
    windowed_shift,
)
from ..calib.spatial import derotate_crop, phase_shift
from ..calib.wavelength import resample_wavelength
from .download import discover_fits

__all__ = ["load_flare_sequence", "robust_shifts", "alignment_shifts"]


def robust_shifts(shifts, threshold=4.0, verbose=False, ref_idx=None):
    """Reject per-frame tracking outliers, treating odd and even scans separately.

    CHASE RSM scans alternate raster direction, so odd and even frames can carry
    a real, fixed geometric offset; each parity is therefore smoothed
    independently and any true alternating pattern survives. Within a parity the
    telescope drift is smooth in time, so a shift deviating from its parity's
    Theil-Sen trend line by more than ``threshold`` pixels is a failed
    correlation (e.g. the tracker locking onto the limb) and is replaced by
    interpolation over the surviving frames of that parity. Theil-Sen (median of pairwise slopes) stays correct
    with up to ~29% outliers, so clustered failures cannot corrupt it the
    way they corrupt a short running median.

    Parameters
    ----------
    shifts : sequence of (sy, sx)
        Per-frame shifts as measured by cross-correlation.
    threshold : float, optional
        Outlier rejection threshold in pixels (default 4).
    verbose : bool, optional
        Print each corrected frame.
    ref_idx : int or None, optional
        Index of the reference frame the shifts were measured against. Its
        shift is (0, 0) by construction, so it is never treated as an outlier
        and is kept as measured; otherwise a reference that lies off the
        trend of the other frames of its parity would itself be moved, which
        shifts the whole tracked series. ``None`` (default) keeps the v2.1.0
        behaviour; :func:`load_flare_sequence` passes ``ref_idx=0``.

    Returns
    -------
    list of (sy, sx) floats, same length, outliers replaced.
    """
    from scipy.stats import theilslopes

    arr = np.asarray(shifts, dtype=float)
    out = arr.copy()
    for parity in (0, 1):
        idx = np.arange(parity, len(arr), 2)
        if len(idx) < 4:
            continue
        x = idx.astype(float)
        for axis in (0, 1):
            series = arr[idx, axis]
            # Theil-Sen: median of pairwise slopes, robust to ~29% outliers,
            # so clustered correlation failures cannot drag the trend.
            slope, intercept, _, _ = theilslopes(series, x)
            fit = intercept + slope * x
            bad = np.abs(series - fit) > threshold
            if ref_idx is not None:
                bad &= idx != ref_idx
            if bad.all() or not bad.any():
                continue
            # Replace by interpolating the surviving neighbours rather than the
            # global fit: real drift is only locally linear (steps happen when
            # the field crossing the limb changes the correlation content).
            good = ~bad
            out[idx[bad], axis] = np.round(np.interp(x[bad], x[good], series[good]))
            if verbose:
                for k in np.nonzero(bad)[0]:
                    ax_name = "y" if axis == 0 else "x"
                    print(f"  Frame {idx[k]:>2d}: {ax_name}-shift {series[k]:+.0f} is an outlier "
                          f"for its scan parity; using {out[idx[k], axis]:+.0f}")
    return [tuple(row) for row in out]


_ALIGN_MODES = ("xcorr", "header", "header+xcorr")
_REFERENCES = ("first", "previous", "running")
_DISC_CENTERS = ("crpix", "limb", "auto")


def _obs_time(hdr, i):
    return hdr.get("DATE-OBS", hdr.get("DATE_OBS", None))


def _disc_centers(names, hdrs, source, derotate, verbose):
    """Disc centre (cx, cy) of every scan and the disc radius, for header modes.

    ``names`` are the files of the tracking channel (Fe I if present, else Hα);
    their last channel is used for the limb fits.
    """
    import astropy.io.fits as fits

    crp = [header_disc_center(h) for h in hdrs]
    r_hdr = hdrs[0].get("R_SUN", None)
    if source == "crpix" and r_hdr is not None:
        return crp, float(r_hdr), "crpix"
    if derotate and source != "crpix":
        raise ValueError("disc_center='limb'/'auto' is not supported with derotate=True; "
                         "use disc_center='crpix'")

    cache = {}

    def limb(i):
        if i not in cache:
            with fits.open(names[i], memmap=True) as h:
                img = np.asarray(h[1].section[h[1].header["NAXIS3"] - 1], dtype=np.float32)
            cache[i] = limb_disc_centers([img])[0]
        return cache[i]

    if source == "crpix":                      # R_SUN missing: radius from one limb fit
        return crp, float(limb(0)[2]), "crpix"
    n = len(hdrs)
    if source == "auto":
        idx = sorted({0, n // 2, n - 1})
        off = np.array([[limb(i)[0] - crp[i][0], limb(i)[1] - crp[i][1]] for i in idx])
        ok = bool(np.all(np.ptp(off, axis=0) <= 1.0) and np.all(np.abs(off.mean(0)) <= 3.0))
        if verbose:
            print(f"   disc_center='auto': limb - CRPIX offset on scans {idx} = "
                  f"{np.round(off, 2).tolist()} px -> using {'CRPIX' if ok else 'limb fits'}")
        if ok:
            return crp, float(r_hdr) if r_hdr is not None else float(limb(0)[2]), "crpix"
        import warnings

        warnings.warn(
            "disc_center='auto': the header CRPIX disagrees with the limb, so per-scan limb fits "
            "are used. On such files (the 2023-03 Level-1 data) the header modes were not "
            "reliable: odd and even scans had field-dependent offsets of 8-11 px that the disc "
            "centre does not follow. Prefer align_mode='xcorr' with Fe I there.", stacklevel=3)
    fitc = [limb(i) for i in range(n)]
    return [(f[0], f[1]) for f in fitc], float(np.median([f[2] for f in fitc])), "limb"


def _shift_crop(buf, cy, cx, H, W, subpixel):
    """Crop ``buf[:, cy:cy+H, cx:cx+W]`` at a possibly fractional origin.

    With ``subpixel`` the fractional part is applied with a cubic spline
    (``scipy.ndimage.shift``); otherwise the origin is rounded.
    """
    if not subpixel:
        iy, ix = int(round(cy)), int(round(cx))
        return buf[:, iy:iy + H, ix:ix + W]
    from scipy.ndimage import shift as ndshift

    iy, ix = int(np.floor(cy)), int(np.floor(cx))
    fy, fx = cy - iy, cx - ix
    m = 3
    bp = np.pad(buf, ((0, 0), (m, m + 1), (m, m + 1)), mode="edge")
    sub = bp[:, iy:iy + H + 2 * m, ix:ix + W + 2 * m].astype(np.float32)
    if abs(fy) < 1e-6 and abs(fx) < 1e-6:
        return sub[:, m:m + H, m:m + W]
    out = ndshift(sub, (0.0, -fy, -fx), order=3, mode="nearest")
    return out[:, m:m + H, m:m + W]


def load_flare_sequence(
    fits_dir: str,
    patch: Optional[Sequence[int]] = None,
    align_patch: Optional[Sequence[int]] = None,
    core_wavelength: float = 6562.8,
    track: bool = True,
    pad: int = 20,
    resample: bool = True,
    smooth_shifts: bool = True,
    derotate: bool = False,
    verbose: bool = True,
    *,
    align_mode: str = "xcorr",
    reference: str = "first",
    reference_window: int = 3,
    drift_control: str = "none",
    disc_center: str = "crpix",
    rotation: bool = True,
    subpixel: bool = False,
    shift_method: str = "phase",
    max_residual: float = 3.0,
    header_sigma: float = 1.0,
    step_sigma: float = 0.25,
):
    """Load and crop-track a CHASE HA(+FE) sequence from a directory.

    Parameters
    ----------
    fits_dir : str
        Directory of ``*HA.fits`` (and optional ``*FE.fits``) cubes.
    patch : [y0, y1, x0, x1] or None, optional
        Data crop window.  ``None`` → full FOV (no tracking).
    align_patch : [y0, y1, x0, x1] or None, optional
        Separate window for shift estimation (defaults to ``patch``).
    core_wavelength : float, optional
        Hα core wavelength in Å (default 6562.8).
    track : bool, optional
        Cross-correlation crop tracking on the flare-free channel.
    pad : int, optional
        Shift-then-crop padding in pixels.
    resample : bool, optional
        Per-frame wavelength resampling onto frame 0's grid.
    smooth_shifts : bool, optional
        Robust outlier rejection on the tracking shifts (see
        :func:`robust_shifts`). Failed correlations otherwise leave whole
        frames misaligned by their full drift.
    derotate : bool, optional
        Rotate every frame to solar north using the header's ``INST_ROT``
        angle about the disc centre (``CRPIX1``, ``CRPIX2``) before any
        cropping (default False). CHASE detectors are mounted ~11 deg from
        solar north; with this on, ``patch`` coordinates refer to the
        derotated (north-up) frame. See :func:`chase.calib.derotate`.
    verbose : bool, optional
    align_mode : {"xcorr", "header", "header+xcorr"}, optional
        How the per-scan crop shift is found (only with a ``patch`` and
        ``track=True``).

        * ``"xcorr"`` (default, the v2.1 behaviour): cross-correlation on the
          tracking channel (Fe I last wavelength, else Hα last wavelength).
        * ``"header"``: from the disc-centre drift of the headers plus the
          differential rotation of the patch centre; no image correlation.
          Recommended for Hα-only data of quiet Sun, filaments and the limb,
          where the far-wing contrast is too low for correlation.
        * ``"header+xcorr"``: each scan's window is first moved by the header
          shift, then the remaining shift is measured by correlation. With
          ``reference="first"`` a residual larger than ``max_residual`` px is
          treated as a failed correlation and set to zero (header only); with
          a chained reference the steps are tied back to the header (see
          ``drift_control``).
    reference : {"first", "previous", "running"}, optional
        Correlation reference for the ``xcorr`` modes (see
        :func:`chase.calib.pointing.track_shifts`). ``"first"`` (default)
        correlates with scan 0, which fails on quiet Sun after about 15
        minutes because the granulation evolves. ``"previous"`` chains
        scan-to-scan steps; ``"running"`` correlates with the mean of the last
        ``reference_window`` aligned scans.
    reference_window : int, optional
        Scans in the running mean (default 3).
    drift_control : {"none", "header"}, optional
        For chained references in ``align_mode="xcorr"``: ``"header"`` ties
        the chained steps back to the header pointing by least squares
        (:func:`chase.calib.pointing.tie_to_header`), so chained errors do not
        random-walk. ``"header+xcorr"`` with a chained reference always does
        this. Default ``"none"``.
    disc_center : {"crpix", "limb", "auto"}, optional
        Source of the disc centre for the header modes and ``drift_control``:
        ``"crpix"`` (default) the header ``CRPIX1``/``CRPIX2``; ``"limb"`` a
        sub-pixel limb fit of every scan (one full channel is read per scan);
        ``"auto"`` checks the header against limb fits of the first, middle
        and last scans and uses the limb fits only when they disagree (as for
        the 2023-03 Level-1 files).
    rotation : bool, optional
        Include differential rotation of the patch centre in the header
        shifts (default True; it is skipped automatically for a patch centre
        off the disc).
    subpixel : bool, optional
        Apply the fractional part of the shifts with a cubic spline instead of
        rounding them to whole pixels (default False). Only useful when the
        shifts are better than 0.5 px, i.e. for the header modes.
    shift_method : {"phase", "windowed"}, optional
        Correlation estimator: ``"phase"`` (default, :func:`phase_shift`) or
        ``"windowed"`` (Hann-windowed unnormalised correlation,
        :func:`chase.calib.pointing.windowed_shift`), which is more robust on
        low-contrast fields.
    max_residual : float, optional
        Largest accepted correlation residual in px for ``"header+xcorr"``
        with ``reference="first"`` (default 3).
    header_sigma, step_sigma : float, optional
        Weights of :func:`chase.calib.pointing.tie_to_header` (defaults 1.0
        and 0.25 px).

    Returns
    -------
    dict
        Keys: ``ha_cubes`` (nframes, nch_ha, H, W), ``fe_cubes`` (or None),
        ``cont_raw``, ``core_raw``, ``align_ref``, ``ha_bg`` (or None),
        ``wavelength_ha``, ``wavelength_fe``, ``times``, ``nframes``, ``H``,
        ``W``, ``nchannels_ha``, ``nchannels_fe``, ``core_idx``, ``patch``,
        ``shifts``, ``inst_rot``, ``derotated``, ``crpix``, ``cdelt`` and
        ``pointing`` (the alignment settings, the header shifts and the
        correlation measurements, when used).
    """
    if align_mode not in _ALIGN_MODES:
        raise ValueError(f"align_mode must be one of {_ALIGN_MODES}, got {align_mode!r}")
    if reference not in _REFERENCES:
        raise ValueError(f"reference must be one of {_REFERENCES}, got {reference!r}")
    if disc_center not in _DISC_CENTERS:
        raise ValueError(f"disc_center must be one of {_DISC_CENTERS}, got {disc_center!r}")
    if drift_control not in ("none", "header"):
        raise ValueError(f"drift_control must be 'none' or 'header', got {drift_control!r}")
    if shift_method not in ("phase", "windowed"):
        raise ValueError(f"shift_method must be 'phase' or 'windowed', got {shift_method!r}")
    import astropy.io.fits as fits

    ha_names, fe_names = discover_fits(fits_dir)
    if not ha_names:
        raise FileNotFoundError(f"No *HA.fits files found in {fits_dir}")
    has_fe = len(fe_names) == len(ha_names) and len(fe_names) > 0
    nframes = len(ha_names)

    hdr0 = fits.open(ha_names[0])[1].header
    wavelength_ha = np.arange(hdr0["NAXIS3"]) * hdr0["CDELT3"] + hdr0["CRVAL3"]
    nch_ha = len(wavelength_ha)
    core_idx = int(np.argmin(np.abs(wavelength_ha - core_wavelength)))
    H_full, W_full = int(hdr0["NAXIS2"]), int(hdr0["NAXIS1"])

    if has_fe:
        hdr0_fe = fits.open(fe_names[0])[1].header
        wavelength_fe = np.arange(hdr0_fe["NAXIS3"]) * hdr0_fe["CDELT3"] + hdr0_fe["CRVAL3"]
        nch_fe = len(wavelength_fe)
    else:
        wavelength_fe, nch_fe = None, 0

    if patch is not None:
        y0, y1, x0, x1 = patch
    else:
        y0, y1, x0, x1 = 0, H_full, 0, W_full
    H, W = y1 - y0, x1 - x0
    ay0, ay1, ax0, ax1 = align_patch if align_patch is not None else (y0, y1, x0, x1)

    py0, py1 = max(0, y0 - pad), min(H_full, y1 + pad)
    px0, px1 = max(0, x0 - pad), min(W_full, x1 + pad)
    oy, ox = y0 - py0, x0 - px0

    inst_rot0 = float(hdr0.get("INST_ROT", 0.0))
    if derotate and verbose:
        print(f"   Derotating to solar north: INST_ROT = {inst_rot0:.2f} deg")

    def _chan_win(hdu, ch, ys, ye, xs, xe):
        """One channel's window, derotated to solar north when requested."""
        hdr = hdu.header
        ang = float(hdr.get("INST_ROT", 0.0)) if derotate else 0.0
        if abs(ang) < 1e-9:
            return np.asarray(hdu.section[ch, ys:ye, xs:xe], dtype=np.float32)
        ctr = (float(hdr.get("CRPIX1", hdr["NAXIS1"] / 2)) - 1.0,
               float(hdr.get("CRPIX2", hdr["NAXIS2"] / 2)) - 1.0)
        return derotate_crop(np.asarray(hdu.section[ch]), ys, ye, xs, xe, ang, ctr)

    def _cube_win(cube, hdr, ys, ye, xs, xe):
        """A cube's spatial window, derotated channel by channel if requested.
        Rotation and crop are composed, so only the window is computed."""
        ang = float(hdr.get("INST_ROT", 0.0)) if derotate else 0.0
        if abs(ang) < 1e-9:
            return cube[:, ys:ye, xs:xe].astype(np.float32)
        ctr = (float(hdr.get("CRPIX1", cube.shape[2] / 2)) - 1.0,
               float(hdr.get("CRPIX2", cube.shape[1] / 2)) - 1.0)
        return np.stack([derotate_crop(cube[k], ys, ye, xs, xe, ang, ctr)
                         for k in range(cube.shape[0])])

    # Reference crop for tracking: prefer FE last wavelength (flare-free).
    # .section reads only the needed tiles from tile-compressed cubes.
    if has_fe:
        with fits.open(fe_names[0], memmap=True) as _h:
            ref_crop = _chan_win(_h[1], -1, ay0, ay1, ax0, ax1)
        if verbose:
            print("   Alignment reference: FE last wavelength (flare-free)")
    else:
        with fits.open(ha_names[0], memmap=True) as _h:
            ref_crop = _chan_win(_h[1], -1, ay0, ay1, ax0, ax1)
        if verbose:
            print("   Alignment reference: HA continuum (FE not available)")

    # Pass 1: measure every frame's shift on the tracking channel, then clean
    # the series as a whole. Per-frame decisions can't tell a failed
    # correlation from real drift; the full series can.
    pointing = {"align_mode": align_mode, "reference": reference,
                "drift_control": drift_control, "subpixel": bool(subpixel)}
    default_path = (align_mode == "xcorr" and reference == "first"
                    and shift_method == "phase")
    if patch is not None and track and default_path:
        shifts = []
        for i in range(nframes):
            track_name = fe_names[i] if has_fe else ha_names[i]
            with fits.open(track_name, memmap=True) as _h:
                cur = _chan_win(_h[1], -1, ay0, ay1, ax0, ax1)
            sy, sx = phase_shift(ref_crop, cur)
            shifts.append((float(sy), float(sx)))
        pointing["measured"] = [list(x) for x in shifts]
        if smooth_shifts:
            shifts = robust_shifts(shifts, verbose=verbose, ref_idx=0)
    elif patch is not None and track:
        shifts = _pass1_modes(
            ha_names, fe_names if has_fe else None, (ay0, ay1, ax0, ax1), _chan_win,
            H_full, W_full, derotate, inst_rot0, align_mode, reference, reference_window,
            drift_control, disc_center, rotation, shift_method, max_residual,
            header_sigma, step_sigma, smooth_shifts, pointing, verbose)
    else:
        shifts = [(0.0, 0.0)] * nframes
    if subpixel:
        shifts = [(float(sy), float(sx)) for sy, sx in shifts]
    else:
        shifts = [(int(round(sy)), int(round(sx))) for sy, sx in shifts]
    if patch is not None:
        big = max(max(abs(a), abs(b)) for a, b in shifts)
        if big > pad - (1 if subpixel else 0):
            raise ValueError(f"tracking shift of {big:.1f} px exceeds pad={pad}; increase pad")

    # Quiet-Sun background patch, offset left by one patch width.
    bg_dx = -(x1 - x0)
    bx0, bx1 = x0 + bg_dx, x1 + bg_dx
    by0, by1 = y0, y1
    has_bg = patch is not None and bx0 >= 0 and by0 >= 0 and bx1 <= W_full and by1 <= H_full

    ha_cubes = np.zeros((nframes, nch_ha, H, W), dtype=np.float32)
    fe_cubes = np.zeros((nframes, nch_fe, H, W), dtype=np.float32) if has_fe else None
    ha_bg = np.zeros((nframes, nch_ha, H, W), dtype=np.float32) if has_bg else None
    cont_raw = np.zeros((nframes, H, W), dtype=np.float32)
    core_raw = np.zeros((nframes, H, W), dtype=np.float32)
    align_ref = np.zeros((nframes, H, W), dtype=np.float32)
    times = []

    for i in range(nframes):
        hdu_ha = fits.open(ha_names[i])[1]
        cube_ha = hdu_ha.data
        hdr_ha = hdu_ha.header
        wav_ha_i = np.arange(hdr_ha["NAXIS3"]) * hdr_ha["CDELT3"] + hdr_ha["CRVAL3"]

        if has_fe:
            hdu_fe = fits.open(fe_names[i])[1]
            cube_fe = hdu_fe.data
            hdr_fe = hdu_fe.header
            wav_fe_i = np.arange(hdr_fe["NAXIS3"]) * hdr_fe["CDELT3"] + hdr_fe["CRVAL3"]

        # Shift-then-crop with the pass-1 (cleaned) shift for this frame.
        sy, sx = shifts[i]
        if verbose and (sy or sx):
            if subpixel:
                print(f"  Frame {i:>2d}: shift y={sy:+.2f}, x={sx:+.2f}")
            else:
                print(f"  Frame {i:>2d}: shift y={sy:+d}, x={sx:+d}")
        cy, cx = oy - sy, ox - sx

        ha_crop = _shift_crop(_cube_win(cube_ha, hdr_ha, py0, py1, px0, px1), cy, cx, H, W, subpixel)
        if resample:
            ha_crop = resample_wavelength(ha_crop, wav_ha_i, wavelength_ha)
        ha_cubes[i] = ha_crop
        cont_raw[i] = ha_cubes[i, -1]
        core_raw[i] = ha_cubes[i, core_idx]

        if has_bg:
            isy, isx = int(round(sy)), int(round(sx))
            bg = _cube_win(cube_ha, hdr_ha, by0 - isy, by1 - isy, bx0 - isx, bx1 - isx)
            ha_bg[i] = resample_wavelength(bg, wav_ha_i, wavelength_ha) if resample else bg

        if has_fe:
            fe_crop = _shift_crop(_cube_win(cube_fe, hdr_fe, py0, py1, px0, px1), cy, cx, H, W, subpixel)
            if resample:
                fe_crop = resample_wavelength(fe_crop, wav_fe_i, wavelength_fe)
            fe_cubes[i] = fe_crop
            align_ref[i] = fe_cubes[i, -1]
        else:
            align_ref[i] = cont_raw[i]

        times.append(hdr_ha.get("DATE-OBS", hdr_ha.get("DATE_OBS", f"frame_{i}")))
        if verbose:
            print(f"  Loading FITS {i + 1}/{nframes}", end="\r")

    if verbose:
        print(f"\nLoaded {nframes} frames, {H}x{W} px; HA {nch_ha} ch, FE {nch_fe} ch")

    return {
        "ha_cubes": ha_cubes,
        "fe_cubes": fe_cubes,
        "cont_raw": cont_raw,
        "core_raw": core_raw,
        "align_ref": align_ref,
        "ha_bg": ha_bg,
        "wavelength_ha": wavelength_ha,
        "wavelength_fe": wavelength_fe,
        "times": times,
        "nframes": nframes,
        "H": H,
        "W": W,
        "nchannels_ha": nch_ha,
        "nchannels_fe": nch_fe,
        "core_idx": core_idx,
        "patch": np.array(patch) if patch is not None else np.array([]),
        "shifts": shifts,
        "inst_rot": inst_rot0,
        "derotated": bool(derotate),
        "crpix": (float(hdr0.get("CRPIX1", W_full / 2)),
                  float(hdr0.get("CRPIX2", H_full / 2))),
        "cdelt": float(hdr0.get("CDELT1", 1.04)),
        "pointing": pointing,
    }


def _pass1_modes(ha_names, fe_names, awin, chan_win, H_full, W_full, derotate, inst_rot0,
                 align_mode, reference, reference_window, drift_control, disc_center,
                 rotation, shift_method, max_residual, header_sigma, step_sigma,
                 smooth_shifts, pointing, verbose):
    """Pass 1 of :func:`load_flare_sequence` for the non-default alignment modes:
    gather the header pointing from the FITS files, then call
    :func:`alignment_shifts`."""
    import astropy.io.fits as fits

    ay0, ay1, ax0, ax1 = awin
    track_names = fe_names if fe_names is not None else ha_names
    need_header = align_mode in ("header", "header+xcorr") or drift_control == "header"
    h = None
    if need_header:
        hdrs = [fits.getheader(f, 1) for f in track_names]
        times = [_obs_time(hd, i) for i, hd in enumerate(hdrs)]
        if any(t is None for t in times):
            raise ValueError("header alignment needs DATE-OBS/DATE_OBS in every file")
        centers, radius, used = _disc_centers(track_names, hdrs, disc_center, derotate, verbose)
        roi_c = ((ay0 + ay1 - 1) / 2.0, (ax0 + ax1 - 1) / 2.0)
        h = header_shifts(centers, times, roi_c, radius, rotation=rotation,
                          north_angle=0.0 if derotate else inst_rot0)
        pointing.update({"disc_center": used, "disc_centers": [list(c) for c in centers],
                         "radius_px": radius})
        if verbose:
            print(f"   Header pointing: disc centre from {used}, rotation={'on' if rotation else 'off'}; "
                  f"last scan shift y={h[-1][0]:+.2f}, x={h[-1][1]:+.2f} px")

    def read_window(i, ys, ye, xs, xe):
        with fits.open(track_names[i], memmap=True) as _h:
            return chan_win(_h[1], -1, ys, ye, xs, xe)

    return alignment_shifts(
        read_window, len(ha_names), awin, (H_full, W_full), align_mode=align_mode,
        header=h, reference=reference, reference_window=reference_window,
        drift_control=drift_control, shift_method=shift_method, max_residual=max_residual,
        header_sigma=header_sigma, step_sigma=step_sigma, smooth_shifts=smooth_shifts,
        pointing=pointing, verbose=verbose)


def alignment_shifts(
    read_window,
    nframes: int,
    align_patch: Sequence[int],
    frame_shape: Sequence[int],
    align_mode: str = "xcorr",
    header=None,
    reference: str = "first",
    reference_window: int = 3,
    drift_control: str = "none",
    shift_method: str = "phase",
    max_residual: float = 3.0,
    header_sigma: float = 1.0,
    step_sigma: float = 0.25,
    smooth_shifts: bool = True,
    pointing: Optional[dict] = None,
    verbose: bool = False,
):
    """Per-scan crop shifts for every alignment mode of :func:`load_flare_sequence`.

    This is the loader's pass 1 without the file handling, so the same logic
    can run on windows that are already in memory.

    Parameters
    ----------
    read_window : callable
        ``read_window(i, y0, y1, x0, x1) -> ndarray (h, w)``: the tracking
        channel of scan ``i`` in the given frame window.
    nframes : int
    align_patch : [y0, y1, x0, x1]
        Nominal window for the correlation.
    frame_shape : (H, W)
        Full frame size, for bounds checks.
    align_mode, reference, reference_window, drift_control, shift_method,
    max_residual, header_sigma, step_sigma, smooth_shifts
        As in :func:`load_flare_sequence`.
    header : list of (sy, sx) or None
        Header shifts (:func:`chase.calib.pointing.header_shifts`); required
        for ``"header"``, ``"header+xcorr"`` and ``drift_control="header"``.
    pointing : dict or None, optional
        Filled with the header shifts, the correlation measurements and the
        number of rejected residuals.
    verbose : bool, optional

    Returns
    -------
    list of (sy, sx) floats, scan 0 = (0, 0).
    """
    if pointing is None:
        pointing = {}
    n = nframes
    ay0, ay1, ax0, ax1 = align_patch
    H_full, W_full = frame_shape
    h = header
    need_header = align_mode in ("header", "header+xcorr") or drift_control == "header"
    if need_header and h is None:
        raise ValueError(f"align_mode={align_mode!r} / drift_control={drift_control!r} "
                         "needs the header shifts")
    if h is not None:
        pointing["header_shifts"] = [list(map(float, x)) for x in h]
    if align_mode == "header":
        return [tuple(map(float, x)) for x in h]

    # Correlation on windows read at the nominal position (xcorr) or at the
    # header-predicted position (header+xcorr).
    pre = [(0, 0)] * n if align_mode == "xcorr" else [(int(round(a)), int(round(b))) for a, b in h]
    windows = []
    for i in range(n):
        py, px = pre[i]
        ys, ye, xs, xe = ay0 - py, ay1 - py, ax0 - px, ax1 - px
        if ys < 0 or xs < 0 or ye > H_full or xe > W_full:
            raise ValueError(f"scan {i}: header-shifted align window leaves the frame")
        windows.append(read_window(i, ys, ye, xs, xe))
    fn = phase_shift if shift_method == "phase" else windowed_shift
    meas, steps = track_shifts(windows, reference=reference,
                               reference_window=reference_window, shift_fn=fn)
    pointing["measured"] = [list(x) for x in meas]
    pointing["steps"] = [list(x) for x in steps]

    if align_mode == "xcorr":
        if reference == "first":
            return robust_shifts(meas, verbose=verbose, ref_idx=0) if smooth_shifts else meas
        if drift_control == "header":
            return tie_to_header(steps, h, sigma_step=step_sigma, sigma_header=header_sigma,
                                 verbose=verbose)
        return meas

    # header+xcorr
    if reference == "first":
        res = np.asarray(meas, dtype=float)
        bad = np.hypot(res[:, 0], res[:, 1]) > max_residual
        if verbose:
            for k in np.nonzero(bad)[0]:
                print(f"  Frame {k:>2d}: correlation residual ({res[k, 0]:+.1f}, {res[k, 1]:+.1f}) px "
                      f"> {max_residual} px; header shift only")
        res[bad] = 0.0
        pointing["n_residual_rejected"] = int(bad.sum())
        return [(pre[i][0] + res[i, 0], pre[i][1] + res[i, 1]) for i in range(n)]
    abs_steps = [(0.0, 0.0)] + [(steps[i][0] + pre[i][0] - pre[i - 1][0],
                                 steps[i][1] + pre[i][1] - pre[i - 1][1]) for i in range(1, n)]
    return tie_to_header(abs_steps, h, sigma_step=step_sigma, sigma_header=header_sigma,
                         verbose=verbose)
