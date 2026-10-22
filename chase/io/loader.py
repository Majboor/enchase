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
  for the saved data).

Power users who want only one of these steps can call the modular functions in
:mod:`chase.calib` directly — this loader simply orchestrates them with the
streaming, memory-aware FITS reads that big full-disk cubes require.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import _compat  # noqa: F401
from ..calib.spatial import derotate_crop, phase_shift
from ..calib.wavelength import resample_wavelength
from .download import discover_fits

__all__ = ["load_flare_sequence", "robust_shifts"]


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

    Returns
    -------
    dict
        Keys: ``ha_cubes`` (nframes, nch_ha, H, W), ``fe_cubes`` (or None),
        ``cont_raw``, ``core_raw``, ``align_ref``, ``ha_bg`` (or None),
        ``wavelength_ha``, ``wavelength_fe``, ``times``, ``nframes``, ``H``,
        ``W``, ``nchannels_ha``, ``nchannels_fe``, ``core_idx``, ``patch``.
    """
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
    if patch is not None and track:
        shifts = []
        for i in range(nframes):
            track_name = fe_names[i] if has_fe else ha_names[i]
            with fits.open(track_name, memmap=True) as _h:
                cur = _chan_win(_h[1], -1, ay0, ay1, ax0, ax1)
            sy, sx = phase_shift(ref_crop, cur)
            shifts.append((float(sy), float(sx)))
        if smooth_shifts:
            shifts = robust_shifts(shifts, verbose=verbose, ref_idx=0)
        shifts = [(int(round(sy)), int(round(sx))) for sy, sx in shifts]
    else:
        shifts = [(0, 0)] * nframes

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
            print(f"  Frame {i:>2d}: shift y={sy:+d}, x={sx:+d}")
        cy, cx = oy - sy, ox - sx

        ha_crop = _cube_win(cube_ha, hdr_ha, py0, py1, px0, px1)[:, cy:cy + H, cx:cx + W]
        if resample:
            ha_crop = resample_wavelength(ha_crop, wav_ha_i, wavelength_ha)
        ha_cubes[i] = ha_crop
        cont_raw[i] = ha_cubes[i, -1]
        core_raw[i] = ha_cubes[i, core_idx]

        if has_bg:
            bg = _cube_win(cube_ha, hdr_ha, by0 - sy, by1 - sy, bx0 - sx, bx1 - sx)
            ha_bg[i] = resample_wavelength(bg, wav_ha_i, wavelength_ha) if resample else bg

        if has_fe:
            fe_crop = _cube_win(cube_fe, hdr_fe, py0, py1, px0, px1)[:, cy:cy + H, cx:cx + W]
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
    }
