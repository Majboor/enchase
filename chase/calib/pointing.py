"""Pointing from the header, differential rotation, and chained tracking.

Cross-correlation tracking against the first scan (the default of
:func:`chase.io.load_flare_sequence`) needs a channel with enough contrast that
does not change during the series. Two cases break it:

* **Hα-only data** (no Fe I file). The fallback tracking channel is the Hα far
  wing, whose contrast is 1-3 % on quiet Sun and filaments, and along a limb
  the image has structure across the limb only.
* **Long quiet-Sun series.** Granulation evolves, so after about 15 minutes a
  scan no longer correlates with the first one, also in Fe I.

This module provides the pieces of the alternatives:

* :func:`header_disc_center` and :func:`limb_disc_centers` — the disc centre
  of each scan, from the Level-1 header (``CRPIX1``, ``CRPIX2``) or from a
  sub-pixel limb fit;
* :func:`rotation_shift` — the displacement of a surface feature by solar
  differential rotation (Snodgrass & Ulrich 1990), including ``B0``;
* :func:`header_shifts` — the per-scan crop shift of a region of interest from
  the disc-centre drift plus differential rotation;
* :func:`track_shifts` — cross-correlation shifts against the first scan, the
  previous scan, or a running mean of the last few aligned scans;
* :func:`tie_to_header` — least-squares combination of chained scan-to-scan
  steps with the header pointing, so that chained errors cannot random-walk.

All shifts use the loader's convention: ``(sy, sx)`` is the shift that
registers a scan onto the reference (the output of :func:`phase_shift`); the
crop of scan ``i`` is taken at ``patch - (sy, sx)``. A feature that moves by
``+d`` pixels therefore has shift ``-d``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "header_disc_center",
    "limb_disc_centers",
    "solar_b0",
    "parse_obs_time",
    "rotation_shift",
    "header_shifts",
    "windowed_shift",
    "track_shifts",
    "tie_to_header",
]

#: Snodgrass & Ulrich (1990) sidereal rotation coefficients, deg/day.
SNODGRASS = (14.713, -2.396, -1.787)
#: Mean orbital motion of the Earth, deg/day (sidereal to synodic).
EARTH_RATE = 0.9856


# ---------------------------------------------------------------------------
# Disc centre
# ---------------------------------------------------------------------------
def header_disc_center(header) -> Tuple[float, float]:
    """Disc centre of one scan from its Level-1 header.

    ``CRPIX1`` is the reference pixel along FITS axis 1 (``NAXIS1``), which is
    the *last* numpy axis (columns, x); ``CRPIX2`` is along axis 2 (rows, y).
    Both are one-based. Checked on real data against a sub-pixel limb fit of
    every scan of the 2023-12-29, 2024-10-03, 2024-12-14 and 2026-07-04 sets:
    with this mapping the header and the limb agree to 0.5-0.7 px and their
    scan-to-scan drifts to 0.10-0.23 px rms; with the axes swapped they differ
    by 20-66 px. In the 2023-03-29 Level-1 files the header centre is 37 px
    from the limb with this mapping and its drift does not follow the limb
    under either mapping (6.8-7.7 px rms). Header recentring is not
    recommended for those files: even with per-scan limb fits
    (``disc_center="limb"``) the odd and even scans had field-dependent
    offsets of 8-11 px that no disc centre follows.

    Parameters
    ----------
    header : mapping
        FITS header with ``CRPIX1``/``CRPIX2`` (falls back to the array centre
        from ``NAXIS1``/``NAXIS2`` when absent).

    Returns
    -------
    (cx, cy) : float, float
        Zero-based column and row of the disc centre.
    """
    cx = float(header.get("CRPIX1", header.get("NAXIS1", 0) / 2 + 1)) - 1.0
    cy = float(header.get("CRPIX2", header.get("NAXIS2", 0) / 2 + 1)) - 1.0
    return cx, cy


def limb_disc_centers(images: Sequence[np.ndarray], **kwargs) -> List[Tuple[float, float, float]]:
    """Disc centre and radius of each full-disc image from a sub-pixel limb fit.

    Thin wrapper over :func:`chase.analysis.systematics.fit_limb_ellipse`
    (axis-aligned ellipse, so the slightly elliptical 2023 limbs are handled).

    Parameters
    ----------
    images : sequence of ndarray (H, W)
        One full-disc continuum or far-wing image per scan.
    **kwargs
        Passed to :func:`fit_limb_ellipse`.

    Returns
    -------
    list of (cx, cy, radius) in zero-based pixels.
    """
    from ..analysis.systematics import fit_limb_ellipse

    out = []
    for img in images:
        f = fit_limb_ellipse(img, **kwargs)
        out.append((f["x0"], f["y0"], f["radius_px"]))
    return out


# ---------------------------------------------------------------------------
# Time and solar ephemeris
# ---------------------------------------------------------------------------
def parse_obs_time(value: Union[str, datetime]) -> datetime:
    """Parse a FITS ``DATE-OBS``/``DATE_OBS`` string (ISO 8601, optional Z)."""
    if isinstance(value, datetime):
        return value
    s = str(value).strip().rstrip("Z")
    return datetime.fromisoformat(s)


def solar_b0(time: Union[str, datetime]) -> float:
    """Heliographic latitude of the disc centre ``B0`` in degrees.

    Low-precision formula (Meeus, *Astronomical Algorithms*, ch. 29) with the
    Sun's apparent longitude from ch. 25; accurate to about 0.01 deg, far
    better than needed for the rotation of a crop window.
    """
    t = parse_obs_time(time)
    # Julian day (UT is close enough here)
    jd = (t - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0 + 2451545.0
    T = (jd - 2451545.0) / 36525.0
    L0 = 280.46646 + 36000.76983 * T
    M = np.deg2rad(357.52911 + 35999.05029 * T)
    C = ((1.914602 - 0.004817 * T) * np.sin(M) + 0.019993 * np.sin(2 * M)
         + 0.000289 * np.sin(3 * M))
    omega = np.deg2rad(125.04 - 1934.136 * T)
    lam = np.deg2rad(L0 + C - 0.00569 - 0.00478 * np.sin(omega))
    K = np.deg2rad(73.6667 + 1.3958333 * (jd - 2396758.0) / 36525.0)
    inc = np.deg2rad(7.25)
    return float(np.rad2deg(np.arcsin(np.sin(lam - K) * np.sin(inc))))


def _synodic_rate(lat_rad: float) -> float:
    """Synodic Snodgrass rotation rate in rad/s at heliographic latitude."""
    a, b, c = SNODGRASS
    s2 = np.sin(lat_rad) ** 2
    return np.deg2rad(a + b * s2 + c * s2 ** 2 - EARTH_RATE) / 86400.0


def rotation_shift(
    y: float,
    x: float,
    center: Tuple[float, float],
    radius: float,
    dt: float,
    b0: float = 0.0,
    north_angle: float = 0.0,
) -> Tuple[float, float]:
    """Image displacement of a surface feature by differential rotation.

    The point is projected orthographically onto the sphere, its Carrington
    longitude is advanced by the synodic Snodgrass rate at its latitude for
    ``dt`` seconds, and it is projected back.

    Parameters
    ----------
    y, x : float
        Pixel position of the feature (zero-based row, column) at time 0.
    center : (cx, cy)
        Disc centre in zero-based pixels at time 0.
    radius : float
        Disc radius in pixels.
    dt : float
        Elapsed time in seconds (may be negative).
    b0 : float, optional
        Heliographic latitude of disc centre in degrees (see :func:`solar_b0`).
    north_angle : float, optional
        Angle of solar north from the image +y axis, degrees, counter-clockwise
        (the ``INST_ROT`` convention of :func:`chase.calib.derotate`). 0 for
        north-up data and for frames derotated with ``derotate=True``.
        Image +x is assumed to point to solar west and +y to north.

    Returns
    -------
    (dy, dx) : float, float
        Displacement in pixels (rows, columns). ``(0, 0)`` for points on or
        outside the limb, where the projected position does not rotate.
    """
    cx, cy = center
    th = np.deg2rad(north_angle)
    u, v = (x - cx) / radius, (y - cy) / radius
    # image -> (west, north) of the solar disc
    X = u * np.cos(th) + v * np.sin(th)
    Y = -u * np.sin(th) + v * np.cos(th)
    rho2 = X * X + Y * Y
    if rho2 >= 1.0:
        return 0.0, 0.0
    Z = np.sqrt(1.0 - rho2)
    B = np.deg2rad(b0)
    sin_lat = Y * np.cos(B) + Z * np.sin(B)
    lat = np.arcsin(np.clip(sin_lat, -1, 1))
    lon = np.arctan2(X, Z * np.cos(B) - Y * np.sin(B))
    lon2 = lon + _synodic_rate(lat) * dt
    X2 = np.cos(lat) * np.sin(lon2)
    Y2 = np.sin(lat) * np.cos(B) - np.cos(lat) * np.cos(lon2) * np.sin(B)
    dX, dY = (X2 - X) * radius, (Y2 - Y) * radius
    du = dX * np.cos(th) - dY * np.sin(th)
    dv = dX * np.sin(th) + dY * np.cos(th)
    return float(dv), float(du)


def header_shifts(
    centers: Sequence[Tuple[float, float]],
    times: Sequence[Union[str, datetime]],
    roi_center: Tuple[float, float],
    radius: float,
    rotation: bool = True,
    b0: Optional[float] = None,
    north_angle: float = 0.0,
) -> List[Tuple[float, float]]:
    """Crop shifts of a region of interest from the disc-centre drift.

    The region is assumed to move with the disc centre plus, if ``rotation``,
    the differential rotation of its centre. No image content is used, so the
    result does not depend on the contrast or the evolution of the field.

    Parameters
    ----------
    centers : sequence of (cx, cy)
        Disc centre of every scan (zero-based px), e.g. from
        :func:`header_disc_center` or :func:`limb_disc_centers`.
    times : sequence of str or datetime
        Observation time of every scan.
    roi_center : (y, x)
        Centre of the region in scan 0 (zero-based px).
    radius : float
        Disc radius in pixels (header ``R_SUN`` or a limb fit).
    rotation : bool, optional
        Add differential rotation (default True). Turn off for off-limb
        regions, which do not rotate in the image.
    b0 : float or None, optional
        ``B0`` in degrees; ``None`` computes it from ``times[0]``.
    north_angle : float, optional
        See :func:`rotation_shift`.

    Returns
    -------
    list of (sy, sx) floats, scan 0 = (0, 0), in the loader's sign convention
    (minus the displacement of the region).
    """
    c = np.asarray(centers, dtype=float)
    t = [parse_obs_time(s) for s in times]
    if rotation and b0 is None:
        b0 = solar_b0(t[0])
    y, x = roi_center
    out = []
    for i in range(len(c)):
        dcx, dcy = c[i, 0] - c[0, 0], c[i, 1] - c[0, 1]
        if rotation:
            dy, dx = rotation_shift(y, x, (c[0, 0], c[0, 1]), radius,
                                    (t[i] - t[0]).total_seconds(), b0, north_angle)
        else:
            dy, dx = 0.0, 0.0
        out.append((-(dcy + dy), -(dcx + dx)))
    return out


# ---------------------------------------------------------------------------
# Cross-correlation tracking
# ---------------------------------------------------------------------------
def windowed_shift(reference: np.ndarray, target: np.ndarray,
                   upsample_factor: int = 20) -> Tuple[float, float]:
    """Shift from Hann-windowed, unnormalised cross-correlation.

    Same estimator as :func:`chase.analysis.validation.residual_pointing`.
    Phase normalisation (the :func:`phase_shift` default) weights all
    spatial frequencies equally, so on low-contrast fields (1-3 % far-wing
    contrast) it follows noise; the unnormalised correlation is dominated by
    the real large-scale structure.

    Returns
    -------
    (sy, sx) : float, float
    """
    from skimage.registration import phase_cross_correlation

    ref = np.asarray(reference, dtype=np.float32)
    cur = np.asarray(target, dtype=np.float32)
    win = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1])).astype(np.float32)
    s, _, _ = phase_cross_correlation((ref - ref.mean()) * win, (cur - cur.mean()) * win,
                                      upsample_factor=upsample_factor, normalization=None)
    return float(s[0]), float(s[1])


def _subpixel_move(img: np.ndarray, sy: float, sx: float) -> np.ndarray:
    from scipy.ndimage import shift as ndshift

    return ndshift(np.asarray(img, dtype=np.float32), (sy, sx), order=1, mode="nearest")


def track_shifts(
    windows: Sequence[np.ndarray],
    reference: str = "first",
    reference_window: int = 3,
    shift_fn: Optional[Callable] = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Cross-correlation shifts of every scan against a chosen reference.

    Parameters
    ----------
    windows : sequence of ndarray (h, w)
        The tracking window of every scan, all read at the same nominal
        position (or each at its header-predicted position, in which case the
        result is the residual on top of the header shifts).
    reference : {"first", "previous", "running"}, optional
        ``"first"`` (default, the v2.1 behaviour) correlates every scan with
        scan 0. ``"previous"`` correlates scan ``i`` with scan ``i-1`` and sums
        the steps. ``"running"`` correlates scan ``i`` with the mean of the
        last ``reference_window`` scans, each moved (sub-pixel, bilinear) onto
        the position of scan ``i-1`` with its own tracked shift. Chained
        references keep working when the field evolves (quiet-Sun granulation
        decorrelates after about 15 minutes), but their errors add up; tie
        them back with :func:`tie_to_header`.
    reference_window : int, optional
        Number of previous scans in the running mean (default 3).
    shift_fn : callable or None, optional
        ``shift_fn(reference, target) -> (sy, sx)``; default :func:`phase_shift`.

    Returns
    -------
    shifts : list of (sy, sx)
        Float shift of every scan relative to scan 0 (scan 0 = (0, 0)).
    steps : list of (sy, sx)
        The measured step of every scan relative to its reference
        (scan 0 = (0, 0)); for ``"first"`` equal to ``shifts``.
    """
    if shift_fn is None:
        from .spatial import phase_shift as shift_fn
    if reference not in ("first", "previous", "running"):
        raise ValueError(f"reference must be 'first', 'previous' or 'running', got {reference!r}")
    n = len(windows)
    shifts = [(0.0, 0.0)]
    steps = [(0.0, 0.0)]
    if reference == "first":
        for i in range(1, n):
            s = shift_fn(windows[0], windows[i])
            shifts.append((float(s[0]), float(s[1])))
            steps.append(shifts[-1])
        return shifts, steps
    k = max(1, int(reference_window))
    for i in range(1, n):
        py, px = shifts[i - 1]
        if reference == "previous" or k == 1:
            ref = windows[i - 1]
        else:
            js = range(max(0, i - k), i)
            ref = np.mean([_subpixel_move(windows[j], shifts[j][0] - py, shifts[j][1] - px)
                           for j in js], axis=0)
        s = shift_fn(ref, windows[i])
        steps.append((float(s[0]), float(s[1])))
        shifts.append((py + float(s[0]), px + float(s[1])))
    return shifts, steps


def tie_to_header(
    steps: Sequence[Tuple[float, float]],
    header: Sequence[Tuple[float, float]],
    sigma_step: float = 0.25,
    sigma_header: float = 1.0,
    threshold: float = 4.0,
    verbose: bool = False,
) -> List[Tuple[float, float]]:
    """Combine chained scan-to-scan steps with the header pointing.

    Solves, per axis, the least-squares problem

    .. math::
        \\min_x \\sum_{i\\ge1} \\frac{(x_i - x_{i-1} - s_i)^2}{\\sigma_s^2}
        + \\sum_{i\\ge1} \\frac{(x_i - h_i)^2}{\\sigma_h^2},\\quad x_0 = 0,

    where ``s_i`` are the measured steps and ``h_i`` the header shifts. Short
    term the chained correlation dominates (it sees the real image motion);
    long term the solution cannot wander further from the header than about
    ``sigma_header``, so chained errors do not random-walk. A step that
    differs from the header step ``h_i - h_{i-1}`` by more than ``threshold``
    pixels is a failed correlation and is dropped (the header step is used).

    Parameters
    ----------
    steps : sequence of (sy, sx)
        Step of scan ``i`` relative to scan ``i-1`` (entry 0 ignored).
        For the ``"running"`` reference these are the steps returned by
        :func:`track_shifts` (relative to the running mean placed at scan
        ``i-1``).
    header : sequence of (sy, sx)
        Header shifts (:func:`header_shifts`), scan 0 = (0, 0).
    sigma_step, sigma_header : float, optional
        Assumed 1-sigma errors in pixels of one step and of the header
        pointing (defaults 0.25 and 1.0).
    threshold : float, optional
        Outlier threshold on the step in pixels (default 4).
    verbose : bool, optional

    Returns
    -------
    list of (sy, sx) floats, scan 0 = (0, 0).
    """
    s = np.asarray(steps, dtype=float)
    h = np.asarray(header, dtype=float)
    n = len(h)
    out = np.zeros((n, 2))
    if n < 2:
        return [tuple(r) for r in out]
    ws, wh = 1.0 / sigma_step ** 2, 1.0 / sigma_header ** 2
    for axis in (0, 1):
        hs = np.diff(h[:, axis])
        st = s[1:, axis].copy()
        bad = np.abs(st - hs) > threshold
        if verbose:
            for k in np.nonzero(bad)[0]:
                print(f"  Frame {k + 1:>2d}: {'yx'[axis]}-step {st[k]:+.1f} px rejected "
                      f"(header step {hs[k]:+.1f}); using the header step")
        st[bad] = hs[bad]
        # unknowns x_1..x_{n-1}; x_0 = 0
        m = n - 1
        A = np.zeros((m, m))
        b = np.zeros(m)
        for i in range(1, n):          # step terms (x_i - x_{i-1} - st)
            r = i - 1
            A[r, r] += ws
            b[r] += ws * st[i - 1]
            if i >= 2:
                A[r - 1, r - 1] += ws
                A[r, r - 1] -= ws
                A[r - 1, r] -= ws
                b[r - 1] -= ws * st[i - 1]
        A[np.diag_indices(m)] += wh
        b += wh * h[1:, axis]
        out[1:, axis] = np.linalg.solve(A, b)
    return [tuple(map(float, r)) for r in out]
