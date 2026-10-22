"""Spatial calibration: disk centring and crop-window tracking.

Two disk-centring strategies are provided:

* :func:`hough_disk_center` — robust limb detection via a Hough-circle fit on
  the Canny edge of the solar limb.  Ported from Finlay Davis's ``satprocess``
  (BSD-3-Clause, see ``NOTICE``); far more reliable than a brightness centroid
  when flares or limb darkening skew the intensity distribution.
* :func:`find_disk_center` — the original brightness-centroid fallback (no
  ``scikit-image`` circle search required).
* :func:`fit_limb` / :func:`limb_disk_center` — sub-pixel least-squares fit
  of a circle or a rotated ellipse to limb points found on radial rays
  (:func:`limb_points`). Use ``model="ellipse"`` when the Level-1 limb is not
  round (e.g. a scan-axis scale error); :func:`limb_shifts` turns the fitted
  centres of a sequence into recentring shifts.

:func:`track_and_crop` implements the **shift-then-crop** rule that is critical
to the flare pipeline: load a padded region, estimate the integer shift on a
flare-free reference channel, then crop the final patch from *inside* the
buffer.  Cropping first and shifting afterwards reintroduces the antisymmetric
pre-flare colour artefact Alex caught.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "derotate",
    "derotate_crop",
    "find_disk_center",
    "fit_limb",
    "hough_disk_center",
    "limb_disk_center",
    "limb_points",
    "limb_shifts",
    "phase_shift",
    "recenter",
    "track_and_crop",
]


# ---------------------------------------------------------------------------
# Derotation to solar north
# ---------------------------------------------------------------------------

def derotate(image: np.ndarray, angle_deg: float,
             center: Tuple[float, float]) -> np.ndarray:
    """Rotate an image about ``center`` to align it with solar north.

    CHASE Level-1 headers record ``INST_ROT``, the angle of solar north with
    respect to the detector y-axis (~11 deg for the 2023 data). Unlike
    instruments where this offset must be fitted against an external reference
    (e.g. THEMIS, Peat et al. 2026), CHASE provides it in every file, so the
    correction is a single rotation about the disc centre
    (``CRPIX1``, ``CRPIX2``).

    Parameters
    ----------
    image : ndarray (H, W)
        A single spatial frame.
    angle_deg : float
        Rotation angle in degrees, OpenCV convention (positive rotates the
        image content clockwise when displayed with ``origin='lower'``).
        Pass the header's ``INST_ROT`` to bring solar north to +y. Validated
        against a co-temporal SDO/HMI continuum image (2023-03-29 02:41 UT):
        sunspot position angles in the detector frame differ from the north-up
        HMI reference by ~``INST_ROT`` and agree to within ~2 deg after this
        rotation.
    center : (cx, cy)
        Rotation centre in zero-based pixel coordinates
        (``CRPIX1 - 1``, ``CRPIX2 - 1`` for FITS one-based headers).

    Returns
    -------
    ndarray (H, W) float32
        The derotated frame. Corners that rotate out of the field are filled
        with 0.
    """
    import cv2

    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])),
                                float(angle_deg), 1.0)
    return cv2.warpAffine(
        np.ascontiguousarray(image, dtype=np.float32), M,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def derotate_crop(image: np.ndarray, ys: int, ye: int, xs: int, xe: int,
                  angle_deg: float, center: Tuple[float, float]) -> np.ndarray:
    """Derotate a full frame and return only the window ``[ys:ye, xs:xe]``.

    The rotation and the crop are composed into a single ``warpAffine`` call,
    so only the requested window is ever computed --- rotating a whole CHASE
    channel (2313x2304) to keep a 140x170 patch would waste two orders of
    magnitude of work. The window coordinates refer to the *derotated* frame,
    i.e. the same frame :func:`derotate` would return.
    """
    import cv2

    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])),
                                float(angle_deg), 1.0)
    M[0, 2] -= xs
    M[1, 2] -= ys
    return cv2.warpAffine(
        np.ascontiguousarray(image, dtype=np.float32), M, (xe - xs, ye - ys),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


# ---------------------------------------------------------------------------
# Disk centring
# ---------------------------------------------------------------------------
def find_disk_center(image: np.ndarray, threshold_ratio: float = 0.2):
    """Brightness-centroid disk centre (original ``chase`` method).

    Parameters
    ----------
    image : ndarray (H, W)
    threshold_ratio : float, optional
        Pixels brighter than ``threshold_ratio * max`` are counted as disk.

    Returns
    -------
    (cx, cy, mask) : int, int, ndarray
    """
    norm = image / np.max(image)
    mask = norm > threshold_ratio
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("No disk detected.")
    return int(xs.mean()), int(ys.mean()), mask


# -- BSD 3-Clause License -----------------------------------------------------
# Copyright (c) 2025, Finlay Davis.  All rights reserved.
# Ported from FinlayDavis/satprocess (preprocess_image + detect_circles).
# -----------------------------------------------------------------------------
def hough_disk_center(
    image: np.ndarray,
    sigma: float = 2.0,
    reference_radius: Optional[float] = None,
    margin_percent: float = 2.0,
    min_radius: int = 500,
    max_radius: int = 1500,
) -> Tuple[float, float, float]:
    """Detect the solar disk centre via a Hough-circle limb fit.

    The image is Gaussian-blurred, Otsu-thresholded, and Canny-edged to isolate
    the limb ring; a Hough-circle transform then finds the best circle.  When a
    ``reference_radius`` is supplied (from a previously fitted frame), the radius
    search is constrained to ``reference_radius ± margin_percent`` for speed and
    stability.

    Parameters
    ----------
    image : ndarray (H, W)
    sigma : float, optional
        Gaussian blur sigma before thresholding.
    reference_radius : float or None, optional
        If given, restrict the radius search around this value.
    margin_percent : float, optional
        Half-width of the constrained radius band, in percent.
    min_radius, max_radius : int, optional
        Wide radius search bounds when ``reference_radius`` is None.

    Returns
    -------
    (cx, cy, radius) : float, float, float
    """
    from skimage.feature import canny
    from skimage.filters import gaussian, threshold_otsu
    from skimage.transform import hough_circle, hough_circle_peaks

    img = image.astype(np.float64)
    blurred = gaussian(img, sigma=sigma, preserve_range=True)
    try:
        thresh = threshold_otsu(blurred)
    except ValueError:
        thresh = blurred.mean()
    binary = blurred > thresh
    edges = canny(binary.astype(float), sigma=1)

    if reference_radius is not None:
        margin = reference_radius * margin_percent / 100.0
        lo = max(1, int(reference_radius - margin))
        hi = int(reference_radius + margin) + 1
    else:
        lo, hi = min_radius, max_radius
    radii = np.arange(lo, hi, 1)
    if len(radii) == 0:
        raise ValueError("Empty radius search range for Hough circle.")

    hough = hough_circle(edges, radii)
    _, cx, cy, rad = hough_circle_peaks(hough, radii, total_num_peaks=1)
    if len(cx) == 0:
        raise ValueError("No circle detected on the limb edge map.")
    return float(cx[0]), float(cy[0]), float(rad[0])



# ---------------------------------------------------------------------------
# Sub-pixel limb fit (circle or rotated ellipse)
# ---------------------------------------------------------------------------
def limb_points(
    image: np.ndarray,
    center: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
    nang: int = 720,
    half_width: Optional[float] = None,
    smooth: float = 1.0,
) -> np.ndarray:
    """Sub-pixel limb points on radial rays.

    On each of ``nang`` rays from ``center`` the limb is the steepest intensity
    drop (minimum of the radial derivative) within ``radius ± half_width``,
    refined with a 3-point parabola. Rays that leave the image are skipped.

    Parameters
    ----------
    image : ndarray (H, W)
        Full-disc image of a continuum or far-wing channel.
    center, radius : optional
        Start point ``(cx, cy)`` and radius guess. ``None`` → brightness
        centroid and the radius of a disc with the same area (pixels above
        20 % of the 99th percentile).
    nang : int, optional
        Number of rays (default 720, every 0.5°).
    half_width : float or None, optional
        Search band half-width in px (default 3 % of the radius, at least 10).
    smooth : float, optional
        Gaussian smoothing (px) before differentiating (default 1).

    Returns
    -------
    ndarray (n, 2)
        ``(x, y)`` of the limb points, in pixels.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    im = np.asarray(image, dtype=np.float64)
    im = np.where(np.isfinite(im), im, 0.0)
    H, W = im.shape
    if center is None or radius is None:
        disc = im > 0.2 * np.percentile(im, 99)
        ys, xs = np.nonzero(disc)
        if len(xs) == 0:
            raise ValueError("No disk detected.")
        if center is None:
            center = (xs.mean(), ys.mean())
        if radius is None:
            radius = np.sqrt(disc.sum() / np.pi)
    cx, cy = float(center[0]), float(center[1])
    hw = max(10.0, 0.03 * radius) if half_width is None else float(half_width)
    if smooth:
        im = gaussian_filter(im, smooth)
    step = 0.25
    rr = np.arange(radius - hw, radius + hw, step)
    th = np.linspace(0, 2 * np.pi, nang, endpoint=False)
    X = cx + np.outer(np.cos(th), rr)
    Y = cy + np.outer(np.sin(th), rr)
    inside = ((X >= 0) & (X <= W - 1) & (Y >= 0) & (Y <= H - 1)).all(axis=1)
    prof = map_coordinates(im, [Y.ravel(), X.ravel()], order=1,
                           mode="nearest").reshape(X.shape)
    g = np.gradient(prof, axis=1)
    pts = []
    for k in np.nonzero(inside)[0]:
        j = int(np.argmin(g[k]))
        if 1 <= j < len(rr) - 1:
            a, b, c = g[k, j - 1], g[k, j], g[k, j + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if den != 0 else 0.0
            r = rr[j] + np.clip(d, -1, 1) * step
            pts.append((cx + r * np.cos(th[k]), cy + r * np.sin(th[k])))
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def _ellipse_residual(q, P):
    """Normalised radial residual of points P to ellipse q (cx, cy, a, b, phi), px."""
    cx, cy, a, b, phi = q
    c, s = np.cos(phi), np.sin(phi)
    dx, dy = P[:, 0] - cx, P[:, 1] - cy
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return (np.hypot(u / a, v / b) - 1.0) * np.sqrt(abs(a * b))


def fit_limb(
    image: np.ndarray,
    model: str = "circle",
    center: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
    nang: int = 720,
    clip: float = 3.0,
    niter: int = 5,
    points: Optional[np.ndarray] = None,
) -> dict:
    """Least-squares limb fit: circle or rotated ellipse.

    Limb points come from :func:`limb_points` (or ``points``). A circle is
    fitted first, with ``niter`` rounds of ``clip``-σ (MAD) rejection that
    drop rays crossing prominences, flare ribbons or detector edges. With
    ``model="ellipse"`` the kept points are then fitted with a general
    ellipse (centre, two semi-axes and the angle of the first axis).

    The ellipse centre is the right disc centre when the Level-1 image is
    stretched along one axis (a scan-axis plate-scale error): a circle or a
    Hough fit to such a limb has a centre that depends on which part of the
    limb dominates the vote.

    Parameters
    ----------
    image : ndarray (H, W)
        Full-disc image of a continuum or far-wing channel.
    model : {"circle", "ellipse"}
    center, radius, nang : optional
        Passed to :func:`limb_points`.
    clip, niter : optional
        Outlier rejection (robust σ, rounds).
    points : ndarray (n, 2), optional
        Use these limb points instead of detecting them.

    Returns
    -------
    dict
        ``cx``, ``cy`` (centre, px), ``a``, ``b`` (semi-axes, px; equal for a
        circle), ``angle_deg`` (angle of the ``a`` axis from +x),
        ``radius_px`` (√(ab)), ``rms_px`` (residual of the kept points),
        ``n_pts``, ``n_kept`` and ``model``.

    Raises
    ------
    ValueError
        For an unknown model or fewer than 10 limb points.
    """
    from scipy.optimize import least_squares

    if model not in ("circle", "ellipse"):
        raise ValueError(f"model must be 'circle' or 'ellipse', got {model!r}")
    P = limb_points(image, center, radius, nang=nang) if points is None else np.asarray(points, float)
    if len(P) < 10:
        raise ValueError(f"fit_limb: only {len(P)} limb points found")
    # algebraic circle start (Kasa), then geometric fit with clipping
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    p = np.array([sol[0] / 2, sol[1] / 2, 0.0])
    p[2] = np.sqrt(max(sol[2] + p[0] ** 2 + p[1] ** 2, 1.0))
    keep = np.ones(len(P), bool)
    for _ in range(niter):
        fc = lambda q: np.hypot(P[keep, 0] - q[0], P[keep, 1] - q[1]) - q[2]
        p = least_squares(fc, p).x
        res = np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]) - p[2]
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        nk = np.abs(res - np.median(res[keep])) < clip * max(s, 0.05)
        if nk.sum() < 10 or (nk == keep).all():
            break
        keep = nk
    out = {"cx": float(p[0]), "cy": float(p[1]), "a": float(p[2]), "b": float(p[2]),
           "angle_deg": 0.0, "radius_px": float(p[2]),
           "rms_px": float(np.sqrt(np.mean(res[keep] ** 2))),
           "n_pts": int(len(P)), "n_kept": int(keep.sum()), "model": model}
    if model == "circle":
        return out
    q0 = np.array([p[0], p[1], p[2] * 1.001, p[2] / 1.001, 0.0])
    for _ in range(niter):
        q = least_squares(_ellipse_residual, q0, args=(P[keep],)).x
        res = _ellipse_residual(q, P)
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        nk = np.abs(res - np.median(res[keep])) < clip * max(s, 0.05)
        q0 = q
        if nk.sum() < 10 or (nk == keep).all():
            break
        keep = nk
    cx, cy, a, b, phi = q0
    a, b = abs(a), abs(b)
    if b > a:                       # report a as the major axis
        a, b, phi = b, a, phi + np.pi / 2
    ang = (np.degrees(phi) + 90.0) % 180.0 - 90.0
    res = _ellipse_residual(q0, P)
    out.update(cx=float(cx), cy=float(cy), a=float(a), b=float(b), angle_deg=float(ang),
               radius_px=float(np.sqrt(a * b)),
               rms_px=float(np.sqrt(np.mean(res[keep] ** 2))), n_kept=int(keep.sum()))
    return out


def limb_disk_center(image: np.ndarray, model: str = "ellipse", **kwargs) -> Tuple[float, float, float]:
    """Sub-pixel disc centre from :func:`fit_limb`.

    A faster, sub-pixel alternative to :func:`hough_disk_center` (about 1 s
    instead of 1-2 min for a 2300×2300 px CHASE frame).

    Returns
    -------
    (cx, cy, radius) : float, float, float
        ``radius`` is √(ab) for an ellipse.
    """
    f = fit_limb(image, model=model, **kwargs)
    return f["cx"], f["cy"], f["radius_px"]


def limb_shifts(
    images: Sequence[np.ndarray],
    model: str = "ellipse",
    ref_idx: int = 0,
    integer: bool = True,
    return_fits: bool = False,
    **kwargs,
):
    """Recentring shifts from a limb fit on every scan.

    Parameters
    ----------
    images : sequence of ndarray (H, W)
        Full-disc continuum or far-wing image of each scan.
    model : {"circle", "ellipse"}
        Passed to :func:`fit_limb` (default ``"ellipse"``).
    ref_idx : int, optional
        Scan whose centre is the reference (shift 0).
    integer : bool, optional
        Round to whole pixels (for :func:`recenter` or an integer crop).
    return_fits : bool, optional
        Also return the list of :func:`fit_limb` dicts.

    Returns
    -------
    shifts : ndarray (n, 2)
        ``(dx, dy)`` that moves each scan's disc centre onto the reference
        centre, i.e. ``recenter(cube_i, shifts[i])`` registers scan ``i``.
        A crop window defined on the reference scan at ``(x, y)`` sits at
        ``(x - dx, y - dy)`` in scan ``i``.
    """
    fits_ = []
    guess = {}
    for im in images:
        f = fit_limb(im, model=model, **{**guess, **kwargs})
        fits_.append(f)
        guess = {"center": (f["cx"], f["cy"]), "radius": f["radius_px"]}
    c = np.array([[f["cx"], f["cy"]] for f in fits_])
    sh = c[ref_idx] - c
    if integer:
        sh = np.round(sh).astype(int)
    return (sh, fits_) if return_fits else sh


def recenter(cube: np.ndarray, shift_xy: Tuple[int, int]) -> np.ndarray:
    """Integer-roll every channel of a cube by ``(dx, dy)``."""
    dx, dy = shift_xy
    out = np.empty_like(cube)
    for i in range(cube.shape[0]):
        out[i] = np.roll(np.roll(cube[i], dy, axis=0), dx, axis=1)
    return out


# ---------------------------------------------------------------------------
# Crop-window tracking (shift-then-crop)
# ---------------------------------------------------------------------------
def phase_shift(reference: np.ndarray, target: np.ndarray, upsample_factor: int = 100):
    """Sub-pixel shift that best aligns ``target`` to ``reference``.

    Thin wrapper over :func:`skimage.registration.phase_cross_correlation`.

    Returns
    -------
    (sy, sx) : float, float
        Shift in pixels along (row, col).
    """
    from skimage.registration import phase_cross_correlation

    shift, _, _ = phase_cross_correlation(
        reference, target, upsample_factor=upsample_factor
    )
    return float(shift[0]), float(shift[1])


def track_and_crop(
    frames: Sequence[np.ndarray],
    patch: Sequence[int],
    ref_channel: int = -1,
    pad: int = 20,
    track: bool = True,
    align_patch: Optional[Sequence[int]] = None,
    reference_index: int = 0,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Track a moving feature across frames and crop a stable patch.

    Implements **shift-then-crop**: for each frame a padded region around
    ``patch`` is taken, the integer shift relative to the reference frame is
    estimated on ``ref_channel`` (a flare-free channel — e.g. the Fe I or Hα
    continuum), and the final patch is cropped from inside the padded buffer at
    the shifted offset.  Every frame therefore covers identical physical
    content, which removes the spectral-averaging artefact that crop-then-shift
    produces.

    Parameters
    ----------
    frames : sequence of ndarray (nchannels, H, W)
        Per-frame spectral cubes (full FOV, or a region large enough to contain
        ``patch`` plus ``pad``).
    patch : [y0, y1, x0, x1]
        The data window to extract.
    ref_channel : int, optional
        Channel index used to estimate the shift (default ``-1`` = last channel,
        which is the flare-free continuum / photosphere).
    pad : int, optional
        Padding in pixels added on each side before shifting (default 20; must
        exceed the largest expected inter-frame shift).
    track : bool, optional
        If False, use a fixed window (zero shift) — useful when optical flow
        will do all the alignment.
    align_patch : [y0, y1, x0, x1] or None, optional
        A *separate, usually smaller* window used only for shift estimation
        (Alex's idea: small box for the shift, large box for the saved data,
        reducing wobble).  Defaults to ``patch``.
    reference_index : int, optional
        Frame whose patch defines the alignment reference.

    Returns
    -------
    cropped : ndarray (nframes, nchannels, h, w)
    shifts : list of (sy, sx) integer shifts applied per frame.
    """
    y0, y1, x0, x1 = patch
    H, W = y1 - y0, x1 - x0
    ay0, ay1, ax0, ax1 = align_patch if align_patch is not None else patch

    full_H, full_W = frames[reference_index].shape[1], frames[reference_index].shape[2]
    py0, py1 = max(0, y0 - pad), min(full_H, y1 + pad)
    px0, px1 = max(0, x0 - pad), min(full_W, x1 + pad)
    oy, ox = y0 - py0, x0 - px0

    ref_crop = frames[reference_index][ref_channel, ay0:ay1, ax0:ax1]

    out = np.empty((len(frames), frames[0].shape[0], H, W), dtype=np.float32)
    shifts: List[Tuple[int, int]] = []
    for i, cube in enumerate(frames):
        if track:
            cur = cube[ref_channel, ay0:ay1, ax0:ax1]
            sy, sx = phase_shift(ref_crop, cur)
            sy, sx = int(round(sy)), int(round(sx))
        else:
            sy, sx = 0, 0
        shifts.append((sy, sx))
        cy, cx = oy - sy, ox - sx
        buf = cube[:, py0:py1, px0:px1]
        out[i] = buf[:, cy:cy + H, cx:cx + W].astype(np.float32)
    return out, shifts
