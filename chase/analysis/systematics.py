"""Instrumental systematics: stray light and throughput change.

Neither is corrected by the pipeline; these functions measure them so the size
of the effect can be quoted.

Stray light (from one full-disc image of a continuum or far-wing channel):

* :func:`fit_limb_ellipse` — sub-pixel limb fit (raster images can be slightly
  elliptical, so an axis-aligned ellipse is fitted, not a circle).
* :func:`limb_coordinates` — normalised radius ``rho`` (1 at the limb) and
  position angle for every pixel.
* :func:`offlimb_profile` — azimuthal median of the off-limb intensity relative
  to disc centre, rejecting sectors with prominences or active-limb emission.
* :func:`psf_stray_light` — fits a PSF-wing (halo) model to that profile and
  convolves the observed disc with it, giving the scattered fraction at disc
  centre.
* :func:`contrast_bias` — the contrast a small dark feature appears to have
  for a given scattered fraction.

Throughput:

* :func:`decimal_year` and :func:`throughput_trend` — exponential fit of the
  atlas factor ``ifact`` against observing date (see
  :func:`chase.calib.intensity.atlas_factor_series`). ``ifact`` converts DN to
  radiance, so a rising ``ifact`` at fixed exposure is a falling throughput.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Dict, Optional, Sequence, Union

import numpy as np

__all__ = [
    "fit_limb_ellipse",
    "limb_coordinates",
    "offlimb_profile",
    "psf_stray_light",
    "contrast_bias",
    "decimal_year",
    "throughput_trend",
]


# --------------------------------------------------------------------- limb
def fit_limb_ellipse(
    image: np.ndarray,
    threshold: float = 0.2,
    nang: int = 720,
    center: Optional[Sequence[float]] = None,
    clip: float = 4.0,
    niter: int = 5,
) -> dict:
    """Sub-pixel limb fit with an axis-aligned ellipse.

    Rays are cast from the disc centroid; on each ray the limb is the
    (linearly interpolated) outermost point where the intensity crosses
    ``threshold`` × the disc-centre level. An axis-aligned ellipse
    ``a x² + b y² + c x + d y = 1`` is fitted to these points by linear least
    squares, with ``niter`` rounds of ``clip``-σ (MAD) rejection, which removes
    rays that cross prominences or detector edges.

    Parameters
    ----------
    image : ndarray (H, W)
        Full-disc image of a continuum or far-wing channel.
    threshold : float, optional
        Limb level as a fraction of the disc-centre intensity (default 0.2).
    nang : int, optional
        Number of rays (default 720, i.e. every 0.5°).
    center : (cx, cy) or None, optional
        Start point of the rays; ``None`` → centroid of the bright disc.
    clip : float, optional
        Rejection threshold in robust standard deviations.
    niter : int, optional
        Number of rejection rounds.

    Returns
    -------
    dict with ``x0``, ``y0`` (centre, px), ``ax``, ``ay`` (semi-axes along x
    and y, px), ``radius_px`` (mean of the two), ``n_pts`` (rays kept) and
    ``rms_px`` (residual of the kept points).

    Raises
    ------
    ValueError
        If fewer than 10 limb points are found.
    """
    from scipy.ndimage import map_coordinates

    im = np.asarray(image, dtype=float)
    H, W = im.shape
    big = im > 0.3 * np.nanpercentile(im, 99)
    if center is None:
        yy, xx = np.nonzero(big)
        cy, cx = yy.mean(), xx.mean()
    else:
        cx, cy = center
    r_est = np.sqrt(big.sum() / np.pi)
    h = max(2, int(0.1 * r_est))
    dc = np.nanmedian(im[int(cy) - h:int(cy) + h, int(cx) - h:int(cx) + h])
    thr = threshold * dc
    step = 0.5
    r = np.arange(0.5 * r_est, 1.6 * r_est, step)
    pts = []
    for a in np.linspace(0, 2 * np.pi, nang, endpoint=False):
        x = cx + r * np.cos(a)
        y = cy + r * np.sin(a)
        ok = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
        if ok.sum() < 0.6 * len(r):
            continue
        prof = map_coordinates(im, [y[ok], x[ok]], order=1)
        above = np.nonzero(prof > thr)[0]
        if len(above) == 0 or above[-1] >= len(prof) - 1:
            continue
        j = above[-1]
        f = (prof[j] - thr) / (prof[j] - prof[j + 1])
        rr = r[ok][j] + f * step
        pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
    pts = np.asarray(pts)
    if len(pts) < 10:
        raise ValueError(f"fit_limb_ellipse: only {len(pts)} limb points found")
    keep = np.ones(len(pts), bool)
    for _ in range(niter):
        X, Y = pts[keep, 0], pts[keep, 1]
        A = np.c_[X ** 2, Y ** 2, X, Y]
        (a2, b2, c1, d1), *_ = np.linalg.lstsq(A, np.ones_like(X), rcond=None)
        x0, y0 = -c1 / (2 * a2), -d1 / (2 * b2)
        G = 1 + a2 * x0 ** 2 + b2 * y0 ** 2
        ax, ay = np.sqrt(G / a2), np.sqrt(G / b2)
        res = np.hypot((pts[:, 0] - x0) / ax, (pts[:, 1] - y0) / ay) - 1
        s = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < clip * max(s, 1e-4)
    return {"x0": float(x0), "y0": float(y0), "ax": float(ax), "ay": float(ay),
            "radius_px": float(0.5 * (ax + ay)), "n_pts": int(keep.sum()),
            "rms_px": float(np.std(res[keep]) * 0.5 * (ax + ay))}


def limb_coordinates(shape, limb: dict):
    """Normalised radius and position angle of every pixel.

    Parameters
    ----------
    shape : (H, W)
    limb : dict
        Output of :func:`fit_limb_ellipse` (``x0``, ``y0``, ``ax``, ``ay``).

    Returns
    -------
    rho : ndarray (H, W)
        Elliptical radius, 1 on the limb.
    theta : ndarray (H, W)
        Position angle in radians (−π…π, measured from +x).
    """
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    rho = np.hypot((xx - limb["x0"]) / limb["ax"], (yy - limb["y0"]) / limb["ay"])
    return rho, np.arctan2(yy - limb["y0"], xx - limb["x0"])


def _sector_profiles(im, rho, th, edges, nsec, min_pix=5):
    sec = ((th + np.pi) / (2 * np.pi) * nsec).astype(int) % nsec
    rb = np.digitize(rho, edges) - 1
    nb = len(edges) - 1
    P = np.full((nsec, nb), np.nan)
    sel = (rb >= 0) & (rb < nb)
    key = sec[sel] * nb + rb[sel]
    v = im[sel]
    order = np.argsort(key, kind="stable")
    key, v = key[order], v[order]
    u, start = np.unique(key, return_index=True)
    ends = np.r_[start[1:], len(key)]
    for kk, s0, e0 in zip(u, start, ends):
        if e0 - s0 >= min_pix:
            P.flat[kk] = np.median(v[s0:e0])
    return P


# ------------------------------------------------------------ stray light
def offlimb_profile(
    image: np.ndarray,
    limb: Optional[dict] = None,
    core_image: Optional[np.ndarray] = None,
    edges: Optional[np.ndarray] = None,
    nsec: int = 72,
    reject: float = 1.5,
    radii: Sequence[float] = (1.02, 1.05, 1.10, 1.20),
) -> dict:
    """Azimuthally averaged off-limb intensity relative to disc centre.

    In the photospheric continuum and the Hα far wings the true off-limb
    signal is negligible, so the light measured just outside the limb is
    scattered light (plus any detector offset). The image is divided into
    ``nsec`` position-angle sectors; in each, the median intensity is taken in
    bins of normalised radius. A sector is rejected when its level at
    ``rho`` = 1.02–1.10 exceeds ``reject`` × the median over sectors (active
    limb), or, if ``core_image`` (the Hα line core) is given, when the core
    at ``rho`` = 1.01–1.10 does (prominences, flare loops). The profile is the
    median over the kept sectors.

    Parameters
    ----------
    image : ndarray (H, W)
        Full-disc continuum or far-wing image.
    limb : dict or None, optional
        Output of :func:`fit_limb_ellipse`; ``None`` → fitted here.
    core_image : ndarray (H, W) or None, optional
        Hα line-core image of the same scan, for prominence rejection.
    edges : ndarray or None, optional
        Bin edges in ``rho`` (default 0.90–1.70 in steps of 0.005).
    nsec : int, optional
        Number of position-angle sectors (default 72, i.e. 5° each).
    reject : float, optional
        Sector rejection factor (default 1.5).
    radii : sequence of float, optional
        Radii at which the profile is reported in ``fraction``.

    Returns
    -------
    dict with ``I_dc`` (median at ``rho`` < 0.05), ``r`` (bin centres),
    ``profile``, ``p16``, ``p84`` (fraction of ``I_dc``; spread over sectors),
    ``n_sectors`` (per bin), ``sectors_kept``, ``sectors_rejected``,
    ``fraction`` (``{"1.05": …}``) and ``limb``.
    """
    im = np.asarray(image, dtype=float)
    if limb is None:
        limb = fit_limb_ellipse(im)
    rho, th = limb_coordinates(im.shape, limb)
    dc = float(np.nanmedian(im[rho < 0.05]))
    if edges is None:
        edges = np.round(np.arange(0.90, 1.70001, 0.005), 4)
    rc = 0.5 * (edges[1:] + edges[:-1])
    P = _sector_profiles(im, rho, th, edges, nsec)
    win = (rc > 1.02) & (rc < 1.10)
    sm = np.nanmedian(P[:, win], axis=1)
    bad = sm > reject * np.nanmedian(sm)
    if core_image is not None:
        Pc = _sector_profiles(np.asarray(core_image, dtype=float), rho, th, edges, nsec)
        cs = np.nanmedian(Pc[:, (rc > 1.01) & (rc < 1.10)], axis=1)
        bad |= cs > reject * np.nanmedian(cs)
    K = P[~bad] / dc
    with warnings.catch_warnings():   # all-NaN bins far off the limb
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.nanmedian(K, axis=0)
        p16, p84 = np.nanpercentile(K, [16, 84], axis=0)
    nsec_ok = np.sum(np.isfinite(K), axis=0)
    frac = {}
    for r0 in radii:
        j = int(np.argmin(np.abs(rc - r0)))
        frac[f"{r0:.2f}"] = float(prof[j]) if np.isfinite(prof[j]) else None
    return {"I_dc": dc, "r": rc, "profile": prof, "p16": p16, "p84": p84,
            "n_sectors": nsec_ok, "sectors_kept": int((~bad).sum()),
            "sectors_rejected": int(bad.sum()), "fraction": frac, "limb": limb}


def psf_stray_light(
    image: np.ndarray,
    profile: dict,
    kinds: Sequence[str] = ("moffat", "gauss2", "exp_gauss"),
    rfit: Sequence[float] = (1.03, 1.45),
    binning: int = 8,
    min_level: float = 0.0015,
    floor: float = 0.001,
    min_sectors: int = 5,
) -> dict:
    """Scattered-light fraction at disc centre from a PSF-wing model.

    The observed image is modelled as ``I = (1 − ε) D + Σ ε_k (D ∗ K_k)``,
    where ``D`` is the observed disc (``rho`` < 1, zero outside) and ``K_k``
    are normalised, radially symmetric halo kernels with widths in units of
    the solar radius. The kernel parameters are fitted, on a ``binning``-binned
    grid with zero-padded FFT convolution, to the off-limb profile from
    :func:`offlimb_profile` between ``rfit[0]`` and ``rfit[1]``. The scattered
    light the model puts at disc centre, ``s_dc = Σ ε_k (D ∗ K_k)(0) / I_dc``,
    is the stray-light fraction there.

    Kernel families (``kinds``):

    * ``"moffat"`` — ``(1 + (r/a)²)^−β``; parameters (ε, a, β).
    * ``"gauss2"`` — two Gaussians; (ε₁, σ₁, ε₂, σ₂).
    * ``"exp_gauss"`` — exponential plus Gaussian; (ε₁, λ, ε₂, σ).

    The fit only constrains the halo beyond ``rfit[0]``; a narrow PSF core
    inside that is not measured. The spread over ``kinds`` gives a feel for
    the model dependence, and the result is an order of magnitude, not a
    deconvolution.

    Parameters
    ----------
    image : ndarray (H, W)
        The same image passed to :func:`offlimb_profile`.
    profile : dict
        Output of :func:`offlimb_profile`.
    kinds : sequence of str, optional
        Kernel families to fit.
    rfit : (float, float), optional
        Radius range (``rho``) of the fit.
    binning : int, optional
        Spatial binning for the convolution (default 8 for a full CHASE disc).
    min_level : float, optional
        Bins below this fraction of ``I_dc`` are not fitted (Level-1 data are
        integers with negative values clipped to 0, so a few DN is the floor).
    floor : float, optional
        Added to the observed level (fraction of ``I_dc``) in the relative
        residual, so near-zero bins do not dominate.
    min_sectors : int, optional
        Bins with fewer kept sectors are ignored.

    Returns
    -------
    dict with ``models`` (per kind: ``params``, ``eps_total``,
    ``fit_rms_rel``, ``s_dc``, ``scatter_frac_of_local_I`` at ``rho`` 0, 0.5,
    0.8, 0.9, 0.95, and ``fit_model`` on ``fit_r``), ``best`` (kind with the
    lowest residual), ``s_dc_range`` (min, max over kinds), ``fit_r`` and
    ``fit_obs``.
    """
    from scipy.optimize import least_squares

    im = np.asarray(image, dtype=float)
    limb = profile["limb"]
    dc = float(profile["I_dc"])
    pr = np.isfinite(profile["profile"]) & (np.asarray(profile["n_sectors"]) >= min_sectors)
    prof_r = np.asarray(profile["r"])[pr]
    prof_v = np.asarray(profile["profile"])[pr] * dc
    rho, _ = limb_coordinates(im.shape, limb)

    b = int(binning)
    H, W = (im.shape[0] // b) * b, (im.shape[1] // b) * b
    Ib = im[:H, :W].reshape(H // b, b, W // b, b).mean(axis=(1, 3))
    R = rho[:H, :W].reshape(H // b, b, W // b, b).mean(axis=(1, 3))
    D = np.where(R < 1.0, Ib, 0.0)
    N = int(2 ** np.ceil(np.log2(2 * max(D.shape))))
    oy, ox = (N - D.shape[0]) // 2, (N - D.shape[1]) // 2
    pad = np.zeros((N, N))
    pad[oy:oy + D.shape[0], ox:ox + D.shape[1]] = D
    FD = np.fft.rfft2(pad)
    k = np.fft.fftfreq(N) * N
    rr = np.hypot(k[:, None], k[None, :])
    Rpad = np.full((N, N), np.nan)
    Rpad[oy:oy + D.shape[0], ox:ox + D.shape[1]] = R
    Ipad = np.full((N, N), np.nan)
    Ipad[oy:oy + D.shape[0], ox:ox + D.shape[1]] = Ib

    fe = np.arange(rfit[0], rfit[1] + 1e-9, 0.01)
    rc = 0.5 * (fe[1:] + fe[:-1])
    obs = np.interp(rc, prof_r, prof_v, left=np.nan, right=np.nan)
    good = np.isfinite(obs) & (obs > min_level * dc)
    rbin = np.digitize(np.nan_to_num(Rpad, nan=99.0), fe) - 1
    rb_sel = np.nonzero(((rbin >= 0) & (rbin < len(rc))).ravel())[0]
    rb_flat = rbin.ravel()[rb_sel]
    rb_cnt = np.bincount(rb_flat, minlength=len(rc)).astype(float)
    rsun_b = 0.5 * (limb["ax"] + limb["ay"]) / b
    cyc, cxc = oy + limb["y0"] / b, ox + limb["x0"] / b
    iyc, ixc = int(round(cyc)), int(round(cxc))

    def kern(kind, p):
        if kind == "moffat":
            return [(p[0], (1 + (rr / (p[1] * rsun_b)) ** 2) ** (-p[2]))]
        if kind == "gauss2":
            return [(p[0], np.exp(-0.5 * (rr / (p[1] * rsun_b)) ** 2)),
                    (p[2], np.exp(-0.5 * (rr / (p[3] * rsun_b)) ** 2))]
        if kind == "exp_gauss":
            return [(p[0], np.exp(-rr / (p[1] * rsun_b))),
                    (p[2], np.exp(-0.5 * (rr / (p[3] * rsun_b)) ** 2))]
        raise ValueError(f"unknown kernel kind {kind!r}")

    def model(kind, p):
        S = np.zeros((N, N))
        for e, K in kern(kind, p):
            S += e * np.fft.irfft2(FD * np.fft.rfft2(K / K.sum()), s=(N, N))
        sm = np.bincount(rb_flat, weights=S.ravel()[rb_sel], minlength=len(rc))
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(rb_cnt > 0, sm / rb_cnt, np.nan)
        return m, S

    fl = floor * dc

    def resid(kind, p):
        m, _ = model(kind, p)
        ok = good & np.isfinite(m)
        return (m[ok] - obs[ok]) / (obs[ok] + fl)

    specs = {
        "moffat": ([[0.05, 0.02, 1.2], [0.05, 0.1, 2.0]], ([1e-4, 0.002, 0.55], [0.9, 3.0, 40.0])),
        "gauss2": ([[0.02, 0.03, 0.03, 0.2], [0.03, 0.05, 0.02, 0.4]],
                   ([0, 0.003, 0, 0.05], [0.9, 0.3, 0.9, 2.0])),
        "exp_gauss": ([[0.03, 0.05, 0.02, 0.3], [0.03, 0.15, 0.01, 0.03]],
                      ([0, 0.003, 0, 0.005], [0.9, 2.0, 0.9, 2.0])),
    }
    yy, xx = np.mgrid[:N, :N]
    centre = np.hypot(yy - cyc, xx - cxc) < 2
    models: Dict[str, dict] = {}
    for kind in kinds:
        starts, bnds = specs[kind]
        best = None
        for p0 in starts:
            try:
                r_ = least_squares(lambda p: resid(kind, p), p0, bounds=bnds, max_nfev=300)
            except (ValueError, FloatingPointError):
                continue
            if best is None or r_.cost < best.cost:
                best = r_
        if best is None:
            continue
        m, S = model(kind, best.x)
        on = {}
        for r0 in (0.0, 0.5, 0.8, 0.9, 0.95):
            sel = centre if r0 == 0 else (np.abs(Rpad - r0) < 0.02)
            on[f"{r0:.2f}"] = float(np.nanmedian(S[sel] / Ipad[sel])) if np.any(sel) else None
        models[kind] = {
            "params": best.x.tolist(),
            "eps_total": float(sum(e for e, _ in kern(kind, best.x))),
            "fit_rms_rel": float(np.sqrt(np.mean(best.fun ** 2))) if best.fun.size else None,
            "s_dc": float(np.median(S[iyc - 1:iyc + 2, ixc - 1:ixc + 2]) / dc),
            "scatter_frac_of_local_I": on,
            "fit_model": (m / dc).tolist(),
        }
    s_all = [v["s_dc"] for v in models.values()]
    return {"models": models,
            "best": min(models, key=lambda kk: models[kk]["fit_rms_rel"]) if models else None,
            "s_dc_range": [float(min(s_all)), float(max(s_all))] if s_all else None,
            "fit_r": rc.tolist(), "fit_r_used": rc[good].tolist(), "fit_obs": (obs / dc).tolist()}


def contrast_bias(c_true, s):
    """Observed contrast of a dark feature in the presence of stray light.

    For a feature much smaller than the halo, the scattered light adds about
    ``s`` × the quiet-Sun intensity, so ``I_obs = (1 − s) I_true + s I_QS`` and

    ``C_obs = (1 − s) C_true + s``.

    Parameters
    ----------
    c_true : float or ndarray
        True contrast ``I/I_QS`` (e.g. 0.2 for an umbra in the continuum).
    s : float or ndarray
        Scattered fraction of the quiet-Sun intensity (e.g. ``s_dc`` from
        :func:`psf_stray_light`).

    Returns
    -------
    float or ndarray
    """
    return (1.0 - np.asarray(s)) * np.asarray(c_true) + np.asarray(s)


# ------------------------------------------------------------- throughput
def decimal_year(t: Union[str, datetime, float]) -> float:
    """Decimal year of an ISO date string (e.g. ``DATE_OBS``) or datetime.

    Numbers are returned unchanged.
    """
    if isinstance(t, (int, float, np.floating)):
        return float(t)
    if isinstance(t, str):
        t = datetime.fromisoformat(t.replace("Z", ""))
    y0, y1 = datetime(t.year, 1, 1), datetime(t.year + 1, 1, 1)
    return t.year + (t - y0).total_seconds() / (y1 - y0).total_seconds()


def throughput_trend(times: Sequence, ifact: Sequence[float]) -> dict:
    """Throughput change per year from the atlas factor at several dates.

    ``ifact`` (radiance per DN) is inversely proportional to the throughput
    when exposure time, binning and Level-1 processing are the same. A
    straight line is fitted to ``ln ifact`` against time; the throughput then
    changes by ``exp(−slope) − 1`` per year.

    Parameters
    ----------
    times : sequence of str, datetime or float
        Observing dates (ISO strings such as ``DATE_OBS``, datetimes, or
        decimal years), one per data set.
    ifact : sequence of float
        Mean atlas factor of each data set (same line and threshold).

    Returns
    -------
    dict with ``n_points``, ``t0`` (first date, decimal year),
    ``slope_ln_per_yr``, ``throughput_pct_per_yr``, ``total_change_pct``
    (throughput change between the first and last date from the data), and,
    with more than two points, ``slope_se``, ``throughput_pct_per_yr_se``,
    ``resid_rms_pct`` and ``jackknife_pct_per_yr`` (min, max with one point
    left out).
    """
    t = np.array([decimal_year(x) for x in times], dtype=float)
    a = np.asarray(ifact, dtype=float)
    order = np.argsort(t)
    t, a = t[order], a[order]
    y = np.log(a)
    t0 = float(t.min())
    A = np.c_[np.ones_like(t), t - t0]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    n = len(t)
    pct = lambda b: 100.0 * (np.exp(-b) - 1.0)  # noqa: E731
    out = {"n_points": n, "t0": t0, "slope_ln_per_yr": float(c[1]),
           "throughput_pct_per_yr": float(pct(c[1])),
           "total_change_pct": float(100.0 * (a[0] / a[-1] - 1.0))}
    if n > 2:
        res = y - A @ c
        cov = np.sum(res ** 2) / (n - 2) * np.linalg.inv(A.T @ A)
        se = float(np.sqrt(cov[1, 1]))
        jk = []
        for i in range(n):
            m = np.ones(n, bool)
            m[i] = False
            cj, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
            jk.append(pct(cj[1]))
        out.update({"slope_se": se,
                    "throughput_pct_per_yr_se": float(100.0 * np.exp(-c[1]) * se),
                    "resid_rms_pct": float(100.0 * np.sqrt(np.mean(res ** 2))),
                    "jackknife_pct_per_yr": [float(min(jk)), float(max(jk))]})
    return out
