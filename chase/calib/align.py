"""Fine alignment via dense optical flow.

Integer crop-tracking (:func:`chase.calib.spatial.track_and_crop`) removes whole
pixel jitter; residual sub-pixel wobble is removed here with a dense optical-flow
warp.

The flow field is computed **once per frame** from a single flare-free reference
channel converted to ``uint8`` (TV-L1 if OpenCV's ``optflow`` module is present,
else Farneback).  The resulting displacement map is then applied with
``cv2.remap`` (cubic) to **every** wavelength channel of the float32 science
cubes — the science data never leaves float precision.

Without Fe I the flow is computed on the Hα far wing, and during a strong flare
that channel brightens too (by up to 77 % at +2.3 Å in the X9.0 flare of
2024-10-03), so the flow partly follows the flare. Two options address this:
:func:`select_flow_reference` picks the wing channel(s) whose intensity changes
least over the series, and :func:`flare_mask` marks the pixels that brighten,
which :func:`optical_flow_align` then ignores (``mask=``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "create_optical_flow_solver",
    "calculate_optical_flow",
    "warp_cube",
    "optical_flow_align",
    "select_flow_reference",
    "flare_mask",
]


def create_optical_flow_solver():
    """Return a TV-L1 optical-flow solver, or ``None`` to signal Farneback."""
    import cv2

    try:
        return cv2.optflow.DualTVL1OpticalFlow_create()
    except AttributeError:
        try:
            return cv2.DualTVL1OpticalFlow_create()
        except AttributeError:
            return None


def calculate_optical_flow(ref_u8: np.ndarray, cur_u8: np.ndarray, solver=None) -> np.ndarray:
    """Compute the dense flow field mapping ``cur`` onto ``ref`` (both uint8)."""
    import cv2

    if solver is not None:
        return solver.calc(ref_u8, cur_u8, None)
    return cv2.calcOpticalFlowFarneback(
        ref_u8, cur_u8, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )


def _to_u8(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _to_u8_masked(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """uint8 version of ``img`` scaled on the unmasked pixels only.

    The min-max range is taken from the pixels outside ``mask``, so a flare
    kernel does not compress the grey scale of the rest of the image (the
    masked pixels are clipped at 255).
    """
    good = ~mask
    if not good.any():
        return _to_u8(img)
    lo, hi = float(np.min(img[good])), float(np.max(img[good]))
    return np.clip((img - lo) / max(hi - lo, 1e-12) * 255.0, 0, 255).astype(np.uint8)


def _fill_flow(flow: np.ndarray, mask: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Replace the flow inside ``mask`` by a smooth interpolation from outside.

    Normalised convolution: ``G * (f w) / G * w`` with ``w = ~mask`` and a
    Gaussian ``G`` of width ``sigma`` px. The flow of the flare kernel itself
    (which follows the brightening, not the pointing) is discarded.
    """
    if not mask.any():
        return flow
    from scipy.ndimage import gaussian_filter

    w = (~mask).astype(np.float32)
    den = gaussian_filter(w, sigma, mode="nearest")
    out = flow.copy()
    for k in range(flow.shape[-1]):
        num = gaussian_filter(flow[..., k] * w, sigma, mode="nearest")
        fill = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
        out[..., k] = np.where(mask, fill, flow[..., k])
    return out


def warp_cube(cube: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Apply a precomputed remap to every channel of a cube (cubic, reflect)."""
    import cv2

    out = np.empty_like(cube)
    for ch in range(cube.shape[0]):
        out[ch] = cv2.remap(
            cube[ch], map_x, map_y,
            interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT,
        )
    return out


def optical_flow_align(
    cubes: np.ndarray,
    reference: Optional[np.ndarray] = None,
    ref_idx: int = 0,
    method: str = "tvl1",
    extra_cubes: Optional[np.ndarray] = None,
    verbose: bool = False,
    mask: Optional[np.ndarray] = None,
):
    """Stabilise a frame sequence with optical-flow warping.

    Parameters
    ----------
    cubes : ndarray (nframes, nchannels, H, W)
        Science cubes to warp (e.g. the Hα stack).  Stays float32.
    reference : ndarray (nframes, H, W) or None, optional
        Per-frame flare-free image used to *compute* the flow.  If None, the last
        channel of ``cubes`` is used.  Pass the Fe I last wavelength here to
        avoid stabilising the eruption away.
    ref_idx : int, optional
        Index of the reference frame everything is aligned to.
    method : {"tvl1", "farneback"}, optional
        Optical-flow algorithm (TV-L1 falls back to Farneback if unavailable).
    extra_cubes : ndarray (nframes, nchannels2, H, W) or None, optional
        A second stack (e.g. Fe I) warped with the *same* per-frame map.
    verbose : bool, optional
    mask : ndarray (nframes, H, W) of bool or None, optional
        Pixels to ignore when computing the flow (e.g. from :func:`flare_mask`).
        For each frame the union of its mask and the reference frame's mask
        is used: the uint8 scaling of both images uses the unmasked pixels
        only, and the flow inside the mask is replaced by a smooth
        interpolation of the flow around it. The warp is still applied to
        every pixel. ``None`` (default) keeps the v2.1 behaviour.

    Returns
    -------
    aligned : ndarray, same shape as ``cubes``
    aligned_extra : ndarray or None
        Present only when ``extra_cubes`` is given.
    """
    import cv2  # noqa: F401  (ensure available early)

    nframes, nch, H, W = cubes.shape
    if reference is None:
        reference = cubes[:, -1, :, :]

    solver = create_optical_flow_solver() if method == "tvl1" else None
    ref_u8 = _to_u8(reference[ref_idx])

    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    aligned = np.empty_like(cubes)
    aligned_extra = np.empty_like(extra_cubes) if extra_cubes is not None else None

    for i in range(nframes):
        if mask is None:
            cur_u8 = _to_u8(reference[i])
            r8 = ref_u8
        else:
            m = np.asarray(mask[i], bool) | np.asarray(mask[ref_idx], bool)
            cur_u8 = _to_u8_masked(reference[i], m)
            r8 = _to_u8_masked(reference[ref_idx], m)
        flow = calculate_optical_flow(r8, cur_u8, solver)
        if mask is not None:
            flow = _fill_flow(flow, m)
        map_x = (X + flow[..., 0]).astype(np.float32)
        map_y = (Y + flow[..., 1]).astype(np.float32)
        aligned[i] = warp_cube(cubes[i], map_x, map_y)
        if aligned_extra is not None:
            aligned_extra[i] = warp_cube(extra_cubes[i], map_x, map_y)
        if verbose:
            print(f"  optical-flow frame {i}/{nframes - 1} aligned")

    if aligned_extra is not None:
        return aligned, aligned_extra
    return aligned


def _smooth(img: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    return uniform_filter(np.asarray(img, dtype=np.float32), size) if size > 1 else np.asarray(img, np.float32)


def _change(stack: np.ndarray, ref_idx: int = 0, smooth: int = 5):
    """Smoothed change ``S(I_t) - S(I_ref)`` of every frame and ``S(I_ref)``."""
    ref = _smooth(stack[ref_idx], smooth)
    return np.stack([_smooth(f, smooth) - ref for f in stack]), ref


def select_flow_reference(
    cubes: np.ndarray,
    wavelengths: np.ndarray,
    core_wavelength: float = 6562.8,
    min_offset: float = 1.0,
    n_channels: int = 1,
    exclude: Sequence[int] = (),
    ref_idx: int = 0,
    smooth: int = 5,
    percentile: float = 99.0,
) -> dict:
    """Pick the flow-reference channel(s) that change least over the series.

    Every channel at least ``min_offset`` Å from the line core is scored by
    how much its (boxcar-smoothed) image changes relative to the reference
    frame, in units of the image's own spatial contrast: the score is the
    largest, over frames, of the ``percentile``-th percentile of
    ``|S(I_t) - S(I_ref)|`` divided by the standard deviation of
    ``S(I_ref)``. A flare kernel that brightens a wing channel raises its
    score; normalising by the contrast stops a featureless channel (which
    changes little but gives the flow nothing to follow) from winning. The
    ``n_channels`` lowest-scoring channels are averaged (each normalised to
    its own mean in the reference frame) to form the flow reference.

    Parameters
    ----------
    cubes : ndarray (nframes, nchannels, H, W)
        Tracked (and wavelength-resampled) cubes, e.g. ``seq["ha_cubes"]``.
    wavelengths : ndarray (nchannels,)
        Common wavelength grid of ``cubes``.
    core_wavelength : float, optional
        Line core in Å (6562.8 for Hα, 6569.214 for Fe I).
    min_offset : float, optional
        Only channels at least this far from the core (Å) are candidates
        (default 1.0, i.e. the far wings).
    n_channels : int, optional
        Number of channels averaged into the reference (default 1).
    exclude : sequence of int, optional
        Channel indices that may not be used (e.g. a channel kept back for
        validation).
    ref_idx : int, optional
        Reference frame of the flow.
    smooth : int, optional
        Boxcar size in pixels before the ratio (default 5).
    percentile : float, optional
        Pixel percentile of the change per frame (default 99).

    Returns
    -------
    dict with ``reference`` (nframes, H, W float32), ``channels`` (chosen
    indices), ``wavelengths`` (their wavelengths), ``scores`` (per candidate
    channel, dict index -> score) and ``score_last_channel`` (the score of the
    v2.1 default, the last channel, for comparison; ``None`` if it was not a
    candidate).
    """
    cubes = np.asarray(cubes)
    wav = np.asarray(wavelengths, dtype=float)
    excl = {int(e) % len(wav) for e in exclude}
    cand = [c for c in range(len(wav))
            if abs(wav[c] - core_wavelength) >= min_offset and c not in excl]
    if not cand:
        raise ValueError("select_flow_reference: no candidate channel; lower min_offset")
    scores = {}
    for c in cand:
        d, ref0 = _change(cubes[:, c], ref_idx, smooth)
        scale = max(float(np.std(ref0)), 1e-12)
        scores[c] = float(max(np.percentile(np.abs(f), percentile) for f in d) / scale)
    best = sorted(cand, key=lambda c: scores[c])[:max(1, int(n_channels))]
    ref = np.mean([cubes[:, c] / max(float(np.mean(cubes[ref_idx, c])), 1e-12) for c in best],
                  axis=0).astype(np.float32)
    last = len(wav) - 1
    return {"reference": ref, "channels": [int(c) for c in best],
            "wavelengths": [float(wav[c]) for c in best], "scores": scores,
            "score_last_channel": scores.get(last)}


def flare_mask(
    reference: np.ndarray,
    threshold: float = 0.1,
    ref_idx: int = 0,
    smooth: int = 5,
    dilate: int = 5,
) -> np.ndarray:
    """Pixels of a flow-reference series that brighten or darken by more than ``threshold``.

    The change ``|S(I_t) - S(I_ref)|`` (boxcar ``S``) is compared with
    ``threshold`` times the median of ``S(I_ref)``, i.e. with a fraction of
    the typical intensity of the field, so dark off-limb pixels with large
    relative noise are not masked.

    Parameters
    ----------
    reference : ndarray (nframes, H, W)
        The flow reference (e.g. ``seq["align_ref"]`` or the output of
        :func:`select_flow_reference`).
    threshold : float, optional
        Change, as a fraction of the median reference intensity, above which
        a pixel is masked (default 0.1). Quiet-Sun far-wing evolution is a few
        percent, the X9.0 flare kernels reach +77 %.
    ref_idx : int, optional
        Reference frame.
    smooth : int, optional
        Boxcar size before the ratio (default 5).
    dilate : int, optional
        Size of the square dilation applied to the mask (default 5 px), so the
        edges of a kernel are masked as well.

    Returns
    -------
    ndarray (nframes, H, W) of bool. Frame ``ref_idx`` is all False.
    """
    from scipy.ndimage import binary_dilation

    ch, ref0 = _change(np.asarray(reference), ref_idx, smooth)
    d = np.abs(ch) > threshold * max(float(np.median(np.abs(ref0))), 1e-12)
    if dilate > 1:
        st = np.ones((dilate, dilate), bool)
        d = np.stack([binary_dilation(m, st) if m.any() else m for m in d])
    d[ref_idx] = False
    return d
