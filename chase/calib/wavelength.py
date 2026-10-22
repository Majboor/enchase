"""Wavelength calibration: resampling and spectral-drift correction.

Each CHASE frame carries a slightly different wavelength zero-point (``CRVAL3``
drifts by ~0.04 Å, ~0.8 channels) and dispersion (``CDELT3``).  Comparing the
same channel index across frames on the steep Hα wings manufactures fake
contrast, so every frame must be resampled onto a common grid *before* any
spectral arithmetic.

Two complementary tools:

* :func:`resample_wavelength` — header-accurate cubic resampling of a cube from
  its own grid onto a target grid (the physically correct correction; use it
  when you have per-frame ``CRVAL3``/``CDELT3``).
* :func:`correct_spectral_drift` / :func:`align_spectrum` — data-driven
  cross-correlation of the mean profile against a reference, for when headers are
  missing or unreliable.  ``align_spectrum`` is ported from Davis's ``satprocess``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "resample_wavelength",
    "correct_spectral_drift",
    "cross_correlation_shift",
    "align_spectrum",
]


def resample_wavelength(
    cube: np.ndarray,
    src_wavelengths: np.ndarray,
    dst_wavelengths: np.ndarray,
    kind: str = "cubic",
) -> np.ndarray:
    """Resample a spectral cube from one wavelength grid onto another.

    Parameters
    ----------
    cube : ndarray (nchannels, H, W)
    src_wavelengths : ndarray (nchannels,)
        This cube's own wavelength axis.
    dst_wavelengths : ndarray (nchannels_out,)
        Target grid (typically frame 0's grid).
    kind : str, optional
        ``scipy.interpolate.interp1d`` interpolation kind (default ``cubic``).

    Returns
    -------
    ndarray (nchannels_out, H, W)
    """
    from scipy.interpolate import interp1d

    # rtol must be 0: numpy's default rtol=1e-5 is ~0.066 Å at 6560 Å, larger
    # than the real CHASE zero-point drift (~0.04 Å), which silently skipped the
    # resampling for every real frame in v2.0–2.1.0.
    if np.allclose(src_wavelengths, dst_wavelengths, rtol=0.0, atol=1e-6):
        return cube.astype(np.float32, copy=True)

    nch, H, W = cube.shape
    out = np.empty((len(dst_wavelengths), H, W), dtype=np.float32)
    for row in range(H):
        f = interp1d(
            src_wavelengths, cube[:, row, :], axis=0,
            kind=kind, fill_value="extrapolate", bounds_error=False,
        )
        out[:, row, :] = f(dst_wavelengths)
    return out


def cross_correlation_shift(ref_spectrum: np.ndarray, target_spectrum: np.ndarray) -> int:
    """Integer spectral lag aligning ``target`` to ``ref`` (z-normalised xcorr).

    Ported from ``satprocess.calculate_shift`` (BSD-3, Finlay Davis).
    """
    from scipy.signal import correlate

    def _znorm(x):
        x = np.asarray(x, dtype=float)
        s = x.std()
        return (x - x.mean()) / s if s > 0 else x - x.mean()

    a, b = _znorm(ref_spectrum), _znorm(target_spectrum)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    corr = correlate(a, b, mode="full")
    return int(np.argmax(corr) - (n - 1))


def align_spectrum(
    ref_w: np.ndarray, ref_i: np.ndarray,
    target_w: np.ndarray, target_i: np.ndarray,
) -> np.ndarray:
    """Roll + interpolate ``target`` spectrum onto the reference grid.

    Ported from ``satprocess.align_spectrum`` (BSD-3, Finlay Davis): integer
    cross-correlation roll (zero-filling the wrapped end) followed by linear
    interpolation onto ``ref_w``.
    """
    from scipy.interpolate import interp1d

    shift = cross_correlation_shift(ref_i, target_i)
    rolled = np.roll(target_i, shift)
    if shift > 0:
        rolled[:shift] = 0
    elif shift < 0:
        rolled[shift:] = 0
    f = interp1d(target_w, rolled, kind="linear", fill_value="extrapolate", bounds_error=False)
    return f(ref_w)


def correct_spectral_drift(ha_cubes: np.ndarray, verbose: bool = False):
    """Remove per-frame wavelength drift by cross-correlating mean profiles.

    Each frame's spatially-averaged spectrum is cross-correlated against frame
    0's, and the whole cube is shifted along the spectral axis by the sub-pixel
    offset (applied with ``scipy.ndimage.shift`` per spatial column).

    Parameters
    ----------
    ha_cubes : ndarray (nframes, nchannels, H, W)
    verbose : bool, optional

    Returns
    -------
    corrected : ndarray, same shape
    shifts : ndarray (nframes,) of sub-pixel spectral shifts applied
    """
    from scipy.ndimage import shift as ndshift
    from skimage.registration import phase_cross_correlation

    nframes, nch, H, W = ha_cubes.shape
    mean_spec = np.mean(ha_cubes, axis=(2, 3))
    ref_2d = mean_spec[0][np.newaxis, :]

    shifts = np.zeros(nframes)
    corrected = np.empty_like(ha_cubes)
    for i in range(nframes):
        if i == 0:
            corrected[i] = ha_cubes[i]
            continue
        cur_2d = mean_spec[i][np.newaxis, :]
        shift_yx, _, _ = phase_cross_correlation(ref_2d, cur_2d, upsample_factor=100)
        spec_shift = float(shift_yx[1])
        shifts[i] = spec_shift
        for y in range(H):
            ndshift(ha_cubes[i, :, y, :], [spec_shift, 0],
                    output=corrected[i, :, y, :], mode="nearest")
        if verbose:
            print(f"  frame {i:2d}: spectral shift = {spec_shift:+.3f} channels")
    return corrected, shifts
