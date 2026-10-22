"""FITS cube reading and product saving.

CHASE RSM cubes store their image data in extension 1 (``hdul[1]``), with the
spectral axis as ``NAXIS3`` and a per-frame wavelength grid encoded by
``CRVAL3``/``CDELT3``.  ``BUNIT`` and ``DATE_OBS`` may be present.

These helpers stay deliberately low-level — a single file in, arrays out — so
they compose into the higher-level sequence loader and the modular SDK.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .. import _compat  # noqa: F401  (installs lzma stub before astropy import)

__all__ = [
    "save_aligned_fits",
    "load_cube",
    "load_fits_data",
    "wavelength_grid",
    "extract_disk_center",
    "save_cube_fits",
    "save_npz",
]


def wavelength_grid(header) -> np.ndarray:
    """Build the wavelength axis (Å) from a FITS header.

    ``wav[k] = CRVAL3 + k * CDELT3`` for ``k`` in ``0 .. NAXIS3-1``.
    """
    n = int(header["NAXIS3"])
    return np.arange(n) * header["CDELT3"] + header["CRVAL3"]


def load_cube(
    filepath: str,
    region: Optional[Tuple[int, int, int, int]] = None,
    dtype=np.float32,
) -> Tuple[np.ndarray, "object", np.ndarray]:
    """Load one FITS spectral cube with its header and wavelength grid.

    Parameters
    ----------
    filepath : str
        Path to a ``*HA.fits`` / ``*FE.fits`` cube.
    region : (y0, y1, x0, x1) or None, optional
        If given, only this spatial window is read (memory-mapped section slice,
        so large full-disk cubes are not pulled fully into RAM).
    dtype : numpy dtype, optional
        Output dtype (default ``float32`` — science data stays float).

    Returns
    -------
    cube : ndarray (nchannels, H, W)
    header : astropy FITS header
    wavelengths : ndarray (nchannels,) in Å

    Examples
    --------
    >>> cube, hdr, wav = load_cube("RSM..._HA.fits")
    >>> cube.shape, wav[0], wav[-1]
    ((118, 2313, 2304), 6559.4, 6565.1)
    """
    import astropy.io.fits as fits

    with fits.open(filepath, memmap=True, do_not_scale_image_data=False) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"{filepath} has no image extension (hdul[1])")
        hdu = hdul[1]
        header = hdu.header
        wav = wavelength_grid(header)
        if region is None:
            cube = np.asarray(hdu.data, dtype=dtype)
        else:
            y0, y1, x0, x1 = region
            cube = np.asarray(hdu.section[:, y0:y1, x0:x1], dtype=dtype)
    return cube, header, wav


def load_fits_data(filepath: str) -> np.ndarray:
    """Backward-compatible loader: return ``hdul[1].data`` only."""
    cube, _, _ = load_cube(filepath, dtype=None if False else np.float32)
    return cube


def extract_disk_center(
    filepath: str,
    size: int = 100,
    center: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Extract the disk-center patch used for absolute calibration.

    Absolute intensity/wavelength calibration is always done at solar disk
    centre (``CRPIX1``/``CRPIX2``), because that location does not change frame
    to frame.  Returns a ``size × size`` cube centred on the reference pixel.

    Parameters
    ----------
    filepath : str
        FITS cube (typically the Fe I file).
    size : int, optional
        Patch side length in pixels (default 100).
    center : (cy, cx) or None, optional
        Patch centre in pixels. ``None`` → the header reference pixel. Pass the
        result of :func:`chase.calib.find_quiet_patch` to avoid active regions.

    Returns
    -------
    cube : ndarray (nchannels, size, size)
    wavelengths : ndarray (nchannels,)
    center : (cy, cx) integer reference-pixel location used.
    """
    import astropy.io.fits as fits

    with fits.open(filepath, memmap=True) as hdul:
        hdu = hdul[1]
        header = hdu.header
        wav = wavelength_grid(header)
        H, W = int(header["NAXIS2"]), int(header["NAXIS1"])
        if center is None:
            cx = int(round(header.get("CRPIX1", W / 2)))
            cy = int(round(header.get("CRPIX2", H / 2)))
        else:
            cy, cx = int(center[0]), int(center[1])
        half = size // 2
        y0, y1 = max(0, cy - half), min(H, cy + half)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        cube = np.asarray(hdu.section[:, y0:y1, x0:x1], dtype=np.float32)
    return cube, wav, (cy, cx)


def save_cube_fits(cube: np.ndarray, path: str, header=None) -> str:
    """Write an aligned/calibrated cube to a FITS file (data in ``hdul[1]``)."""
    import astropy.io.fits as fits

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    primary = fits.PrimaryHDU()
    image = fits.ImageHDU(data=np.asarray(cube, dtype=np.float32), header=header)
    fits.HDUList([primary, image]).writeto(path, overwrite=True)
    return path


def save_aligned_fits(out_dir, seq, ha_al, plate_scale=1.04, prefix="aligned"):
    """Write each aligned scan as a FITS cube with an updated header.

    The header records the crop origin in original detector pixels, a linear
    arcsec WCS from the plate scale, the common wavelength axis, and HISTORY
    entries naming each calibration applied - so downstream tools (and future
    selves) know exactly what the data are.
    """
    import astropy.io.fits as fits

    os.makedirs(out_dir, exist_ok=True)
    patch = np.asarray(seq.get("patch"))
    y0, x0 = (int(patch[0]), int(patch[2])) if patch.size == 4 else (0, 0)
    wav = seq["wavelength_ha"]
    shifts = seq.get("shifts") or [(0, 0)] * ha_al.shape[0]
    paths = []
    for i in range(ha_al.shape[0]):
        hdr = fits.Header()
        hdr["TELESCOP"] = "CHASE-HIS"
        hdr["DATE-OBS"] = str(seq["times"][i])
        hdr["BUNIT"] = ("DN", "detector counts (not radiometrically calibrated)")
        hdr["CTYPE1"], hdr["CTYPE2"], hdr["CTYPE3"] = "SOLAR-X", "SOLAR-Y", "WAVE"
        hdr["CUNIT1"] = hdr["CUNIT2"] = "arcsec"
        hdr["CUNIT3"] = "Angstrom"
        hdr["CDELT1"] = hdr["CDELT2"] = (plate_scale, "arcsec/pixel")
        hdr["CDELT3"] = float(wav[1] - wav[0])
        hdr["CRPIX1"] = hdr["CRPIX2"] = 1.0
        hdr["CRPIX3"] = 1.0
        crpx, crpy = seq.get("crpix", (0.0, 0.0))
        hdr["CRVAL1"] = ((x0 - (crpx - 1.0)) * plate_scale,
                         "helioprojective X of crop origin [arcsec]")
        hdr["CRVAL2"] = ((y0 - (crpy - 1.0)) * plate_scale,
                         "helioprojective Y of crop origin [arcsec]")
        hdr["INSTROT"] = (float(seq.get("inst_rot", 0.0)),
                          "solar north vs detector y-axis [deg]")
        hdr["DEROT"] = (bool(seq.get("derotated", False)),
                        "frames rotated to solar north by enchase")
        hdr["CRVAL3"] = float(wav[0])
        hdr["PATCHY0"], hdr["PATCHX0"] = y0, x0
        hdr["TRKSH_Y"] = (int(shifts[i][0]), "tracking shift applied [px]")
        hdr["TRKSH_X"] = (int(shifts[i][1]), "tracking shift applied [px]")
        hdr["HISTORY"] = "enchase: shift-then-crop tracking (robust per-parity shifts)"
        if seq.get("derotated"):
            hdr["HISTORY"] = "enchase: derotated to solar north (INST_ROT from header)"
        hdr["HISTORY"] = "enchase: wavelength resampled onto scan-0 grid (cubic)"
        hdr["HISTORY"] = "enchase: optical-flow stabilised (flare-free reference)"
        path = os.path.join(out_dir, f"{prefix}_{i:04d}_HA.fits")
        save_cube_fits(ha_al[i], path, header=hdr)
        paths.append(path)
    return paths


def save_npz(path: str, compressed: bool = True, **arrays) -> str:
    """Save named arrays to an ``.npz`` archive (aligned cubes, wavelengths…)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
    return path
