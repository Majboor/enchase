"""Input/output: data discovery, downloading, and FITS cube loading."""

from .download import discover_fits, download_file, resolve_inputs
from .fits_io import (
    extract_disk_center,
    load_cube,
    load_fits_data,
    save_cube_fits,
    save_npz,
    wavelength_grid,
)
from .loader import alignment_shifts, load_flare_sequence, robust_shifts

__all__ = [
    "discover_fits",
    "download_file",
    "resolve_inputs",
    "extract_disk_center",
    "load_cube",
    "load_fits_data",
    "save_cube_fits",
    "save_npz",
    "wavelength_grid",
    "load_flare_sequence",
    "robust_shifts",
    "alignment_shifts",
]
