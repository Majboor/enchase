"""Calibration stages: spatial, alignment, wavelength, intensity.

Every function here is usable on its own — nothing forces a full pipeline.
"""

from .spatial import (
    derotate,
    derotate_crop,
    find_disk_center,
    hough_disk_center,
    phase_shift,
    recenter,
    track_and_crop,
)
from .align import create_optical_flow_solver, optical_flow_align, warp_cube
from .wavelength import (
    align_spectrum,
    correct_spectral_drift,
    cross_correlation_shift,
    resample_wavelength,
)
from .intensity import (
    activity_mask,
    atlas_calibrate,
    atlas_factor_series,
    atlas_sensitivity,
    atlas_fit_quality,
    find_quiet_patch,
    integral_scaling,
    normalize_intensity,
    planck_inversion,
    quiet_sun_mask,
    wing_channels,
)

__all__ = [
    "derotate",
    "derotate_crop",
    # spatial
    "find_disk_center",
    "hough_disk_center",
    "phase_shift",
    "recenter",
    "track_and_crop",
    # align
    "create_optical_flow_solver",
    "optical_flow_align",
    "warp_cube",
    # wavelength
    "align_spectrum",
    "correct_spectral_drift",
    "cross_correlation_shift",
    "resample_wavelength",
    # intensity
    "activity_mask",
    "atlas_calibrate",
    "atlas_factor_series",
    "atlas_fit_quality",
    "atlas_sensitivity",
    "find_quiet_patch",
    "quiet_sun_mask",
    "wing_channels",
    "integral_scaling",
    "normalize_intensity",
    "planck_inversion",
]
