"""chase — a modular CHASE/HIS calibration + flare-analysis pipeline.

Nothing here is a forced pipeline: every stage is an importable function you can
call on its own, *and* there is a one-call orchestrator (:func:`run_pipeline`), a
config-file workflow (:class:`Config`), a CLI (``chase``), and a TUI
(``chase-tui``) layered on top of the exact same functions.

Quick start (SDK)
-----------------
>>> import chase
>>> cube, hdr, wav = chase.load_cube("RSM..._HA.fits")
>>> seq = chase.load_flare_sequence("/path/to/fits", patch=[940, 1080, 1870, 2040])
>>> aligned = chase.optical_flow_align(seq["ha_cubes"], reference=seq["align_ref"])
>>> contrast, _ = chase.contrast_profile(aligned, background=seq["ha_bg"])

Quick start (lazy)
------------------
>>> from chase import Config, run_pipeline
>>> run_pipeline(Config(fits_dir="/path/to/fits", patch=[940, 1080, 1870, 2040],
...                     contrast=True, temperature="double"))
"""

from __future__ import annotations

from . import _compat  # noqa: F401  (lzma stub + headless matplotlib)

__version__ = "2.1.1"

# --- I/O -------------------------------------------------------------------
from .io import (
    discover_fits,
    download_file,
    extract_disk_center,
    load_cube,
    load_fits_data,
    load_flare_sequence,
    resolve_inputs,
    save_cube_fits,
    save_npz,
)
download = download_file  # SDK alias

# --- Calibration -----------------------------------------------------------
from .calib import (
    align_spectrum,
    derotate,
    derotate_crop,
    activity_mask,
    atlas_calibrate,
    atlas_factor_series,
    atlas_fit_quality,
    atlas_sensitivity,
    correct_spectral_drift,
    find_quiet_patch,
    quiet_sun_mask,
    wing_channels,
    create_optical_flow_solver,
    find_disk_center,
    hough_disk_center,
    integral_scaling,
    normalize_intensity,
    optical_flow_align,
    phase_shift,
    planck_inversion,
    recenter,
    resample_wavelength,
    track_and_crop,
    warp_cube,
)

# --- Analysis --------------------------------------------------------------
from .analysis import (
    contrast_profile,
    fe_eb_temperature,
    fe_planck_temperature,
    fe_voigt_temperature,
    frame_rms,
    halpha_width_map,
    line_centroid,
    residual_pointing,
    residual_wavelength,
    stray_light_estimate,
    fit_limb_ellipse,
    offlimb_profile,
    psf_stray_light,
    contrast_bias,
    throughput_trend,
    halpha_width_temperature,
    width_to_temperature,
)

# --- Visualisation ---------------------------------------------------------
from .viz import (
    make_animation,
    make_contrast_animation,
    make_temperature_animation,
)

# --- Orchestration ---------------------------------------------------------
from .config import Config, load_config
from .pipeline import run_pipeline

__all__ = [
    "__version__",
    # io
    "download", "download_file", "resolve_inputs", "discover_fits",
    "load_cube", "load_fits_data", "load_flare_sequence", "extract_disk_center",
    "save_cube_fits", "save_npz",
    # calib
    "derotate", "derotate_crop",
    "find_disk_center", "hough_disk_center", "recenter", "phase_shift",
    "track_and_crop", "optical_flow_align", "warp_cube",
    "create_optical_flow_solver", "resample_wavelength", "correct_spectral_drift",
    "align_spectrum", "normalize_intensity", "integral_scaling",
    "atlas_calibrate", "atlas_fit_quality", "quiet_sun_mask", "find_quiet_patch",
    "activity_mask", "wing_channels", "atlas_sensitivity", "atlas_factor_series",
    "planck_inversion",
    # analysis
    "contrast_profile", "halpha_width_map", "halpha_width_temperature",
    "width_to_temperature", "fe_planck_temperature", "fe_eb_temperature",
    "fe_voigt_temperature",
    # validation
    "residual_pointing", "line_centroid", "residual_wavelength", "frame_rms",
    "stray_light_estimate",
    # systematics
    "fit_limb_ellipse", "offlimb_profile", "psf_stray_light", "contrast_bias",
    "throughput_trend",
    # viz
    "make_animation", "make_contrast_animation", "make_temperature_animation",
    # orchestration
    "Config", "load_config", "run_pipeline",
]
