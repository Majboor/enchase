"""Science analysis: contrast profiles, temperature maps, validation and systematics."""

from .contrast import contrast_profile
from .temperature import (
    fe_eb_temperature,
    fe_planck_temperature,
    fe_voigt_temperature,
    halpha_width_map,
    halpha_width_temperature,
    measure_halpha_width,
    width_to_temperature,
)
from .systematics import (
    contrast_bias,
    decimal_year,
    fit_limb_ellipse,
    limb_coordinates,
    offlimb_profile,
    psf_stray_light,
    throughput_trend,
)
from .validation import (
    frame_rms,
    line_centroid,
    residual_pointing,
    residual_wavelength,
    stray_light_estimate,
)

__all__ = [
    "contrast_profile",
    "halpha_width_map",
    "halpha_width_temperature",
    "measure_halpha_width",
    "width_to_temperature",
    "fe_planck_temperature",
    "fe_eb_temperature",
    "fe_voigt_temperature",
    "residual_pointing",
    "line_centroid",
    "residual_wavelength",
    "frame_rms",
    "stray_light_estimate",
    "fit_limb_ellipse",
    "limb_coordinates",
    "offlimb_profile",
    "psf_stray_light",
    "contrast_bias",
    "decimal_year",
    "throughput_trend",
]
