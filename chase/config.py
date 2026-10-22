"""Configuration: a dataclass that mirrors every pipeline option, loadable from
TOML.

This is the "lazy user" surface Alex asked for — point at a folder, set the FOV
and a handful of true/false flags in a text file, run once.  The same object is
what the CLI builds from flags and what the TUI builds from its forms, so all
three front-ends drive identical behaviour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

__all__ = ["Config", "load_config"]


@dataclass
class Config:
    """All pipeline options in one place.

    Attributes
    ----------
    fits_dir : str
        Folder of ``*HA.fits`` (+ ``*FE.fits``) cubes, a single file, a ``.txt``
        URL list, or a URL.
    out_dir : str
        Where outputs are written.
    patch : list[int] or None
        Data crop ``[y0, y1, x0, x1]``; ``None`` (or ``full_fov=True``) → full FOV.
    align_patch : list[int] or None
        Separate smaller window for shift estimation.
    full_fov : bool
        Ignore ``patch`` and process the whole field.
    core_wavelength : float
        Hα core wavelength in Å.
    track : bool
        Cross-correlation crop tracking (shift-then-crop).
    smooth_shifts : bool
        Robust outlier rejection on the tracking shifts, per scan parity.
    derotate : bool
        Rotate frames to solar north using the header INST_ROT angle before
        cropping; patch coordinates then refer to the north-up frame.
    align_mode : str
        ``"xcorr"`` (default), ``"header"`` or ``"header+xcorr"``: how the
        crop shift is found (see :func:`chase.io.load_flare_sequence`).
        ``"header"`` is the recommended mode for Hα-only data of quiet Sun,
        filaments and the limb.
    reference : str
        Correlation reference: ``"first"`` (default), ``"previous"`` or
        ``"running"``.
    reference_window : int
        Scans in the running-mean reference.
    drift_control : str
        ``"none"`` (default) or ``"header"``: tie chained correlation steps
        back to the header pointing.
    disc_center : str
        ``"crpix"`` (default), ``"limb"`` or ``"auto"``: disc-centre source for
        the header modes.
    rotation : bool
        Add differential rotation of the patch in the header modes.
    subpixel : bool
        Apply fractional shifts with a cubic spline instead of rounding.
    shift_method : str
        ``"phase"`` (default) or ``"windowed"`` correlation estimator.
    resample_wavelength : bool
        Per-frame wavelength resampling onto frame 0's grid.
    optical_flow : bool
        Fine optical-flow stabilisation.
    flow_method : str
        ``"farneback"`` (default, as used in the paper) or ``"tvl1"``. TV-L1
        misregisters the flare peak, where the brightening breaks brightness
        constancy.
    flow_reference : str
        Image the optical flow is computed on: ``"default"`` (Fe I last
        wavelength, else Hα last wavelength), ``"stable"`` (the far-wing
        channel(s) that change least, see
        :func:`chase.calib.select_flow_reference`), ``"masked"`` (default
        image with brightening pixels ignored, see
        :func:`chase.calib.flare_mask`) or ``"stable+masked"``.
    flow_channels : int
        Channels averaged by ``flow_reference="stable"``.
    flare_threshold : float
        Relative change above which ``"masked"`` ignores a pixel.
    ref_channel : str
        Alignment reference: ``"fe_last"`` or ``"ha_cont"``.
    contrast : bool
        Compute the contrast profile.
    correct_drift : bool
        Data-driven spectral drift correction inside the contrast stage
        (cross-correlate each frame's mean spectrum against frame 0).
    temperature : str
        ``"none" | "halpha" | "fe_planck" | "fe_eb" | "fe_voigt" | "double"``.
    resume : bool
        Reuse ``out_dir/aligned_data.npz`` from a previous run: skip the load
        and align stages and rerun only the output/analysis steps.  Falls back
        to a full run when no checkpoint exists.
    gif, freeze_clim, save_npz, save_fits, diagnostics : bool
        Output toggles.
    flarestart, flarepeak : int
        Frame markers drawn on the contrast plot.
    flareclass : str
        Label for the contrast plot (e.g. ``"X1.2"``).
    """

    fits_dir: str = "."
    out_dir: str = "./chase_out"
    patch: Optional[List[int]] = None
    align_patch: Optional[List[int]] = None
    full_fov: bool = False
    core_wavelength: float = 6562.8

    track: bool = True
    smooth_shifts: bool = True
    derotate: bool = False
    align_mode: str = "xcorr"
    reference: str = "first"
    reference_window: int = 3
    drift_control: str = "none"
    disc_center: str = "crpix"
    rotation: bool = True
    subpixel: bool = False
    shift_method: str = "phase"
    resample_wavelength: bool = True
    optical_flow: bool = True
    flow_method: str = "farneback"
    flow_reference: str = "default"
    flow_channels: int = 1
    flare_threshold: float = 0.1
    ref_channel: str = "fe_last"

    contrast: bool = False
    correct_drift: bool = True
    temperature: str = "none"
    resume: bool = False

    gif: bool = True
    freeze_clim: bool = False
    save_npz: bool = True
    save_fits: bool = False
    diagnostics: bool = True

    flarestart: int = 15
    flarepeak: int = 24
    flareclass: str = "X1.2"

    def effective_patch(self) -> Optional[List[int]]:
        """Return the patch to use, honouring ``full_fov``."""
        if self.full_fov:
            return None
        return self.patch

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str) -> Config:
    """Load a :class:`Config` from a TOML file.

    Unknown keys are ignored with a warning so configs survive minor version
    drift.  Uses the stdlib ``tomllib`` (3.11+) or the ``tomli`` backport.
    """
    try:
        import tomllib  # Python 3.11+
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except ModuleNotFoundError:  # pragma: no cover - <3.11
        import tomli
        with open(path, "rb") as f:
            data = tomli.load(f)

    # Allow a [chase] table or a flat document.
    if "chase" in data and isinstance(data["chase"], dict):
        data = data["chase"]

    valid = {f.name for f in Config.__dataclass_fields__.values()}
    clean = {}
    for k, v in data.items():
        if k in valid:
            clean[k] = v
        else:
            print(f"  [config] ignoring unknown key: {k!r}")
    return Config(**clean)
