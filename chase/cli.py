"""Command-line interface.

Backward-compatible with the original ``chase`` flags (``--input/-i``,
``--no-spatial`` …), plus the new patch / FOV / config-file / contrast /
temperature options.  ``--config`` loads a TOML that mirrors every
:class:`chase.config.Config` field — Alex's "set a few flags in a text file and
run once" workflow.
"""

from __future__ import annotations

import argparse
import sys

from .config import Config, load_config
from .pipeline import run_pipeline

BANNER = "CHASE pipeline — modular CHASE/HIS calibration + flare analysis"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chase", description=BANNER)
    p.add_argument("fits_dir", nargs="?", default=None,
                   help="Folder of *HA.fits (+*FE.fits), a single .fits, a .txt URL list, or a URL")
    # Legacy alias kept so old invocations still parse:
    p.add_argument("--input", "-i", dest="input_legacy", default=None,
                   help="(legacy) data source; same as the positional argument")
    p.add_argument("--config", help="TOML config file (mirrors all options below)")
    p.add_argument("--out", "--fig-dir", dest="out_dir", default=None, help="Output directory")

    # FOV / patch
    p.add_argument("--patch", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"),
                   help="Data crop window [y0 y1 x0 x1]")
    p.add_argument("--align-patch", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"),
                   help="Separate (smaller) window used only for shift estimation")
    p.add_argument("--full-fov", action="store_true", help="Process the whole field")
    p.add_argument("--core-wavelength", type=float, default=None, help="Hα core wavelength (Å)")
    p.add_argument("--ref-channel", choices=["fe_last", "ha_cont"], default=None,
                   help="Alignment reference channel")

    # Stage toggles
    p.add_argument("--no-track", action="store_true", help="Disable crop tracking")
    p.add_argument("--no-smooth-shifts", action="store_true",
                   help="Disable outlier rejection on tracking shifts")
    p.add_argument("--derotate", action="store_true",
                   help="Rotate frames to solar north using the header INST_ROT angle")
    p.add_argument("--align-mode", choices=["xcorr", "header", "header+xcorr"], default=None,
                   help="Crop shifts from cross-correlation (default), the header disc centre "
                        "plus solar rotation, or the header plus a correlation residual")
    p.add_argument("--reference", choices=["first", "previous", "running"], default=None,
                   help="Correlation reference scan (default: first)")
    p.add_argument("--reference-window", type=int, default=None,
                   help="Scans in the running-mean reference (default 3)")
    p.add_argument("--drift-control", choices=["none", "header"], default=None,
                   help="Tie chained correlation steps back to the header pointing")
    p.add_argument("--disc-center", choices=["crpix", "limb", "auto"], default=None,
                   help="Disc-centre source for the header modes (default: crpix)")
    p.add_argument("--no-rotation", action="store_true",
                   help="Leave out differential rotation in the header modes")
    p.add_argument("--subpixel", action="store_true",
                   help="Apply fractional crop shifts with a cubic spline")
    p.add_argument("--shift-method", choices=["phase", "windowed"], default=None,
                   help="Correlation estimator (default: phase)")
    p.add_argument("--no-resample", action="store_true", help="Disable wavelength resampling")
    p.add_argument("--no-drift", action="store_true",
                   help="Disable spectral drift correction in the contrast stage")
    p.add_argument("--no-optical-flow", action="store_true", help="Disable optical-flow stabilisation")
    p.add_argument("--flow-method", choices=["tvl1", "farneback"], default=None)
    p.add_argument("--flow-reference", choices=["default", "stable", "masked", "stable+masked"],
                   default=None, help="Image the optical flow is computed on (default: Fe I or "
                                      "Hα last wavelength)")
    p.add_argument("--flow-channels", type=int, default=None,
                   help="Channels averaged by --flow-reference stable (default 1)")
    p.add_argument("--flare-threshold", type=float, default=None,
                   help="Relative change masked by --flow-reference masked (default 0.1)")

    # Resume / step selection
    p.add_argument("--resume", action="store_true",
                   help="Reuse out_dir/aligned_data.npz from a previous run (skips load + align)")
    p.add_argument("--only", metavar="STEP[,STEP…]", default=None,
                   help="Run only the listed output steps, reusing the checkpoint when present "
                        "(implies --resume). Steps: gif, contrast, temperature, npz, fits, "
                        "diagnostics. E.g.: --only gif  or  --only contrast,gif")

    # Analysis
    p.add_argument("--contrast", action="store_true", help="Compute the contrast profile")
    p.add_argument("--temperature", choices=["none", "halpha", "fe_planck", "fe_eb", "fe_voigt", "double"],
                   default=None, help="Temperature-map method")

    # Output
    p.add_argument("--gif", dest="gif", action="store_true", default=None)
    p.add_argument("--no-gif", dest="gif", action="store_false")
    p.add_argument("--freeze-clim", action="store_true", help="Fix animation colour limits across frames")
    p.add_argument("--no-npz", action="store_true", help="Do not save aligned_data.npz")
    p.add_argument("--save-fits", action="store_true", help="Save aligned FITS cubes")
    p.add_argument("--no-diagnostics", action="store_true", help="Skip diagnostic PNGs")

    # Contrast labels
    p.add_argument("--flarestart", type=int, default=None)
    p.add_argument("--flarepeak", type=int, default=None)
    p.add_argument("--flareclass", default=None)

    # Accepted-but-ignored legacy flags (kept so old scripts don't crash)
    for legacy in ("--no-spatial", "--no-subpixel", "--no-spectral", "--no-intensity",
                   "--no-fov", "--save"):
        p.add_argument(legacy, action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--threshold", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--accuracy", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--qsbox", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--fov-width", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--fov-height", type=int, default=None, help=argparse.SUPPRESS)

    p.add_argument("--tui", action="store_true", help="Launch the interactive TUI instead")
    return p


def config_from_args(args) -> Config:
    """Build a Config from CLI args, layered over an optional --config file."""
    cfg = load_config(args.config) if args.config else Config()

    source = args.fits_dir or args.input_legacy
    if source is not None:
        cfg.fits_dir = source
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.patch is not None:
        cfg.patch = list(args.patch)
    if args.align_patch is not None:
        cfg.align_patch = list(args.align_patch)
    if args.full_fov:
        cfg.full_fov = True
    if args.core_wavelength is not None:
        cfg.core_wavelength = args.core_wavelength
    if args.ref_channel is not None:
        cfg.ref_channel = args.ref_channel

    if args.no_track:
        cfg.track = False
    if args.no_smooth_shifts:
        cfg.smooth_shifts = False
    if args.derotate:
        cfg.derotate = True
    for name in ("align_mode", "reference", "reference_window", "drift_control", "disc_center",
                 "shift_method", "flow_reference", "flow_channels", "flare_threshold"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    if args.no_rotation:
        cfg.rotation = False
    if args.subpixel:
        cfg.subpixel = True
    if args.no_resample:
        cfg.resample_wavelength = False
    if args.no_optical_flow:
        cfg.optical_flow = False
    if args.flow_method is not None:
        cfg.flow_method = args.flow_method

    if args.contrast:
        cfg.contrast = True
    if args.temperature is not None:
        cfg.temperature = args.temperature

    if args.gif is not None:
        cfg.gif = args.gif
    if args.freeze_clim:
        cfg.freeze_clim = True
    if args.no_npz:
        cfg.save_npz = False
    if args.save_fits:
        cfg.save_fits = True
    if args.no_diagnostics:
        cfg.diagnostics = False

    if args.flarestart is not None:
        cfg.flarestart = args.flarestart
    if args.flarepeak is not None:
        cfg.flarepeak = args.flarepeak
    if args.flareclass is not None:
        cfg.flareclass = args.flareclass

    if args.no_drift:
        cfg.correct_drift = False
    if args.resume:
        cfg.resume = True
    if args.only is not None:
        _apply_only(cfg, args.only)
    return cfg


_ONLY_STEPS = ("gif", "contrast", "temperature", "npz", "fits", "diagnostics")


def _apply_only(cfg: Config, only: str) -> None:
    """Turn every output step off except the ones listed in ``--only``."""
    steps = {s.strip() for s in only.split(",") if s.strip()}
    unknown = steps - set(_ONLY_STEPS)
    if unknown:
        raise SystemExit(
            f"--only: unknown step(s) {', '.join(sorted(unknown))}; "
            f"choose from: {', '.join(_ONLY_STEPS)}"
        )
    cfg.gif = "gif" in steps
    cfg.contrast = "contrast" in steps
    cfg.save_npz = "npz" in steps
    cfg.save_fits = "fits" in steps
    cfg.diagnostics = "diagnostics" in steps
    if "temperature" in steps:
        if cfg.temperature == "none":
            raise SystemExit(
                "--only temperature needs a method: add --temperature "
                "halpha|fe_planck|fe_eb|fe_voigt|double"
            )
    else:
        cfg.temperature = "none"
    cfg.resume = True


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.tui:
        from .tui import run as run_tui

        return run_tui()

    cfg = config_from_args(args)
    if cfg.fits_dir in (None, "."):
        print(BANNER)
        print("\nNo data source given. Examples:")
        print("  chase /path/to/fits --patch 940 1080 1870 2040 --contrast --temperature double")
        print("  chase --config examples/config.toml")
        print("  chase-tui            # interactive")
        return 1

    print(BANNER)
    results = run_pipeline(cfg)
    print("\nOutputs:")
    for key in ("npz", "gif", "contrast_gif", "temperature_gif"):
        if key in results:
            print(f"  {key:16s} {results[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
