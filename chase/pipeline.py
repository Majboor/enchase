"""End-to-end orchestrator that wires the modular stages together.

``run_pipeline(config)`` is the one call the CLI, the config-file workflow, and
the TUI all funnel into — it contains no analysis logic of its own, only the
glue.  A ``progress`` callback receives ``(stage, message)`` updates so a UI can
show live progress; pass ``None`` for plain stdout.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

from . import _compat  # noqa: F401
from .analysis.contrast import contrast_profile
from .calib.align import optical_flow_align
from .config import Config
from .io.fits_io import extract_disk_center, save_aligned_fits, save_cube_fits, save_npz
from .io.loader import load_flare_sequence

__all__ = ["run_pipeline"]

ProgressFn = Callable[[str, str], None]


def _noop(stage: str, message: str) -> None:
    print(f"[{stage}] {message}")


def run_pipeline(config: Config, progress: Optional[ProgressFn] = None) -> dict:
    """Run the configured stages and return a dict of output products + paths.

    Parameters
    ----------
    config : Config
        Fully-specified options (see :class:`chase.config.Config`).
    progress : callable or None, optional
        ``progress(stage, message)`` for live UI updates.

    Returns
    -------
    dict
        Keys include ``out_dir`` and any of ``npz``, ``gif``, ``contrast_gif``,
        ``temperature_gif``, ``diagnostics`` (list), plus the in-memory
        ``sequence`` and ``aligned`` arrays for further SDK use.
    """
    p = progress or _noop
    os.makedirs(config.out_dir, exist_ok=True)
    results: dict = {"out_dir": config.out_dir}
    npz_path = os.path.join(config.out_dir, "aligned_data.npz")

    # --- 1+2. Load + align (or resume from a previous run's checkpoint) ------
    resumed = False
    if config.resume:
        checkpoint = _load_checkpoint(npz_path)
        if checkpoint is not None:
            seq, ha_al, fe_al = checkpoint
            resumed = True
            p("load", f"resuming from {npz_path} — load + align skipped")
        else:
            p("load", "no usable checkpoint in out_dir; running load + align from scratch")

    if not resumed:
        # --- 1. Load + crop-track --------------------------------------------
        p("load", "reading FITS and tracking the patch (shift-then-crop)…")
        seq = load_flare_sequence(
            config.fits_dir,
            patch=config.effective_patch(),
            align_patch=config.align_patch,
            core_wavelength=config.core_wavelength,
            track=config.track,
            smooth_shifts=config.smooth_shifts,
            derotate=config.derotate,
            resample=config.resample_wavelength,
            verbose=True,
            align_mode=config.align_mode,
            reference=config.reference,
            reference_window=config.reference_window,
            drift_control=config.drift_control,
            disc_center=config.disc_center,
            rotation=config.rotation,
            subpixel=config.subpixel,
            shift_method=config.shift_method,
        )
    results["sequence"] = seq
    has_fe = seq["fe_cubes"] is not None

    ha = seq["ha_cubes"]
    fe = seq["fe_cubes"]
    if not resumed:
        # --- 2. Optical-flow stabilisation -----------------------------------
        if config.optical_flow:
            p("align", f"optical-flow stabilisation ({config.flow_method}) on flare-free channel…")
            ref, mask = _flow_reference(config, seq, p)
            if has_fe:
                ha_al, fe_al = optical_flow_align(
                    ha, reference=ref, method=config.flow_method, extra_cubes=fe, mask=mask
                )
            else:
                ha_al = optical_flow_align(ha, reference=ref, method=config.flow_method, mask=mask)
                fe_al = None
        else:
            ha_al, fe_al = ha, fe

    cont_al = ha_al[:, -1, :, :]
    core_al = ha_al[:, seq["core_idx"], :, :]
    results["aligned"] = {"ha": ha_al, "fe": fe_al, "cont": cont_al, "core": core_al}

    # --- 3. Save cubes -------------------------------------------------------
    if config.save_npz and resumed:
        results["npz"] = npz_path  # checkpoint already on disk; don't rewrite
    elif config.save_npz:
        p("save", "writing aligned_data.npz…")
        save_dict = {
            "ha_cubes_al": ha_al,
            "cont_al": cont_al,
            "core_al": core_al,
            "ha_cubes_raw": ha,
            "wavelength_ha": seq["wavelength_ha"],
            "times": np.array(seq["times"], dtype=str),
            "patch": seq["patch"],
            "core_idx": seq["core_idx"],
        }
        if has_fe:
            save_dict["fe_cubes_al"] = fe_al
            save_dict["fe_cubes_raw"] = fe
            save_dict["wavelength_fe"] = seq["wavelength_fe"]
        if seq["ha_bg"] is not None:
            save_dict["ha_bg"] = seq["ha_bg"]
        results["npz"] = save_npz(npz_path, **save_dict)

    if config.save_fits:
        p("save", "writing aligned FITS cubes with updated headers…")
        results["fits"] = save_aligned_fits(
            os.path.join(config.out_dir, "aligned_fits"), seq, ha_al)

    # --- 4. Hero animation ---------------------------------------------------
    if config.gif:
        p("animate", "rendering 2-panel Photosphere|Chromosphere GIF…")
        from .viz.animate import make_animation

        results["gif"] = make_animation(
            cont_al, core_al, times=seq["times"], patch=seq["patch"],
            output_path=os.path.join(config.out_dir, "quicklook.gif"),
            freeze_clim=config.freeze_clim,
        )

    # --- 5. Contrast profile -------------------------------------------------
    if config.contrast:
        p("contrast", "computing (flare-bg)/bg - frame0 contrast profile…")
        contrast, _ = contrast_profile(
            ha_al, background=seq["ha_bg"], baseline_frame=0,
            correct_drift=config.correct_drift,
        )
        results["contrast"] = contrast
        contrast_npy = os.path.join(config.out_dir, "contrast_profile.npy")
        np.save(contrast_npy, contrast)
        results["contrast_npy"] = contrast_npy
        if config.gif:
            from .viz.animate import make_contrast_animation

            results["contrast_gif"] = make_contrast_animation(
                ha_al, seq["wavelength_ha"], contrast, seq["core_idx"],
                times=seq["times"], patch=seq["patch"],
                output_path=os.path.join(config.out_dir, "contrast_profile.gif"),
                flarestart=config.flarestart, flarepeak=config.flarepeak,
                flareclass=config.flareclass,
            )

    # --- 6. Temperature maps -------------------------------------------------
    if config.temperature != "none":
        p("temperature", f"building temperature maps ({config.temperature})…")
        results.update(_run_temperature(config, seq, ha_al, fe_al, has_fe, p))

    # --- 7. Diagnostics ------------------------------------------------------
    if config.diagnostics:
        p("diagnostics", "writing diagnostic PNGs…")
        results["diagnostics"] = _run_diagnostics(config, seq, ha_al)

    p("done", f"outputs written to {config.out_dir}")
    return results


def _flow_reference(config, seq, p):
    """Flow reference image and optional flare mask for ``config.flow_reference``."""
    from .calib.align import flare_mask, select_flow_reference

    mode = config.flow_reference
    if mode not in ("default", "stable", "masked", "stable+masked"):
        raise ValueError(f"flow_reference must be 'default', 'stable', 'masked' or "
                         f"'stable+masked', got {mode!r}")
    ref = seq["align_ref"]
    if mode.startswith("stable"):
        if seq["fe_cubes"] is not None:
            sel = select_flow_reference(seq["fe_cubes"], seq["wavelength_fe"],
                                        core_wavelength=6569.214, min_offset=0.3,
                                        n_channels=config.flow_channels)
        else:
            sel = select_flow_reference(seq["ha_cubes"], seq["wavelength_ha"],
                                        core_wavelength=config.core_wavelength,
                                        n_channels=config.flow_channels)
        ref = sel["reference"]
        p("align", f"flow reference: channel(s) {sel['channels']} at "
                   f"{[round(w, 2) for w in sel['wavelengths']]} Å (score "
                   f"{min(sel['scores'].values()):.3f}; last channel "
                   f"{sel['score_last_channel']})")
    mask = None
    if mode.endswith("masked"):
        mask = flare_mask(ref, threshold=config.flare_threshold)
        p("align", f"flare mask: up to {100 * mask.mean(axis=(1, 2)).max():.1f} % of pixels ignored")
    return ref, mask


def _run_temperature(config, seq, ha_al, fe_al, has_fe, p) -> dict:
    from .analysis.temperature import (
        fe_eb_temperature,
        fe_planck_temperature,
        fe_voigt_temperature,
        halpha_width_temperature,
    )
    from .viz.animate import make_temperature_animation

    out: dict = {}
    wav_ha = seq["wavelength_ha"]
    core_wav = wav_ha[seq["core_idx"]]
    nframes = ha_al.shape[0]

    want_ha = config.temperature in ("halpha", "double")
    want_fe = config.temperature in ("fe_planck", "fe_eb", "fe_voigt", "double") and has_fe

    # Atlas calibration (optional ISPy) for absolute Fe I temperatures.
    ifact = None
    if want_fe:
        try:
            from .calib.intensity import atlas_calibrate

            ha0, fe0 = _find_first_files(config.fits_dir)
            if fe0 is not None:
                dc_cube, dc_wav, _ = extract_disk_center(fe0)
                ifact, woff = atlas_calibrate(dc_wav, dc_cube)
                p("temperature", f"atlas calibration: ifact={ifact:.3e}, woff={woff:.4f} Å")
        except Exception as e:  # ISPy missing or atlas fetch failed
            p("temperature", f"atlas calibration unavailable ({e}); Fe I T uses ifact=1")
            ifact = 1.0

    ha_temps, fe_temps, core_imgs = [], [], []
    for i in range(nframes):
        if want_ha:
            tmap, _ = halpha_width_temperature(ha_al[i], wav_ha, core_wavelength=core_wav)
            ha_temps.append(tmap)
        if want_fe:
            method = "fe_planck" if config.temperature == "double" else config.temperature
            if method == "fe_eb":
                _, tcore, _ = fe_eb_temperature(fe_al[i], seq["wavelength_fe"], ifact)
                fe_temps.append(tcore)
            elif method == "fe_voigt":
                tmap, _, _ = fe_voigt_temperature(fe_al[i], seq["wavelength_fe"], ifact)
                fe_temps.append(tmap)
            else:
                tmap, _ = fe_planck_temperature(fe_al[i], seq["wavelength_fe"], ifact)
                fe_temps.append(tmap)
        core_imgs.append(ha_al[i, seq["core_idx"]])

    ha_arr = np.array(ha_temps) if ha_temps else None
    fe_arr = np.array(fe_temps) if fe_temps else None
    if ha_arr is not None:
        np.save(os.path.join(config.out_dir, "ha_temp_maps.npy"), ha_arr)
        out["ha_temp"] = ha_arr
    if fe_arr is not None:
        np.save(os.path.join(config.out_dir, "fe_temp_maps.npy"), fe_arr)
        out["fe_temp"] = fe_arr

    if config.gif and (ha_arr is not None):
        out["temperature_gif"] = make_temperature_animation(
            ha_arr, fe_arr, np.array(core_imgs), times=seq["times"], patch=seq["patch"],
            output_path=os.path.join(config.out_dir, "temperature.gif"),
        )
    return out


def _run_diagnostics(config, seq, _diag_after=None) -> list:
    from .viz.diagnostics import plot_patch_overlay, plot_qs_spectrum, plot_shift_track

    paths = []
    d = config.out_dir
    frame0 = seq["cont_raw"][0]
    if seq["patch"].size == 4:
        paths.append(plot_patch_overlay(frame0, list(seq["patch"]),
                                        os.path.join(d, "diag_patch_overlay.png"),
                                        align_patch=config.align_patch))
    if config.optical_flow and _diag_after is not None:
        from .viz.diagnostics import plot_alignment_difference

        mid = seq["ha_cubes"].shape[0] // 2
        paths.append(plot_alignment_difference(
            seq["ha_cubes"][:, -1], _diag_after[:, -1], mid,
            os.path.join(d, "diag_alignment_difference.png"),
            patch=list(seq["patch"]) if seq["patch"].size == 4 else None))
    qs = np.mean(seq["ha_cubes"][0], axis=(1, 2))
    paths.append(plot_qs_spectrum(qs, seq["wavelength_ha"],
                                  os.path.join(d, "diag_qs_spectrum.png"),
                                  core_wavelength=seq["wavelength_ha"][seq["core_idx"]]))
    return paths


def _load_checkpoint(npz_path: str):
    """Rebuild ``(seq, ha_al, fe_al)`` from a previous run's ``aligned_data.npz``.

    Returns ``None`` when the file is missing or lacks the raw cubes needed to
    reconstruct the sequence dict (checkpoints from very old runs).
    """
    if not os.path.exists(npz_path):
        return None
    z = np.load(npz_path, allow_pickle=False)
    required = {"ha_cubes_al", "ha_cubes_raw", "wavelength_ha", "times", "patch", "core_idx"}
    if not required.issubset(set(z.files)):
        return None

    ha = z["ha_cubes_raw"]
    has_fe = "fe_cubes_al" in z.files
    core_idx = int(z["core_idx"])
    fe = z["fe_cubes_raw"] if has_fe else None
    seq = {
        "ha_cubes": ha,
        "fe_cubes": fe,
        "cont_raw": ha[:, -1],
        "core_raw": ha[:, core_idx],
        "align_ref": fe[:, -1] if has_fe else ha[:, -1],
        "ha_bg": z["ha_bg"] if "ha_bg" in z.files else None,
        "wavelength_ha": z["wavelength_ha"],
        "wavelength_fe": z["wavelength_fe"] if has_fe else None,
        "times": [str(t) for t in z["times"]],
        "nframes": ha.shape[0],
        "H": ha.shape[2],
        "W": ha.shape[3],
        "nchannels_ha": ha.shape[1],
        "nchannels_fe": fe.shape[1] if has_fe else 0,
        "core_idx": core_idx,
        "patch": z["patch"],
    }
    return seq, z["ha_cubes_al"], (z["fe_cubes_al"] if has_fe else None)


def _find_first_files(fits_dir):
    from .io.download import discover_fits

    if os.path.isdir(fits_dir):
        ha, fe = discover_fits(fits_dir)
        return (ha[0] if ha else None), (fe[0] if fe else None)
    return fits_dir, None
