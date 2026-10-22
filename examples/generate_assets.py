#!/usr/bin/env python3
"""Generate the README/doc assets from precomputed aligned products.

Reuses the real 2023-03-29 X1.2 ``aligned_data.npz`` + ``disk_center.npz`` (so we
don't re-run alignment on 25 x 375 MB cubes) and drives the **chase SDK** to
render every figure embedded in the docs.  Run from anywhere:

    python examples/generate_assets.py \
        --npz /path/to/aligned_data.npz \
        --disk-center /path/to/disk_center.npz \
        --out assets
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import chase._compat  # noqa: F401
import numpy as np

import chase


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--disk-center", required=True)
    ap.add_argument("--out", default="assets")
    ap.add_argument("--temp-frames", type=int, nargs="*", default=[0, 5, 10, 15, 20, 22, 24])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.npz}")
    d = np.load(args.npz, allow_pickle=True)
    ha = d["ha_cubes_al"]
    fe = d["fe_cubes_al"]
    cont, core = d["cont_al"], d["core_al"]
    wav_ha, wav_fe = d["wavelength_ha"], d["wavelength_fe"]
    times = d["times"]
    patch = d["patch"]
    core_idx = int(d["core_idx"])
    ha_bg = d["ha_bg"] if "ha_bg" in d else None
    nframes = ha.shape[0]

    # 1) Hero 2-panel GIF. Per-frame autoscale (freeze_clim=False) matches the
    #    signed-off v50 animation: with the flare peak in-sequence, frozen limits
    #    would crush the photosphere to a flat dark frame. (freeze_clim remains
    #    available via the SDK/CLI for like-for-like brightness comparisons.)
    print("→ quicklook.gif")
    chase.make_animation(cont, core, times=times, patch=patch,
                         output_path=str(out / "quicklook.gif"), fps=4, freeze_clim=False)

    # 2) Static hero PNG at the flare peak
    print("→ flare_peak.png")
    _two_panel_png(cont, core, times, patch, frame=min(24, nframes - 1),
                   path=str(out / "flare_peak.png"))

    # 3) Contrast profile (wavelength vs time) GIF + static
    print("→ contrast_profile.gif")
    contrast, _ = chase.contrast_profile(ha, background=ha_bg, baseline_frame=0, correct_drift=True)
    chase.make_contrast_animation(ha, wav_ha, contrast, core_idx, times=times, patch=patch,
                                  output_path=str(out / "contrast_profile.gif"),
                                  flarestart=15, flarepeak=24, flareclass="X1.2")
    _contrast_png(contrast, wav_ha, nframes, str(out / "contrast_profile.png"))

    # 4) Atlas calibration chart (ISPy)
    print("→ atlas_calibration.png")
    dc = np.load(args.disk_center, allow_pickle=True)
    fe0, wav_fe0 = dc["fe_cubes"][0], dc["wavelength_fe"][0]
    ifact, woff = chase.atlas_calibrate(wav_fe0, fe0)
    print(f"   ifact={ifact:.4e}, woff={woff:.4f} Å")
    from chase.viz.diagnostics import plot_atlas_calibration
    plot_atlas_calibration(wav_fe0, fe0, ifact, woff, str(out / "atlas_calibration.png"))

    # 5) Double temperature maps (Hα width + Fe I Planck) for a frame subset
    print("→ temperature maps")
    frames = [f for f in args.temp_frames if f < nframes]
    ha_temps, fe_temps, cores = [], [], []
    for f in frames:
        ht, _ = chase.halpha_width_temperature(ha[f], wav_ha, core_wavelength=wav_ha[core_idx])
        ft, _ = chase.fe_planck_temperature(fe[f], wav_fe, ifact)
        ha_temps.append(ht); fe_temps.append(ft); cores.append(ha[f, core_idx])
        print(f"   frame {f}: Hα median {np.nanmedian(ht):.0f} K, Fe median {np.nanmedian(ft):.0f} K")
    ha_temps, fe_temps, cores = np.array(ha_temps), np.array(fe_temps), np.array(cores)
    chase.make_temperature_animation(ha_temps, fe_temps, cores,
                                     times=[times[f] for f in frames], patch=patch,
                                     output_path=str(out / "temperature.gif"))
    _temp_png(ha_temps[-1], fe_temps[-1], cores[-1], times[frames[-1]], patch,
              str(out / "double_temp_map_peak.png"))

    # 6) Diagnostics: patch overlay, QS spectrum, optical-flow before/after
    print("→ diagnostics")
    from chase.viz.diagnostics import plot_patch_overlay, plot_qs_spectrum
    if patch.size == 4:
        # overlay on the full first frame would need raw FITS; use the crop itself
        plot_patch_overlay(cont[0], [0, cont.shape[1], 0, cont.shape[2]],
                           str(out / "diag_patch.png"), title="Tracked patch (frame 0, continuum)")
    qs = np.mean(ha[0], axis=(1, 2))
    plot_qs_spectrum(qs, wav_ha, str(out / "diag_qs_spectrum.png"),
                     core_wavelength=wav_ha[core_idx], title="Patch-averaged Hα spectrum")
    _optical_flow_png(d, str(out / "diag_optical_flow.png"))

    print(f"\nAll assets written to {out}/")


def _two_panel_png(cont, core, times, patch, frame, path):
    import matplotlib.pyplot as plt
    ext = _extent(patch)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 6), facecolor="white")
    a.imshow(cont[frame], cmap="hot", origin="lower", extent=ext); a.set_title("Photosphere (Hα continuum)")
    b.imshow(core[frame], cmap="hot", origin="lower", extent=ext); b.set_title("Chromosphere (Hα core)")
    fig.suptitle(f"2023-03-29 X1.2  —  {times[frame]} (frame {frame})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=130); plt.close(fig)


def _contrast_png(contrast, wav, nframes, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    im = ax.imshow(contrast, aspect="auto", origin="lower", cmap="PiYG",
                   extent=[wav[0], wav[-1], 0, nframes], vmin=-0.15, vmax=0.15)
    ax.axhline(24, color="black", ls="dashed", lw=0.8)
    ax.set_xlabel(r"Wavelength [$\rm\AA$]"); ax.set_ylabel("Time [frame]")
    ax.set_title("Contrast profile — X1.2 (CME Doppler feature in the Hα wing)")
    plt.colorbar(im, ax=ax, label="contrast"); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def _temp_png(ha_t, fe_t, core_img, time_str, patch, path):
    import matplotlib.pyplot as plt
    ext = _extent(patch)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    im0 = axes[0].imshow(ha_t, origin="lower", cmap="inferno", extent=ext, vmin=5000, vmax=15000)
    axes[0].set_title("Chromosphere (Hα width)"); plt.colorbar(im0, ax=axes[0], shrink=0.8, label="T [K]")
    im1 = axes[1].imshow(fe_t, origin="lower", cmap="inferno", extent=ext, vmin=4000, vmax=6000)
    axes[1].set_title("Photosphere (Fe I Planck)"); plt.colorbar(im1, ax=axes[1], shrink=0.8, label="T [K]")
    axes[2].imshow(core_img, origin="lower", cmap="hot", extent=ext); axes[2].set_title("Hα core")
    fig.suptitle(f"Double temperature map — {time_str}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=130); plt.close(fig)


def _optical_flow_png(d, path):
    import matplotlib.pyplot as plt
    if "cont_raw" not in d:
        return
    raw, al = d["cont_raw"], d["cont_al"]
    f = min(22, raw.shape[0] - 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="white")
    axes[0].imshow(raw[f], cmap="hot", origin="lower"); axes[0].set_title("Before optical flow (cross-corr only)")
    axes[1].imshow(al[f], cmap="hot", origin="lower"); axes[1].set_title("After optical flow (stabilised)")
    fig.suptitle(f"Optical-flow stabilisation (frame {f})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=130); plt.close(fig)


def _extent(patch):
    patch = np.asarray(patch)
    if patch.size == 4:
        y0, y1, x0, x1 = [int(v) for v in patch]
        return [x0, x1, y0, y1]
    return None


if __name__ == "__main__":
    main()
