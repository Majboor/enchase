#!/usr/bin/env python3
"""SDK quick-start: drive individual stages, then the one-call orchestrator.

    python examples/sdk_quickstart.py /data/20230329_X12/fits
"""

from __future__ import annotations

import sys

import chase


def main(fits_dir: str) -> None:
    patch = [940, 1080, 1870, 2040]  # [y0, y1, x0, x1] for the X1.2 sunspot/flare

    # 1) Load + crop-track a sequence (shift-then-crop + per-frame wavelength resample)
    seq = chase.load_flare_sequence(fits_dir, patch=patch, core_wavelength=6562.8)
    print("HA cube:", seq["ha_cubes"].shape, "| FE cube:",
          None if seq["fe_cubes"] is None else seq["fe_cubes"].shape)

    # 2) Fine-stabilise on the flare-free channel; science data stays float32
    aligned = chase.optical_flow_align(seq["ha_cubes"], reference=seq["align_ref"], method="tvl1")

    # 3) Contrast profile (CME Doppler feature in the Hα wing)
    contrast, _ = chase.contrast_profile(aligned, background=seq["ha_bg"], baseline_frame=0)
    print("contrast map:", contrast.shape)

    # 4) Chromospheric temperature at the flare peak (Molnar 2019)
    peak = min(24, aligned.shape[0] - 1)
    ha_T, width = chase.halpha_width_temperature(aligned[peak], seq["wavelength_ha"])
    import numpy as np
    print(f"peak-frame median Hα T = {np.nanmedian(ha_T):.0f} K")

    # 5) Save a 2-panel GIF (frozen colour limits)
    cont = aligned[:, -1]
    core = aligned[:, seq["core_idx"]]
    chase.make_animation(cont, core, times=seq["times"], patch=seq["patch"],
                         output_path="quicklook.gif", freeze_clim=True)
    print("wrote quicklook.gif")

    # …or run everything in one call:
    # from chase import Config, run_pipeline
    # run_pipeline(Config(fits_dir=fits_dir, patch=patch, contrast=True, temperature="double"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
