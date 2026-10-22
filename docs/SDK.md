# Python SDK

Everything imports from the top level:

```python
import chase
```

Every public function has a NumPy-style docstring, so `help(chase.track_and_crop)` is authoritative. This page shows how to actually use each piece: what to call, what comes back, what to do with it.

Array convention: spectral cubes are `(nλ, H, W)`, sequences are `(nframes, nλ, H, W)`, always float32.

## Get data

```python
# Download one file (resumable, retries)
path = chase.download("https://ssdc.nju.edu.cn/.../RSM..._HA.fits", save_dir="./data")

# Or resolve any input kind: a folder, one file, a .txt of URLs, or a URL
paths = chase.resolve_inputs("urls.txt", save_dir="./data")

# Find the cubes in a folder
ha_files, fe_files = chase.discover_fits("./data")
```

Portal walkthrough with screenshots: [DATA_ACQUISITION.md](DATA_ACQUISITION.md).

## Load

One cube:

```python
cube, header, wav = chase.load_cube("RSM..._HA.fits")
cube.shape      # (118, H_full, W_full)
wav[:3]         # array([6559.4 , 6559.45, 6559.49])  from CRVAL3/CDELT3
```

A whole flare sequence, tracked and resampled in one call:

```python
seq = chase.load_flare_sequence("/data/fits",
                                patch=[940, 1080, 1870, 2040],       # [y0, y1, x0, x1]
                                align_patch=[980, 1040, 1920, 1990]) # optional, smaller
```

What you get back:

| Key | Shape / type | Meaning |
|---|---|---|
| `ha_cubes` | `(nframes, nλ_ha, H, W)` | tracked, resampled Hα stack |
| `fe_cubes` | same or `None` | Fe I stack, if `*FE.fits` present |
| `cont_raw`, `core_raw` | `(nframes, H, W)` | continuum / core images |
| `align_ref` | `(nframes, H, W)` | flare-free channel used for alignment |
| `ha_bg` | `(nframes, nλ_ha, H, W)` or `None` | quiet-Sun background patch |
| `wavelength_ha`, `wavelength_fe` | `(nλ,)` | common wavelength grids (Å) |
| `times` | list of str | `DATE-OBS` per frame |
| `core_idx` | int | index of the Hα core channel |
| `patch` | ndarray | the crop window |

Switches: `track=False` disables crop tracking, `resample=False` keeps each frame's own wavelength grid. `derotate=True` rotates every frame to solar north using the header `INST_ROT` angle about the disc centre before cropping (patch coordinates then refer to the north-up frame).

## Stabilise

```python
ha_al, fe_al = chase.optical_flow_align(seq["ha_cubes"],
                                        reference=seq["align_ref"],
                                        method="farneback",          # or "tvl1"
                                        extra_cubes=seq["fe_cubes"])
```

The flow is estimated once on the reference channel and the same warp is applied to every wavelength of both cubes, in float32. Without `extra_cubes` it returns just the aligned HA stack. Use `farneback` for flares; TV-L1 turns the flare peak into a rigid block ([FLARE_STABILIZATION.md](FLARE_STABILIZATION.md)).

Lower-level pieces if you want the steps separately:

```python
sy, sx = chase.phase_shift(ref_img, target_img)          # subpixel xcorr shift
cropped, shifts = chase.track_and_crop(frames, patch)    # shift-then-crop
warped = chase.warp_cube(cube, map_x, map_y)             # apply a flow map yourself
cx, cy, r = chase.hough_disk_center(full_disk_image)     # limb fit (satprocess)
```

## Wavelength

Two corrections, both optional everywhere:

```python
# Header-based: resample a cube from its own grid onto a target grid
fixed = chase.resample_wavelength(cube, src_wav, dst_wav)     # cubic interp

# Data-driven: cross-correlate mean spectra against frame 0, shift subpixel
corrected, shifts = chase.correct_spectral_drift(ha_cubes)
shifts          # array([ 0.  , -0.31, -0.62, ...])  channels
```

`load_flare_sequence(..., resample=False)` skips the first. `contrast_profile(..., correct_drift=False)` skips the second. Why they exist: each frame's zero-point drifts ~0.8 channels, and on the steep Hα wings that manufactures fake contrast ([CALIBRATION.md](CALIBRATION.md) §2).

## Intensity

```python
norm = chase.normalize_intensity(cube, ref_cube)              # quiet-Sun normalisation
factor = chase.integral_scaling(ref_spec, target_spec)        # satprocess integral match

# Absolute calibration against the FTS atlas (pip install 'enchase[atlas]')
dc_cube, dc_wav, _ = chase.extract_disk_center("RSM..._FE.fits")
ifact, woff = chase.atlas_calibrate(dc_wav, dc_cube)
ifact           # intensity factor to physical units
woff            # wavelength offset (Å)
```

## Contrast

```python
contrast, spectra = chase.contrast_profile(ha_al, background=seq["ha_bg"])
contrast.shape  # (nframes, nλ)   (flare - bg)/bg - frame0
contrast[24].max()   # peak-frame line-core enhancement, e.g. 0.12
```

`background=None` falls back to frame 0 of the flare patch itself. `correct_drift=False` skips the spectral drift correction.

## Temperature

Chromosphere, from the Hα line width (Molnar et al. 2019):

```python
T, width = chase.halpha_width_temperature(ha_al[24], seq["wavelength_ha"])
T.shape         # (H, W), Kelvin, ~1e4 K
```

Photosphere, from Fe I. Three estimators, all needing the atlas `ifact`:

```python
T, core = chase.fe_planck_temperature(fe_al[24], seq["wavelength_fe"], ifact)
import numpy as np
np.nanmedian(T)      # ~5100 K quiet photosphere

T_cube, T_core, T_cont = chase.fe_eb_temperature(fe_al[24], seq["wavelength_fe"], ifact)
T, core, center = chase.fe_voigt_temperature(fe_al[24], seq["wavelength_fe"], ifact)
```

Planck fits a Gaussian core. Eddington-Barbier gives core and continuum temperatures. Voigt handles under-sampled lines best. Pick per [CALIBRATION.md](CALIBRATION.md) §8.

## Animate

```python
gif = chase.make_animation(cont_al, core_al, times=seq["times"],
                           output_path="quicklook.gif", freeze_clim=True)

gif = chase.make_contrast_animation(ha_al, seq["wavelength_ha"], contrast,
                                    seq["core_idx"], output_path="contrast.gif",
                                    flarestart=15, flarepeak=24, flareclass="X1.2")

gif = chase.make_temperature_animation(ha_T_stack, fe_T_stack, core_imgs,
                                       output_path="temperature.gif")
```

Each returns the output path. Diagnostic single plots live in `chase.viz.diagnostics`: `plot_patch_overlay`, `plot_qs_spectrum`, `plot_shift_track`, `plot_atlas_calibration`.

## Save

```python
chase.save_npz("out.npz", ha=ha_al, wav=seq["wavelength_ha"])   # compressed
chase.save_cube_fits(ha_al[0], "frame0.fits", header=header)
```

## Run the whole thing

```python
from chase import Config, run_pipeline

cfg = Config(fits_dir="/data/fits", patch=[940, 1080, 1870, 2040],
             contrast=True, temperature="double")
results = run_pipeline(cfg)

results["aligned"]["ha"]     # in-memory stabilised stack
results["npz"]               # path to aligned_data.npz
results["contrast_npy"]      # path to contrast_profile.npy
```

Rerun a single output later. `resume=True` reads `out_dir/aligned_data.npz` back instead of touching the FITS:

```python
run_pipeline(Config(fits_dir="/data/fits", out_dir="./chase_out",
                    resume=True, temperature="fe_voigt", gif=True))
```

`progress=` takes a `(stage, message)` callback for live UIs. `load_config("file.toml")` builds a `Config` from TOML; every field maps one to one. The full field list is the `Config` docstring, or [examples/config.toml](../examples/config.toml) annotated.

## Backward compatibility

The pre-2.0 flat API still works: `from chase.core import run_pipeline, load_fits_data, find_disk_center, ...`. New code should import from the top level as above.
