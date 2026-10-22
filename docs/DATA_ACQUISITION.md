# Data acquisition — from the CHASE portal to a processed cube

This walks a brand-new user from zero to an aligned, calibrated data cube.

## 1. What CHASE data looks like

CHASE (the Chinese Hα Solar Explorer) observes the full disk with its
Hα Imaging Spectrograph (HIS) in **Raster Scanning Mode (RSM)**. Each scan gives
two spectral cubes:

| File | Line | Channels | Range | Size | Use |
|------|------|----------|-------|------|-----|
| `RSM<date>T<time>_<frame>_HA.fits` | Hα 6562.8 Å | **118** | 6559.4–6565.1 Å | ~300–375 MB | chromosphere (the science) |
| `RSM<date>T<time>_<frame>_FE.fits` | Fe I ~6569 Å | **46** | ~6567–6571 Å | ~155 MB | photosphere (clean disk; alignment + absolute calibration) |

Each cube is `(nλ, 2313, 2304)` with the image data in **`hdul[1]`** and the
wavelength axis encoded by `CRVAL3` / `CDELT3` (these **drift slightly per frame** —
the pipeline corrects for it). Example filename:
`RSM20230329T021231_0000_HA.fits`.

## 2. Download from the SSDC portal

The portal has a captcha, so you log in once in a real browser window; the
session cookie is saved and everything after that is command line:

```bash
pip install enchase[portal] && playwright install chromium

chase-portal login    # opens the portal; log in, close the window
```

Then search a UTC window and mint signed download links:

```bash
chase-portal search --start "2023-03-29 02:00" --end "2023-03-29 03:00"
# files: 50   HA: 25   FE: 25   total: 16.9 GiB

chase-portal urls --start "2023-03-29 02:00" --end "2023-03-29 03:00" -o links.txt
```

`--line HA` restricts to one spectral line. Links expire in **~24 h**, so mint
them right before downloading. When the session expires, run
`chase-portal login` again.

You can also click through the portal by hand: choose **CHASE / HIS**, **RSM**
(full-disk) mode and your date/time window, select the frames (both **HA** and
**FE** for each time step) and copy the download links. Note: the archive is
typically **1–2 weeks behind** real time.

> The exact event used throughout this repo is the **2023-03-29 X1.2** flare:
> 25 time steps, ~1–2 min cadence, peaking around **scan 17 (02:32 UT)**.

## 3. Feed the URLs to the downloader

Put one signed URL per line in a text file:

```text
# urls.txt
https://ssdc.nju.edu.cn/.../RSM20230329T021231_0000_HA.fits?sig=...
https://ssdc.nju.edu.cn/.../RSM20230329T021231_0000_FE.fits?sig=...
https://ssdc.nju.edu.cn/.../RSM20230329T021343_0001_HA.fits?sig=...
...
```

Then download (resumable — safe to re-run if interrupted):

```python
import chase
files = chase.resolve_inputs("urls.txt", save_dir="data/20230329_X12/fits")
```

or from a single URL / a directory you already have:

```python
chase.resolve_inputs("https://.../RSM...HA.fits", save_dir="data")  # one file
chase.discover_fits("data/20230329_X12/fits")                       # -> (ha_files, fe_files)
```

> Some collaborators host private mirrors (e.g. AIP `cloud.aip.de` share links).
> Those need their own auth and are not hardcoded here — drop their direct URLs
> into your `urls.txt`.

## 4. Run the pipeline

Once you have a folder of `*HA.fits` (+ `*FE.fits`):

```bash
# interactive
chase-tui

# one-shot config
chase --config examples/config.toml

# explicit
chase data/20230329_X12/fits --patch 940 1080 1870 2040 --contrast --temperature double
```

Outputs (in `out_dir`): `aligned_data.npz` (all 118 HA + 46 FE channels, aligned),
`quicklook.gif`, optionally `contrast_profile.gif`, `temperature.gif`, temperature
`.npy` arrays, and diagnostic PNGs.

## 5. Finding your patch

A patch is `[y0, y1, x0, x1]` in pixel coordinates of the full disk. The quickest
way to find one: load a frame, display the continuum, and read off the box:

```python
import chase, matplotlib.pyplot as plt
cube, hdr, wav = chase.load_cube("data/.../RSM...HA.fits")
plt.imshow(cube[-1], origin="lower", cmap="hot"); plt.show()  # continuum
```

For the 2023-03-29 X2.1 event the sunspot/flare sits at `[940, 1080, 1870, 2040]`.
Use `full_fov = true` (config) or `--full-fov` to process the whole disk instead.
