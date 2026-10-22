# chase-pipeline

Calibration and flare analysis for CHASE/HIS solar spectroscopy. Install one package, point it at data, get aligned cubes, animations, contrast profiles, and temperature maps.

[![tests](https://github.com/Majboor/chase-pipeline/actions/workflows/test.yml/badge.svg)](https://github.com/Majboor/chase-pipeline/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/chasepy)](https://pypi.org/project/chasepy/)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

<p align="center">
  <img src="https://raw.githubusercontent.com/Majboor/chase-pipeline/feat/unified-pipeline-tui-sdk/assets/flare.gif" width="760" alt="Stabilised 2-panel animation of the 2023-03-29 X2.1 flare"><br>
  <em>The 2023-03-29 X2.1 flare, stabilised by this pipeline. Photosphere left, chromosphere right.</em>
</p>

## Install

```bash
pip install chasepy
```

That gives you the `chase` module, the `chase` command, and the `chase-tui` interactive app. Extras: `pip install 'chasepy[tui]'` (TUI), `'chasepy[atlas]'` (absolute Fe I calibration), `'chasepy[all]'` (everything).

## Get data

Two ways.

**Option 1: our hosted dataset.** We host ready-to-use CHASE observations on an S3 server, including the 2026-07-04 M3.2 flare (48 files, 8.65 GiB). No portal account, no expiring links:

```python
import boto3
s3 = boto3.client("s3", endpoint_url="https://s3.preservemy.world",
                  aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY,
                  region_name="garage")
s3.download_file("chase-data",
                 "events/2026/07/04/M3.2/raw/HA/RSM20260704T132940_0000_HA.fits",
                 "data/RSM20260704T132940_0000_HA.fits")
```

Read credentials are free for research use: see [docs/S3_DATA.md](docs/S3_DATA.md) for the full guide (bucket layout, Colab setup, s3fs) and how to get keys.

**Option 2: the CHASE portal.** The portal ([ssdc.nju.edu.cn](https://ssdc.nju.edu.cn)) has a captcha, so you log in once in a real browser; after that everything is command line:

```bash
pip install chasepy[portal] && playwright install chromium

chase-portal login          # opens the portal once; log in, close the window
chase-portal search --start "2023-03-29 02:00" --end "2023-03-29 03:00"
chase-portal urls   --start "2023-03-29 02:00" --end "2023-03-29 03:00" -o links.txt
chase links.txt             # downloads the files and runs the pipeline
```

`urls` writes one signed link per line (they expire in ~24h, so mint right before downloading). `chase` also accepts a single URL or a folder of already-downloaded `*.fits`. Walkthrough: [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md).

## Run it

**Run this:**

```bash
chase ./data --patch 940 1080 1870 2040 --contrast --temperature halpha
```

**What happens:** loads every scan, tracks the patch across frames, resamples wavelengths onto one grid, stabilises with optical flow, then writes results to `./chase_out/`:

| You get | What it is |
|---|---|
| `quicklook.gif` | photosphere / chromosphere animation |
| `contrast_profile.gif` + `.npy` | wavelength-vs-time flare contrast |
| `ha_temp_maps.npy` + `temperature.gif` | chromospheric temperature maps |
| `aligned_data.npz` | the aligned cubes (also the resume checkpoint) |
| `diag_*.png` | patch overlay, quiet-Sun spectrum |

No patch coordinates yet? Run with `--full-fov --no-optical-flow --no-track` first, look at `diag_qs_spectrum.png` and the GIF, then pick a `[y0 y1 x0 x1]` box.

## Rerun just one thing

The first run saved a checkpoint, so reruns skip the slow loading and alignment:

```bash
chase ./data --out ./chase_out --only gif           # re-render the animation. Takes seconds
chase ./data --out ./chase_out --only contrast      # recompute contrast only
chase ./data --out ./chase_out --only temperature --temperature fe_voigt
```

## Turn things off

Every stage has a switch. Mix freely:

| Flag | What it disables |
|---|---|
| `--no-track` | cross-correlation patch tracking |
| `--no-resample` | per-frame wavelength resampling |
| `--no-drift` | spectral drift correction in the contrast stage |
| `--no-optical-flow` | optical-flow stabilisation |
| `--no-gif`, `--no-npz`, `--no-diagnostics` | those outputs |

`--no-resample --no-drift` together = completely untouched wavelength axis.

## The same thing in Python

```python
from chase import Config, run_pipeline

results = run_pipeline(Config(fits_dir="./data", patch=[940, 1080, 1870, 2040],
                              contrast=True, temperature="halpha"))

results["aligned"]["ha"].shape    # (nframes, 118, H, W) stabilised float32
results["contrast"].shape         # (nframes, 118)
results["gif"]                    # './chase_out/quicklook.gif'
```

Rerun one step from the checkpoint: add `resume=True` and only the toggles you want.

Or skip the pipeline and call single functions:

```python
import chase

cube, hdr, wav = chase.load_cube("RSM..._HA.fits")          # one cube: (118, H, W) + wavelengths
seq = chase.load_flare_sequence("./data", patch=[...])      # tracked + resampled sequence
ha_al = chase.optical_flow_align(seq["ha_cubes"], reference=seq["align_ref"])
contrast, _ = chase.contrast_profile(ha_al, background=seq["ha_bg"])
T, width = chase.halpha_width_temperature(ha_al[12], seq["wavelength_ha"])
```

Each line works on its own. Full list with snippets: [docs/SDK.md](docs/SDK.md).

## Prefer clicking to typing?

```bash
chase-tui
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Majboor/chase-pipeline/feat/unified-pipeline-tui-sdk/assets/tui_e2e_walkthrough.gif" width="800" alt="End-to-end chase-tui walkthrough">
</p>

Every checkbox is one of the switches above. There is also a one-file config workflow: `chase --config my.toml`, keys mirror the flags ([examples/config.toml](examples/config.toml)).

## Why this exists (statement of need)

CHASE (the Chinese Hα Solar Explorer) scans the full solar disk in Hα (118 wavelengths, 6559.4 to 6565.1 Å) and Fe I (46 wavelengths near 6569 Å). The portal ships full-disk cubes with no co-alignment and a wavelength zero-point that drifts by ~0.8 channels per frame. Getting a science-ready region out of that takes real calibration work, which is why the data is under-used. Earlier scripts (the original `chase-pipeline`, Finlay Davis's `satprocess`) each solved part of the problem. This package merges them into one installable tool that preserves the physics: shift-then-crop tracking, per-frame wavelength resampling, alignment on a flare-free channel, absolute atlas calibration, and the correct temperature inversion per line. The reasoning behind each rule is in [docs/CALIBRATION.md](docs/CALIBRATION.md).

## What lives where

| Subpackage | Does | |
|---|---|---|
| `chase.io` | resumable download, FITS discovery, cube + sequence loading | <img src="assets/diag_patch.png" width="220"> |
| `chase.calib.spatial` / `.align` | shift-then-crop tracking, optical-flow stabilisation | <img src="assets/diag_optical_flow.png" width="220"> |
| `chase.calib.wavelength` | per-frame resampling onto a common grid, spectral-drift xcorr | <img src="assets/diag_qs_spectrum.png" width="220"> |
| `chase.calib.intensity` | quiet-Sun norm, integral scaling, FTS-atlas absolute calibration | <img src="assets/atlas_calibration.png" width="220"> |
| `chase.analysis.contrast` | wavelength-vs-time flare contrast | <img src="assets/contrast_profile.png" width="220"> |
| `chase.analysis.temperature` | Hα width to T (Molnar), Fe I to T (Planck / EB / Voigt) | <img src="assets/double_temp_map_peak.png" width="220"> |

<p align="center">
  <img src="https://raw.githubusercontent.com/Majboor/chase-pipeline/feat/unified-pipeline-tui-sdk/assets/temperature.gif" width="720" alt="Double temperature map animation"><br>
  <em>Double temperature map: chromosphere from Hα width (~10⁴ K), photosphere from Fe I Planck inversion (~5100 K).</em>
</p>

## Documentation

- [docs/S3_DATA.md](docs/S3_DATA.md): use our hosted CHASE datasets from anywhere (incl. Google Colab)
- [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md): CHASE portal to processed cube, end to end
- [docs/SDK.md](docs/SDK.md): every function, with snippets
- [docs/TUI.md](docs/TUI.md): TUI walkthrough
- [docs/CALIBRATION.md](docs/CALIBRATION.md): the science behind each calibration step
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md): dev setup, tests, style

## Development

```bash
git clone https://github.com/Majboor/chase-pipeline.git
cd chase-pipeline
pip install -e '.[all]'
pytest -q            # 27 tests, synthetic FITS fixtures, no big data needed
```

## Acknowledgements

Built with the guidance and corrections of **Dr. Alexander Pietrow** (AIP) and **Dr. Malcolm Druett**. The Hough-circle limb centring, spectral cross-correlation, integral intensity scaling, and CSV shift-cache are adapted from **Finlay Davis**'s [`satprocess`](https://github.com/FinlayDavis/satprocess) (BSD-3-Clause; see [`NOTICE`](NOTICE)). Affiliation support from **Prof. Dr. Sayed Amer Mahmood** (University of the Punjab). Data courtesy of the **CHASE/HIS** mission and the [Solar Science Data Center, Nanjing University](https://ssdc.nju.edu.cn/NdchaseSatellite).

## Citing

If you use this pipeline, cite the methods it builds on:

- Molnar et al. 2019, *ApJ* **881**, 99. Hα width to temperature ([10.3847/1538-4357/ab2ba3](https://doi.org/10.3847/1538-4357/ab2ba3))
- The Eddington-Barbier photospheric inversion (A&A 2013, aa21259-13)
- The **ISPy** FTS solar-atlas calibration ([ISP-SST/ISPy](https://github.com/ISP-SST/ISPy))

## License

MIT. Ported `satprocess` code stays BSD-3-Clause, listed in [`NOTICE`](NOTICE).
