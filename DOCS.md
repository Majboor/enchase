# 🌞 CHASE H-alpha Calibration Pipeline Documentation

Welcome to the official documentation for the **CHASE Satellite FITS Calibration Pipeline** — a tool designed to process and visualize H-alpha observation data from the CHASE satellite.

---

## 📦 Installation

```bash
git clone https://github.com/Majboor/enchase.git
cd enchase
pip install -r requirements.txt
```

> Requirements include: `numpy`, `matplotlib`, `astropy`, `scikit-image`, `tqdm`, `requests`

---

## ⚙️ CLI Usage

Run the pipeline from terminal:

```bash
PYTHONPATH=. python -m chase.cli --input PATH_OR_URL [options]
```

### ✅ Common CLI Arguments

```bash
--input           Path to FITS URL or .txt list of URLs (required)  
--output          Output directory for downloaded + processed files (default: ./downloads)  
--threshold       Brightness threshold for disk detection (default: 0.2)  
--qs_box_size     Quiet Sun box size (default: 100)  
--fov_width       Field of view width in pixels (default: 300)  
--fov_height      Field of view height in pixels (default: 300)  
--no-spatial      Skip spatial calibration  
--no-subpixel     Skip subpixel alignment  
--no-spectral     Skip spectral calibration  
--no-intensity    Skip intensity normalization  
--no-fov          Skip subregion cropping  
--save            Save diagnostic plots (QS spectrum, recenter diff, sub-FOV, etc.)  
--fig-dir         Directory to store output figures (default: ./figures)  
```

### 🔁 Example CLI Command

```bash
PYTHONPATH=. python -m chase.cli \
  --input ./downloads/myfile.fits \
  --save \
  --fig-dir ./figures
```

---

## 🧪 Python API Usage

Use CHASE as a Python library for interactive exploration or integration into other tools.

### 🔹 Import Core Functions

```python
from chase.core import (
    load_fits_data,
    recenter_image_cube,
    align_subpixel,
    extract_qs_region,
    calibrate_wavelength,
    normalize_intensity,
    extract_subregion,
    run_pipeline
)
```

### 🧠 Example: Full Pipeline Execution

```python
run_pipeline(
    fits_file = "./downloads/sample.fits",
    do_spatial   = True,
    do_subpixel  = True,
    do_spectral  = True,
    do_intensity = True,
    do_fov       = True,
    save_figs    = True,
    fig_dir      = "./figures",
    threshold_ratio   = 0.2,
    subpixel_accuracy = 100,
    qs_box_size       = 100,
    fov_width         = 300,
    fov_height        = 300
)
```

### 🧩 Example: Step-by-Step Custom Calibration

```python
data = load_fits_data("./downloads/sample.fits")

# Spatial alignment
recentered, _ = recenter_image_cube(data, threshold_ratio=0.2)

# Subpixel alignment
ref = recentered[20]
target = recentered[25]
aligned, shift = align_subpixel(ref, target)

# Spectral calibration
qs = extract_qs_region(recentered, size=100)
qs_profile = qs.mean(axis=(1, 2))
wavelengths = calibrate_wavelength(qs_profile)

# Intensity normalization
normalized = normalize_intensity(recentered, qs)

# Sub-FOV
subcube = extract_subregion(normalized, 1156, 1156, width=300, height=300)
```

---

## 📂 Output Figures

When `--save` is passed or `save_figs=True` in the function call, the following graphics are saved to `fig_dir`:

| Calibration Step       | Output Filename Suffix      |
| ---------------------- | --------------------------- |
| Disk recentering diff  | `_spatial_diff.png`         |
| QS spectral profile    | `_qs_spectrum.png`          |
| Wavelength calibration | `_wavelength_spectrum.png`  |
| Intensity comparison   | `_intensity_comparison.png` |
| Sub-FOV slice          | `_subfov_slice.png`         |

All figures are saved per input FITS file, prefixed with the filename.

---

## 🧰 Function Reference

| Function                                         | Description                                 |
| ------------------------------------------------ | ------------------------------------------- |
| `load_fits_data(path)`                           | Load FITS data cube from file               |
| `recenter_image_cube(cube, threshold_ratio)`     | Center disk spatially                       |
| `align_subpixel(ref, target, upsample_factor)`   | Register slices at subpixel level           |
| `extract_qs_region(cube, size)`                  | Extract Quiet Sun region                    |
| `calibrate_wavelength(profile)`                  | Map profile to wavelength                   |
| `normalize_intensity(cube, qs_cube)`             | Normalize intensity using QS                |
| `extract_subregion(cube, cx, cy, width, height)` | Crop subregion around center                |
| `run_pipeline(...)`                              | Full calibration and visualization pipeline |

---

## 🧑‍💻 Contributing

PRs and feature suggestions welcome! To contribute:

1. Fork the repo
2. Create a branch
3. Submit a pull request

---

## 📜 License

MIT License © 2025 – Your Name / Institution

---


