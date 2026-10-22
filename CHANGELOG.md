# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [2.2.0] — 2026-10-23

Alignment, limb-fit and centre-to-limb options from the Solar Physics referee
revision. All defaults are unchanged, so existing results do not change
(checked: the 2.1.1 tests pass unmodified, and default `run_pipeline` runs
on real data give bit-identical aligned cubes and shifts to 2.1.1: X1.2
Fe I + Hα, 10 scans; Hα-only quiet Sun 2023-12-29, 18 scans; M3.2, 13 scans).

### Added
- **Header recentring:** `load_flare_sequence(..., align_mode="header")`
  (config `align_mode`, CLI `--align-mode header`) takes each scan's crop shift
  from the disc-centre drift in the headers (`CRPIX1`/`CRPIX2`) plus the
  differential rotation of the patch centre (Snodgrass & Ulrich 1990, with
  `B0`), without image correlation. `align_mode="header+xcorr"` adds a
  correlation residual on top (rejected above `max_residual`, 3 px).
  `disc_center="limb"|"auto"` uses sub-pixel limb fits instead of `CRPIX`
  (`auto` checks the header against limb fits of three scans first).
  `subpixel=True` (`--subpixel`) applies the fractional part of the shift with
  a cubic spline instead of rounding.
- **Chained references for long series:** `reference="previous"|"running"`
  (`--reference`, `reference_window`) correlates each scan with the previous
  scan or with the mean of the last few aligned scans, instead of scan 0.
  `drift_control="header"` (`--drift-control header`) ties the chained steps
  back to the header pointing by least squares, so chained errors do not
  random-walk. `shift_method="windowed"` uses the Hann-windowed unnormalised
  correlation of `chase.analysis.validation` instead of phase correlation.
- **Flow reference options:** `flow_reference="stable"|"masked"|"stable+masked"`
  (`--flow-reference`) in the pipeline. `stable` computes the optical flow on
  the far-wing channel(s) whose image changes least relative to its own
  contrast (`chase.calib.select_flow_reference`); `masked` ignores pixels
  that change by more than `flare_threshold` (default 0.1) of the median
  intensity (`chase.calib.flare_mask`, `optical_flow_align(mask=...)`).
- New module `chase.calib.pointing`: `header_disc_center`,
  `limb_disc_centers`, `solar_b0`, `rotation_shift`, `header_shifts`,
  `windowed_shift`, `track_shifts`, `tie_to_header`; and
  `chase.io.alignment_shifts` (the loader's pass 1 on windows in memory).
- **Sub-pixel limb fit** (`chase.calib.spatial`): `limb_points`, `fit_limb`
  (`model="circle"` or `"ellipse"`, a rotated ellipse with centre, two
  semi-axes and angle, with MAD clipping of rays that cross prominences or
  the detector edge), `limb_disk_center` and `limb_shifts` (per-scan
  recentring shifts, integer or sub-pixel). 0.6–2 s per 2300×2300 frame,
  against 70–120 s for `hough_disk_center`. On the X1.2 and M3.2 full-disc
  data the residual disc-centre motion, measured independently on on-disc
  structure, is 2.7″ and 0.6″ RMS with the limb fit, against 5.0″ and 1.0″
  with Hough. `hough_disk_center` is unchanged.
- **Centre-to-limb variation for the atlas calibration**
  (`chase.calib.intensity`): `atlas_calibrate(..., mu=, clv_table=)` and
  `atlas_fit_quality(..., mu=, clv_table=)` multiply the FTS atlas by the
  quiet-Sun CLV of the averaged pixels, so a box away from disc centre gives
  the disc-centre DN→radiance factor. `limb_darkening` (Neckel & Labs 1994,
  the table shipped with ISPy), `download_clv_table` / `load_clv_table`
  (Pietrow et al. 2023, A&A 671, A130, CDS J/A+A/671/A130, SHA-256 checked),
  `clv_factor`, `clv_reference` (quiet-Sun reference profile at μ) and
  `mu_map`. The default `mu=1.0` keeps the 2.1 behaviour.

### Checked
- **No `CRPIX1`/`CRPIX2` axis swap.** `CRPIX1` is the column (x) and
  `CRPIX2` the row (y). Against per-scan sub-pixel limb fits the header
  centre agrees to 0.5-0.7 px and its drift to 0.10-0.23 px rms in the
  2023-12-29, 2024-10-03, 2024-12-14 and 2026-07-04 sets; swapped, they
  differ by 20-66 px. The 2023-03-29 Level-1 headers are 37 px off the limb
  and do not follow it under either reading (6.8-7.7 px rms). A test covers
  the axis mapping. `disc_center="auto"` detects such files and warns: on the
  2023-03-29 X1.2 data the header modes are not usable even with limb fits,
  because odd and even scans have field-dependent offsets of 8-11 px.

## [2.1.1] — 2026-10-22

Second round of changes for the Solar Physics referee report: code used for
the validation, stray-light, throughput and threshold analyses, plus two bug
fixes found while measuring them.

### Fixed
- **`resample_wavelength` now resamples real CHASE data.** Its "grids are
  equal" shortcut used `np.allclose` with numpy's default `rtol=1e-5`, which
  is about 0.066 Å at 6560 Å and larger than the real per-scan zero-point
  drift (0.01–0.06 Å). Every real frame was returned unchanged in 2.0–2.1.0,
  so the drift was only partly removed (by the cross-correlation fallback).
  The check now uses `rtol=0`. With the fix the scan-to-scan scatter of the
  Fe I and Hα line cores drops from 12–28 mÅ to 0.9–1.4 mÅ on the test data.
  Outputs made with 2.1.0 or earlier should be regenerated.
- **`robust_shifts` no longer moves the reference frame.** New `ref_idx`
  argument: the reference (shift 0 by construction) is never treated as an
  outlier. `load_flare_sequence` passes `ref_idx=0`. Before, a reference that
  lay off its parity's trend was "repaired" (e.g. by 8 px on a near-limb
  field), which shifted the whole tracked sequence.
- `atlas_calibrate` warns when no channel passes `wing_continuum_frac`
  (the case for Hα at the 0.85 default, where all channels, line core
  included, were silently given equal weight).

### Added
- **`chase.analysis.systematics`** — `fit_limb_ellipse` (sub-pixel limb,
  axis-aligned ellipse), `limb_coordinates`, `offlimb_profile` (sector median
  of the off-limb level with prominence / active-limb rejection),
  `psf_stray_light` (halo-model fit, scattered fraction at disc centre),
  `contrast_bias`, `decimal_year` and `throughput_trend` (exponential fit of
  the atlas factor against date, % per year with standard error and
  jackknife range).
- `chase.calib.intensity`: `activity_mask` (Fe I continuum within 3 % and
  Hα core no more than 10 % above the median), `wing_channels` (which channels
  the atlas fit up-weights), `atlas_sensitivity` (ifact change and
  brightness-temperature change per threshold and mask) and
  `atlas_factor_series` (ifact for every scan, with scatter).

## [2.1.0] — 2026-10-22 (renamed to `enchase`)

Changes made for the Solar Physics referee report on the enCHASE paper.

### Added
- **`chase.analysis.validation`** — numbers for every calibration step:
  `residual_pointing` (global cross-correlation and local dense-flow residual,
  px and arcsec), `line_centroid` / `residual_wavelength` (scan-to-scan Fe I
  core scatter in mÅ and km/s), `frame_rms` (scan-to-scan RMS in % and in
  erg s⁻¹ cm⁻² sr⁻¹ Hz⁻¹), `stray_light_estimate` (off-limb continuum upper
  limit on scattered light).
- **Active-region-safe atlas calibration** — `atlas_calibrate(mask_active=True)`
  averages only quiet pixels (`quiet_sun_mask`: continuum within 5 %, line core
  within 10 % of the patch median), or takes an explicit `mask`;
  `find_quiet_patch` picks the quietest box near disc centre;
  `extract_disk_center(center=...)` extracts it.
- `atlas_calibrate(return_quality=True)` returns the fit RMS (% of continuum)
  and reduced χ² via `atlas_fit_quality`.
- `import enchase` — alias of the `chase` package.

### Changed
- Flare class label corrected from X2.1 to X1.2 (README, docs, examples,
  `ChaseConfig.flareclass` default).
- **Distribution renamed `chasepy` → `enchase`**, repository
  `Majboor/chase-pipeline` → `Majboor/enchase` (old URLs redirect). `chasepy`
  stays on PyPI as an empty shim that installs `enchase`.
- Default `flow_method` is now `"farneback"`, matching the paper; TV-L1 is
  still available.
- `atlas_calibrate` is deterministic (seeded optimiser, `seed=0`). The docstring
  now states that `wing_continuum_frac` selects wavelength channels (weighted
  20× in ISPy's fit), not pixels.
- `main` now carries all feature branches (derotation, robust tracking, portal
  client, resume, citation metadata).

## [2.0.1] — 2026-08-02

### Changed
- Docs rewritten code-first around the PyPI package: README, `docs/SDK.md`
  (usage snippets with expected outputs for every stage), and
  `docs/CONTRIBUTING.md` (adds the PyPI release procedure). No code changes.

## [2.0.0] — 2026-08-02 (first PyPI release, as `chasepy`)

A ground-up rebuild into a modular, multi-front-end package.

### Added
- **Derotation to solar north** — CHASE detectors are mounted ~11 deg from
  solar north (`INST_ROT` in every Level-1 header). `derotate=True` on
  `load_flare_sequence` / `Config` (CLI `--derotate`, TUI checkbox) rotates
  every frame about the disc centre (`CRPIX1`, `CRPIX2`) before cropping,
  composing rotation and crop into one `warpAffine` so only the requested
  window is computed. Standalone `chase.derotate` / `chase.derotate_crop`.
  Unlike THEMIS-style calibrations the angle comes from the header rather
  than being fitted. Exported FITS record the correction (`DEROT`,
  `INSTROT`, HISTORY) and now carry true helioprojective `CRVAL1/2`
  (origin at disc centre) instead of raw pixel*scale values.
- **PyPI packaging as `chasepy`** — `pip install chasepy` installs the `chase`
  module and the `chase` / `chase-tui` commands.
- **Resume / step skipping** — `Config.resume` (CLI `--resume`, TUI checkbox)
  reuses `out_dir/aligned_data.npz` from a previous run, skipping the slow
  load + align stages so individual output steps can be rerun without
  restarting the whole pipeline. `chase --only gif,contrast` (implies
  `--resume`) runs just the listed output steps: `gif`, `contrast`,
  `temperature`, `npz`, `fits`, `diagnostics`. The TUI gains the matching
  per-output checkboxes (GIF, npz, FITS, diagnostics) plus a resume toggle,
  so single steps can be rerun from any front-end.
- **`correct_drift` toggle** — the contrast stage's data-driven spectral drift
  correction (previously always on) is now `Config.correct_drift`
  (CLI `--no-drift`, TUI checkbox); together with `resample_wavelength = false`
  this turns wavelength correction off entirely.
- The contrast profile is now saved to `out_dir/contrast_profile.npy` (it
  previously existed only in memory / in the contrast GIF).
- **Modular package layout** (`chase.io`, `chase.calib`, `chase.analysis`,
  `chase.viz`) — every stage is an importable function; nothing is a forced
  pipeline.
- **Python SDK** re-exported from `chase` (`load_cube`, `load_flare_sequence`,
  `track_and_crop`, `optical_flow_align`, `resample_wavelength`,
  `correct_spectral_drift`, `contrast_profile`, `halpha_width_temperature`,
  `fe_planck_temperature`, `atlas_calibrate`, `make_animation`, …).
- **TUI** (`chase-tui`, Textual) — data source → FOV → toggles → live-progress run
  → results; a thin layer over the SDK, with graceful degradation.
- **Config-file workflow** (`chase --config file.toml`) and a `Config` dataclass.
- **Shift-then-crop** crop tracking with a padded buffer and a separate optional
  alignment patch.
- **Optical-flow stabilisation** (TV-L1 / Farneback) applied to float32 cubes.
- **Per-frame wavelength resampling** onto a common grid.
- **Contrast profile** `(flare−bg)/bg − frame0` with a quiet-Sun background patch.
- **Temperature maps**: Hα width→T (Molnar 2019); Fe I →T via Planck (Gaussian),
  Eddington–Barbier, and Voigt estimators; absolute calibration against the FTS
  atlas (ISPy, optional).
- **`satprocess` integration** (BSD-3): Hough-circle disk centring, spectral
  cross-correlation + roll, integral intensity scaling, CSV shift-cache.
- **pytest** suite on synthetic FITS fixtures + GitHub Actions CI.
- **Docs**: data-acquisition tutorial, SDK reference, TUI guide, calibration
  science, contributing guide; README with real embedded outputs.

### Changed
- Disk centring now defaults to a robust Hough-circle limb fit (centroid kept as
  a fallback).
- `chase.core` is now a thin backward-compatibility shim re-exporting the old
  flat API.
- `pyproject.toml` packaging with `chase` and `chase-tui` entry points and
  `tui` / `atlas` / `dev` extras.

### Removed
- The 90 MB `assets/tutorial.gif` is no longer referenced (replaced by the real
  X2.1 `flare.gif`).

## [1.x] — original `chase-pipeline`
- Resumable download, brightness-centroid disk centring, QS spectral/intensity
  calibration, single-file argparse CLI, tutorial notebook.
