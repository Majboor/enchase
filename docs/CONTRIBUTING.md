# Contributing

## Dev setup

```bash
git clone https://github.com/Majboor/enchase
cd enchase
pip install -e '.[all]'      # core + tui + atlas + dev
pytest -q                    # 27 tests, all on synthetic FITS fixtures
```

You do not need the real multi-hundred-MB CHASE cubes to develop or test. `tests/conftest.py` synthesises tiny HA/FE cubes with a known spatial shift and a known CRVAL3 drift, so every physics rule can be asserted against ground truth.

Users install from PyPI instead: `pip install enchase`. The import name is `chase` (`import enchase` works too).

## Project layout

```
chase/
  io/        download + discovery + cube/sequence loading
  calib/     spatial · align · wavelength · intensity
  analysis/  contrast · temperature
  viz/       animate · diagnostics
  config.py pipeline.py cli.py tui.py core.py(shim) shifts_cache.py
tests/  examples/  docs/  assets/
```

## Conventions

- **Modular first.** New functionality is a small, pure function in the relevant submodule, re-exported from the package `__init__`. Don't force it into the pipeline; `pipeline.py` is glue only.
- **Every knob reaches all front-ends.** A new option gets a `Config` field, a CLI flag, and a TUI control. TOML support comes free with the `Config` field.
- **NumPy-style docstrings** on every public function. Keep parameter names and array shapes explicit.
- **Don't break the physics.** The ten rules in [CALIBRATION.md](CALIBRATION.md) are requirements, not suggestions.
- **Keep science data float32.** Convert to uint8 only where an algorithm demands it (the optical-flow reference channel).
- **Ported `satprocess` code keeps its BSD-3 header** and is listed in [`NOTICE`](../NOTICE).

## Tests and CI

- Add a test for any new stage. Assert against ground truth from the synthetic fixtures where possible.
- CI runs `pytest` on Python 3.9 / 3.11 / 3.12 (`.github/workflows/test.yml`).
- Keep tests fast and hermetic. No network, no large data.

## Commits and PRs

- Branch from `feat/unified-pipeline-tui-sdk`, open PRs against it.
- Small, readable, incremental commits. This project tracks toward a JOSS submission, which values a clear history.
- Describe why, not just what. Name the calibration rule a change touches.
- Run `pytest -q` before pushing.

## Releasing to PyPI

The package is published as [`enchase`](https://pypi.org/project/enchase/); the old name [`chasepy`](https://pypi.org/project/chasepy/) is kept as a shim that depends on it. PyPI rejects re-uploads of an existing version, so every release needs a bump.

```bash
# 1. bump version in pyproject.toml, update CHANGELOG.md
# 2. build and check
pip install build twine
python -m build
twine check dist/*
# 3. sanity-test the wheel in a clean venv, then upload
twine upload dist/*          # username __token__, password = a pypi- API token
```

## Optional dependencies

- `textual`: the TUI. Gate any TUI-only import; `chase.tui` must import cleanly without it.
- `ISPy`: absolute Fe I atlas calibration. Gate the import and degrade gracefully; the core pipeline must work without it.
