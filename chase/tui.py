"""Interactive terminal UI (Textual) — a thin front-end over the SDK.

The TUI contains **no analysis logic**: every control maps to a
:class:`chase.config.Config` field and the Run button calls
:func:`chase.pipeline.run_pipeline`, exactly like the CLI and config-file paths.

Launch with ``chase-tui`` or ``python -m chase.tui``.  On a terminal without
Textual installed (or a non-interactive/dumb terminal) it prints install/usage
instructions and exits cleanly instead of crashing.
"""

from __future__ import annotations

import sys

from .config import Config


def _textual_available() -> bool:
    try:
        import textual  # noqa: F401

        return True
    except ImportError:
        return False


def run() -> int:
    """Entry point for the ``chase-tui`` console script."""
    if not _textual_available():
        print(
            "The TUI needs Textual.  Install it with:\n"
            "    pip install 'enchase[tui]'   (or: pip install textual)\n\n"
            "Meanwhile you can use the CLI or a config file:\n"
            "    chase /path/to/fits --patch 940 1080 1870 2040 --contrast\n"
            "    chase --config examples/config.toml"
        )
        return 1
    if not sys.stdout.isatty():
        print("chase-tui needs an interactive terminal. Use the `chase` CLI in non-interactive contexts.")
        return 1

    app = ChaseTUI()
    app.run()
    return 0


# --- The Textual app is defined lazily so importing this module never requires
#     Textual to be installed (the CLI imports chase.tui to discover `run`). ----
if _textual_available():
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, ScrollableContainer, Vertical
    from textual.widgets import (
        Button,
        Checkbox,
        Footer,
        Header,
        Input,
        Label,
        Log,
        Select,
        Static,
    )

    class ChaseTUI(App):
        """Menu-driven front-end: data source → FOV → toggles → run → results."""

        CSS = """
        Screen { layout: vertical; }
        #cols { height: 1fr; }
        #left { width: 46%; border: round $accent; padding: 1; }
        #right { width: 1fr; border: round $secondary; padding: 1; }
        .h { text-style: bold; color: $accent; margin-top: 1; }
        Input { margin-bottom: 1; }
        Log { height: 1fr; }
        """
        BINDINGS = [("q", "quit", "Quit"), ("r", "run", "Run pipeline")]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="cols"):
                with ScrollableContainer(id="left"):
                    yield Static("Data source", classes="h")
                    yield Input(placeholder="/path/to/fits (folder, file, .txt, or URL)", id="src")
                    yield Button("Discover", id="discover", variant="primary")
                    yield Label("", id="discovered")

                    yield Static("Field of view", classes="h")
                    yield Checkbox("Full FOV (ignore patch)", id="full_fov")
                    yield Input(placeholder="patch y0 y1 x0 x1 (e.g. 940 1080 1870 2040)", id="patch")
                    yield Input(placeholder="align-patch y0 y1 x0 x1 (optional)", id="align_patch")

                    yield Static("Calibration", classes="h")
                    yield Checkbox("Crop tracking (shift-then-crop)", value=True, id="track")
                    yield Checkbox("Smooth tracking shifts (outlier rejection)", value=True, id="smooth_shifts")
                    yield Checkbox("Derotate to solar north (INST_ROT)", id="derotate")
                    yield Checkbox("Wavelength resampling", value=True, id="resample")
                    yield Checkbox("Optical-flow stabilisation", value=True, id="optical_flow")
                    yield Checkbox("Contrast profile", id="contrast")
                    yield Checkbox("Spectral drift correction (contrast)", value=True, id="correct_drift")
                    yield Checkbox("Freeze animation colour limits", id="freeze_clim")

                    yield Static("Temperature", classes="h")
                    yield Select(
                        [(m, m) for m in ["none", "halpha", "fe_planck", "fe_eb", "fe_voigt", "double"]],
                        value="none", id="temperature", allow_blank=False,
                    )

                    yield Static("Output", classes="h")
                    yield Input(value="./chase_out", id="out")
                    yield Checkbox("Resume from aligned_data.npz (skip load + align)", id="resume")
                    yield Checkbox("Animations (GIF)", value=True, id="gif")
                    yield Checkbox("Save aligned_data.npz", value=True, id="save_npz")
                    yield Checkbox("Save aligned FITS", id="save_fits")
                    yield Checkbox("Diagnostic PNGs", value=True, id="diagnostics")
                    yield Button("Run pipeline", id="run", variant="success")

                with Vertical(id="right"):
                    yield Static("Progress / log", classes="h")
                    yield Log(id="log", highlight=True)
            yield Footer()

        # -- helpers ------------------------------------------------------
        def _log(self, msg: str) -> None:
            self.query_one("#log", Log).write_line(msg)

        def _parse_patch(self, raw: str):
            raw = raw.strip()
            if not raw:
                return None
            parts = raw.replace(",", " ").split()
            return [int(v) for v in parts] if len(parts) == 4 else None

        def _build_config(self) -> Config:
            cfg = Config()
            cfg.fits_dir = self.query_one("#src", Input).value.strip() or "."
            cfg.out_dir = self.query_one("#out", Input).value.strip() or "./chase_out"
            cfg.full_fov = self.query_one("#full_fov", Checkbox).value
            cfg.patch = self._parse_patch(self.query_one("#patch", Input).value)
            cfg.align_patch = self._parse_patch(self.query_one("#align_patch", Input).value)
            cfg.track = self.query_one("#track", Checkbox).value
            cfg.smooth_shifts = self.query_one("#smooth_shifts", Checkbox).value
            cfg.derotate = self.query_one("#derotate", Checkbox).value
            cfg.resample_wavelength = self.query_one("#resample", Checkbox).value
            cfg.optical_flow = self.query_one("#optical_flow", Checkbox).value
            cfg.contrast = self.query_one("#contrast", Checkbox).value
            cfg.correct_drift = self.query_one("#correct_drift", Checkbox).value
            cfg.resume = self.query_one("#resume", Checkbox).value
            cfg.gif = self.query_one("#gif", Checkbox).value
            cfg.save_npz = self.query_one("#save_npz", Checkbox).value
            cfg.save_fits = self.query_one("#save_fits", Checkbox).value
            cfg.diagnostics = self.query_one("#diagnostics", Checkbox).value
            cfg.freeze_clim = self.query_one("#freeze_clim", Checkbox).value
            cfg.temperature = str(self.query_one("#temperature", Select).value)
            return cfg

        # -- events -------------------------------------------------------
        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "discover":
                self._discover()
            elif event.button.id == "run":
                self.action_run()

        def _discover(self) -> None:
            from .io.download import discover_fits
            import os

            src = self.query_one("#src", Input).value.strip()
            if os.path.isdir(src):
                ha, fe = discover_fits(src)
                msg = f"Found {len(ha)} HA + {len(fe)} FE cubes."
            elif src:
                msg = f"Source: {src} (file/URL/.txt — resolved at run time)"
            else:
                msg = "Enter a data source first."
            self.query_one("#discovered", Label).update(msg)
            self._log(msg)

        def action_run(self) -> None:
            cfg = self._build_config()
            self._log(f"Running pipeline on {cfg.fits_dir} → {cfg.out_dir}")
            self.run_worker(lambda: self._run_pipeline(cfg), thread=True, exclusive=True)

        def _run_pipeline(self, cfg: Config) -> None:
            from .pipeline import run_pipeline

            def progress(stage: str, message: str) -> None:
                self.call_from_thread(self._log, f"[{stage}] {message}")

            try:
                results = run_pipeline(cfg, progress=progress)
                self.call_from_thread(self._log, "Done. Outputs:")
                for key in ("npz", "gif", "contrast_gif", "temperature_gif"):
                    if key in results:
                        self.call_from_thread(self._log, f"  {key}: {results[key]}")
            except Exception as e:  # surface errors in the log pane
                self.call_from_thread(self._log, f"ERROR: {e}")

else:  # pragma: no cover - Textual not installed
    class ChaseTUI:  # type: ignore
        def run(self):
            run()


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())
