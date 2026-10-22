# Flare GIF stabilization: use Farneback, not TV-L1

The headline `assets/flare.gif` (the stabilised 2-panel X1.2 animation in the
README) must be rendered with the **Farneback** optical-flow method, not TV-L1.

## Why

Optical-flow stabilisation assumes **brightness constancy** — a pixel keeps its
value and merely moves. At the flare peak a region **brightens with no real
motion**, violating that assumption. The two solvers fail very differently:

- **TV-L1** (`flow_method = "tvl1"`) has a total-variation regularizer that forces
  the motion field to be piecewise-constant. It "explains" the brightening as
  motion and snaps it into a solid **block** that all shifts together, which
  `cv2.remap` then smears. This is the blocky artifact that appears on the
  flare-peak frames (≈ frame 16+).
- **Farneback** (`flow_method = "farneback"`) has no such regularizer; it produces
  a small, smooth, diffuse flow on the same bad input, so the frame stays clean.

## How this silently regressed

`create_optical_flow_solver()` prefers TV-L1 when OpenCV's `optflow` module is
present, else falls back to Farneback. The original clean GIF was produced on a
machine **without** `cv2.optflow` (Farneback). On a machine **with** it, the same
code+data silently switched to TV-L1 and the GIF came out blocky — with no code or
data change.

## The fix

Render the flare animation with Farneback:

```bash
chase /data/20230329_X12/fits \
  --patch 940 1080 1870 2040 --flow-method farneback --freeze-clim
```

`examples/config.toml` sets `flow_method = "farneback"` for this reason. The
library default in `chase/config.py` is left as `"tvl1"`, since TV-L1 is generally
more accurate on well-behaved (non-flare) data — the flare brightening is the
pathological case. Pass `--flow-method tvl1` to opt back in.

Verified 2026-06-16: TV-L1 (OpenCV 4.10 and 4.13) → blocky at flare peak;
Farneback → clean. The committed `assets/flare.gif` is the Farneback render.
