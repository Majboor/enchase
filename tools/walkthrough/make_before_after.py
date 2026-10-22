"""Render side-by-side Before | After stabilisation frames from the real
2023-03-29 X1.2 flare data.

  Before = fixed crop, no tracking, no wavelength resample (raw portal data)
  After  = shift-then-crop tracking + per-frame resample + optical-flow
           stabilisation on the flare-free reference (farneback)

Writes /tmp/tui_rec/before_after/NNN.png (side-by-side, labelled) and
assets-ready before_after.gif.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import chase

DATA = "/tmp/CHASE/20230329_X12/fits"
PATCH = [940, 1080, 1870, 2040]
OUT = "/tmp/tui_rec/before_after"
os.makedirs(OUT, exist_ok=True)

FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

print("loading BEFORE (track=False, resample=False) ...")
raw = chase.load_flare_sequence(DATA, patch=PATCH, track=False, resample=False,
                                verbose=False)
print("loading AFTER (tracked) ...")
seq = chase.load_flare_sequence(DATA, patch=PATCH, verbose=False)
print("optical-flow stabilising (farneback) ...")
aligned = chase.optical_flow_align(seq["ha_cubes"], reference=seq["align_ref"],
                                   method="farneback", verbose=False)

core = seq["core_idx"]
n = min(raw["ha_cubes"].shape[0], aligned.shape[0])


def to_u8(img):
    lo, hi = np.percentile(img, (1, 99.5))
    return np.clip((img - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)


PANEL = 560  # each panel rendered square-ish, side by side with a gutter
GUT = 24
LABEL_H = 64
W = PANEL * 2 + GUT
H = PANEL + LABEL_H

font = ImageFont.truetype(FONT_B, 30)
fidx = 0
for t in range(n):
    left = Image.fromarray(to_u8(raw["ha_cubes"][t, core])).convert("RGB")
    right = Image.fromarray(to_u8(aligned[t, core])).convert("RGB")
    left = left.resize((PANEL, PANEL), Image.LANCZOS)
    right = right.resize((PANEL, PANEL), Image.LANCZOS)

    canvas = Image.new("RGB", (W, H), "#0b0b10")
    canvas.paste(left, (0, LABEL_H))
    canvas.paste(right, (PANEL + GUT, LABEL_H))
    d = ImageDraw.Draw(canvas)
    for x0, txt, col in [(0, "BEFORE  ·  raw crop", "#f7768e"),
                         (PANEL + GUT, "AFTER  ·  stabilised", "#9ece6a")]:
        b = d.textbbox((0, 0), txt, font=font)
        d.text((x0 + (PANEL - (b[2] - b[0])) // 2, (LABEL_H - (b[3] - b[1])) // 2 - b[1]),
               txt, font=font, fill=col)
    ftxt = f"Hα core   ·   frame {t + 1:02d}/{n}"
    b = d.textbbox((0, 0), ftxt, font=ImageFont.truetype(FONT_B, 20))
    d.text((W - (b[2] - b[0]) - 16, 18), ftxt,
           font=ImageFont.truetype(FONT_B, 20), fill="#565f89")

    canvas.save(f"{OUT}/{fidx:03d}.png")
    fidx += 1

frames = [Image.open(f"{OUT}/{i:03d}.png") for i in range(fidx)]
gif = "/tmp/tui_rec/before_after.gif"
small = [f.resize((880, 880 * H // W), Image.LANCZOS) for f in frames]
small[0].save(gif, save_all=True, append_images=small[1:], duration=160, loop=0)
print(f"frames={fidx}  wrote {gif}")
