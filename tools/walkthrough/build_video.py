"""Assemble the captured TUI frames + title cards + before/after comparison
+ the real output GIFs into the e2e walkthrough MP4. v2 (post-review):

  - opens on an introductory title screen
  - no Run/Outputs log sections
  - a Before | After stabilisation segment before the finale
"""
import json, os, subprocess
from PIL import Image, ImageDraw, ImageFont

REC = "/tmp/tui_rec"
WORK = f"{REC}/build"
os.makedirs(WORK, exist_ok=True)
W, H = 1280, 858
ASSETS = os.environ.get("ASSETS_DIR", "/Volumes/moodular/chase_tmp/chase-pipeline/assets")
BA_DIR = os.environ.get("BA_DIR", "/tmp/tui_rec/before_after")  # PNG frames from make_before_after.py
FINAL = f"{REC}/tui_e2e_walkthrough.mp4"

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

CARDS = {
    "intro": ("chase-tui", "Interactive CHASE / HIS pipeline  —  end-to-end walkthrough", "#7aa2f7"),
    "data":  ("1  ·  Data Source", "Point at a folder of RSM…_HA.fits cubes, then Discover", "#7dcfff"),
    "fov":   ("2  ·  Field of View", "Crop to the flare patch    [ y0  y1  x0  x1 ]", "#9ece6a"),
    "calib": ("3  ·  Calibration", "Crop-tracking · wavelength resample · optical flow · contrast", "#e0af68"),
    "temp":  ("4  ·  Temperature", "double   =   Hα width   +   Fe I Planck  (FTS-atlas calibrated)", "#bb9af7"),
    "before_after": ("Before  |  After", "Raw fixed crop   vs.   tracked + optical-flow stabilised", "#f7768e"),
    "finale": ("The Result", "2023-03-29  X1.2 flare, stabilised   ·   photosphere | chromosphere", "#73daca"),
}
CARD_DUR = {"intro": 3.2, "finale": 2.6, "before_after": 2.8}
DEFAULT_CARD_DUR = 2.6


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _center(draw, y, text, font, fill):
    b = draw.textbbox((0, 0), text, font=font)
    w = b[2] - b[0]
    draw.text(((W - w) // 2, y), text, font=font, fill=fill)


def make_card(key, path):
    title, sub, accent = CARDS[key]
    img = Image.new("RGB", (W, H), "#0b0b10")
    d = ImageDraw.Draw(img)
    d.rectangle([W // 2 - 120, H // 2 - 70, W // 2 + 120, H // 2 - 66], fill=accent)
    _center(d, H // 2 - 40, title, _font(FONT_B, 64), accent)
    _center(d, H // 2 + 48, sub, _font(FONT, 26), "#c0caf5")
    d.rectangle([W // 2 - 120, H // 2 + 110, W // 2 + 120, H // 2 + 114], fill=accent)
    _center(d, H - 70, "chase-pipeline  ·  feat/unified-pipeline-tui-sdk", _font(FONT, 18), "#565f89")
    img.save(path)


def pad_to_canvas(src_img, path):
    im = src_img.convert("RGB")
    if im.size != (W, H):
        im.thumbnail((W, H))
        canvas = Image.new("RGB", (W, H), "#0b0b10")
        canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        im = canvas
    im.save(path)


def svg_to_png(svg, png):
    subprocess.run(["rsvg-convert", "-w", str(W), svg, "-o", png + ".raw.png"], check=True)
    raw = Image.open(png + ".raw.png")
    pad_to_canvas(raw, png)
    os.remove(png + ".raw.png")


def main():
    manifest = json.load(open(f"{REC}/manifest.json"))
    entries = []

    def add(png, dur):
        entries.append((png, dur))

    seen_seg = set()
    for item in manifest:
        seg = item["segment"]
        if seg not in seen_seg:
            seen_seg.add(seg)
            card = f"{WORK}/card_{seg}.png"
            make_card(seg, card)
            add(card, CARD_DUR.get(seg, DEFAULT_CARD_DUR))
        png = f"{WORK}/f_{os.path.basename(item['file']).replace('.svg', '')}.png"
        svg_to_png(item["file"], png)
        add(png, item["dur"])

    # Before | After stabilisation comparison (pre-rendered side-by-side PNGs)
    ba_frames = sorted(
        os.path.join(BA_DIR, f) for f in os.listdir(BA_DIR) if f.endswith(".png")
    ) if os.path.isdir(BA_DIR) else []
    if ba_frames:
        make_card("before_after", f"{WORK}/card_before_after.png")
        add(f"{WORK}/card_before_after.png", CARD_DUR["before_after"])
        for loop in range(2):
            for i, fp in enumerate(ba_frames):
                png = f"{WORK}/ba_{loop}_{i:03d}.png"
                pad_to_canvas(Image.open(fp), png)
                add(png, 0.16)
        add(png, 1.2)
    else:
        print("WARNING: no before/after frames found in", BA_DIR)

    # Finale: real output GIFs
    make_card("finale", f"{WORK}/card_finale.png")
    add(f"{WORK}/card_finale.png", CARD_DUR["finale"])
    for gif in ["flare.gif", "temperature.gif", "contrast_profile.gif"]:
        gpath = f"{ASSETS}/{gif}"
        if not os.path.exists(gpath):
            print("WARNING: missing", gpath)
            continue
        im = Image.open(gpath)
        # NB: seek + convert per frame — do NOT materialise the iterator into a
        # list (all entries would alias the last-seeked frame).
        for fi in range(getattr(im, "n_frames", 1)):
            im.seek(fi)
            png = f"{WORK}/g_{gif.replace('.gif','')}_{fi:03d}.png"
            pad_to_canvas(im.convert("RGB"), png)
            add(png, 0.12)
        add(png, 0.8)

    listf = f"{WORK}/concat.txt"
    with open(listf, "w") as f:
        for png, dur in entries:
            f.write(f"file '{png}'\n")
            f.write(f"duration {dur:.3f}\n")
        f.write(f"file '{entries[-1][0]}'\n")

    total = sum(d for _, d in entries)
    print(f"entries={len(entries)} total_duration={total:.1f}s")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listf,
        "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium",
        "-crf", "20", "-movflags", "+faststart", FINAL,
    ], check=True)
    print("WROTE", FINAL)


if __name__ == "__main__":
    main()
