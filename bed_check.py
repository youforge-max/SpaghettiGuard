#!/usr/bin/env python3
"""Bed-clear check for the SV08 via webcam reference-diff (no cloud, stdlib+numpy+PIL).

Problem: the spaghetti detector's Obico model finds *failures*, not "is the bed
empty". Detecting a clear vs occupied bed is a different task, solved the
OctoPrint-BedReady way: keep one reference snapshot of a known-clear bed, then
before a print snapshot again and compare over the bed region. A big enough
change = something is on the bed.

Key trick (per the gcode-position insight): before *both* the reference capture
and every check, park the toolhead to the SAME fixed corner. Because the nozzle
then sits in an identical spot in both frames, it cancels out in the difference —
no masking needed. We read gcode_position to confirm the park landed.

Camera caveat: the SV08 cam is a low front-grazing view, not top-down. A tall
leftover part or a tool is caught easily; a thin flat skirt remnant may not be.
For high reliability add a top-down camera. Framing MUST be identical between
reference and check (same park, same lighting).

Usage:
  python3 bed_check.py --capture-reference   # run once, bed actually clear
  python3 bed_check.py --check               # returns CLEAR / OCCUPIED + match%
  python3 bed_check.py --check --no-park      # skip parking (bed already parked)
Exit code: 0 = clear, 2 = occupied, 1 = error/no reference.
"""
import argparse
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
MOONRAKER = "http://192.168.0.72"
SNAPSHOT_URL = f"{MOONRAKER}/webcam/?action=snapshot"
REF_PATH = HERE / "bed_reference.png"

# Park target — a corner well away from bed centre. SV08 bed is 350x350.
PARK_X, PARK_Y, PARK_Z_MIN = 5.0, 5.0, 15.0
# Bed region of interest in the snapshot (fractions of W,H): left,top,right,bottom.
# The visible plate sits in the mid band of this grazing cam; tune per-camera.
ROI = (0.00, 0.22, 1.00, 0.86)
# Preprocessing (calibrated on this webcam 2026-07-15): Gaussian blur then
# downsample to DOWNSAMPLE, then exposure-normalise, before per-pixel diff.
# This crushed the empty-vs-empty noise floor from ~6% (raw pixels) to 0.0%.
DOWNSAMPLE = (64, 32)
BLUR_RADIUS = 3
# A block counts as "changed" if grayscale abs-diff exceeds this (0-255).
PIXEL_DELTA = 30
# Occupied if more than this fraction of ROI blocks changed. Noise floor is 0%,
# so this is pure margin; a bed object lights up far more than this.
CHANGE_THRESHOLD = 0.015


def moonraker_post(path, timeout=15):
    req = urllib.request.Request(f"{MOONRAKER}{path}", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def moonraker_get(path):
    with urllib.request.urlopen(f"{MOONRAKER}{path}", timeout=10) as r:
        return json.load(r)


def gcode(script, timeout=90):
    return moonraker_post(
        f"/printer/gcode/script?script={urllib.parse.quote(script)}", timeout=timeout)


def snapshot(retries=4):
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(SNAPSHOT_URL, timeout=8) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception as e:      # mjpg-streamer occasionally stalls a frame
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"snapshot failed after {retries} tries: {last}")


def printer_state():
    s = moonraker_get("/printer/objects/query?print_stats&toolhead")["result"]["status"]
    return s["print_stats"]["state"], s["toolhead"]["homed_axes"], s["toolhead"]["position"]


def park():
    state, homed, _ = printer_state()
    if state == "printing":
        sys.exit("refusing to park: a print is running")
    if "xyz" not in homed:
        print("  homing (axes not homed)...")
        gcode("G28")
    gcode(f"G90\nG1 Z{PARK_Z_MIN} F600\nG1 X{PARK_X} Y{PARK_Y} F6000\nM400")
    # confirm landing
    for _ in range(20):
        time.sleep(0.5)
        _, _, pos = printer_state()
        if abs(pos[0] - PARK_X) < 1 and abs(pos[1] - PARK_Y) < 1:
            print(f"  parked at X{pos[0]:.1f} Y{pos[1]:.1f} Z{pos[2]:.1f}")
            return
    print("  WARN: park position not confirmed")


def roi_gray(img):
    """Crop to bed ROI, blur, downsample, return float32 grayscale blocks."""
    w, h = img.size
    l, t, r, b = ROI
    crop = img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    crop = crop.convert("L").filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    crop = crop.resize(DOWNSAMPLE)
    return np.asarray(crop, dtype=np.float32)


def compare(ref, cur):
    if ref.shape != cur.shape:
        cur_img = Image.fromarray(cur.astype(np.uint8)).resize(
            (ref.shape[1], ref.shape[0]))
        cur = np.asarray(cur_img, dtype=np.float32)
    cur = cur + (ref.mean() - cur.mean())          # cancel auto-exposure drift
    diff = np.abs(ref - cur)
    changed = float((diff > PIXEL_DELTA).mean())
    match_pct = 100.0 * (1.0 - changed)
    return changed, match_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-reference", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-park", action="store_true")
    ap.add_argument("--save", help="also save the analysed snapshot to this path")
    args = ap.parse_args()

    if not (args.capture_reference or args.check):
        ap.error("pick --capture-reference or --check")

    if not args.no_park:
        park()
        time.sleep(1.0)  # let the frame settle

    img = snapshot()
    if args.save:
        img.save(args.save)

    if args.capture_reference:
        img.save(REF_PATH)
        print(f"CLEAR-BED REFERENCE saved -> {REF_PATH}  ({img.size[0]}x{img.size[1]})")
        print("  (ensure the bed was actually empty when you ran this)")
        return 0

    # --check
    if not REF_PATH.exists():
        print("ERROR: no reference image. Run --capture-reference on a clear bed first.")
        return 1
    ref = roi_gray(Image.open(REF_PATH).convert("RGB"))
    cur = roi_gray(img)
    changed, match = compare(ref, cur)
    occupied = changed > CHANGE_THRESHOLD
    verdict = "OCCUPIED" if occupied else "CLEAR"
    print(f"{verdict}  match={match:.1f}%  changed_pixels={changed*100:.1f}%  "
          f"(threshold {CHANGE_THRESHOLD*100:.0f}%)")
    return 2 if occupied else 0


if __name__ == "__main__":
    sys.exit(main())
