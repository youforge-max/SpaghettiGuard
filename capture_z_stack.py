#!/usr/bin/env python3
"""Capture a clear-bed reference at every Z height (1mm steps), X/Y locked.

Post-print the SV08 leaves the head at a fixed X/Y; only Z varies. So a single
reference is wrong for a grazing cam (nozzle sits at a different apparent height
per Z). This builds a Z-indexed reference stack of the EMPTY bed once, so a later
check can pick the reference matching the current Z. Bed MUST be empty when run.
"""
import datetime as dt
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

MOONRAKER = "http://192.168.0.72"
SNAPSHOT_URL = f"{MOONRAKER}/webcam/?action=snapshot"
# Per-hour reference dirs (lighting shifts over the day: window sun moves the
# shadows, not just brightness). Default = current hour bucket bed_ref_z/hHH/.
# Override with argv[1] (an hour tag like "h07") for testing.
_ROOT = Path(__file__).resolve().parent / "bed_ref_z"
_TAG = sys.argv[1] if len(sys.argv) > 1 else f"h{dt.datetime.now():%H}"
OUTDIR = _ROOT / _TAG

PARK_X, PARK_Y = 5.0, 345.0     # locked X/Y (current print-end-style pose)
Z_START, Z_END, Z_STEP = 5, 300, 1   # Z305 hits enclosure top (measured 2026-07-15); capped 300 per user


def post(path, timeout=90):
    req = urllib.request.Request(f"{MOONRAKER}{path}", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(f"{MOONRAKER}{path}", timeout=10) as r:
        return json.load(r)


def gcode(script):
    return post(f"/printer/gcode/script?script={urllib.parse.quote(script)}")


def state_pos():
    s = get("/printer/objects/query?print_stats&toolhead")["result"]["status"]
    return s["print_stats"]["state"], s["toolhead"]["position"]


def snapshot(retries=5):
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(SNAPSHOT_URL, timeout=8) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"snapshot failed: {last}")


def main():
    st, _ = state_pos()
    if st == "printing":
        sys.exit("refusing: print running")
    OUTDIR.mkdir(exist_ok=True)
    gcode(f"G90\nG1 X{PARK_X} Y{PARK_Y} F6000\nM400")
    n = 0
    for z in range(Z_START, Z_END + 1, Z_STEP):
        gcode(f"G1 Z{z} F900\nM400")
        # confirm Z landed
        for _ in range(20):
            time.sleep(0.2)
            _, pos = state_pos()
            if abs(pos[2] - z) < 0.2:
                break
        time.sleep(0.6)  # frame settle
        img = snapshot()
        img.save(OUTDIR / f"z{z:03d}.png")
        n += 1
        print(f"z={z:3d}  saved z{z:03d}.png  (X{pos[0]:.0f} Y{pos[1]:.0f})", flush=True)
    print(f"DONE: {n} references in {OUTDIR}")


if __name__ == "__main__":
    sys.exit(main())
