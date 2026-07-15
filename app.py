#!/usr/bin/env python3
"""
Spaghetti Detector — self-hosted, no-cloud print-failure monitor for the SV08.

Inference is done by a local Obico ml_api container (pretrained ONNX "failure"
model, CPU). This app polls the printer webcam, asks ml_api to score the frame,
debounces detections, draws boxes, and serves a small dark dashboard on :8110.

On a confirmed failure it pauses the print via Moonraker. That is ON by default; set
AUTO_PAUSE=0 for notify-only. It is also toggleable live from the dashboard
(POST /api/auto_pause) — the env var only sets the value it starts with.
"""
import io
import os
import sys
import json
import time
import signal
import subprocess
import threading
import datetime as dt
import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops, ImageStat


def env(k, d=None):
    v = os.environ.get(k)
    return v if v not in (None, "") else d


# --- Bed-clear detection (reference-diff, PIL-only, no numpy) ------------------
# Separate from the Obico spaghetti model: that finds tangled-filament failures,
# this answers "is a solid object on the bed" by diffing against a stored clear-bed
# reference. Only valid while the toolhead sits where the reference was captured
# (the head must not move between reference and check) — same nozzle position
# cancels in the diff. Camera is low/grazing: objects behind the toolhead or
# very flat can hide (see bed_check.py CAMERA CAVEAT).
BED_REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bed_reference.png")
BED_ROI = (0.00, 0.22, 1.00, 0.86)   # frac left,top,right,bottom of the frame
BED_DOWNSAMPLE = (128, 72)
BED_BLUR = 3
BED_PIXEL_DELTA = 22                  # per-pixel grayscale abs-diff = "changed" (lowered: catch flat/faint objects)
BED_CHANGE_THRESHOLD = 0.005          # occupied if > this fraction of blocks changed (lowered: false-positive OK, never let head slam an object)


def bed_roi_gray(img):
    """Crop to BED_ROI, grayscale, blur, downsample — the comparison form."""
    w, h = img.size
    l, t, r, b = BED_ROI
    crop = img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    crop = crop.convert("L").filter(ImageFilter.GaussianBlur(BED_BLUR))
    return crop.resize(BED_DOWNSAMPLE)


BED_REF_Z_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bed_ref_z")


def load_bed_reference():
    try:
        return bed_roi_gray(Image.open(BED_REF_PATH).convert("RGB"))
    except Exception as e:  # noqa: BLE001
        log(f"bed: no reference ({e}); bed-clear disabled until --capture-reference")
        return None


SLOTS_PER_DAY = 96               # 15-min buckets (1440 / 15)


def _slot_cyclic_dist(a, b):
    d = abs(a - b) % SLOTS_PER_DAY
    return min(d, SLOTS_PER_DAY - d)


def slot_now(now=None):
    """15-min slot 0..95 for the given/current wall clock."""
    now = now or dt.datetime.now()
    return (now.hour * 60 + now.minute) // 15


class BedRefStack:
    """Clear-bed reference stack indexed by (15-min time slot, Z), X/Y locked.

    Two things move the empty-bed image: (1) Z — the grazing cam sees the nozzle
    at a different apparent height per Z; (2) time of day — the printer is by a
    window, so daylight shifts the *shadows*, not just brightness (exposure-
    normalise only cancels global brightness). Daylight near a window moves fast
    at dawn/dusk, so time is bucketed every 15 min (96 slots/day), not hourly.
    References live in per-slot dirs bed_ref_z/qNN/zNNN.png. Each frame we pick
    the reference nearest the current Z within the slot-dir nearest the current
    slot (cyclic). Legacy hourly dirs hHH map to slot HH*4. Falls back to a
    single legacy bed_reference.png if no stack.
    """

    def __init__(self):
        self.slots = {}          # slot_int -> (tag_dirname, sorted [z_int, ...])
        self.cache = {}          # (slot_int, z_int) -> ref_small
        self.single = None
        try:
            for tag in os.listdir(BED_REF_Z_DIR):
                d = os.path.join(BED_REF_Z_DIR, tag)
                if not os.path.isdir(d):
                    continue
                if tag.startswith("q"):
                    try:
                        slot = int(tag[1:]) % SLOTS_PER_DAY
                    except ValueError:
                        continue
                elif tag.startswith("h"):     # legacy hourly dir
                    try:
                        slot = (int(tag[1:]) * 4) % SLOTS_PER_DAY
                    except ValueError:
                        continue
                else:
                    continue
                zs = []
                for fn in os.listdir(d):
                    if fn.startswith("z") and fn.endswith(".png"):
                        try:
                            zs.append(int(fn[1:-4]))
                        except ValueError:
                            pass
                if zs:
                    self.slots[slot] = (tag, sorted(zs))
        except FileNotFoundError:
            pass
        if not self.slots:
            self.single = load_bed_reference()
            log("bed: no Z/slot-stack; using single bed_reference.png")
        else:
            tot = sum(len(v[1]) for v in self.slots.values())
            log(f"bed: stack loaded {tot} refs across {len(self.slots)} slots "
                f"{sorted(self.slots)}")

    def ready(self):
        return bool(self.slots) or self.single is not None

    def get(self, z, slot):
        """Return (ref_small, chosen_z, chosen_slot); (single, None, None) if flat."""
        if not self.slots:
            return self.single, None, None
        if slot is None:
            slot = slot_now()
        ns = min(self.slots, key=lambda s: (_slot_cyclic_dist(s, slot), s))
        tag, zs = self.slots[ns]
        zref = zs[len(zs) // 2] if z is None else min(zs, key=lambda a: abs(a - z))
        key = (ns, zref)
        if key not in self.cache:
            try:
                self.cache[key] = bed_roi_gray(Image.open(
                    os.path.join(BED_REF_Z_DIR, tag, f"z{zref:03d}.png")
                ).convert("RGB"))
            except Exception as e:  # noqa: BLE001
                log(f"bed: failed loading {tag}/z{zref:03d}.png: {e}")
                return self.single, None, None
        return self.cache[key], zref, ns


def bed_detect(frame_img, ref_small):
    """Diff frame vs clear reference. Return (occupied, changed_frac, box_or_None).
    box is [x0,y0,x1,y1] in full-frame pixel coords."""
    cur = bed_roi_gray(frame_img)
    # cancel auto-exposure drift: shift cur so its mean matches the reference
    delta = int(round(ImageStat.Stat(ref_small).mean[0] - ImageStat.Stat(cur).mean[0]))
    if delta:
        cur = cur.point(lambda v, d=delta: max(0, min(255, v + d)))
    diff = ImageChops.difference(ref_small, cur)
    mask = diff.point(lambda v: 255 if v > BED_PIXEL_DELTA else 0)
    total = BED_DOWNSAMPLE[0] * BED_DOWNSAMPLE[1]
    changed = mask.histogram()[255] / total
    occupied = changed > BED_CHANGE_THRESHOLD
    box = None
    if occupied:
        bb = mask.getbbox()  # in downsample coords
        if bb:
            fw, fh = frame_img.size
            l, t, r, b = BED_ROI
            rx = (r - l) * fw / BED_DOWNSAMPLE[0]
            ry = (b - t) * fh / BED_DOWNSAMPLE[1]
            ox, oy = l * fw, t * fh
            box = [ox + bb[0] * rx, oy + bb[1] * ry,
                   ox + bb[2] * rx, oy + bb[3] * ry]
    return occupied, changed, box


CFG = {
    "SNAPSHOT_URL": env("SNAPSHOT_URL", "http://192.0.2.72/webcam/?action=snapshot"),  # set to your printer in .env
    "ML_API_URL": env("ML_API_URL", "http://127.0.0.1:3333"),
    "ML_API_TOKEN": env("ML_API_TOKEN"),          # blank = ml_api auth disabled
    "HOST": env("HOST", "0.0.0.0"),
    "PORT": int(env("PORT", "8110")),
    "CONF": float(env("CONF", "0.25")),           # client-side min confidence for a "positive"
    "POLL_SEC": float(env("POLL_SEC", "6")),
    "ALERT_STREAK": int(env("ALERT_STREAK", "3")),
    "CLEAR_STREAK": int(env("CLEAR_STREAK", "5")),
    "AUTO_PAUSE": env("AUTO_PAUSE", "1") == "1",   # ON by default; toggleable at runtime
    "MOONRAKER_URL": env("MOONRAKER_URL", "http://192.0.2.72:7125"),
    "SMTP_HOST": env("SMTP_HOST"),
    "SMTP_PORT": int(env("SMTP_PORT", "587")),
    "SMTP_USER": env("SMTP_USER"),
    "SMTP_PASS": env("SMTP_PASS"),
    "ALERT_EMAIL_TO": env("ALERT_EMAIL_TO"),
}

LOCK = threading.Lock()
STATE = {
    "started": dt.datetime.now().isoformat(timespec="seconds"),
    "last_grab": None,
    "last_error": None,
    "frames": 0,
    "score_ms": 0.0,
    "pos_streak": 0,
    "clean_streak": 0,
    "alert": False,
    "alert_since": None,
    "last_detections": [],   # [{conf,box:[cx,cy,w,h]}]
    "max_conf": 0.0,
    "paused_print": False,
    "ml_ok": False,
    "auto_pause": CFG["AUTO_PAUSE"],   # live-toggleable; env sets only the initial value
    "conf": CFG["CONF"],               # live-toggleable min confidence (sensitivity)
    "printer_state": "?",              # from Moonraker print_stats: printing/paused/complete/standby/...
    "printer_file": None,
    "printer_ok": False,               # False = Moonraker unreachable
    "bed_ok": False,                   # False = no clear-bed reference loaded
    "bed_occupied": False,             # True = object detected on the bed
    "bed_changed": 0.0,                # fraction of bed blocks changed vs reference
    "bed_ref_z": None,                 # Z (mm) of the reference chosen this frame (Z-stack)
    "bed_ref_slot": None,              # 15-min time slot (0..95) of the reference chosen this frame
    "bed_calibrating": False,          # True while a Z-stack sweep is running
    "bed_calib_aborting": False,       # True once Abort pressed, until the sweep exits
    "bed_calib_msg": "",               # last calibration result/status line
}
LATEST_JPG = None


def log(msg):
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def http_get(url, timeout=8, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def grab_frame_bytes():
    return http_get(CFG["SNAPSHOT_URL"], timeout=8, headers={"User-Agent": "spaghetti-detector"})


def score_via_mlapi():
    """Ask ml_api to fetch + score the webcam frame. Returns list of [label,conf,[cx,cy,w,h]]."""
    img_param = urllib.parse.quote(CFG["SNAPSHOT_URL"], safe="")
    url = f"{CFG['ML_API_URL'].rstrip('/')}/p/?img={img_param}"
    headers = {}
    if CFG["ML_API_TOKEN"]:
        headers["Authorization"] = f"Bearer {CFG['ML_API_TOKEN']}"
    raw = http_get(url, timeout=20, headers=headers)
    data = json.loads(raw.decode())
    return data.get("detections", [])


def moonraker_print_state():
    """Return (state, filename) from Moonraker print_stats, or (None, None) if unreachable."""
    url = CFG["MOONRAKER_URL"].rstrip("/") + "/printer/objects/query?print_stats"
    try:
        raw = http_get(url, timeout=6)
        ps = json.loads(raw.decode())["result"]["status"]["print_stats"]
        return ps.get("state"), (ps.get("filename") or None)
    except Exception:  # noqa: BLE001
        return None, None


def moonraker_z():
    """Current toolhead Z (mm), or None if unreachable/unhomed."""
    url = CFG["MOONRAKER_URL"].rstrip("/") + "/printer/objects/query?toolhead"
    try:
        raw = http_get(url, timeout=6)
        return float(json.loads(raw.decode())["result"]["status"]["toolhead"]["position"][2])
    except Exception:  # noqa: BLE001
        return None


def purge_calib_cron():
    """Remove any leftover build cron so the head NEVER sweeps unprompted.
    Calibration is button-only now — no scheduled sweeps exist. Called at
    startup and after every manual sweep as a safety net."""
    try:
        cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        lines = [ln for ln in (cur.stdout or "").splitlines()
                 if "capture_15min.sh" not in ln and "capture_hourly.sh" not in ln]
        subprocess.run(["crontab", "-"], input="\n".join(lines) + ("\n" if lines else ""),
                       text=True, check=True)
    except Exception as e:  # noqa: BLE001
        log(f"calibrate: cron purge failed: {e}")


CALIB_PROC = None       # running capture_z_stack.py Popen, or None. Guarded by LOCK.


def run_calibration():
    """Run ONE clear-bed Z sweep on button press (blocking; call in a thread).

    Captures a full Z-stack into bed_ref_z/qNN/ for the CURRENT 15-min slot only.
    No cron, no scheduled sweeps — the head moves solely as a direct result of the
    press and stops when this sweep ends (or Abort). Z only, X/Y locked; the sweep
    script refuses if a print is running. Press again (any slot, any time of day)
    to add references; the live BedRefStack picks up new slot-dirs at next reload.
    """
    global CALIB_PROC
    here = os.path.dirname(os.path.abspath(__file__))
    purge_calib_cron()  # belt-and-braces: kill any stray scheduled sweep
    try:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(here, "capture_z_stack.py")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=here)
    except Exception as e:  # noqa: BLE001
        with LOCK:
            STATE["bed_calibrating"] = False
            STATE["bed_calib_msg"] = f"error: {e}"
        return
    with LOCK:
        CALIB_PROC = proc
        STATE["bed_calibrating"] = True
        STATE["bed_calib_msg"] = f"sweep started {dt.datetime.now():%H:%M} — capturing this slot..."
    aborted = False
    try:
        out, err = proc.communicate(timeout=1800)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        err = (err or "") + " [timeout 1800s — killed]"
    tail = (out or "").strip().splitlines()
    tail = tail[-1] if tail else ""
    with LOCK:
        aborted = STATE.get("bed_calib_aborting", False)
        STATE["bed_calib_aborting"] = False
    if aborted:
        msg = f"aborted {dt.datetime.now():%H:%M} (head stopped, fans restored)"
    elif proc.returncode == 0:
        msg = f"sweep done {dt.datetime.now():%H:%M}: {tail}"
    else:
        msg = f"failed rc={proc.returncode}: {((err or tail) or '')[:180]}"
    log(f"calibrate: {msg}")
    with LOCK:
        CALIB_PROC = None
        STATE["bed_calibrating"] = False
        STATE["bed_calib_msg"] = msg


def abort_calibration():
    """Stop a running sweep. SIGINT (not kill) so capture_z_stack's finally runs:
    restores fans + releases the flock. Escalates to kill if it ignores SIGINT."""
    with LOCK:
        proc = CALIB_PROC
        if proc is None or proc.poll() is not None:
            return False
        STATE["bed_calib_aborting"] = True
        STATE["bed_calib_msg"] = "aborting — stopping head, restoring fans..."
    try:
        proc.send_signal(signal.SIGINT)
    except Exception as e:  # noqa: BLE001
        log(f"calibrate: abort signal failed: {e}")
        return False
    # give the finally block a few seconds; escalate if the process hangs
    for _ in range(30):
        if proc.poll() is not None:
            break
        time.sleep(0.2)
    else:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    log("calibrate: abort requested (SIGINT sent)")
    return True


def moonraker_pause():
    url = CFG["MOONRAKER_URL"].rstrip("/") + "/printer/print/pause"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=8) as r:
            r.read()
        log("ACTION: sent Moonraker pause")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"ERROR: Moonraker pause failed: {e}")
        return False


def send_email(subject, body):
    if not (CFG["SMTP_HOST"] and CFG["SMTP_USER"] and CFG["ALERT_EMAIL_TO"]):
        return
    import smtplib
    from email.mime.text import MIMEText
    try:
        m = MIMEText(body)
        m["Subject"], m["From"], m["To"] = subject, CFG["SMTP_USER"], CFG["ALERT_EMAIL_TO"]
        with smtplib.SMTP(CFG["SMTP_HOST"], CFG["SMTP_PORT"], timeout=15) as s:
            s.starttls()
            s.login(CFG["SMTP_USER"], CFG["SMTP_PASS"])
            s.sendmail(CFG["SMTP_USER"], [CFG["ALERT_EMAIL_TO"]], m.as_string())
        log("ACTION: alert email sent")
    except Exception as e:  # noqa: BLE001
        log(f"ERROR: email failed: {e}")


def on_alert(max_conf):
    log(f"ALERT: spaghetti/failure confirmed (conf={max_conf:.2f})")
    send_email(
        "SV08 print-failure detected",
        f"Failure detected at {dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"max confidence {max_conf:.2f}\nDashboard: http://<host>:{CFG['PORT']}/",
    )
    with LOCK:
        want_pause = STATE["auto_pause"]
    if want_pause and moonraker_pause():
        with LOCK:
            STATE["paused_print"] = True


def annotate(frame_bytes, dets, alert, streak, bed_box=None):
    """Draw failure boxes (+ optional bed-object box) on the frame; return jpeg bytes."""
    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    d = ImageDraw.Draw(img)
    box_col = (248, 81, 73) if alert else (210, 153, 34)
    for det in dets:
        conf = det["conf"]
        cx, cy, w, h = det["box"]
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        d.rectangle([x0, y0, x1, y1], outline=box_col, width=3)
        d.text((x0 + 2, max(0, y0 - 12)), f"failure {conf:.2f}", fill=box_col)
    if bed_box:                       # cyan = "object on bed" (distinct from failure red/amber)
        x0, y0, x1, y1 = bed_box
        d.rectangle([x0, y0, x1, y1], outline=(57, 197, 187), width=3)
        d.text((x0 + 2, max(0, y0 - 12)), "bed object", fill=(57, 197, 187))
    # HUD banner
    d.rectangle([0, 0, img.width, 22], fill=(20, 22, 28))
    hud = f"{'ALERT' if alert else 'OK'}  streak={streak}  {dt.datetime.now().strftime('%H:%M:%S')}"
    d.text((6, 5), hud, fill=(255, 123, 114) if alert else (126, 231, 135))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=80)
    return out.getvalue()


def poller():
    global LATEST_JPG
    log(f"ml_api target {CFG['ML_API_URL']}  webcam {CFG['SNAPSHOT_URL']}")
    bed_stack = BedRefStack()
    bed_stack_loaded = time.time()
    while True:
        t0 = time.time()
        if t0 - bed_stack_loaded > 300:   # reload every 5min: pick up new 15-min slot dirs
            bed_stack = BedRefStack()
            bed_stack_loaded = t0
        try:
            with LOCK:
                conf_min = STATE["conf"]
            raw_dets = score_via_mlapi()
            score_ms = (time.time() - t0) * 1000.0
            frame = grab_frame_bytes()
            pstate, pfile = moonraker_print_state()

            # bed-clear check (independent of the spaghetti model)
            bed_occupied, bed_changed, bed_box, bed_ref_z, bed_ref_h = False, 0.0, None, None, None
            if bed_stack.ready():
                try:
                    cur_z = moonraker_z()
                    bed_ref, bed_ref_z, bed_ref_h = bed_stack.get(cur_z, slot_now())
                    if bed_ref is not None:
                        bed_occupied, bed_changed, bed_box = bed_detect(
                            Image.open(io.BytesIO(frame)).convert("RGB"), bed_ref)
                except Exception as e:  # noqa: BLE001
                    log(f"bed: detect failed: {e}")

            dets, max_conf = [], 0.0
            for row in raw_dets:
                # ml_api format: [label, confidence, [cx, cy, w, h]]
                try:
                    conf = float(row[1])
                    box = [float(v) for v in row[2]]
                except (IndexError, TypeError, ValueError):
                    continue
                if conf >= conf_min:
                    dets.append({"conf": round(conf, 3), "box": box})
                    max_conf = max(max_conf, conf)
            positive = len(dets) > 0
            jpg = annotate(frame, dets, False, 0)  # banner refined after state update below

            fire = False
            with LOCK:
                STATE["frames"] += 1
                STATE["last_grab"] = dt.datetime.now().isoformat(timespec="seconds")
                STATE["last_error"] = None
                STATE["ml_ok"] = True
                STATE["score_ms"] = round(score_ms, 1)
                STATE["last_detections"] = dets
                STATE["max_conf"] = round(max_conf, 3)
                STATE["printer_ok"] = pstate is not None
                if pstate is not None:
                    STATE["printer_state"] = pstate
                    STATE["printer_file"] = pfile
                STATE["bed_ok"] = bed_stack.ready()
                STATE["bed_occupied"] = bed_occupied
                STATE["bed_changed"] = round(bed_changed, 4)
                STATE["bed_ref_z"] = bed_ref_z
                STATE["bed_ref_slot"] = bed_ref_h
                if positive:
                    STATE["pos_streak"] += 1
                    STATE["clean_streak"] = 0
                else:
                    STATE["clean_streak"] += 1
                    STATE["pos_streak"] = 0
                if not STATE["alert"] and STATE["pos_streak"] >= CFG["ALERT_STREAK"]:
                    STATE["alert"] = True
                    STATE["alert_since"] = STATE["last_grab"]
                    fire = True
                elif STATE["alert"] and STATE["clean_streak"] >= CFG["CLEAR_STREAK"]:
                    STATE["alert"] = False
                    STATE["alert_since"] = None
                    log("alert cleared (frames went clean)")
                alert_now, streak_now = STATE["alert"], STATE["pos_streak"]
            # redraw banner with final alert state (+ bed-object box)
            LATEST_JPG = annotate(frame, dets, alert_now, streak_now, bed_box)
            if fire:
                on_alert(max_conf)
        except Exception as e:  # noqa: BLE001
            with LOCK:
                STATE["last_error"] = str(e)
                STATE["ml_ok"] = False
            log(f"ERROR: poll: {e}")
        sleep = CFG["POLL_SEC"] - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Spaghetti Detector — SV08</title><link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🍝</text></svg>">
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{background:#0d1117;color:#c9d1d9;font:14px/1.5 system-ui,sans-serif;margin:0}
 header{padding:14px 20px;border-bottom:1px solid #21262d;display:flex;align-items:center;gap:12px}
 h1{font-size:17px;margin:0;color:#e6edf3}
 .dot{width:12px;height:12px;border-radius:50%;display:inline-block}
 .wrap{display:flex;flex-wrap:wrap;gap:20px;padding:20px}
 .cam{flex:1 1 640px;max-width:900px}
 .cam img{width:100%;border:1px solid #21262d;border-radius:8px;display:block}
 .panel{flex:0 0 300px;background:#161b22;border:1px solid #21262d;border-radius:8px;padding:16px}
 .k{color:#8b949e}.v{color:#e6edf3;font-weight:600}
 table{width:100%;border-collapse:collapse}td{padding:4px 0;vertical-align:top}
 td:last-child{text-align:right}
 a{color:#58a6ff}
 .badge{padding:2px 10px;border-radius:12px;font-weight:700;font-size:12px}
 .ok{background:#1a3d1a;color:#7ee787}.alert{background:#4d1414;color:#ff7b72}
 .tgl{margin-top:14px;display:flex;align-items:center;justify-content:space-between;gap:10px;
      padding-top:12px;border-top:1px solid #21262d}
 button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;
        padding:5px 14px;font:600 12px system-ui,sans-serif;cursor:pointer}
 button:hover{border-color:#8b949e}
 button.on{background:#1a3d1a;color:#7ee787;border-color:#2ea043}
 .hint{color:#8b949e;font-size:12px;margin:8px 0 0}
</style></head><body>
<header>
 <span class=dot id=dot></span>
 <h1>🍝 Spaghetti Detector — SV08</h1>
 <span id=badge class="badge ok">…</span>
 <span id=bedbadge class="badge" style="margin-left:auto">…</span>
 <span id=pbadge class="badge">…</span>
</header>
<div class=wrap>
 <div class=cam><img id=cam src="/snapshot.jpg" alt="camera"></div>
 <div class=panel>
  <table id=stats></table>
  <div class=tgl>
   <span class=k>Auto-pause print</span>
   <button id=ap onclick=toggleAP()>…</button>
  </div>
  <p class=hint id=aphint></p>
  <div class=tgl style="display:block">
   <div style="display:flex;justify-content:space-between">
    <span class=k>Sensitivity</span>
    <span class=v><span id=confval>…</span> min conf</span>
   </div>
   <input id=conf type=range min=5 max=95 step=1 style="width:100%;margin-top:8px" onchange=setConf(this.value)>
   <p class=hint>Lower = more sensitive (flags sooner, more false positives). Higher = stricter.</p>
  </div>
  <div style="margin-top:14px">
   <button id=calibbtn onclick=calibrate()
     style="width:100%;padding:9px;background:#0d3b3a;color:#39c5bb;border:1px solid #39c5bb;border-radius:6px;cursor:pointer;font:inherit">Capture bed reference</button>
   <button id=calibabort onclick=abortCalib() disabled
     style="width:100%;margin-top:6px;padding:9px;background:#3b0d0d;color:#ff7b72;border:1px solid #ff7b72;border-radius:6px;cursor:pointer;font:inherit">Abort sweep</button>
   <p class=hint id=calibhint>One Z sweep of the current time slot (homes, parks back-left, Z 5→300). Head MOVES only while running — no schedule. Bed must be EMPTY. Press again anytime to add slots.</p>
  </div>
  <p style="margin-top:14px"><a href=/api/status>/api/status</a> · <a href=/healthz>/healthz</a></p>
 </div>
</div>
<script>
 let AP=null;
 function refreshCam(){document.getElementById('cam').src='/snapshot.jpg?t='+Date.now();}
 function paintAP(on){
  AP=on;
  const b=document.getElementById('ap');
  b.textContent=on?'ON':'OFF';
  b.className=on?'on':'';
  document.getElementById('aphint').textContent=on
   ?'A confirmed failure will pause the print via Moonraker.'
   :'Notify-only — the print will not be touched.';
 }
 async function toggleAP(){
  if(AP===null)return;
  try{
   const r=await fetch('/api/auto_pause',{method:'POST',headers:{'Content-Type':'application/json'},
                                          body:JSON.stringify({enabled:!AP})});
   paintAP((await r.json()).auto_pause);
  }catch(e){}
 }
 async function setConf(v){
  const c=(v/100).toFixed(2);
  document.getElementById('confval').textContent=c;
  try{await fetch('/api/sensitivity',{method:'POST',headers:{'Content-Type':'application/json'},
                                       body:JSON.stringify({conf:parseFloat(c)})});}catch(e){}
 }
 const PBADGE={printing:['printing','#1a3d1a','#7ee787'],paused:['paused','#4d3a14','#e3b341'],
   complete:['complete','#1a3d1a','#7ee787'],cancelled:['cancelled','#30363d','#8b949e'],
   standby:['idle','#30363d','#8b949e'],error:['error','#4d1414','#ff7b72']};
 function paintPrinter(s){
  const b=document.getElementById('pbadge');
  if(!s.printer_ok){b.textContent='no printer';b.style.background='#4d3a14';b.style.color='#e3b341';return;}
  const m=PBADGE[s.printer_state]||[s.printer_state||'?','#30363d','#8b949e'];
  b.textContent=m[0];b.style.background=m[1];b.style.color=m[2];
 }
 function paintBed(s){
  const b=document.getElementById('bedbadge');
  if(!s.bed_ok){b.textContent='bed: no ref';b.style.background='#30363d';b.style.color='#8b949e';return;}
  if(s.bed_occupied){b.textContent='bed: OBJECT';b.style.background='#0d3b3a';b.style.color='#39c5bb';}
  else{b.textContent='bed: clear';b.style.background='#1a3d1a';b.style.color='#7ee787';}
 }
 async function calibrate(){
  const b=document.getElementById('calibbtn');
  if(b.disabled)return;                       // already running — ignore repeat press
  if(!confirm('Capture one empty-bed reference sweep now? The head homes, parks back-left and steps Z 5 to 300 (several min). The bed must be EMPTY. No schedule is created — this runs once.'))return;
  b.disabled=true; b.style.opacity='0.6'; b.textContent='Starting…';   // lock immediately
  try{
   const r=await fetch('/api/calibrate',{method:'POST'});
   const j=await r.json();
   if(!r.ok){alert('Cannot calibrate: '+(j.error||r.status)); b.disabled=false; b.style.opacity='1';}
  }catch(e){alert('calibrate failed: '+e); b.disabled=false; b.style.opacity='1';}
 }
 async function abortCalib(){
  const a=document.getElementById('calibabort');
  if(a.disabled)return;
  a.disabled=true; a.textContent='Aborting…';
  try{await fetch('/api/calib_abort',{method:'POST'});}catch(e){}
 }
 function paintCalib(s){
  const b=document.getElementById('calibbtn'),h=document.getElementById('calibhint'),
        a=document.getElementById('calibabort');
  b.disabled=!!s.bed_calibrating;
  b.textContent=s.bed_calibrating?'Calibrating… (head moving)':'Capture bed reference';
  b.style.opacity=s.bed_calibrating?'0.6':'1';
  a.disabled=!s.bed_calibrating||!!s.bed_calib_aborting;   // always visible; greyed when idle
  a.style.opacity=(!s.bed_calibrating||!!s.bed_calib_aborting)?'0.5':'1';
  if(!s.bed_calib_aborting)a.textContent='Abort sweep';
  if(s.bed_calib_msg)h.textContent=s.bed_calib_msg;
 }
 async function refresh(){
  try{
   const s=await (await fetch('/api/status')).json();
   const a=s.alert;
   document.getElementById('dot').style.background=a?'#f85149':(s.last_error?'#d29922':'#3fb950');
   const b=document.getElementById('badge');
   b.textContent=a?'ALERT — failure':(s.ml_ok?'OK':'no ml_api');
   b.className='badge '+(a?'alert':'ok');
   const rows=[
    ['Bed',!s.bed_ok?'no reference':(s.bed_occupied?'OBJECT ('+(s.bed_changed*100).toFixed(1)+'%)':'clear')],
    ['Bed ref',s.bed_ref_z!=null?('z'+s.bed_ref_z+' / q'+s.bed_ref_slot):'—'],
    ['Printer',s.printer_ok?s.printer_state:'unreachable'],
    ['Print file',s.printer_file||'—'],
    ['ml_api',s.ml_ok?'up':'DOWN'],['Score',s.score_ms+' ms'],
    ['Frames',s.frames],['Pos streak',s.pos_streak],['Clean streak',s.clean_streak],
    ['Max conf',s.max_conf],['Min conf',s.conf],['Detections',s.last_detections.length],
    ['Paused print',s.paused_print?'YES':'no'],
    ['Last grab',s.last_grab||'—'],['Last error',s.last_error||'none'],['Started',s.started],
   ];
   document.getElementById('stats').innerHTML=rows.map(r=>`<tr><td class=k>${r[0]}</td><td class=v>${r[1]}</td></tr>`).join('');
   paintAP(s.auto_pause);
   paintPrinter(s);
   paintBed(s);
   paintCalib(s);
   const cs=document.getElementById('conf');
   if(document.activeElement!==cs){cs.value=Math.round(s.conf*100);
    document.getElementById('confval').textContent=Number(s.conf).toFixed(2);}
  }catch(e){}
 }
 setInterval(refreshCam,4000);setInterval(refresh,2000);refresh();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        if p in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif p == "/snapshot.jpg":
            with LOCK:
                jpg = LATEST_JPG
            self._send(200, "image/jpeg", jpg) if jpg else self._send(503, "text/plain", b"no frame yet")
        elif p == "/api/status":
            with LOCK:
                body = dict(STATE)
            self._send(200, "application/json", json.dumps(body).encode())
        elif p == "/healthz":
            with LOCK:
                ok = STATE["ml_ok"] and STATE["frames"] > 0
            self._send(200 if ok else 503, "text/plain", b"ok" if ok else b"degraded")
        else:
            self._send(404, "text/plain", b"not found")

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1024:
            raise ValueError("body too large")
        return json.loads(self.rfile.read(n).decode() or "{}")

    def do_POST(self):
        p = urllib.parse.urlparse(self.path).path
        if p == "/api/auto_pause":
            try:
                enabled = self._read_json()["enabled"]
                if not isinstance(enabled, bool):
                    raise ValueError("'enabled' must be a boolean")
            except (ValueError, KeyError, UnicodeDecodeError) as e:
                self._send(400, "application/json", json.dumps({"error": str(e)}).encode())
                return
            with LOCK:
                STATE["auto_pause"] = enabled
            log(f"auto-pause toggled {'ON' if enabled else 'OFF'} via API")
            self._send(200, "application/json", json.dumps({"auto_pause": enabled}).encode())
        elif p == "/api/sensitivity":
            try:
                val = float(self._read_json()["conf"])
                if not 0.05 <= val <= 0.95:
                    raise ValueError("'conf' must be between 0.05 and 0.95")
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as e:
                self._send(400, "application/json", json.dumps({"error": str(e)}).encode())
                return
            with LOCK:
                STATE["conf"] = round(val, 2)
            log(f"sensitivity set: min conf={val:.2f} via API")
            self._send(200, "application/json", json.dumps({"conf": round(val, 2)}).encode())
        elif p == "/api/calibrate":
            pstate, _ = moonraker_print_state()
            if pstate == "printing":
                self._send(409, "application/json",
                           json.dumps({"error": "print running"}).encode())
                return
            with LOCK:
                busy = STATE["bed_calibrating"]
            if busy:
                self._send(409, "application/json",
                           json.dumps({"error": "already calibrating"}).encode())
                return
            threading.Thread(target=run_calibration, daemon=True).start()
            log("calibrate: started via API (head will move, Z sweep)")
            self._send(200, "application/json", json.dumps({"started": True}).encode())
        elif p == "/api/calib_abort":
            ok = abort_calibration()
            if not ok:
                self._send(409, "application/json",
                           json.dumps({"error": "no sweep running"}).encode())
                return
            self._send(200, "application/json", json.dumps({"aborting": True}).encode())
        else:
            self._send(404, "text/plain", b"not found")


def main():
    log(f"config: {json.dumps({k: v for k, v in CFG.items() if 'PASS' not in k and 'TOKEN' not in k})}")
    purge_calib_cron()  # calibration is button-only; kill any leftover scheduled sweep
    threading.Thread(target=poller, daemon=True).start()
    srv = ThreadingHTTPServer((CFG["HOST"], CFG["PORT"]), Handler)
    log(f"serving on http://{CFG['HOST']}:{CFG['PORT']}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
