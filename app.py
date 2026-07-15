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
import json
import time
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


class BedRefStack:
    """Z-indexed clear-bed reference stack (bed_ref_z/zNNN.png, X/Y locked).

    The grazing cam sees the nozzle at a different apparent height per Z, so a
    single reference is only valid at one Z. Post-print the head sits at a fixed
    X/Y but a varying Z; pick the reference nearest the current Z so the nozzle
    cancels in the diff. Falls back to the single bed_reference.png if no stack.
    """

    def __init__(self):
        self.zs = []
        self.cache = {}
        self.single = None
        try:
            for fn in os.listdir(BED_REF_Z_DIR):
                if fn.startswith("z") and fn.endswith(".png"):
                    try:
                        self.zs.append(int(fn[1:-4]))
                    except ValueError:
                        pass
            self.zs.sort()
        except FileNotFoundError:
            pass
        if not self.zs:
            self.single = load_bed_reference()
            log("bed: no Z-stack; using single bed_reference.png")
        else:
            log(f"bed: Z-stack loaded {len(self.zs)} refs "
                f"(z{self.zs[0]}..z{self.zs[-1]})")

    def ready(self):
        return bool(self.zs) or self.single is not None

    def get(self, z):
        """Return (ref_small, chosen_z) nearest z, or (single, None)."""
        if not self.zs:
            return self.single, None
        if z is None:
            z = self.zs[len(self.zs) // 2]
        nz = min(self.zs, key=lambda a: abs(a - z))
        if nz not in self.cache:
            try:
                self.cache[nz] = bed_roi_gray(
                    Image.open(os.path.join(BED_REF_Z_DIR, f"z{nz:03d}.png")).convert("RGB"))
            except Exception as e:  # noqa: BLE001
                log(f"bed: failed loading z{nz:03d}.png: {e}")
                return self.single, None
        return self.cache[nz], nz


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
    while True:
        t0 = time.time()
        try:
            with LOCK:
                conf_min = STATE["conf"]
            raw_dets = score_via_mlapi()
            score_ms = (time.time() - t0) * 1000.0
            frame = grab_frame_bytes()
            pstate, pfile = moonraker_print_state()

            # bed-clear check (independent of the spaghetti model)
            bed_occupied, bed_changed, bed_box, bed_ref_z = False, 0.0, None, None
            if bed_stack.ready():
                try:
                    cur_z = moonraker_z()
                    bed_ref, bed_ref_z = bed_stack.get(cur_z)
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
<title>Spaghetti Detector — SV08</title>
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
        else:
            self._send(404, "text/plain", b"not found")


def main():
    log(f"config: {json.dumps({k: v for k, v in CFG.items() if 'PASS' not in k and 'TOKEN' not in k})}")
    threading.Thread(target=poller, daemon=True).start()
    srv = ThreadingHTTPServer((CFG["HOST"], CFG["PORT"]), Handler)
    log(f"serving on http://{CFG['HOST']}:{CFG['PORT']}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
