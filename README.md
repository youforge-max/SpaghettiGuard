# 🍝 Spaghetti Detector

Self-hosted, no-cloud, free print-failure ("spaghetti") monitor for the Sovol SV08.
Polls the printer webcam, runs a YOLO model on each frame, and raises a debounced
**ALERT** when a failure is seen on N consecutive frames. Small dark dashboard on
`:8110` shows the live annotated frame + status. Notify-only by default; optional
Moonraker auto-pause and email.

Runs on the workstation GPU box (GTX 1050) but inference defaults to **CPU** — one
frame every ~6 s is trivial for CPU YOLOv8n, and the 2 GB card has too little free
VRAM once the desktop is loaded.

## Layout
```
app.py            poller thread + stdlib web UI (:8110)
.env.example      config (copy to .env)
models/           YOLO weights (.pt) — gitignored
requirements.txt  ultralytics (pulls torch + opencv)
```

## Run
```bash
cd ~/spaghetti-detector
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env            # edit MODEL_PATH / DEVICE / thresholds
./venv/bin/python app.py
# open http://localhost:8110
```

## The weights problem
Stock `yolov8n.pt` (COCO classes) **cannot** detect spaghetti — use it only to prove
the pipeline. A real detector needs spaghetti-trained weights. Options, all free /
local at runtime:

1. **Obico `ml_api`** — the maintained open-source Spaghetti Detective model
   (Darknet YOLO), shipped as a Docker container with its trained weights. CPU-capable,
   self-hosted, no cloud. Point this app's detector at its `/p/` endpoint. *(recommended
   for actual detection)*
2. **Train YOLOv8n** on a labelled 3D-print-failure dataset (e.g. Roboflow Universe
   "3d-printing-flaws", ~9.4k images). Needs a GPU with ≥6 GB VRAM to train in
   reasonable time — the local 1050 (2 GB) can't; train elsewhere, then drop `best.pt`
   into `models/`.

## Alerting
- Debounce: `ALERT_STREAK` positive frames → ALERT; `CLEAR_STREAK` clean frames → clear.
- `AUTO_PAUSE=1` sends Moonraker `printer/print/pause` on alert (default off — notify-only).
- SMTP block (reuses the SOC gmail pattern) sends an email if configured.

## Endpoints
`/` dashboard · `/snapshot.jpg` latest annotated frame · `/api/status` JSON · `/healthz`
