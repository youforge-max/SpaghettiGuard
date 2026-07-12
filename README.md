# 🍝 SpaghettiGuard

> Self-hosted, no-cloud, free 3D-print spaghetti/failure detector — Obico ml_api (CPU) + webcam poller + live dashboard. Notify-only or optional Moonraker auto-pause.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![Docker required](https://img.shields.io/badge/docker-required-2496ED?logo=docker&logoColor=white)
![Self-hosted](https://img.shields.io/badge/self--hosted-yes-brightgreen)
![No cloud](https://img.shields.io/badge/cloud-none-brightgreen)
![Cost](https://img.shields.io/badge/cost-free-brightgreen)

**Self-hosted, no-cloud, free print-failure ("spaghetti") detector for Klipper 3D printers.**

SpaghettiGuard watches your printer's webcam, runs a pretrained failure-detection model
on each frame, and raises a debounced alert when a print starts turning into spaghetti.
A small dashboard shows the live annotated camera feed and detector status. Everything
runs on your own hardware — no accounts, no subscriptions, no images leaving your network.

![SpaghettiGuard detecting a failed print](docs/demo.jpg)

*Live detection: red `failure` boxes over a tangled print; the toolhead and bed are ignored.*

---

## How it works

```
Printer webcam (mjpg-streamer)
      │  snapshot every few seconds
      ▼
Obico ml_api  ──►  pretrained ONNX "failure" model (CPU)   [Docker container]
      │  detections: [[label, confidence, [cx,cy,w,h]], ...]
      ▼
SpaghettiGuard app  ──►  debounce ──► draw boxes ──► dashboard :8110
      │
      └──► on confirmed failure: log · Moonraker pause (on by default) · optional email
```

Inference is done by the open-source [Obico](https://github.com/TheSpaghettiDetective/obico-server)
`ml_api` model — the same failure detector used by The Spaghetti Detective — packaged as a
small CPU-only Docker container. SpaghettiGuard is the thin, dependency-light layer that polls
the camera, debounces detections into stable alerts, and gives you a dashboard and actions.

## Features

- **100% self-hosted & free** — no cloud, no API keys, no telemetry.
- **CPU-only** — no GPU required; ~150–250 ms per frame on a typical CPU.
- **Debounced alerts** — needs N consecutive positive frames to alert, and M clean frames
  to clear, so a single noisy frame won't cry wolf.
- **Stops the bleeding** — on a confirmed failure it pauses the print via Moonraker.
  On by default; flip it to notify-only whenever you like.
- **Live auto-pause toggle** — switch it on/off from the dashboard (or `POST /api/auto_pause`)
  without restarting the service or editing `.env`.
- **Optional email** — get mailed when a failure is confirmed.
- **Tiny dashboard** — live annotated feed, detector health, `/api/status` JSON, `/healthz`.

## Requirements

- A 3D printer running **Klipper + Moonraker** with a webcam exposing an MJPEG **snapshot URL**
  (e.g. `mjpg-streamer`, common on Mainsail/Fluidd setups).
- **Docker** (to run the inference container).
- **Python 3.8+** (to run the app; only dependency is Pillow).

## Install

```bash
git clone https://github.com/youforge-max/SpaghettiGuard.git
cd SpaghettiGuard

# 1) Build + run the Obico ml_api inference container (CPU)
git clone --depth 1 https://github.com/TheSpaghettiDetective/obico-server.git
docker build -f obico-server/ml_api/Containerfile.cpu -t obico-ml-cpu obico-server/ml_api
docker run -d --name obico-ml -p 3333:3333 --restart unless-stopped obico-ml-cpu

# 2) Set up the app
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env          # then edit SNAPSHOT_URL to your printer's webcam
./venv/bin/python app.py

# open http://localhost:8110
```

Find your snapshot URL in your printer's webcam settings — it usually looks like
`http://<printer-ip>/webcam/?action=snapshot`.

## Configuration (`.env`)

| Key | Default | Description |
|-----|---------|-------------|
| `SNAPSHOT_URL` | — | Printer webcam MJPEG snapshot URL (**set this**) |
| `ML_API_URL` | `http://127.0.0.1:3333` | Obico ml_api endpoint |
| `ML_API_TOKEN` | *(blank)* | Only if you set a token on the container |
| `PORT` | `8110` | Dashboard port |
| `CONF` | `0.25` | Min confidence for a frame to count as a positive |
| `POLL_SEC` | `6` | Seconds between webcam grabs |
| `ALERT_STREAK` | `3` | Consecutive positive frames before ALERT |
| `CLEAR_STREAK` | `5` | Consecutive clean frames to clear an alert |
| `AUTO_PAUSE` | `1` | Pause the print via Moonraker on alert. `0` = notify-only. **Starting value only** — toggleable at runtime |
| `MOONRAKER_URL` | — | Moonraker base URL (**set this** if auto-pause is on) |
| `SMTP_*`, `ALERT_EMAIL_TO` | *(blank)* | Optional email alerts |

> **Note:** systemd's `EnvironmentFile` does not strip inline `# comments` — keep values on
> their own line in `.env`.

## Alerting

Debounce: `ALERT_STREAK` positive frames → **ALERT**; `CLEAR_STREAK` clean frames → clear.

On a confirmed failure SpaghettiGuard sends Moonraker `printer/print/pause`, and emails you
if the `SMTP_*` block is filled in.

### Auto-pause

Auto-pause is **on by default**, so a false positive can pause a healthy print. That is the
intended trade — a paused print is recoverable, a bed full of spaghetti is not. If you'd
rather watch it prove itself first, start in notify-only mode with `AUTO_PAUSE=0`.

`AUTO_PAUSE` only sets the value the app *starts* with. After that it's a live toggle:

- **Dashboard** — the *Auto-pause print* button in the status panel.
- **API** — `POST /api/auto_pause` with `{"enabled": true|false}`.

```bash
curl -X POST http://localhost:8110/api/auto_pause \
     -H 'Content-Type: application/json' -d '{"enabled": false}'
# -> {"auto_pause": false}
```

The current value is always in `/api/status` as `auto_pause`. It is **not** persisted —
a restart goes back to whatever `AUTO_PAUSE` says in `.env`.

> Auto-pause needs `MOONRAKER_URL` pointing at your printer. If it's wrong, the pause
> request fails and is logged as an error — the alert itself still fires.

## Run as a service (optional)

A `spaghetti-detector.service` unit is included. Edit the paths/user to match your system,
then:

```bash
sudo cp spaghetti-detector.service /etc/systemd/system/
sudo systemctl enable --now spaghetti-detector
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Dashboard |
| `GET` | `/snapshot.jpg` | Latest annotated frame |
| `GET` | `/api/status` | Detector status (JSON), including `auto_pause` |
| `GET` | `/healthz` | Health check |
| `POST` | `/api/auto_pause` | Toggle auto-pause — body `{"enabled": true\|false}` |

There is no authentication. Keep it on your LAN, behind a reverse proxy, or firewalled —
anything that can reach the port can toggle auto-pause.

## Tuning

Start with the defaults. If you get false positives, raise `CONF` or `ALERT_STREAK`.
If it misses failures, lower them.

Detection quality is mostly about the camera: a stable mount, even lighting, and a view
that actually frames the part will do more than any threshold. Lighting that changes through
the print (a window at dusk) is the usual source of false positives.

If you're not yet convinced it can tell spaghetti from your prints, run with `AUTO_PAUSE=0`
for a few jobs and watch the alerts, then turn it on.

## What it is not

SpaghettiGuard detects **spaghetti and print failures**. It does not tell you whether the
bed is clear, whether the right part is on it, or whether a print finished cleanly — a
finished part sitting on the bed produces no detections, exactly like an empty bed does.
Don't use "no detections" as proof that anything is present or absent.

## Credits

- Failure-detection model & `ml_api`: [Obico / The Spaghetti Detective](https://github.com/TheSpaghettiDetective/obico-server) (AGPL).
- SpaghettiGuard app: MIT (see [LICENSE](LICENSE)).

> This project is not affiliated with or endorsed by Obico. It simply runs their
> open-source model container locally.
