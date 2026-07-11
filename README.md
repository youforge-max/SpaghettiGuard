# 🍝 SpaghettiGuard

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
      └──► on confirmed failure: log · optional email · optional Moonraker pause
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
- **Notify-only by default** — it never touches your print unless you opt in.
- **Optional actions** — pause the print via Moonraker and/or send an email on failure.
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
| `AUTO_PAUSE` | `0` | `1` = pause the print via Moonraker on alert |
| `MOONRAKER_URL` | — | Moonraker base URL (for auto-pause) |
| `SMTP_*`, `ALERT_EMAIL_TO` | *(blank)* | Optional email alerts |

> **Note:** systemd's `EnvironmentFile` does not strip inline `# comments` — keep values on
> their own line in `.env`.

## Alerting

- Debounce: `ALERT_STREAK` positive frames → **ALERT**; `CLEAR_STREAK` clean frames → clear.
- `AUTO_PAUSE=1` sends Moonraker `printer/print/pause` when a failure is confirmed
  (off by default — start in notify-only mode until you trust the detector on your printer).
- Fill the `SMTP_*` block to receive an email on failure.

## Run as a service (optional)

A `spaghetti-detector.service` unit is included. Edit the paths/user to match your system,
then:

```bash
sudo cp spaghetti-detector.service /etc/systemd/system/
sudo systemctl enable --now spaghetti-detector
```

## Endpoints

| Path | Purpose |
|------|---------|
| `/` | Dashboard |
| `/snapshot.jpg` | Latest annotated frame |
| `/api/status` | Detector status (JSON) |
| `/healthz` | Health check |

## Tuning

Start with the defaults. If you get false positives, raise `CONF` or `ALERT_STREAK`.
If it misses failures, lower them. Enable `AUTO_PAUSE` only after you've seen it reliably
catch failures on your own printer and lighting.

## Credits

- Failure-detection model & `ml_api`: [Obico / The Spaghetti Detective](https://github.com/TheSpaghettiDetective/obico-server) (AGPL).
- SpaghettiGuard app: MIT (see [LICENSE](LICENSE)).

> This project is not affiliated with or endorsed by Obico. It simply runs their
> open-source model container locally.
