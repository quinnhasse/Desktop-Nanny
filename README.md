# Desktop Nanny

Real-time desktop monitor using YOLOv8 object detection. Watches a webcam feed or screen capture, classifies detected objects, and fires configurable responses — alerts, automatic application blocks, or timestamped logs. Ships with a full container stack: app, Prometheus, and Grafana pre-wired.

[![CI](https://github.com/quinnhasse/Desktop-Nanny/actions/workflows/ci.yml/badge.svg)](https://github.com/quinnhasse/Desktop-Nanny/actions/workflows/ci.yml)

## What it does

- Captures frames from a webcam or screen in real time
- Runs YOLOv8 inference on each frame
- Matches detections against a configurable rule set
- Fires per-rule actions: terminal alert, system notification, process kill, or log entry
- Exposes a Prometheus metrics endpoint at `:8000`

## Quick start (container)

```bash
git clone https://github.com/quinnhasse/Desktop-Nanny.git
cd Desktop-Nanny
docker compose up
```

| Service    | URL                        |
|------------|----------------------------|
| Prometheus | http://localhost:9090       |
| Grafana    | http://localhost:3000       |

Grafana default credentials: `admin / nanny`. The Desktop Nanny dashboard loads automatically.

## Quick start (local)

```bash
pip install -r requirements.txt

# Webcam
python nanny.py --source 0

# Screen capture
python nanny.py --source screen

# Custom config and metrics port
python nanny.py --source 0 --config config.yaml --metrics-port 8000
```

## Configuration

Edit `config.yaml`:

```yaml
device: cpu          # or "cuda"
model: yolov8n.pt
fps: 10
metrics_port: 8000

rules:
  - object: cell phone
    confidence_threshold: 0.6
    actions:
      - alert
      - log
  - object: person
    confidence_threshold: 0.5
    actions:
      - log
```

Actions per rule: `alert`, `log`, `block`.

## Metrics

The app exposes Prometheus metrics at `http://localhost:8000/`.

| Metric | Type | Description |
|--------|------|-------------|
| `desktop_nanny_fps_current` | Gauge | Current capture-loop FPS |
| `desktop_nanny_inference_duration_seconds` | Histogram | Per-frame YOLOv8 latency |
| `desktop_nanny_detections_total` | Counter | Detection count, labeled by object class |
| `desktop_nanny_frames_processed_total` | Counter | Total frames processed since start |

The Grafana dashboard renders:
- Inference latency p50 and p95 (1-minute rolling window)
- FPS gauge and time-series
- Detection rate by class
- Detection share pie chart
- Stat panels for p50, p95, total frames, and total detections

## Container variants

Build with `--build-arg VARIANT=cpu` (default) or `--build-arg VARIANT=cuda`:

```bash
docker build --build-arg VARIANT=cpu -t desktop-nanny:cpu .
docker build --build-arg VARIANT=cuda -t desktop-nanny:cuda .
```

Published images are on GHCR. Semver tags trigger the CI build-push job:

```
ghcr.io/quinnhasse/desktop-nanny:1.0.0
ghcr.io/quinnhasse/desktop-nanny:1.0.0-cuda
```

## CI/CD

GitHub Actions runs on every push:

1. **Lint** — `ruff check .`
2. **Test** — `pytest tests/`
3. **Build + push** — Docker build for CPU and CUDA variants, pushed to GHCR on `v*` tags

## Project structure

```
Desktop-Nanny/
├── nanny.py                          # Capture loop, rule dispatch, metrics server
├── detector.py                       # YOLOv8 wrapper with latency histogram
├── actions.py                        # Alert, block, and log handlers
├── config.yaml                       # Rule configuration
├── requirements.txt
├── Dockerfile                        # Multi-stage CPU/CUDA build
├── docker-compose.yml                # App + Prometheus + Grafana stack
├── prometheus/
│   └── prometheus.yml                # Scrape config
├── grafana/
│   ├── dashboards/
│   │   └── desktop_nanny.json        # Pre-built Grafana dashboard
│   └── provisioning/
│       ├── datasources/prometheus.yml
│       └── dashboards/default.yml
├── tests/
│   ├── test_detector.py
│   └── test_actions.py
└── .github/workflows/
    └── ci.yml
```

## Testing

```bash
pip install pytest prometheus-client pyyaml
pytest tests/ -v
```

Tests mock the ultralytics model so no GPU or weights download is needed.
