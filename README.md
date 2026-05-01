# Desktop Nanny

Real-time desktop monitor using YOLOv8 object detection. Watches a webcam feed (or screen capture), classifies detected objects, and fires configurable responses — alerts, automatic application blocks, or timestamped logs.

## What it does

- Captures frames from a webcam or screen in real time
- Runs YOLOv8 inference on each frame to detect and classify objects
- Matches detections against a configurable ruleset
- Triggers one or more actions per rule: terminal alert, system notification, process block, or log entry

## Tech stack

| Component | Library |
|---|---|
| Object detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Frame capture | OpenCV (`cv2`) |
| Screen capture | `mss` |
| Rule evaluation | Python `dataclasses` + YAML config |
| Notifications | `plyer` |

## Setup

### Prerequisites

- Python 3.10+
- A webcam or screen capture source

### Install

```bash
git clone https://github.com/quinnhasse/Desktop-Nanny.git
cd Desktop-Nanny
pip install -r requirements.txt
```

### Configure rules

Edit `config.yaml` to define which objects trigger which actions:

```yaml
rules:
  - object: phone
    confidence_threshold: 0.6
    actions:
      - alert
      - log
  - object: person
    confidence_threshold: 0.5
    actions:
      - log
```

### Run

```bash
# Webcam mode (default)
python nanny.py

# Screen capture mode
python nanny.py --source screen

# Specify webcam index
python nanny.py --source 0
```

## How it works

1. A capture loop reads frames at a configurable FPS.
2. Each frame is passed to the YOLOv8 model for inference.
3. Detections above the configured confidence threshold are matched against the ruleset.
4. Matching rules dispatch their action handlers.
5. All detections and triggered actions are written to `nanny.log`.

The model runs on CPU by default. Set `device: cuda` in `config.yaml` if a GPU is available.

## Project structure

```
Desktop-Nanny/
├── nanny.py          # Entry point — capture loop and rule dispatch
├── detector.py       # YOLOv8 wrapper
├── actions.py        # Action handlers (alert, block, log)
├── config.yaml       # Rule configuration
├── requirements.txt
└── nanny.log         # Runtime detection log (generated on first run)
```

## Requirements

```
ultralytics
opencv-python
mss
plyer
pyyaml
```
