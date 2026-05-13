"""Desktop Nanny — main capture loop with rule dispatch and Prometheus metrics."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Dict, List

import yaml
from prometheus_client import Counter, Gauge, start_http_server

from actions import dispatch
from detector import Detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

FPS_GAUGE = Gauge("desktop_nanny_fps_current", "Current capture-loop frames per second")
FRAMES_TOTAL = Counter("desktop_nanny_frames_processed_total", "Total frames processed")


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _open_capture(source: str):
    """Return an OpenCV VideoCapture or an mss screenshot callable.

    Args:
        source: ``"screen"`` for mss screen capture, or a digit string / path
                for cv2.VideoCapture.

    Returns:
        A tuple ``(capture_fn, release_fn)`` where *capture_fn()* returns a
        BGR numpy frame and *release_fn()* frees resources.
    """
    if source == "screen":
        import mss  # type: ignore[import]
        import numpy as np

        sct = mss.mss()
        monitor = sct.monitors[0]

        def grab() -> Any:
            shot = sct.grab(monitor)
            frame = np.array(shot)
            return frame[:, :, :3]  # drop alpha channel

        return grab, sct.close

    import cv2  # type: ignore[import]

    index = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open capture source: {source!r}")

    def read() -> Any:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Frame read failed")
        return frame

    return read, cap.release


def run(config_path: str, source: str, metrics_port: int) -> None:
    """Start the nanny capture loop.

    Args:
        config_path: Path to ``config.yaml``.
        source: Capture source — ``"screen"`` or webcam index/path.
        metrics_port: Port on which to expose Prometheus metrics.
    """
    cfg = _load_config(config_path)
    rules: List[Dict[str, Any]] = cfg.get("rules", [])
    device: str = cfg.get("device", "cpu")
    target_fps: float = float(cfg.get("fps", 10))
    frame_interval = 1.0 / target_fps

    logger.info("Starting metrics server on :%d", metrics_port)
    start_http_server(metrics_port)

    detector = Detector(model_path=cfg.get("model", "yolov8n.pt"), device=device)
    capture, release = _open_capture(source)

    logger.info(
        "Nanny running | source=%s device=%s fps=%.1f rules=%d",
        source,
        device,
        target_fps,
        len(rules),
    )

    try:
        while True:
            t0 = time.perf_counter()

            frame = capture()
            detections = detector.detect(frame)
            FRAMES_TOTAL.inc()

            for rule in rules:
                obj_class: str = rule["object"]
                threshold: float = float(rule.get("confidence_threshold", 0.5))
                actions: List[str] = rule.get("actions", [])

                matched = detector.filter_by_confidence(
                    [d for d in detections if d.label == obj_class],
                    threshold,
                )
                for det in matched:
                    for action in actions:
                        dispatch(action, det.label, det.confidence, rule)

            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            actual_fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            FPS_GAUGE.set(actual_fps)

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop Nanny — YOLOv8 object monitor")
    parser.add_argument(
        "--source",
        default="0",
        help="Capture source: webcam index (default 0) or 'screen'",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=8000,
        help="Port for Prometheus metrics HTTP endpoint (default: 8000)",
    )
    args = parser.parse_args()
    run(config_path=args.config, source=args.source, metrics_port=args.metrics_port)


if __name__ == "__main__":
    main()
