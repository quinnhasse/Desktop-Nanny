"""YOLOv8 inference wrapper with Prometheus latency instrumentation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from prometheus_client import Histogram, Counter

INFERENCE_LATENCY = Histogram(
    "desktop_nanny_inference_duration_seconds",
    "Per-frame YOLOv8 inference latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0),
)

DETECTIONS_TOTAL = Counter(
    "desktop_nanny_detections_total",
    "Object detections by class label",
    ["label"],
)


@dataclass
class Detection:
    """Single bounding-box detection result."""

    label: str
    confidence: float
    box: List[float] = field(default_factory=list)  # [x1, y1, x2, y2]


class Detector:
    """Thin wrapper around a YOLOv8 model.

    Loads the model once on construction; each call to :meth:`detect` runs
    inference and records latency to the ``desktop_nanny_inference_duration_seconds``
    Prometheus histogram.
    """

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu") -> None:
        """Load YOLOv8 model.

        Args:
            model_path: Path or name of the Ultralytics model weights file.
            device: ``"cpu"`` or ``"cuda"``.
        """
        try:
            from ultralytics import YOLO  # type: ignore[import]

            self._model = YOLO(model_path)
            self._model.to(device)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required: pip install ultralytics"
            ) from exc

        self._device = device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame.

        Args:
            frame: HxWx3 numpy array in BGR color order (OpenCV default).

        Returns:
            List of :class:`Detection` objects for boxes above the model's
            built-in confidence threshold. Use :meth:`filter_by_confidence` to
            apply per-rule thresholds afterward.
        """
        t0 = time.perf_counter()
        results = self._model(frame, verbose=False)
        elapsed = time.perf_counter() - t0
        INFERENCE_LATENCY.observe(elapsed)

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                label = result.names[int(box.cls)]
                confidence = float(box.conf)
                xyxy = box.xyxy[0].tolist()
                det = Detection(label=label, confidence=confidence, box=xyxy)
                detections.append(det)
                DETECTIONS_TOTAL.labels(label=label).inc()

        return detections

    @staticmethod
    def filter_by_confidence(
        detections: List[Detection], threshold: float
    ) -> List[Detection]:
        """Return only detections at or above *threshold*."""
        return [d for d in detections if d.confidence >= threshold]
