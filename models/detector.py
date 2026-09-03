"""YOLO model wrapper and helper detector utilities.

This module exposes:
- `ModelNotFoundError` and `PosterDetector` — the project-facing detector used
  by analysis/detection.py (returns list[dict]).
- `Detector` — a lightweight convenience wrapper that returns
  `analysis.types.Detection` dataclasses when `ultralytics` is available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from config.settings import CLASS_NAMES, DEFAULT_MODEL_WEIGHTS

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # ultralytics optional at runtime
    YOLO = None

# Import Detection only when needed to avoid circular imports at module import time


class ModelNotFoundError(FileNotFoundError):
    """Raised when trained weights are missing."""


class PosterDetector:
    """Ultralytics YOLO inference wrapper used by higher-level analysis.

    This class prefers an on-disk `.pt` checkpoint and returns simple dicts.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        class_names: dict[int, str] | None = None,
        conf_threshold: float = 0.25,
    ) -> None:
        self.weights_path = Path(weights_path or DEFAULT_MODEL_WEIGHTS)
        self.class_names = class_names or CLASS_NAMES
        self.conf_threshold = conf_threshold
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        if not self.weights_path.is_file():
            raise ModelNotFoundError(
                f"Model weights not found at: {self.weights_path}\n"
                "Train a model with: python training/train.py\n"
                "Or set MODEL_WEIGHTS to an existing .pt checkpoint."
            )

        if YOLO is None:
            raise ImportError("ultralytics is required for inference. Install with: pip install ultralytics")

        self._model = YOLO(str(self.weights_path))
        return self._model

    @property
    def is_available(self) -> bool:
        return self.weights_path.is_file()

    def predict(self, image_path: str | Path, *, conf: float | None = None) -> list[dict[str, Any]]:
        model = self._ensure_model()
        threshold = self.conf_threshold if conf is None else conf
        results = model.predict(source=str(image_path), conf=threshold, verbose=False)

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[dict[str, Any]] = []
        for box in boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                    "confidence": float(box.conf.item()),
                    "x1": int(round(x1)),
                    "y1": int(round(y1)),
                    "x2": int(round(x2)),
                    "y2": int(round(y2)),
                }
            )

        return detections

    def predict_array(self, image, *, conf: float | None = None) -> list[dict[str, Any]]:
        model = self._ensure_model()
        threshold = self.conf_threshold if conf is None else conf
        results = model.predict(source=image, conf=threshold, verbose=False)

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[dict[str, Any]] = []
        for box in boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                    "confidence": float(box.conf.item()),
                    "x1": int(round(x1)),
                    "y1": int(round(y1)),
                    "x2": int(round(x2)),
                    "y2": int(round(y2)),
                }
            )

        return detections


class Detector:
    """Lightweight ultralytics wrapper returning `analysis.types.Detection`.

    This is optional and useful for quick scripts. Use `PosterDetector` in
    higher-level code which expects dict outputs.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self.model = None

        if weights_path:
            if YOLO is None:
                raise ImportError("ultralytics package is not installed. Install it or run without weights.")
            self.model = YOLO(weights_path)

    def load_weights(self, weights_path: str):
        if YOLO is None:
            raise ImportError("ultralytics package is not installed.")
        self.weights_path = weights_path
        self.model = YOLO(weights_path)

    def detect(self, image, conf_threshold: float = 0.25) -> List[Detection]:
        if self.model is None:
            raise RuntimeError("No model loaded. Set `weights_path` and call `load_weights()`.")

        try:
            from PIL import Image

            if isinstance(image, Image.Image):
                img = np.array(image.convert("RGB"))[:, :, ::-1]
            else:
                img = image
        except Exception:
            img = image

        # import here to avoid circular import at module load time
        from analysis.types import Detection

        results = self.model.predict(img, conf=conf_threshold, verbose=False)
        if not results:
            return []

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        h, w = img.shape[:2]
        detections: List[Detection] = []

        for box, cls_id, conf in zip(xyxy, cls_ids, confidences):
            x1, y1, x2, y2 = box.tolist()
            width = float(x2 - x1)
            height = float(y2 - y1)
            area = width * height
            norm_area = area / float(max(1, w * h))
            cx = float(x1 + width / 2.0)
            cy = float(y1 + height / 2.0)
            rel_x = cx / float(max(1, w))
            rel_y = cy / float(max(1, h))

            class_name = str(int(cls_id))
            try:
                class_name = self.model.names.get(int(cls_id), class_name)
            except Exception:
                pass

            det = Detection(
                class_id=int(cls_id),
                class_name=class_name,
                confidence=float(conf),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                width=width,
                height=height,
                area=area,
                normalized_area=norm_area,
                center_x=cx,
                center_y=cy,
                rel_center_x=rel_x,
                rel_center_y=rel_y,
            )

            detections.append(det)

        return detections
