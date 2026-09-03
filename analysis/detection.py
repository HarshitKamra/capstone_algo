"""High-level poster element detection API."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from analysis.aoi import AOIRecord, build_aoi_record, parse_yolo_label_lines, parse_yolo_labels
from config.settings import CLASS_NAMES, DATASET_LABEL_DIR
from models.detector import ModelNotFoundError, PosterDetector


def detections_to_aoi_records(
    detections: list[dict],
    image_shape: tuple[int, ...],
) -> list[AOIRecord]:
    """Convert YOLO inference output to standardized AOI records."""
    height, width = image_shape[:2]
    records: list[AOIRecord] = []

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        box_w = (x2 - x1) / max(width, 1)
        box_h = (y2 - y1) / max(height, 1)
        x_center = ((x1 + x2) / 2) / max(width, 1)
        y_center = ((y1 + y2) / 2) / max(height, 1)

        records.append(
            build_aoi_record(
                class_id=det["class_id"],
                x_center_norm=x_center,
                y_center_norm=y_center,
                width_norm=box_w,
                height_norm=box_h,
                image_width=width,
                image_height=height,
                confidence=det["confidence"],
            )
        )

    return records


def load_aois_from_labels(
    poster_name: str,
    image_shape: tuple[int, ...],
    label_dir: str | Path | None = None,
) -> list[AOIRecord]:
    """Load ground-truth AOIs from a YOLO label file matching the poster."""
    label_dir = Path(label_dir or DATASET_LABEL_DIR)
    stem = os.path.splitext(poster_name)[0]
    label_path = label_dir / f"{stem}.txt"

    if not label_path.is_file():
        raise FileNotFoundError(f"Matching label file not found: {label_path.name}")

    return parse_yolo_labels(str(label_path), image_shape)


def detect_poster_elements(
    image: np.ndarray,
    *,
    image_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    conf: float = 0.25,
    fallback_to_labels: bool = False,
    poster_name: str | None = None,
    label_dir: str | Path | None = None,
) -> tuple[list[AOIRecord], str]:
    """
    Detect poster elements using YOLO inference.

    Returns (aoi_records, source) where source is 'model' or 'labels'.

    If weights are missing and fallback_to_labels=True, uses dataset labels
    when poster_name is provided.
    """
    detector = PosterDetector(weights_path=weights_path, conf_threshold=conf)

    if detector.is_available:
        if image_path is not None:
            detections = detector.predict(image_path, conf=conf)
        else:
            detections = detector.predict_array(image, conf=conf)
        return detections_to_aoi_records(detections, image.shape), "model"

    if fallback_to_labels and poster_name:
        records = load_aois_from_labels(poster_name, image.shape, label_dir)
        return records, "labels"

    raise ModelNotFoundError(
        "No trained model weights available. "
        "Train with: python training/train.py "
        "Or pass fallback_to_labels=True with a labelled dataset poster."
    )


def load_label_lines(label_dir: str | Path, poster_name: str) -> list[str]:
    """Load raw YOLO label lines for a poster (legacy helper)."""
    label_dir = Path(label_dir)
    stem = os.path.splitext(poster_name)[0]
    label_path = label_dir / f"{stem}.txt"

    if not label_path.is_file():
        raise FileNotFoundError(f"Matching label file not found: {label_path.name}")

    with open(label_path, encoding="utf-8") as file:
        return file.readlines()


def label_lines_to_aoi_records(lines: list[str], image_shape: tuple[int, ...]) -> list[AOIRecord]:
    return parse_yolo_label_lines(lines, image_shape, CLASS_NAMES, confidence=1.0)
