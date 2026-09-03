from __future__ import annotations

from typing import Dict, List
import math

from analysis.aoi import AOIRecord, build_aoi_record


def aoi_feature_vector(record: AOIRecord, image_shape: tuple[int, int]) -> Dict[str, float]:
    """Compute basic features for a single AOIRecord."""
    h, w = image_shape[:2]
    area_pct = record.normalized_area

    # distance from center (0=center, 1=corner)
    center_x = record.center_x
    center_y = record.center_y
    dx = abs(center_x - w / 2) / (w / 2)
    dy = abs(center_y - h / 2) / (h / 2)
    dist = math.sqrt(dx * dx + dy * dy) / math.sqrt(2)
    center_score = max(0.0, 1.0 - dist)

    return {
        "area_pct": float(area_pct),
        "center_score": float(center_score),
        "confidence": float(record.confidence),
    }


def aggregate_features(records: List[AOIRecord], image_shape: tuple[int, int]) -> Dict[str, Dict[str, float]]:
    """Aggregate features by AOI class name.

    Returns a mapping from class_name -> aggregated features like total_area,
    mean_center_score, count, max_confidence.
    """
    groups: dict[str, list[AOIRecord]] = {}
    for r in records:
        groups.setdefault(r.class_name, []).append(r)

    out: dict[str, dict[str, float]] = {}
    for name, items in groups.items():
        feats = [aoi_feature_vector(r, image_shape) for r in items]
        total_area = sum(f["area_pct"] for f in feats)
        mean_center = sum(f["center_score"] for f in feats) / max(1, len(feats))
        max_conf = max(f["confidence"] for f in feats)
        out[name] = {
            "total_area_pct": float(total_area),
            "mean_center_score": float(mean_center),
            "count": float(len(items)),
            "max_confidence": float(max_conf),
        }

    return out
