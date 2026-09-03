from __future__ import annotations

"""Attention metrics and visualization helpers.

Lightweight helpers to summarize attention scores and produce simple visual
outputs (heatmap generation is intentionally simple and testable).
"""

from typing import Dict, Any
import numpy as np

from analysis.aoi import AOIRecord, get_present_aoi_labels
from config.settings import BACKGROUND_LABEL


def summarize_attention(attention_scores: Dict[str, float]) -> Dict[str, Any]:
    """Return total attention and per-AOI percentage breakdown."""
    total = sum(attention_scores.values())
    if total <= 0:
        return {"total_ms": 0, "breakdown": {}, "percentages": {}}

    breakdown = dict(attention_scores)
    percentages = {k: (v / total) * 100.0 for k, v in breakdown.items()}

    return {"total_ms": total, "breakdown": breakdown, "percentages": percentages}


def calculate_attention_percentages(
    scores: dict[str, float],
    aoi_records: list[AOIRecord],
) -> dict[str, float]:
    labels = get_present_aoi_labels(aoi_records)
    report_labels = labels + [BACKGROUND_LABEL]
    total_attention = sum(scores.values())
    if total_attention == 0:
        return {label: 0.0 for label in report_labels}
    return {
        label: (scores.get(label, 0) / total_attention) * 100
        for label in report_labels
    }


def simple_heatmap(points, image_shape):
    """Create a simple heatmap array from gaze points.

    `points` expected as iterable of (x, y) coordinates in pixel space. Returns
    a NumPy 2D array shaped like image (height, width) with counts.
    """
    h, w = image_shape[:2]
    heat = np.zeros((h, w), dtype=np.uint32)
    for x, y in points:
        if x is None or y is None:
            continue
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= iy < h and 0 <= ix < w:
            heat[iy, ix] += 1

    return heat


def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    return max(minimum, min(maximum, value))
