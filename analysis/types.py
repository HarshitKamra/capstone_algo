"""Shared lightweight dataclasses for analysis outputs."""
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    area: float
    normalized_area: float
    center_x: float
    center_y: float
    rel_center_x: float
    rel_center_y: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
