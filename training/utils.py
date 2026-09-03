from __future__ import annotations

from pathlib import Path
from typing import List
import json

from models.detector import PosterDetector, ModelNotFoundError
from analysis.detection import detections_to_aoi_records


def batch_inference(source_dir: Path, weights_path: Path | None, out_dir: Path, conf: float = 0.25) -> List[Path]:
    """Run inference over all images in a directory and write JSON results to out_dir.

    Returns list of output JSON paths.
    """
    detector = PosterDetector(weights_path=weights_path, conf_threshold=conf)
    if not detector.is_available:
        raise ModelNotFoundError("Model weights not available for batch inference.")

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for p in sorted(Path(source_dir).glob(ext)):
            detections = detector.predict(p, conf=conf)
            # attempt to read image to get shape for AOIs
            try:
                import cv2

                img = cv2.imread(str(p))
                shape = img.shape
            except Exception:
                shape = (0, 0, 3)
            records = detections_to_aoi_records(detections, shape)
            out_json = out_dir / f"{p.stem}.detections.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(detections, f, indent=2)
            results.append(out_json)
    return results
