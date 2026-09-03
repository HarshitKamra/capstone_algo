import tempfile
from pathlib import Path
import numpy as np

from training.utils import batch_inference


class FakeDetector:
    def __init__(self, weights_path=None, conf_threshold=0.25):
        self.is_available = True

    def predict(self, image_path, conf=0.25):
        return [
            {"class_id": 3, "class_name": "Product", "confidence": 0.8, "x1": 10, "y1": 10, "x2": 100, "y2": 100}
        ]


def test_batch_inference_monkeypatch(monkeypatch, tmp_path):
    # create two dummy images
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    import cv2

    cv2.imwrite(str(img1), np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.imwrite(str(img2), np.zeros((120, 80, 3), dtype=np.uint8))

    monkeypatch.setattr("training.utils.PosterDetector", FakeDetector)

    out = tmp_path / "out"
    res = batch_inference(tmp_path, None, out)
    assert len(res) >= 2
    for p in res:
        assert p.exists()
