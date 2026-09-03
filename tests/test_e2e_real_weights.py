import os
from pathlib import Path

import pytest
import cv2

from analysis.detection import detect_poster_elements


MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS")
SAMPLE_DIR = Path("Capstone.yolov8/train/images")


@pytest.mark.skipif(not MODEL_WEIGHTS or not Path(MODEL_WEIGHTS).exists(), reason="No MODEL_WEIGHTS provided or file not found")
def test_e2e_with_real_weights():
    imgs = list(SAMPLE_DIR.glob("*"))
    assert imgs, f"No sample images found in {SAMPLE_DIR}"
    img_path = imgs[0]
    img = cv2.imread(str(img_path))
    assert img is not None, f"Failed to read sample image {img_path}"

    records, source = detect_poster_elements(img, image_path=None, weights_path=MODEL_WEIGHTS, conf=0.25)
    assert isinstance(records, list)
    # basic sanity: records items should have to_dict method
    if records:
        assert hasattr(records[0], "to_dict")
