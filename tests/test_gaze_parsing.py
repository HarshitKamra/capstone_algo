import os
from analysis.gaze import analyze_gaze_with_aoi


def test_gaze_parsing_basic(tmp_path):
    # Prepare sample paths
    repo_root = os.path.dirname(os.path.dirname(__file__))
    sample_gaze = os.path.join(repo_root, "data", "samples", "sample_gaze.csv")

    # Create a single AOI box (label, x1, y1, x2, y2) for a 960x640 poster
    # corresponds to the sample poster label: centered small box
    aoi_boxes = [("Product", 384, 256, 576, 384)]  # x1,y1,x2,y2

    image_shape = (640, 960)  # height, width

    result = analyze_gaze_with_aoi(sample_gaze, aoi_boxes, image_shape)

    assert "stats" in result
    assert "attention_scores" in result
