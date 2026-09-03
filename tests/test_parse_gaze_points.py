import tempfile
import csv
from analysis.gaze import parse_gaze_points


def test_parse_gaze_points_basic():
    headers = ["timestamp_ms", "mapped gaze point x", "mapped gaze point y"]
    rows = [
        {"timestamp_ms": "0", "mapped gaze point x": "0.5", "mapped gaze point y": "0.5"},
        {"timestamp_ms": "100", "mapped gaze point x": "0.1", "mapped gaze point y": "0.2"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv", newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        tmp_path = tmp.name

    points = parse_gaze_points(tmp_path, (600, 800))
    assert len(points) == 2
    x0, y0, t0 = points[0]
    assert t0 == 0 or t0 == 0.0
    assert 0 <= x0 <= 800
    assert 0 <= y0 <= 600
