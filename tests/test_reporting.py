from analysis.reporting import build_structured_report
from analysis.aoi import build_aoi_record


def make_record(class_id, cx, cy, w, h):
    return build_aoi_record(class_id, cx, cy, w, h, 800, 600, confidence=0.9)


def test_build_report_basic():
    rec = make_record(3, 0.4, 0.5, 0.2, 0.2)
    attention = {"Product": 40.0, "CTA": 10.0}
    rpt = build_structured_report([rec], attention)
    assert "scores" in rpt
    assert "summary" in rpt
    assert isinstance(rpt["records"], list)
