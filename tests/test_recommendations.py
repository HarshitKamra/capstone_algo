from analysis.recommendations import build_recommendations
from analysis.aoi import build_aoi_record


def make_record(class_id, cx, cy, w, h):
    return build_aoi_record(class_id, cx, cy, w, h, 800, 600, confidence=0.9)


def test_recommendations_basic():
    # product present but low attention; CTA missing
    prod = make_record(3, 0.4, 0.5, 0.2, 0.2)
    records = [prod]
    percentages = {"Product": 15.0, "CTA": 0.0, "Headline": 5.0, "BACKGROUND": 10.0}
    recs = build_recommendations(percentages, records, {"overall_score": 30})
    assert any("Increase the Product" in r or "Increase the Product size" in r or "Increase the Product size or centrality" in r for r in recs)
    assert any("Add a prominent CTA" in r or "Add a prominent CTA" in r for r in recs)
