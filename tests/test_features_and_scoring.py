import numpy as np

from analysis.aoi import build_aoi_record
from analysis.features import aggregate_features
from analysis.scoring import score_poster


def make_record(class_id, cx_norm, cy_norm, w_norm, h_norm):
    return build_aoi_record(class_id, cx_norm, cy_norm, w_norm, h_norm, 800, 600, confidence=0.9)


def test_features_and_scoring_happy_path():
    # Create Product and CTA centered elements
    prod = make_record(3, 0.35, 0.5, 0.2, 0.3)
    cta = make_record(0, 0.6, 0.55, 0.15, 0.1)
    records = [prod, cta]

    feats = aggregate_features(records, (600, 800))
    assert "Product" in feats
    assert "CTA" in feats

    attention = {"CTA": 40.0, "Product": 40.0, "Headline": 10.0, "logo": 10.0}
    scored = score_poster(records, attention)
    assert scored["overall_score"] >= 0 and scored["overall_score"] <= 100
    assert scored["presence_score"] == 100.0


def test_scoring_no_attention():
    prod = make_record(3, 0.35, 0.5, 0.2, 0.3)
    records = [prod]
    attention = {"CTA": 0.0, "Product": 0.0}
    scored = score_poster(records, attention)
    assert scored["attention_score"] == 0.0
