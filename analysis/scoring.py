from __future__ import annotations

from typing import Dict, List

from analysis.aoi import AOIRecord, get_present_aoi_labels


LABEL_WEIGHTS = {
    "CTA": 0.4,
    "Product": 0.35,
    "Headline": 0.15,
    "Price": 0.05,
    "logo": -0.1,
}


def score_attention(attention_percentages: Dict[str, float]) -> float:
    """Compute a 0-100 attention score using label weights and attention percentages."""
    score = 0.0
    for label, pct in attention_percentages.items():
        weight = LABEL_WEIGHTS.get(label, 0.0)
        score += weight * pct

    # map expected range roughly to 0-100, clamp
    val = max(0.0, min(100.0, score))
    return float(val)


def presence_score(records: List[AOIRecord]) -> float:
    labels = set(get_present_aoi_labels(records))
    # simple heuristic: require Product and CTA for good ad
    score = 0.0
    if "Product" in labels:
        score += 50.0
    if "CTA" in labels:
        score += 50.0
    return float(score)


def score_poster(records: List[AOIRecord], attention_percentages: Dict[str, float]) -> Dict[str, float]:
    att = score_attention(attention_percentages)
    pres = presence_score(records)
    overall = 0.7 * att + 0.3 * pres
    overall = max(0.0, min(100.0, overall))
    return {"attention_score": att, "presence_score": pres, "overall_score": overall}


def calculate_pes(
    percentages: dict[str, float],
    aoi_records: list[AOIRecord],
) -> dict[str, Any]:
    component_scores = {
        "Product Attention": score_for_ideal_range(
            percentages.get("Product", 0), *IDEAL_ATTENTION_RANGES["Product"]
        ),
        "CTA Visibility": score_for_ideal_range(
            percentages.get("CTA", 0), *IDEAL_ATTENTION_RANGES["CTA"]
        ),
        "Headline Engagement": score_for_ideal_range(
            percentages.get("Headline", 0), *IDEAL_ATTENTION_RANGES["Headline"]
        ),
        "Attention Balance": calculate_balance_score(percentages, aoi_records),
        "Visual Hierarchy": calculate_hierarchy_score(percentages, aoi_records),
    }

    score = sum(component_scores[name] * weight / 100 for name, weight in PES_WEIGHTS.items())

    return {
        "score": clamp(score),
        "category": get_pes_category(score),
        "components": component_scores,
        "weights": PES_WEIGHTS,
        "insights": build_design_insights(percentages, aoi_records, component_scores),
    }
