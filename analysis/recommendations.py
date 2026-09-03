from __future__ import annotations

from typing import Dict, List, Any

from analysis.aoi import get_present_aoi_labels


def build_recommendations(percentages: Dict[str, float], aoi_records: list[Any], scores: Dict[str, float]) -> List[str]:
    """Return a list of rule-based recommendations given attention percentages and AOIs."""
    labels = set(get_present_aoi_labels(aoi_records))
    recs: List[str] = []

    background = percentages.get("BACKGROUND", percentages.get("Background", 0) if isinstance(percentages, dict) else 0)

    if "Product" not in labels:
        recs.append("Add a clear Product image as a primary AOI to communicate the item.")
    else:
        prod_pct = percentages.get("Product", 0)
        if prod_pct < 25:
            recs.append("Increase the Product size or centrality to attract more attention.")
        elif prod_pct > 55:
            recs.append("Product dominates attention; consider increasing support content (CTA/headline).")

    if "CTA" not in labels:
        recs.append("Add a prominent CTA (action) with clear contrast and wording.")
    else:
        cta_pct = percentages.get("CTA", 0)
        if cta_pct < 8:
            recs.append("Increase CTA contrast, size, or placement to improve click-through.")

    if "Headline" not in labels:
        recs.append("Add a short, readable Headline to explain the offering.")
    else:
        head_pct = percentages.get("Headline", 0)
        if head_pct < 10:
            recs.append("Improve Headline readability or placement to boost engagement.")

    if background > 25:
        recs.append("Reduce background clutter or emphasize AOIs to bring focus back to content.")

    # Use presence/overall scores to suggest design-level changes
    overall = scores.get("overall_score", 0)
    if overall < 40:
        recs.append("Consider a design iteration: rebalance elements, improve CTA visibility, and simplify background.")
    elif overall < 70:
        recs.append("Minor adjustments suggested: tweak CTA and headline contrast and placement.")
    else:
        recs.append("Poster performs well; consider A/B testing small variants to optimize conversions.")

    return recs
