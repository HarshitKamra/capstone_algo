from __future__ import annotations

from typing import Dict, Any, List
import json

from analysis.aoi import AOIRecord
from analysis.scoring import score_poster
from analysis.attention import summarize_attention


def build_structured_report(
    records: List[AOIRecord], attention_scores: Dict[str, float]
) -> Dict[str, Any]:
    """Return a machine-friendly report dict with metrics and findings."""
    attention_summary = summarize_attention(attention_scores)
    percentages = attention_summary.get("percentages", {})

    scores = score_poster(records, percentages)

    findings: List[str] = []
    if scores["overall_score"] < 50:
        findings.append("Poster needs design improvements to improve attention and presence.")
    else:
        findings.append("Poster shows adequate attention and component presence.")

    report = {
        "summary": attention_summary,
        "scores": scores,
        "records": [r.to_dict() for r in records],
        "findings": findings,
    }
    return report


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2)
