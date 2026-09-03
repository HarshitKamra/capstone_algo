from __future__ import annotations

from typing import Dict, Any, List


def format_report_for_llm(report: Dict[str, Any], recommendations: List[str]) -> str:
    """Return a concise LLM-friendly text template summarizing the report.

    This function only formats text; it does not call any external model.
    """
    summary = report.get("summary", {})
    scores = report.get("scores", {})
    records = report.get("records", [])

    lines: List[str] = []
    lines.append("Poster Analysis Summary:")
    lines.append(f"- Total attention (ms): {summary.get('total_ms', 0)}")
    lines.append("- Scores:")
    for k, v in scores.items():
        lines.append(f"  - {k}: {v}")
    lines.append(f"- Detected elements: {', '.join([r.get('class_name','') for r in records])}")
    lines.append("")
    lines.append("Recommendations:")
    for rec in recommendations:
        lines.append(f"- {rec}")

    lines.append("")
    lines.append("Notes: Provide these structured findings as JSON and ask the LLM to expand each recommendation into a short actionable paragraph if human-readable text is required.")

    return "\n".join(lines)
