# Roadmap & Effort Estimates

This document summarizes missing features, blockers, a phased roadmap, milestones, and rough effort estimates to complete the project into a production-ready poster analysis platform.

## Current State (baseline)
- Core analysis modules implemented: `analysis/` (detection, gaze, attention, features, scoring, reporting).
- Model wrapper: `models/detector.py` with graceful handling when weights are missing.
- Streamlit UI scaffold at `app/app.py`.
- Training/inference scaffolds under `training/`.
- Tests covering parsing, detection integration, features, scoring, and reporting.

## Missing Features & Blockers
- Trained YOLOv8 weights (external) — required for full model inference.
- End-to-end integration test using a real detector + sample images.
- Gaze-to-AOI export and UI controls (playback, smoothing, time filters).
- More robust scoring configuration (external YAML) and calibration.
- LLM integration (optional) limited to explanation/formatting only.
- Deployment artifacts (Docker Compose, production config, secrets management).

## Phased Roadmap

Phase 1 — Stabilize core analysis (Done)
- Validate detection integration with model weights (if available).
- Add more unit tests and fixture data.
- Deliverables: green CI tests, stable `analysis` API.
Estimated effort: 1–2 days (mostly verification + small fixes).

Phase 2 — Robust inference & labeling fallbacks
- Add deterministic mocked detector for CI, add integration tests that exercise prediction pipeline.
- Implement dataset utilities for labels and batch inference.
- Deliverables: `training/inference.py` improvements, label utilities.
Estimated effort: 2–3 days.

Phase 3 — Gaze UX and metrics
- Enhance `app/app.py` with gaze overlay controls, time-range slicing, heatmap visualization, and CSV export.
- Add attention calibration tools and unit tests for `analysis/gaze.py` and `analysis/attention.py`.
- Deliverables: interactive UI controls, export features, more tests.
Estimated effort: 3–5 days.

Phase 4 — Scoring, recommendations, and report templates
- Externalize scoring config into `config/scoring.yaml` and support rule edits.
- Implement deterministic recommendation engine (rule-based) and LLM output templates (LLM only formats findings; no generation required at core).
- Deliverables: `config/scoring.yaml`, `analysis/recommendations.py`, `analysis/reporting.py` enhancements.
Estimated effort: 3–4 days.

Phase 5 — UI polish & deployment
- Improve UI styling, add export (JSON/CSV/PDF), enable Docker-based deployment and optional cloud deployment docs.
- Add GitHub Actions to build Docker image and run smoke tests; add Docker Compose for local stacks.
- Deliverables: polished Streamlit app, Docker Compose, deployment docs.
Estimated effort: 2–4 days.

Phase 6 — Production readiness & monitoring
- Add logging, error reporting, usage metrics, and unit/integration test coverage target (>=80%).
- Deliverables: monitoring docs, alerts, CI gates.
Estimated effort: 3–5 days.

## Risks & Mitigations
- Missing model weights: provide clear instructions and a small synthetic fixture for QA.
- Gaze data variability: provide calibration UI and default fallbacks for missing fixation durations.
- LLM misuse: keep LLM role limited to formatting and explanation; keep quantitative results separate.

## Next Immediate Steps (I'll do now)
1. Produce a short deployable Docker Compose and CI job to build the image (optional).
2. Add gaze overlay UI controls in `app/app.py` (play/pause, smoothing, export).
3. Start Phase 3 work: implement playback controls and export (ETA 3–5 days).

If you want, I will start implementing the gaze UI controls next — tell me to proceed or change priority.
