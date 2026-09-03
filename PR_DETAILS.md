PR branch and commit instructions
================================

Branch name (suggested): feature/pr-ready

Recommended commit message (single commit):

Title: "feat: add analysis modules, UI, tests, CI, and deployment scaffolding"

Body:
- Add YOLO detector wrapper (`models/detector.py`) and high-level detection API (`analysis/detection.py`).
- Implement AOI representation and parsing (`analysis/aoi.py`) and gaze parsing/analysis (`analysis/gaze.py`).
- Add attention, feature extraction, scoring, reporting, recommendations, and LLM-friendly templates (`analysis/*.py`).
- Add Streamlit UI at `app/app.py` with gaze controls and heatmap overlay.
- Add training/inference scaffolds (`training/*.py`) and batch inference utils.
- Add Dockerfile, docker-compose files, and GitHub Actions CI workflow.
- Add tests covering parsing, detection integration, features, scoring, reporting, recommendations, and batch inference.
- Add ROADMAP, RELEASE notes, and README updates.

Files changed (high level):
 - models/detector.py
 - analysis/*.py (detection, aoi, gaze, attention, features, scoring, reporting, recommendations, llm_templates)
 - app/app.py
 - training/*.py (train.py, inference.py, utils.py)
 - tests/* (multiple tests)
 - Dockerfile, docker-compose.yml, docker-compose.prod.yml
 - .github/workflows/ci.yml
 - README.md, ROADMAP.md, RELEASE.md, PROJECT_ARCHITECTURE.md

Smoke tests status (local):
 - `python -m pytest -q` → all tests pass (9 passed at time of commit)

Suggested git commands:

```bash
# create branch
git checkout -b feature/pr-ready

# add all changes (or selectively add files)
git add -A

# commit
git commit -m "feat: add analysis modules, UI, tests, CI, and deployment scaffolding"

# push branch
git push -u origin feature/pr-ready

# Then open a PR on GitHub with the same title and the body above.
```

Notes:
- If you want the history split into logical commits, consider squashing related changes (e.g., analysis modules, UI, tests) into separate commits instead of one large commit.
- Do NOT commit model weights; keep them referenced via `MODEL_WEIGHTS` or external artifact storage.
