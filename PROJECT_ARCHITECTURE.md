# Project Architecture & Repository Audit

This document records the Phase 0 repository audit, a summary of existing functionality, reuseable components, identified gaps, blockers, proposed target architecture, and an implementation roadmap aligned with the capstone product vision.

## 1) Snapshot / Inventory
- Root files of interest:
  - `aoi_visualizer.py` — main poster AOI loader, YOLO label parser, Tobii export parser, attention scoring, WebGazer generator and CLI.
  - `webgazer_choose_poster.html` — polished WebGazer-based capture UI (browser-side).
  - `requirements.txt` — currently: `opencv-python`, `matplotlib`, `pillow`, `numpy`.
  - `Capstone.yolov8/` — dataset in YOLOv8 format (images, labels, `data.yaml`, `README.roboflow.txt`).
  - `.gitignore` exists.

## 2) Existing functionality (what's implemented)
- Poster image reading and robust fallback for AVIF (OpenCV + Pillow) (`read_image`).
- Listing posters, interactive poster selection, and non-interactive poster selection (`load_poster`).
- YOLO label file loading and conversion to pixel bounding boxes and drawing (`load_labels`, `draw_aoi_boxes`).
- Tobii export parsing logic: delimiter detection, column-finding heuristics, coordinate normalization, duration & timestamp parsing, AOI-hit detection, per-row event building, iris-quality filtering, and a pipeline to accumulate attention scores (`analyze_raw_gaze_file`, many helper functions).
- WebGazer HTML UI to capture browser gaze data and export CSV for local analysis.

Files implementing this are primarily `aoi_visualizer.py` and `webgazer_choose_poster.html`.

## 3) Reusable components
- Gaze/Tobii parsing utilities (column detection, coordinate normalization).
- AOI pixel conversion and drawing functions.
- WebGazer page for capture — can be reused or embedded in a web UI.

## 4) Incomplete / missing components
- Inference wrapper (YOLOv8 detect API) — missing `models/detector.py` or `training/inference.py`.
- Training script to recreate or fine-tune the model (`training/train.py`).
- Model weights / checkpoints — not present in repo.
- Standardized AOI/internal detection representation (JSON schema or class) — currently ad-hoc tuples/lists.
- Poster feature extraction modules (layout, color, text/OCR, CTA analyzer) separated into `analysis/` submodules.
- Recommendation engine (`recommendations/`) and configuration for scoring weights.
- Target-market structured input and mapping logic.
- Web app (Streamlit / Flask) to host upload → questionnaire → analyze → report.
- Tests and CI.
- Sample Tobii / WebGazer CSV examples and a minimal sample poster+label pair for deterministic tests.

## 5) Broken code / risk areas
- The main code in `aoi_visualizer.py` is large (monolithic). Risk of regressions if refactored without tests.
- Some functions assume specific fieldnames or formats; robust heuristics exist but need unit tests with sample files.
- No explicit error handling for missing dependencies like `pandas` (used often for CSVs) or `ultralytics` (YOLOv8) — these are not present.

## 6) Missing dependencies & recommended additions
- Additions likely needed:
  - `pandas` — robust CSV handling and tests
  - `ultralytics` — YOLOv8 inference/training wrapper (or `yolov8` official package)
  - `streamlit` (or `flask`) — web UI prototype
  - `pytesseract` OR `easyocr` — OCR for text-region estimation (optional)
  - `scikit-image` / `scikit-learn` — color analysis utilities (optional)
  - `pytest` — testing

Do not add these until required by implemented features; prefer minimal additions.

## 7) Model & data blockers
- No trained model weights in the repository. According to project rules: do not fabricate or auto-download arbitrary weights.
- If the user has trained weights, provide a configuration option `MODEL_WEIGHTS_PATH` (env var or config file) and document how to add them.
- The dataset exists (`Capstone.yolov8`) but appears small (Roboflow says 74 images). Training from scratch may be limited by dataset size.

## 8) Proposed target architecture (adapted to repo)
capstone/
├── app/                # Streamlit / Flask app
│   └── app.py
├── analysis/           # modular analysis code (layout, color, text, scoring)
│   ├── poster.py
│   ├── detection.py
│   ├── aoi.py
│   ├── gaze.py
│   ├── attention.py
│   ├── layout.py
│   ├── color.py
│   ├── text.py
│   └── scoring.py
├── recommendations/
│   ├── rules.py
│   ├── engine.py
│   └── llm.py
├── models/
│   ├── detector.py
│   └── weights/         # not committed; path configured in .env
├── training/
│   ├── train.py
│   └── inference.py
├── data/
│   ├── posters/
│   ├── gaze/
│   └── samples/         # deterministic fixtures for tests
├── tests/
├── Capstone.yolov8/
├── webgazer_choose_poster.html
├── requirements.txt
├── README.md
└── PROJECT_ARCHITECTURE.md

Notes: adapt names to existing `aoi_visualizer.py` and `Capstone.yolov8` rather than deleting them. Initially keep `aoi_visualizer.py` but move refactored code into `analysis/` modules incrementally.

## 9) File mapping (what to create / modify)
- Create `models/detector.py` — abstraction for loading weights and running inference. Public API: `detect(pil_or_cv_image) -> List[Detection]` where `Detection` is a dataclass with class, confidence, bbox, area, normalized area, center, relative position.
- Create `analysis/poster.py` — poster loader, image normalization helpers.
- Create `analysis/detection.py` — conversion between raw model outputs and internal AOI schema; utility to save/load detections as JSON.
- Move gaze parsing helper functions from `aoi_visualizer.py` into `analysis/gaze.py` and keep `aoi_visualizer.py` as a thin CLI delegating to new modules until refactor complete.
- Create `analysis/attention.py` — fixation/dwell/AOI assignment and attention metrics and visualizations (heatmap generator).
- Create `training/train.py` and `training/inference.py` — wrappers around `ultralytics` training/inference using `Capstone.yolov8/data.yaml`.
- Create `recommendations/engine.py` and `recommendations/rules.py` — scoring config & explicit rules that map measured features to recommendation artifacts.
- Add `data/samples/` — poster image + YOLO label + small gaze CSV fixture for tests.
- Add `tests/` with unit tests for CSV parsing, coordinate normalization, AOI assignment, and one smoke test.
- Add `README.md` with quickstart and `docs/` or expand `PROJECT_ARCHITECTURE.md` if desired.

## 10) Implementation order (phase-by-phase incremental plan)
Phase 0 — Audit (complete): inventory and PROJECT_ARCHITECTURE.md.

Phase 1 — Stabilize inference & detections (high priority)
- Task 1.1: Add `models/detector.py` with config to point to `MODEL_WEIGHTS_PATH` (env var), implement a safe no-weights mode that returns an explanatory error.
- Task 1.2: Create `analysis/detection.py` and define `Detection` dataclass and JSON schema.
- Task 1.3: Break out `aoi_visualizer.py` helpers into `analysis/gaze.py` and `analysis/aoi.py` (incremental—keep original file working).

Phase 2 — Poster feature extraction and scoring
- Implement `analysis/layout.py`, `analysis/color.py`, `analysis/text.py` (OCR optional), and `analysis/scoring.py` with a central scoring config.

Phase 3 — Gaze & attention pipeline
- Harden `analysis/gaze.py` and `analysis/attention.py`, add visualizations and sample gaze files.

Phase 4 — Recommendations & LLM integration
- Implement `recommendations/engine.py` and `recommendations/llm.py` (LLM optional; must be used only to format structured findings into human text; configurable API keys via env). Keep LLM optional/fallback.

Phase 5 — Web App & UX
- Implement minimal Streamlit app `app/app.py` for upload → questionnaire → analyze → download report.

Phase 6 — Training, tests, CI, and deployment
- Add `training/*`, tests, GitHub Actions CI, and a deployment guide + Dockerfile.

## 11) Short-term concrete next steps (first 2 days)
1. Add `models/detector.py` (inference wrapper) + config for `MODEL_WEIGHTS_PATH` (priority: HIGH; estimate 6–10 hours).
2. Refactor a small portion of `aoi_visualizer.py` into `analysis/gaze.py` and `analysis/detection.py` leaving the CLI working (priority: HIGH; estimate 6 hours).
3. Add deterministic sample data into `data/samples/` (poster + label + gaze CSV) and a small `tests/test_gaze_parsing.py` (priority: HIGH; estimate 2–4 hours).

## 12) Blockers (must be resolved by user or documented)
- Model weights: add a trained YOLOv8 checkpoint and set `MODEL_WEIGHTS_PATH` in config/env.
- Ground-truth gaze exports for robust parser testing: add at least one Tobii CSV and one WebGazer CSV that matches the project's HTML exporter.
- Clarify whether you will use `ultralytics`/YOLOv8 or another detector.

## 13) Estimated effort (rough)
- Phase 1 (inference + refactor + samples + tests): 2–5 days.
- Phase 2 (feature extraction & scoring): 3–7 days.
- Phase 3 (gaze/attention full): 2–5 days.
- Phase 4 (recommendations + LLM formatting): 2–4 days.
- Phase 5 (web UI + polishing): 3–7 days.

Total rough: 3–4 weeks for MVP with production-quality tests and documentation.

---
If you confirm, I will start Phase 1: create `models/detector.py`, an inference wrapper, and add a small deterministic sample in `data/samples/` plus a basic unit test scaffold. If you prefer a different next step, tell me which one.
# PROJECT ARCHITECTURE — AI Advertisement Analysis Platform

**Document version:** Phase 0 audit (2026-09-03)  
**Repository:** `capstone_algo`  
**Status:** Early-stage research prototype → production platform (planned)

---

## 1. Executive Summary

The repository contains a **working offline prototype** for poster Areas-of-Interest (AOI) visualization, Tobii/WebGazer gaze import, and a **Poster Effectiveness Score (PES)** based on empirical attention data. It does **not** yet include YOLO inference, a web application, target-market analysis, a recommendation engine, LLM integration, training/inference scripts, tests, or deployment configuration.

The highest-value existing assets are:

1. **`aoi_visualizer.py`** — monolithic but functional CLI for gaze → AOI → PES pipeline  
2. **`webgazer_choose_poster.html`** — polished browser gaze-collection UI (GazeLab)  
3. **`Capstone.yolov8/`** — 74 annotated food/beverage advertisement posters (YOLO format)

The critical blockers for the full product vision are: **no trained model weights**, **no YOLO inference layer**, **no modular package structure**, and **no web UI**.

---

## 2. Current Repository Inventory

### 2.1 File tree (actual)

```
capstone_algo/
├── aoi_visualizer.py              # ~2,600 lines — main Python application (CLI)
├── webgazer_choose_poster.html    # Pre-generated GazeLab gallery (74 posters)
├── requirements.txt               # Minimal: opencv, matplotlib, pillow, numpy
├── .gitignore
├── Capstone.yolov8.zip            # Dataset archive (~50 MB)
├── Capstone.yolov8/
│   ├── data.yaml                  # YOLO dataset config (5 classes)
│   ├── README.roboflow.txt        # Roboflow export metadata
│   └── train/
│       ├── images/                # 74 poster images
│       └── labels/                # 74 YOLO label files
└── PROJECT_ARCHITECTURE.md        # This document
```

### 2.2 What is missing today

| Component | Status |
|-----------|--------|
| `README.md` | Missing |
| `app/` (Streamlit/web UI) | Missing |
| `analysis/`, `recommendations/`, `models/`, `training/` | Missing |
| `tests/` | Missing |
| YOLO model weights (`.pt`) | Missing |
| `valid/` and `test/` dataset splits | Missing (referenced in `data.yaml` but not present) |
| Sample gaze CSV/TSV in repo | Missing (gitignored) |
| Training script | Missing |
| Inference wrapper | Missing |
| Heatmap generation | **Claimed in UI text, not implemented** |
| Target-market questionnaire | Missing |
| Recommendation engine | Missing (only simple `build_design_insights`) |
| LLM integration | Missing |
| Deployment config (Docker, Streamlit Cloud, etc.) | Missing |

---

## 3. Dataset — `Capstone.yolov8`

### 3.1 Source

- Exported from **Roboflow** on 2026-05-08  
- **74 images**, YOLOv8 bounding-box format  
- Domain: **food & beverage advertisement posters** (India/global brands, flyers, social posts)

### 3.2 Classes (`data.yaml`)

| ID | Class name | Instance count (approx.) |
|----|------------|---------------------------|
| 0 | CTA | 32 |
| 1 | Headline | 68 |
| 2 | Price | 34 |
| 3 | Product | 80 |
| 4 | logo | 61 |

**Total bounding boxes:** ~275 across 74 images (2 files unreadable on Windows — see §6.3).

### 3.3 Annotation coverage notes

- **43 of 72 readable posters** have **no CTA** annotated (CTA absence is common in dataset, not necessarily in poster)
- Multiple instances per class are common (e.g., multiple Price boxes, multiple logo boxes)
- **7 bounding boxes** extend outside normalized `[0, 1]` coordinates (annotation noise)
- Some posters have **overlapping / oversized boxes** (e.g., logo covering large regions)

### 3.4 Dataset split problem

`data.yaml` declares:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
```

Only `train/` exists. Before training, either:

- Create stratified train/val/test splits from the 74 images, or  
- Update `data.yaml` to reflect a single split with manual holdout

### 3.5 Supported image formats

Code supports: `.jpg`, `.jpeg`, `.png`, `.webp`, `.avif` (AVIF via Pillow fallback)

---

## 4. Existing Functionality (Detailed)

### 4.1 `aoi_visualizer.py` — CLI monolith

Single file containing **all** current backend logic. Major capability groups:

#### A. Poster & AOI loading (uses **ground-truth labels**, not YOLO inference)

| Function | Purpose |
|----------|---------|
| `list_available_posters` | Scan dataset image folder |
| `load_poster` | Interactive or `--poster` selection |
| `load_labels` | Load matching YOLO `.txt` label file |
| `draw_aoi_boxes` | Convert YOLO normalized coords → pixel boxes + matplotlib preview |

**Important:** AOIs come from **dataset annotation files**, not from a trained detector. Uploading a new poster without a label file will not work today.

#### B. Tobii gaze import (production-quality parser)

| Function | Purpose |
|----------|---------|
| `detect_delimiter` | Auto-detect TSV/CSV/semicolon |
| `find_coordinate_columns` | Heuristic column detection for many Tobii export formats |
| `find_aoi_hit_column` / `find_aoi_indicator_columns` | AOI-hit exports |
| `raw_coordinates_to_image_coordinates` | pixel / normalized / percent / auto |
| `analyze_raw_gaze_file` | Full pipeline: read export → filter fixations → accumulate dwell time |
| `row_passes_iris_quality_filter` | WebGazer iris quality gate |

Supports:

- Tobii Pro Lab / Glasses exports (TSV/CSV)  
- WebGazer CSV exports (via compatible column names)  
- Optional `--stimulus-filter` for multi-poster exports  
- Fixation-only filtering when eye-movement-type column exists

#### C. Attention metrics

| Function | Purpose |
|----------|---------|
| `get_matched_aoi_labels` | Point-in-rectangle AOI assignment |
| `add_attention` / `apply_raw_event_attention` | Dwell time accumulation |
| `calculate_attention_percentages` | % attention per AOI + Background |
| `calculate_balance_score` | Penalize dominance & high background attention |
| `calculate_hierarchy_score` | Rank-based Product > Headline/CTA heuristic |

#### D. Poster Effectiveness Score (PES)

| Function | Purpose |
|----------|---------|
| `score_for_ideal_range` | Compare attention % to ideal ranges |
| `calculate_pes` | Weighted composite score 0–100 |
| `build_design_insights` | Rule-based text recommendations |

**PES component weights** (hard-coded):

| Component | Weight |
|-----------|--------|
| Product Attention | 25% |
| CTA Visibility | 20% |
| Headline Engagement | 20% |
| Attention Balance | 15% |
| Visual Hierarchy | 20% |

**Ideal attention ranges** (hard-coded heuristics):

| AOI | Range (%) |
|-----|-----------|
| Product | 30–45 |
| Headline | 15–30 |
| CTA | 10–20 |
| Price | 5–15 |
| logo | 3–10 |

#### E. WebGazer HTML generation

| Function | Purpose |
|----------|---------|
| `build_webgazer_html` | Generate self-contained GazeLab page (~1,300 lines HTML/CSS/JS) |
| `write_webgazer_session` | Single-poster capture page (base64 embedded image) |
| `write_webgazer_gallery` | Multi-poster chooser (URL-referenced images) |

CLI flags:

```bash
python aoi_visualizer.py --webgazer-gallery webgazer_choose_poster.html
python aoi_visualizer.py --poster POSTER.jpg --webgazer-session session.html
python aoi_visualizer.py --poster POSTER.jpg --gaze-file export.csv --raw-coordinate-mode pixel
```

#### F. Global mutable state (refactor needed)

```python
aoi_boxes = []           # set per session
attention_scores = {}    # set per session
```

Not thread-safe or suitable for a web server without encapsulation.

### 4.2 `webgazer_choose_poster.html` — GazeLab

Pre-generated gallery including **74 posters** with embedded AOI metadata (JSON `POSTERS` array).

**Features (working):**

- Poster search + dropdown selection  
- WebGazer ridge regression + Kalman filter  
- 9-point calibration + 9-point accuracy validation  
- MediaPipe Face Mesh iris quality tracking  
- Gaze smoothing, jump rejection  
- Live AOI overlay on poster  
- CSV export: `timestamp_ms`, `gaze_x`, `gaze_y`, `duration_ms`, `aoi_hit`, `poster`, iris columns

**Requirements to run:**

- Serve repo root via HTTP (`python -m http.server`) — **not** `file://`  
- Webcam + browser permission  
- CDN access (WebGazer, MediaPipe, Google Fonts)

**Relationship to Python:** Exported CSV is consumed by `aoi_visualizer.py --gaze-file`.

### 4.3 Current end-to-end workflow (what works today)

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│ Dataset poster  │     │ GazeLab (browser)    │     │ aoi_visualizer.py   │
│ + YOLO labels   │────▶│ WebGazer capture     │────▶│ Tobii/WebGazer CSV  │
│ (ground truth)  │     │ → CSV export         │     │ → PES + CLI report  │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
```

**What does NOT work:** Upload arbitrary poster → detect elements → analyze without gaze → target market → web dashboard.

---

## 5. Architecture Classification (Current vs Target)

### 5.1 Current data flow

```
POSTER (from dataset)
    → YOLO LABEL FILE (ground truth, not inference)
    → AOI BOXES
    → [optional] GAZE CSV (Tobii or WebGazer)
    → ATTENTION METRICS
    → PES SCORE + rule-based insights
    → CLI print + matplotlib AOI preview
```

### 5.2 Target data flow (product vision)

```
POSTER (user upload)
    → COMPUTER VISION (YOLO inference)
    → FEATURE EXTRACTION (layout, color, text, branding, CTA)
    → AOI / ATTENTION ANALYSIS (gaze when available)
    → SCORING ENGINE (explainable, configurable)
    → TARGET MARKET ANALYSIS
    → RECOMMENDATION ENGINE (evidence-backed)
    → [optional] LLM (natural language only)
    → WEB UI + EXPORT
```

### 5.3 Separation principle (must preserve)

| Layer | Source of truth |
|-------|-----------------|
| Measurements & scores | Computer vision, gaze math, documented formulas |
| Recommendations | Rule engine with evidence objects |
| Explanations | LLM receives structured JSON; cannot alter numbers |

---

## 6. Problems & Technical Debt

### 6.1 Critical

1. **No YOLO inference** — system depends on pre-existing label files; cannot analyze uploaded posters  
2. **No model weights** — training required before detection works on new images  
3. **Monolithic `aoi_visualizer.py`** — ~2,600 lines; mixing CLI, gaze parsing, scoring, HTML generation  
4. **No web application** — CLI + standalone HTML only  
5. **`data.yaml` invalid splits** — `valid/` and `test/` directories missing  

### 6.2 Functional gaps

6. **Heatmaps not implemented** — UI text promises heatmaps; code only shows AOI bounding boxes  
7. **PES requires gaze data** — no CV-only scoring path for posters without eye tracking  
8. **No layout/color/text/OCR analysis** — not started  
9. **No target-market module** — not started  
10. **`build_design_insights` is not a full recommendation engine** — unstructured strings, no priority/evidence schema  
11. **Interactive CLI** — `input()` blocks automation and web integration  

### 6.3 Data / environment issues

12. **Windows long-path errors** — 2 label files fail to open under default Windows path limits  
13. **Duplicate WebGazer template** — HTML embedded in Python string AND standalone `webgazer_choose_poster.html` (regenerate drift risk)  
14. **Hard-coded relative paths** — `Capstone.yolov8/train/images` (acceptable for now, needs config for deployment)  
15. **No sample gaze files in repo** — `.csv`/`.tsv` gitignored; tests need fixtures  

### 6.4 Missing engineering infrastructure

16. No tests  
17. No logging configuration  
18. No environment variable config  
19. No README or run documentation  
20. `requirements.txt` missing: `ultralytics`, `streamlit`, OCR libs, test deps  

---

## 7. Reusable Components (Preserve & Extract)

These functions/modules should be **refactored out** of `aoi_visualizer.py`, not rewritten:

| Module (planned) | Source functions | Notes |
|------------------|------------------|-------|
| `analysis/aoi.py` | `draw_aoi_boxes`, `get_present_aoi_labels`, YOLO label parsing | Add standardized AOI dataclass |
| `analysis/gaze.py` | `detect_delimiter`, `find_*_column`, `parse_float`, `build_raw_event` | Excellent Tobii compatibility |
| `analysis/attention.py` | `add_attention`, `calculate_attention_percentages`, `calculate_balance_score` | |
| `analysis/scoring.py` | `calculate_pes`, `score_for_ideal_range`, `calculate_hierarchy_score` | Move weights to config |
| `recommendations/rules.py` | `build_design_insights` | Extend to structured recommendations |
| `app/static/gaze/` | `build_webgazer_html` JS/CSS | Consider template file vs f-string |
| `data/samples/` | Dataset posters + labels | Use for tests & demos |

---

## 8. Proposed Target Structure

Adapted to this repository (not a blind scaffold):

```
capstone_algo/
├── app/
│   ├── app.py                      # Streamlit entrypoint
│   ├── pages/                      # upload, questionnaire, results
│   └── components/                 # score cards, AOI overlay, export
├── analysis/
│   ├── poster.py                   # image load, validation
│   ├── detection.py                # YOLO inference wrapper
│   ├── aoi.py                      # AOI dataclass + label parsing
│   ├── gaze.py                     # Tobii/WebGazer import (from existing)
│   ├── attention.py                # dwell, percentages, rankings
│   ├── layout.py                   # NEW: spacing, density, hierarchy
│   ├── color.py                    # NEW: dominant colors, contrast
│   ├── text.py                     # NEW: OCR regions
│   ├── scoring.py                  # PES + extended scores + config
│   └── target_market.py            # NEW: audience profile + rules
├── recommendations/
│   ├── rules.py                    # measurable thresholds
│   ├── engine.py                   # priority, evidence, categories
│   └── llm.py                      # optional narrative layer
├── models/
│   ├── detector.py                 # Ultralytics YOLO wrapper
│   └── weights/                    # .gitignore; README for obtaining weights
├── training/
│   ├── train.py
│   └── inference.py
├── data/
│   ├── posters/                    # symlinks or copies for samples
│   ├── gaze/                       # sample CSV fixtures (un-gitignore subset)
│   └── samples/
├── tests/
├── config/
│   ├── scoring.yaml                # PES weights, ideal ranges
│   └── target_market.yaml          # audience rules (heuristic, documented)
├── Capstone.yolov8/                # unchanged dataset location
├── webgazer_choose_poster.html     # keep; later move to app/static/
├── aoi_visualizer.py               # keep as CLI shim during migration
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── PROJECT_ARCHITECTURE.md
└── .env.example
```

### 8.1 Standardized AOI representation (planned)

Every detection (from labels or YOLO) should map to:

```python
{
    "class": "CTA",
    "confidence": 0.87,          # 1.0 for ground-truth labels
    "x1": 34, "y1": 384, "x2": 161, "y2": 403,
    "width": 127, "height": 19,
    "area": 2413,
    "normalized_area": 0.0095,
    "center_x": 97.5, "center_y": 393.5,
    "relative_position": "bottom-left"   # derived
}
```

### 8.2 Standardized recommendation (planned)

```python
{
    "category": "CTA",
    "issue": "Low CTA prominence",
    "evidence": "CTA normalized area 0.95%; attention 4.2% (ideal 10–20%)",
    "recommendation": "Increase CTA size and contrast in lower third",
    "priority": "HIGH",
    "expected_impact": "Improved action visibility",
    "confidence": "medium",
    "source": "rule"   # rule | heuristic | gaze
}
```

---

## 9. Dependencies

### 9.1 Current (`requirements.txt`)

```
opencv-python
matplotlib
pillow
numpy
```

### 9.2 Required additions (by phase)

| Phase | Packages |
|-------|----------|
| 1 — YOLO | `ultralytics`, `torch` (CPU or CUDA) |
| 2 — Features | `scikit-image` or `colorspacious`, `pytesseract` or `easyocr` (optional) |
| 9 — Web UI | `streamlit`, `plotly` or `altair` |
| 8 — LLM | `openai` or `anthropic` (optional, env-gated) |
| 11 — Tests | `pytest`, `pytest-cov` |
| 13 — Deploy | `gunicorn` N/A for Streamlit; use `streamlit` + platform config |

**Principle:** Add dependencies incrementally per phase; do not install everything upfront.

---

## 10. Blockers & Prerequisites

| Blocker | Impact | Resolution |
|---------|--------|------------|
| **No `best.pt` / trained weights** | Cannot detect AOIs on new uploads | Run `training/train.py` on dataset; document weight path via `MODEL_WEIGHTS` env var |
| **Only 74 training images** | Model may generalize poorly | Document limitation; consider Roboflow re-export with val/test splits |
| **Missing val/test folders** | Training script cannot use `data.yaml` as-is | Split dataset or fix yaml |
| **No gaze sample files** | Cannot CI-test gaze pipeline | Add anonymized sample CSV to `data/gaze/samples/` (exempt from gitignore) |
| **Windows long paths** | 2 posters fail label read | Enable long paths or shorten filenames in dataset |
| **WebGazer requires HTTPS/localhost** | Deployment must serve static assets | Document `http.server` for local; HTTPS for production gaze capture |

**Policy:** Do not download arbitrary pretrained weights or fabricate gaze data.

---

## 11. Recommended Implementation Order

Each phase should end with tests + a short verification before proceeding.

| Order | Phase | Goal | Depends on |
|-------|-------|------|------------|
| **0** | Repository audit | This document | — |
| **1** | YOLO inference layer | Detect 5 classes on uploaded image; standardized AOI output | Weights (train or obtain) |
| **2** | Refactor extract modules | Split `aoi_visualizer.py` into `analysis/` without breaking CLI | Phase 0 |
| **3** | Feature extraction | Layout, color, text (OCR), branding, CTA metrics | Phase 1 AOIs |
| **4** | Gaze pipeline hardening | Heatmaps, fixation metrics, CV vs gaze mode flag | Phase 2 |
| **5** | Scoring engine | Config-driven scores + explanations | Phases 3–4 |
| **6** | Target market model | Questionnaire schema + rule mappings | Phase 5 |
| **7** | Recommendation engine | Structured recs with evidence | Phases 5–6 |
| **8** | LLM narrative layer | Optional; structured input only | Phase 7 |
| **9** | Streamlit web app | Upload → questionnaire → analyze → results → export | Phases 1–7 |
| **10** | Training pipeline | `train.py`, `inference.py`, split fix | Dataset |
| **11** | Tests | Unit + smoke E2E | All core modules |
| **12** | Documentation | README | Phases 1–11 |
| **13** | Deployment | Streamlit Cloud / Docker, env config | Phase 9 |

### Phase 1 immediate next steps (when approved)

1. Fix `data.yaml` paths or create val split  
2. Implement `training/train.py` and train YOLOv8n/s on 74 images  
3. Implement `models/detector.py` + `analysis/detection.py`  
4. Add `config/settings.py` with `MODEL_WEIGHTS` env var  
5. Keep `aoi_visualizer.py` working via imports from new modules  

---

## 12. Methodology Labels (for thesis / defense)

| Component | Type |
|-----------|------|
| YOLO element detection | **Machine learning / computer vision** |
| Layout, color, contrast metrics | **Computer vision / image processing** |
| OCR text analysis | **Computer vision + NLP (tool-assisted)** |
| Tobii import & fixation filtering | **Eye tracking / signal processing** |
| WebGazer capture | **Eye tracking (browser-based, calibrated)** |
| AOI dwell & attention % | **Eye tracking analytics** |
| PES ideal ranges & hierarchy rules | **Heuristic / rule-based** |
| Target market rules | **Heuristic / documented assumptions** |
| LLM explanations | **LLM-based narrative (non-measurement)** |

---

## 13. Verification Checklist (Phase 0 complete)

- [x] Full repository scanned  
- [x] Python, HTML, config, dataset reviewed  
- [x] AOI/Tobii/WebGazer implementation understood  
- [x] Reusable vs missing components identified  
- [x] Model/data blockers documented  
- [x] Proposed architecture documented  
- [ ] Phase 1+ implementation (awaiting approval)

---

## 14. Appendix — Key CLI Reference (existing)

```bash
# Install
pip install -r requirements.txt

# Regenerate WebGazer gallery
python aoi_visualizer.py --webgazer-gallery webgazer_choose_poster.html

# Serve gallery
python -m http.server 8001
# Open http://localhost:8001/webgazer_choose_poster.html

# Analyze WebGazer export
python aoi_visualizer.py \
  --poster "POSTER.jpg" \
  --gaze-file "export.csv" \
  --raw-coordinate-mode pixel

# Analyze Tobii export
python aoi_visualizer.py \
  --poster "POSTER.jpg" \
  --gaze-file "tobii_export.tsv" \
  --stimulus-filter "poster_name_fragment"
```

---

*Next action: Review this audit, confirm implementation order, then proceed to **Phase 1 (YOLO inference layer)** and **Phase 2 (module extraction)**.*
