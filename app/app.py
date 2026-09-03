from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from utils.logger import get_logger

logger = get_logger("capstone.app")

from analysis.detection import detect_poster_elements
from analysis.aoi import draw_aoi_boxes
from analysis.gaze import analyze_gaze_with_aoi
from analysis.aoi import aoi_records_to_legacy_boxes
from analysis.recommendations import build_recommendations
from analysis.llm_templates import format_report_for_llm
from analysis.gaze import parse_gaze_points
from analysis.attention import simple_heatmap
import numpy as np
import cv2
from monitoring.metrics import (
    inference_requests,
    gaze_analysis_requests,
    processing_seconds,
    start_metrics_server,
    timeit,
)
import time


st.set_page_config(page_title="Poster AOI Visualizer", layout="wide")

st.title("Poster AOI Visualizer")

# Start a local metrics server on port 8000 (Prometheus scrape target)
try:
    start_metrics_server(8000)
    logger.info("Started metrics server on :8000")
except Exception:
    logger.exception("Failed to start metrics server")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload poster image", type=["jpg", "jpeg", "png", "webp"])
    weights = st.text_input("Path to model weights (.pt)", value="")
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25)
    run = st.button("Detect AOIs")
    logger.info("Uploaded poster image: %s", getattr(uploaded, "name", "<uploaded>"))
    gaze_file = st.file_uploader("Upload gaze export (CSV/TSV)", type=["csv", "tsv"])
    stimulus_filter = st.text_input("Stimulus filter (optional)")
    smoothing = st.slider("Smoothing (ms)", 0, 1000, 0)
    analyze_gaze = st.button("Analyze gaze with AOIs")

with col2:
    st.write("Preview")
    preview_placeholder = st.empty()


def load_image(file) -> np.ndarray:
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


if uploaded is not None:
    image = load_image(uploaded)
    preview_rgb, _ = draw_aoi_boxes(image, [])
    preview_placeholder.image(preview_rgb, use_column_width=True)

if run:
    if uploaded is None:
        st.error("Upload an image first.")
    else:
        img = load_image(uploaded)
        try:
            inference_requests.inc()
            with timeit(processing_seconds):
                records, source = detect_poster_elements(img, image_path=None, weights_path=weights or None, conf=conf)
        except Exception as exc:
            st.error(f"Detection error: {exc}")
            logger.error("Detector initialization failed. weights=%s", weights)
        else:
            preview_rgb, legacy = draw_aoi_boxes(img, records)
            st.success(f"Detected {len(records)} AOIs (source={source})")
            preview_placeholder.image(preview_rgb, use_column_width=True)
            st.json([r.to_dict() for r in records])

        # store records in session state for later gaze analysis
        st.session_state["last_records"] = records
        st.session_state["last_image_shape"] = img.shape
        logger.info("Detection produced %d AOIs (source=%s)", len(records), source)

if analyze_gaze:
    if uploaded is None:
        st.error("Upload an image first.")
    elif gaze_file is None:
        st.error("Upload a gaze export file (CSV/TSV).")
    else:
        records = st.session_state.get("last_records")
        image_shape = st.session_state.get("last_image_shape")
        if not records or not image_shape:
            st.error("Run detection first to build AOIs.")
        else:
            # write uploaded gaze to a temporary file
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(gaze_file.read())
                tmp_path = tmp.name

            aoi_boxes = aoi_records_to_legacy_boxes(records)
            try:
                gaze_analysis_requests.inc()
                with timeit(processing_seconds):
                    result = analyze_gaze_with_aoi(tmp_path, aoi_boxes, image_shape, coordinate_mode="auto", stimulus_filter=stimulus_filter or None)
            except Exception as exc:
                st.error(f"Gaze analysis failed: {exc}")
                logger.exception("Gaze analysis failed")
            else:
                st.subheader("Gaze Analysis")
                st.json(result["stats"])
                st.subheader("Attention Scores")
                st.json(result["attention_scores"])
                import json
                st.download_button("Download attention JSON", data=json.dumps(result), file_name="attention.json", mime="application/json")
                # CSV export for attention
                import csv, io
                csv_buf = io.StringIO()
                writer = csv.writer(csv_buf)
                writer.writerow(["label", "attention_ms"])
                for k, v in result["attention_scores"].items():
                    writer.writerow([k, v])
                st.download_button("Download attention CSV", data=csv_buf.getvalue(), file_name="attention.csv", mime="text/csv")
                # Build recommendations and LLM template
                recs = build_recommendations(result["attention_scores"], records, {"overall_score": 0})
                st.subheader("Recommendations")
                for r in recs:
                    st.write("- ", r)

                report = {"summary": result.get("stats", {}), "scores": {}, "records": [r.to_dict() for r in records]}
                template = format_report_for_llm(report, recs)
                st.subheader("LLM-friendly Template")
                st.text_area("LLM Template", value=template, height=220)
                st.download_button("Download LLM template", data=template, file_name="llm_template.txt", mime="text/plain")

                # Heatmap visualization and playback controls
                points = parse_gaze_points(tmp_path, image_shape)
                if points:
                    times = [p[2] for p in points]
                    tmin, tmax = int(min(times)), int(max(times))
                    window = st.slider("Time window (ms)", tmin, tmax, (tmin, tmax))
                    # filter points by window
                    filtered = [(x, y) for x, y, t in points if window[0] <= t <= window[1]]
                    heat = simple_heatmap(filtered, image_shape)
                    # apply smoothing
                    sm = smoothing
                    k = max(1, int(sm / 10) | 1)
                    heat_blur = cv2.GaussianBlur(heat.astype('float32'), (k, k), 0)
                    # normalize heatmap to 0-255 and apply colormap
                    norm = cv2.normalize(heat_blur, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                    heat_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                    # overlay
                    overlay = cv2.addWeighted(cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR), 0.7, heat_color, 0.3, 0)
                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    st.subheader("Gaze Heatmap Overlay")
                    st.image(overlay_rgb, use_column_width=True)
                    # allow download
                    import io, base64
                    _, buf = cv2.imencode('.png', cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                    st.download_button("Download heatmap PNG", data=buf.tobytes(), file_name="heatmap.png", mime="image/png")

                    # PDF export combining heatmap + LLM template
                    try:
                        from utils.pdf_export import create_pdf_with_image_and_text

                        pdf_bytes = create_pdf_with_image_and_text(buf.tobytes(), template)
                        st.download_button("Download combined PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")
                    except Exception as exc:
                        logger.exception("PDF export failed: %s", exc)
