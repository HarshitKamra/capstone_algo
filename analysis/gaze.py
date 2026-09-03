from __future__ import annotations

"""Gaze analysis wrappers that orchestrate Tobii/WebGazer parsing with AOIs.

This module exposes a small, testable API used by the higher-level app.
It leverages the parsing logic already implemented in `aoi_visualizer.py`.
"""
from typing import List, Dict, Any, Tuple


def analyze_gaze_with_aoi(
    gaze_file: str,
    aoi_boxes: List[Tuple[str, int, int, int, int]],
    image_shape: Tuple[int, int],
    coordinate_mode: str = "auto",
    stimulus_filter: str = None,
):
    """Analyze a Tobii/WebGazer export file using the provided AOI boxes.

    This function is intentionally self-contained to avoid circular imports
    with the legacy `aoi_visualizer` module. It uses the local parsing helpers
    in this module to build `stats` and `attention_scores`.
    """
    # Prepare labels and stats
    labels = [label for (label, *_rest) in aoi_boxes]
    attention_scores: Dict[str, float] = {label: 0.0 for label in labels}
    from config.settings import BACKGROUND_LABEL

    attention_scores[BACKGROUND_LABEL] = 0.0

    def add_attention(labels_list, duration_ms: float):
        duration_ms = max(0, duration_ms)
        if labels_list:
            for lbl in labels_list:
                attention_scores[lbl] = attention_scores.get(lbl, 0) + duration_ms
        else:
            attention_scores[BACKGROUND_LABEL] = attention_scores.get(BACKGROUND_LABEL, 0) + duration_ms

    def apply_raw_event_attention_local(event, duration_ms, image_shape_inner, coord_mode):
        lbls = event["labels"]
        if not lbls and event["x"] is not None and event["y"] is not None:
            x, y = raw_coordinates_to_image_coordinates(
                event["x"], event["y"], image_shape_inner, coord_mode, event.get("x_column"), event.get("y_column")
            )
            lbls = get_matched_aoi_labels(x, y, aoi_boxes)
        add_attention(lbls, duration_ms)

    delimiter = detect_delimiter(gaze_file)
    stats = {
        "rows_read": 0,
        "rows_used": 0,
        "events_used": 0,
        "rows_quality_rejected": 0,
        "aoi_hit_mode": False,
        "coordinate_mode": False,
        "iris_quality_filter": False,
    }

    with open(gaze_file, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        fieldnames = reader.fieldnames or []

        columns = {
            "aoi_hit": find_aoi_hit_column(fieldnames),
            "aoi_indicators": find_aoi_indicator_columns(fieldnames, labels),
            "duration": find_column_by_tokens(
                fieldnames,
                [
                    ("fixation", "duration"),
                    ("eye", "movement", "duration"),
                    ("gaze", "event", "duration"),
                    ("duration",),
                ],
            ),
            "timestamp": find_column_by_tokens(
                fieldnames,
                [
                    ("recording", "timestamp"),
                    ("timestamp",),
                ],
            ),
            "movement_type": find_column_by_tokens(
                fieldnames,
                [
                    ("eye", "movement", "type"),
                    ("movement", "type"),
                ],
                forbidden_tokens=("index",),
            ),
            "movement_index": find_column_by_tokens(
                fieldnames,
                [
                    ("eye", "movement", "type", "index"),
                    ("fixation", "index"),
                    ("event", "index"),
                ],
            ),
        }

        columns["x"], columns["y"] = find_coordinate_columns(fieldnames)
        iris_quality_columns = find_iris_quality_columns(fieldnames)
        stimulus_columns = find_stimulus_columns(fieldnames)

        stats["aoi_hit_mode"] = bool(columns["aoi_hit"] or columns["aoi_indicators"])
        stats["coordinate_mode"] = bool(columns["x"] and columns["y"])
        stats["iris_quality_filter"] = bool(iris_quality_columns)

        previous_event = None
        seen_movement_indexes = set()

        for row in reader:
            stats["rows_read"] += 1

            if not row_matches_stimulus(row, stimulus_columns, stimulus_filter):
                continue

            if not is_fixation_row(row, columns["movement_type"]):
                continue

            if not row_passes_iris_quality_filter(row, iris_quality_columns):
                stats["rows_quality_rejected"] += 1
                continue

            event = build_raw_event(row, labels, columns)
            if event is None:
                continue

            if columns["duration"] and columns["movement_index"] and event["movement_index"]:
                if event["movement_index"] in seen_movement_indexes:
                    continue
                seen_movement_indexes.add(event["movement_index"])

            stats["rows_used"] += 1

            if previous_event is not None:
                duration_ms = previous_event.get("duration")
                if duration_ms is None:
                    duration_ms = RAW_DATA_DEFAULT_FIXATION_MS
                apply_raw_event_attention_local(previous_event, duration_ms, image_shape, coordinate_mode)
                stats["events_used"] += 1

            previous_event = event

        if previous_event is not None:
            duration_ms = previous_event.get("duration") or RAW_DATA_DEFAULT_FIXATION_MS
            apply_raw_event_attention_local(previous_event, duration_ms, image_shape, coordinate_mode)
            stats["events_used"] += 1

    return {"stats": stats, "attention_scores": attention_scores}

import csv
import re
from typing import Any

from analysis.aoi import AOIRecord, aoi_records_to_legacy_boxes, get_present_aoi_labels
from config.settings import BACKGROUND_LABEL, RAW_DATA_DEFAULT_FIXATION_MS


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def find_column_by_tokens(fieldnames, token_groups, forbidden_tokens=None):
    forbidden_tokens = forbidden_tokens or ()
    for tokens in token_groups:
        for fieldname in fieldnames:
            normalized = normalize_column_name(fieldname)
            if any(token in normalized for token in forbidden_tokens):
                continue
            if all(token in normalized for token in tokens):
                return fieldname
    return None


def detect_delimiter(file_path: str) -> str:
    extension = file_path.rsplit(".", 1)[-1].lower()
    if extension == "tsv":
        return "\t"
    if extension == "csv":
        return ","

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)

    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        return "\t"


def parse_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+(?:[\.,]\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def parse_duration_ms(row, duration_column) -> float | None:
    if not duration_column:
        return None
    value = parse_float(row.get(duration_column))
    if value is None:
        return None
    normalized = normalize_column_name(duration_column)
    if "ms" not in normalized and ("second" in normalized or normalized.endswith(" s")):
        value *= 1000
    return value


def parse_timestamp_ms(row, timestamp_column) -> float | None:
    if not timestamp_column:
        return None
    value = parse_float(row.get(timestamp_column))
    if value is None:
        return None
    normalized = normalize_column_name(timestamp_column)
    if "ms" not in normalized and ("second" in normalized or normalized.endswith(" s")):
        value *= 1000
    return value


def find_aoi_hit_column(fieldnames):
    return find_column_by_tokens(
        fieldnames,
        [("aoi", "hit"), ("area", "of", "interest"), ("aoi", "name")],
    )


def find_aoi_indicator_columns(fieldnames, labels):
    indicator_columns = {}
    for fieldname in fieldnames:
        normalized = normalize_column_name(fieldname)
        for label in labels:
            normalized_label = normalize_column_name(label)
            if normalized == normalized_label or (
                normalized_label in normalized and ("aoi" in normalized or "hit" in normalized)
            ):
                indicator_columns[fieldname] = label
    return indicator_columns


def is_truthy_cell(value) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in ("", "0", "false", "no", "none", "nan", "-", "not hit"):
        return False
    if text in ("1", "true", "yes", "y", "x", "hit", "inside"):
        return True
    number = parse_float(text)
    if number is not None:
        return number > 0
    return True


def extract_aoi_labels_from_row(row, labels, aoi_hit_column, indicator_columns):
    matched_labels = []
    if aoi_hit_column:
        hit_value = row.get(aoi_hit_column, "")
        normalized_hit = normalize_column_name(hit_value)
        for label in labels + [BACKGROUND_LABEL]:
            normalized_label = normalize_column_name(label)
            if normalized_label and normalized_label in normalized_hit:
                matched_labels.append(label)
    for column, label in indicator_columns.items():
        if label not in matched_labels and is_truthy_cell(row.get(column)):
            matched_labels.append(label)
    return matched_labels


def find_coordinate_columns(fieldnames):
    exact_pairs = [
        ("mapped gaze point x", "mapped gaze point y"),
        ("gaze point x", "gaze point y"),
        ("gaze x", "gaze y"),
        ("fixation point x", "fixation point y"),
        ("fixation x", "fixation y"),
        ("snapshot x", "snapshot y"),
        ("stimulus x", "stimulus y"),
        ("image x", "image y"),
        ("poster x", "poster y"),
    ]
    normalized_to_original = {normalize_column_name(f): f for f in fieldnames}

    for x_name, y_name in exact_pairs:
        if x_name in normalized_to_original and y_name in normalized_to_original:
            return normalized_to_original[x_name], normalized_to_original[y_name]

    def coordinate_score(fieldname, axis):
        normalized = normalize_column_name(fieldname)
        tokens = normalized.split()
        if axis not in tokens:
            return -1
        if any(token in normalized for token in ("origin", "pupil", "diameter")):
            return -1
        score = 0
        if "mapped" in normalized:
            score += 6
        if any(token in normalized for token in ("snapshot", "stimulus", "image", "poster")):
            score += 5
        if "fixation" in normalized:
            score += 4
        if "gaze" in normalized:
            score += 3
        if any(token in normalized for token in ("left", "right")):
            score -= 1
        return score

    x_candidates = [(coordinate_score(f, "x"), f) for f in fieldnames]
    y_candidates = [(coordinate_score(f, "y"), f) for f in fieldnames]
    x_candidates = [c for c in x_candidates if c[0] >= 3]
    y_candidates = [c for c in y_candidates if c[0] >= 3]
    if not x_candidates or not y_candidates:
        return None, None
    x_candidates.sort(reverse=True)
    y_candidates.sort(reverse=True)
    return x_candidates[0][1], y_candidates[0][1]


def find_stimulus_columns(fieldnames):
    stimulus_columns = []
    for fieldname in fieldnames:
        normalized = normalize_column_name(fieldname)
        if any(token in normalized for token in ("stimulus", "snapshot", "recording", "media", "poster")):
            stimulus_columns.append(fieldname)
    return stimulus_columns


def row_matches_stimulus(row, stimulus_columns, stimulus_filter):
    if not stimulus_filter:
        return True
    normalized_filter = normalize_column_name(stimulus_filter)
    for column in stimulus_columns:
        if normalized_filter in normalize_column_name(row.get(column, "")):
            return True
    return False


def is_fixation_row(row, eye_movement_type_column):
    if not eye_movement_type_column:
        return True
    movement_type = str(row.get(eye_movement_type_column, "")).strip().lower()
    if not movement_type:
        return True
    return "fixation" in movement_type


def raw_coordinates_to_image_coordinates(
    raw_x,
    raw_y,
    image_shape,
    coordinate_mode,
    x_column=None,
    y_column=None,
):
    if raw_x is None or raw_y is None:
        return None, None

    height, width = image_shape[:2]
    mode = coordinate_mode

    if coordinate_mode == "auto":
        column_text = normalize_column_name(f"{x_column or ''} {y_column or ''}")
        if "%" in str(x_column) or "%" in str(y_column) or "percent" in column_text:
            mode = "percent"
        elif 0 <= raw_x <= 1 and 0 <= raw_y <= 1:
            mode = "normalized"
        else:
            mode = "pixel"

    if mode == "normalized":
        return int(raw_x * (width - 1)), int(raw_y * (height - 1))
    if mode == "percent":
        return int((raw_x / 100) * (width - 1)), int((raw_y / 100) * (height - 1))
    return int(raw_x), int(raw_y)


def get_matched_aoi_labels(
    x: int | None,
    y: int | None,
    aoi_boxes: list[tuple[str, int, int, int, int]],
) -> list[str]:
    if x is None or y is None:
        return []
    matched = []
    for label, x1, y1, x2, y2 in aoi_boxes:
        if x1 <= x <= x2 and y1 <= y <= y2 and label not in matched:
            matched.append(label)
    return matched


def parse_gaze_points(file_path: str, image_shape: tuple[int, int], stimulus_filter: str | None = None) -> list[tuple[int, int, float]]:
    """Parse gaze export file and return list of (x_px, y_px, timestamp_ms).

    Only rows with mapped or pixel coordinates are returned. Timestamp may be None
    for rows without a timestamp and those rows are skipped.
    """
    delimiter = detect_delimiter(file_path)
    points: list[tuple[int, int, float]] = []

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return points

        columns = {}
        columns["timestamp"] = find_column_by_tokens(
            fieldnames, [("recording", "timestamp"), ("timestamp",)]
        )
        columns["x"], columns["y"] = find_coordinate_columns(fieldnames)
        stimulus_columns = find_stimulus_columns(fieldnames)

        for row in reader:
            if not row_matches_stimulus(row, stimulus_columns, stimulus_filter):
                continue
            if not columns.get("x") or not columns.get("y"):
                continue
            raw_x = parse_float(row.get(columns["x"]))
            raw_y = parse_float(row.get(columns["y"]))
            ts = parse_timestamp_ms(row, columns["timestamp"]) if columns.get("timestamp") else None
            if raw_x is None or raw_y is None or ts is None:
                continue
            x_px, y_px = raw_coordinates_to_image_coordinates(raw_x, raw_y, image_shape, "auto", columns["x"], columns["y"])
            if x_px is None or y_px is None:
                continue
            points.append((x_px, y_px, ts))

    return points


def find_exact_normalized_column(fieldnames, target_name):
    normalized_target = normalize_column_name(target_name)
    for fieldname in fieldnames:
        if normalize_column_name(fieldname) == normalized_target:
            return fieldname
    return None


def find_iris_quality_columns(fieldnames):
    columns = {
        "face_detected": find_exact_normalized_column(fieldnames, "face_detected"),
        "eyes_detected": find_exact_normalized_column(fieldnames, "eyes_detected"),
        "iris_tracking_ok": find_exact_normalized_column(fieldnames, "iris_tracking_ok"),
    }
    return {name: column for name, column in columns.items() if column}


def row_passes_iris_quality_filter(row, iris_quality_columns):
    if not iris_quality_columns:
        return True
    for column in iris_quality_columns.values():
        if not is_truthy_cell(row.get(column)):
            return False
    return True


def build_raw_event(row, labels, columns):
    aoi_labels = extract_aoi_labels_from_row(
        row, labels, columns["aoi_hit"], columns["aoi_indicators"]
    )
    raw_x = parse_float(row.get(columns["x"])) if columns["x"] else None
    raw_y = parse_float(row.get(columns["y"])) if columns["y"] else None
    if not aoi_labels and (raw_x is None or raw_y is None):
        return None
    return {
        "labels": aoi_labels,
        "x": raw_x,
        "y": raw_y,
        "x_column": columns["x"],
        "y_column": columns["y"],
        "timestamp": parse_timestamp_ms(row, columns["timestamp"]),
        "duration": parse_duration_ms(row, columns["duration"]),
        "movement_index": row.get(columns["movement_index"]) if columns["movement_index"] else None,
    }


class GazeAnalyzer:
    """Stateful gaze import pipeline for one poster session."""

    def __init__(self, aoi_records: list[AOIRecord]) -> None:
        self.aoi_records = aoi_records
        self.aoi_boxes = aoi_records_to_legacy_boxes(aoi_records)
        self.attention_scores = self.initialize_attention_scores()

    def initialize_attention_scores(self) -> dict[str, float]:
        scores = {label: 0.0 for label in get_present_aoi_labels(self.aoi_records)}
        scores[BACKGROUND_LABEL] = 0.0
        return scores

    def add_attention(self, labels: list[str], duration_ms: float) -> None:
        duration_ms = max(0, duration_ms)
        if labels:
            for label in labels:
                self.attention_scores[label] = self.attention_scores.get(label, 0) + duration_ms
            return
        self.attention_scores[BACKGROUND_LABEL] = (
            self.attention_scores.get(BACKGROUND_LABEL, 0) + duration_ms
        )

    def apply_raw_event_attention(
        self,
        event: dict[str, Any],
        duration_ms: float,
        image_shape,
        coordinate_mode: str,
    ) -> None:
        labels = event["labels"]
        if not labels and event["x"] is not None and event["y"] is not None:
            x, y = raw_coordinates_to_image_coordinates(
                event["x"],
                event["y"],
                image_shape,
                coordinate_mode,
                event["x_column"],
                event["y_column"],
            )
            labels = get_matched_aoi_labels(x, y, self.aoi_boxes)
        self.add_attention(labels, duration_ms)

    def analyze_raw_gaze_file(
        self,
        file_path: str,
        image_shape,
        coordinate_mode: str = "auto",
        stimulus_filter: str | None = None,
    ) -> dict[str, Any]:
        delimiter = detect_delimiter(file_path)
        labels = get_present_aoi_labels(self.aoi_records)
        stats = {
            "rows_read": 0,
            "rows_used": 0,
            "events_used": 0,
            "rows_quality_rejected": 0,
            "aoi_hit_mode": False,
            "coordinate_mode": False,
            "iris_quality_filter": False,
        }

        with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                raise ValueError("The gaze export file has no header row.")

            columns = {
                "aoi_hit": find_aoi_hit_column(fieldnames),
                "aoi_indicators": find_aoi_indicator_columns(fieldnames, labels),
                "duration": find_column_by_tokens(
                    fieldnames,
                    [
                        ("fixation", "duration"),
                        ("eye", "movement", "duration"),
                        ("gaze", "event", "duration"),
                        ("duration",),
                    ],
                ),
                "timestamp": find_column_by_tokens(
                    fieldnames,
                    [("recording", "timestamp"), ("timestamp",)],
                ),
                "movement_type": find_column_by_tokens(
                    fieldnames,
                    [("eye", "movement", "type"), ("movement", "type")],
                    forbidden_tokens=("index",),
                ),
                "movement_index": find_column_by_tokens(
                    fieldnames,
                    [
                        ("eye", "movement", "type", "index"),
                        ("fixation", "index"),
                        ("event", "index"),
                    ],
                ),
            }
            columns["x"], columns["y"] = find_coordinate_columns(fieldnames)
            iris_quality_columns = find_iris_quality_columns(fieldnames)
            stimulus_columns = find_stimulus_columns(fieldnames)

            stats["aoi_hit_mode"] = bool(columns["aoi_hit"] or columns["aoi_indicators"])
            stats["coordinate_mode"] = bool(columns["x"] and columns["y"])
            stats["iris_quality_filter"] = bool(iris_quality_columns)

            if not stats["aoi_hit_mode"] and not stats["coordinate_mode"]:
                raise ValueError(
                    "Could not find AOI-hit columns or mapped gaze x/y columns in "
                    "the export. Export AOI hit data, or export gaze points "
                    "mapped to the poster/snapshot coordinate system."
                )

            previous_event = None
            seen_movement_indexes: set[Any] = set()

            for row in reader:
                stats["rows_read"] += 1
                if not row_matches_stimulus(row, stimulus_columns, stimulus_filter):
                    continue
                if not is_fixation_row(row, columns["movement_type"]):
                    continue
                if not row_passes_iris_quality_filter(row, iris_quality_columns):
                    stats["rows_quality_rejected"] += 1
                    continue

                event = build_raw_event(row, labels, columns)
                if event is None:
                    continue

                if columns["duration"] and columns["movement_index"] and event["movement_index"]:
                    if event["movement_index"] in seen_movement_indexes:
                        continue
                    seen_movement_indexes.add(event["movement_index"])

                stats["rows_used"] += 1

                if previous_event is not None:
                    duration_ms = previous_event["duration"]
                    if duration_ms is None:
                        current_ts = event["timestamp"]
                        previous_ts = previous_event["timestamp"]
                        if current_ts is not None and previous_ts is not None:
                            duration_ms = current_ts - previous_ts
                    if duration_ms is None or duration_ms <= 0:
                        duration_ms = RAW_DATA_DEFAULT_FIXATION_MS
                    self.apply_raw_event_attention(
                        previous_event, duration_ms, image_shape, coordinate_mode
                    )
                    stats["events_used"] += 1

                previous_event = event

            if previous_event is not None:
                duration_ms = previous_event["duration"]
                if duration_ms is None or duration_ms <= 0:
                    duration_ms = RAW_DATA_DEFAULT_FIXATION_MS
                self.apply_raw_event_attention(
                    previous_event, duration_ms, image_shape, coordinate_mode
                )
                stats["events_used"] += 1

        return stats
