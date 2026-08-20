import argparse
import base64
import csv
import html
import json
import os
import re
from urllib.parse import quote

import cv2
import matplotlib.pyplot as plt


# DATASET PATHS
image_path = "Capstone.yolov8/train/images"
label_path = "Capstone.yolov8/train/labels"

# CLASS NAMES FROM data.yaml
classes = {
    0: "CTA",
    1: "Headline",
    2: "Price",
    3: "Product",
    4: "logo",
}

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".avif")
RAW_DATA_DEFAULT_FIXATION_MS = 300
BACKGROUND_LABEL = "Background"
CORE_AOI_ORDER = ["Product", "Headline", "CTA", "Price", "logo"]

IDEAL_ATTENTION_RANGES = {
    "Product": (30, 45),
    "Headline": (15, 30),
    "CTA": (10, 20),
    "Price": (5, 15),
    "logo": (3, 10),
}

PES_WEIGHTS = {
    "Product Attention": 25,
    "CTA Visibility": 20,
    "Headline Engagement": 20,
    "Attention Balance": 15,
    "Visual Hierarchy": 20,
}

aoi_boxes = []
attention_scores = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Poster PES analyzer with Tobii export or WebGazer capture support"
    )
    parser.add_argument(
        "--gaze-file",
        help="Path to exported Tobii Glasses/Pro Lab data (.tsv or .csv).",
    )
    parser.add_argument(
        "--stimulus-filter",
        help="Optional text used to keep rows for one poster/stimulus/recording.",
    )
    parser.add_argument(
        "--raw-coordinate-mode",
        choices=["auto", "pixel", "normalized", "percent"],
        default="auto",
        help="How exported gaze x/y columns should be interpreted.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip showing the AOI preview window after analysis.",
    )
    parser.add_argument(
        "--webgazer-session",
        metavar="HTML_PATH",
        help=(
            "Generate a browser-based WebGazer capture page for the selected "
            "poster and exit. The page exports a CSV that can be analyzed with "
            "--gaze-file."
        ),
    )
    parser.add_argument(
        "--webgazer-gallery",
        metavar="HTML_PATH",
        help=(
            "Generate one browser-based WebGazer page with a poster chooser "
            "for every labelled poster in the dataset."
        ),
    )
    parser.add_argument(
        "--poster",
        help=(
            "Poster filename to use without the interactive selector. Useful "
            "with --webgazer-session."
        ),
    )
    return parser.parse_args()


def list_available_posters(folder_path):
    """Return sorted poster filenames filtered by supported image extensions."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    posters = [
        name
        for name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, name))
        and name.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    return sorted(posters)


def load_poster(folder_path, requested_poster=None):
    """Let the user select a poster and return its filename plus image array."""
    posters = list_available_posters(folder_path)

    if not posters:
        raise FileNotFoundError("No supported poster images found.")

    lower_to_original = {name.lower(): name for name in posters}

    if requested_poster:
        selected_poster = lower_to_original.get(requested_poster.lower())
        if not selected_poster:
            raise FileNotFoundError(
                f"Poster not found: {requested_poster}. Choose one from {folder_path}."
            )
        poster_full_path = os.path.join(folder_path, selected_poster)
        image = read_image(poster_full_path)

        if image is None:
            raise ValueError(f"Unable to read selected poster: {selected_poster}")

        print(f"Selected Poster: {selected_poster}")
        return selected_poster, image

    print("Available Posters:")
    for index, poster_name in enumerate(posters, start=1):
        print(f"{index}. {poster_name}")

    while True:
        selection = input("\nSelect poster by number OR type filename: ").strip()

        if selection.isdigit():
            selected_index = int(selection)
            if 1 <= selected_index <= len(posters):
                selected_poster = posters[selected_index - 1]
                break

            print(f"Invalid number. Choose between 1 and {len(posters)}.")
            continue

        manual_name = selection.lower()
        if manual_name in lower_to_original:
            selected_poster = lower_to_original[manual_name]
            break

        print("Filename not found. Please choose from the listed posters.")

    poster_full_path = os.path.join(folder_path, selected_poster)
    image = read_image(poster_full_path)

    if image is None:
        raise ValueError(f"Unable to read selected poster: {selected_poster}")

    print(f"Selected Poster: {selected_poster}")
    return selected_poster, image


def read_image(file_path):
    """
    Read poster images as OpenCV BGR arrays.
    Pillow fallback supports formats like AVIF when OpenCV cannot read them.
    """
    image = cv2.imread(file_path)

    if image is not None:
        return image

    try:
        import numpy as np
        from PIL import Image

        with Image.open(file_path) as pil_image:
            rgb_image = pil_image.convert("RGB")
            return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_labels(folder_path, poster_name):
    """Load the YOLO label file that matches the selected poster filename."""
    poster_stem, _ = os.path.splitext(poster_name)
    label_file = f"{poster_stem}.txt"
    label_full_path = os.path.join(folder_path, label_file)

    if not os.path.isfile(label_full_path):
        raise FileNotFoundError(f"Matching label file not found: {label_file}")

    with open(label_full_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    print(f"Loaded Label File: {label_file}")
    return lines


def draw_aoi_boxes(image, lines, class_names):
    """Convert YOLO AOIs to pixel boxes and draw them on the poster preview."""
    preview = image.copy()
    height, width, _ = preview.shape
    poster_aoi_boxes = []

    for line in lines:
        data = line.strip().split()
        if len(data) < 5:
            continue

        class_id = int(data[0])
        x_center = float(data[1]) * width
        y_center = float(data[2]) * height
        box_width = float(data[3]) * width
        box_height = float(data[4]) * height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        label = class_names.get(class_id, f"class_{class_id}")
        poster_aoi_boxes.append((label, x1, y1, x2, y2))

        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    return preview, poster_aoi_boxes


def request_gaze_file(gaze_file=None):
    """Ask for the exported Tobii file path when it was not given by CLI."""
    if gaze_file and os.path.isfile(gaze_file):
        return gaze_file

    if gaze_file:
        print(f"Gaze file not found: {gaze_file}")

    while True:
        entered_path = input("\nEnter Tobii raw data .tsv/.csv path: ").strip()
        entered_path = entered_path.strip('"')

        if os.path.isfile(entered_path):
            return entered_path

        print("File not found. Enter a valid exported Tobii data file path.")


def normalize_column_name(name):
    """Normalize export column names from different Tobii tools."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def find_column_by_tokens(fieldnames, token_groups, forbidden_tokens=None):
    """Find the first column containing all tokens from any token group."""
    forbidden_tokens = forbidden_tokens or ()

    for tokens in token_groups:
        for fieldname in fieldnames:
            normalized = normalize_column_name(fieldname)

            if any(token in normalized for token in forbidden_tokens):
                continue

            if all(token in normalized for token in tokens):
                return fieldname

    return None


def detect_delimiter(file_path):
    """Detect whether an exported gaze file is TSV, CSV, or semicolon-separated."""
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".tsv":
        return "\t"

    if extension == ".csv":
        return ","

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)

    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        return "\t"


def parse_float(value):
    """Parse numeric values from Tobii export cells."""
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


def parse_duration_ms(row, duration_column):
    """Return duration in milliseconds when an export provides a duration field."""
    if not duration_column:
        return None

    value = parse_float(row.get(duration_column))
    if value is None:
        return None

    normalized = normalize_column_name(duration_column)
    if "ms" not in normalized and (
        "second" in normalized or normalized.endswith(" s")
    ):
        value *= 1000

    return value


def parse_timestamp_ms(row, timestamp_column):
    """Return timestamp in milliseconds when available."""
    if not timestamp_column:
        return None

    value = parse_float(row.get(timestamp_column))
    if value is None:
        return None

    normalized = normalize_column_name(timestamp_column)
    if "ms" not in normalized and (
        "second" in normalized or normalized.endswith(" s")
    ):
        value *= 1000

    return value


def find_aoi_hit_column(fieldnames):
    """Find a generic AOI-hit column from a Tobii export."""
    return find_column_by_tokens(
        fieldnames,
        [
            ("aoi", "hit"),
            ("area", "of", "interest"),
            ("aoi", "name"),
        ],
    )


def find_aoi_indicator_columns(fieldnames, labels):
    """Find per-AOI hit columns such as Product/Headline/CTA indicators."""
    indicator_columns = {}

    for fieldname in fieldnames:
        normalized = normalize_column_name(fieldname)

        for label in labels:
            normalized_label = normalize_column_name(label)

            if normalized == normalized_label or (
                normalized_label in normalized
                and ("aoi" in normalized or "hit" in normalized)
            ):
                indicator_columns[fieldname] = label

    return indicator_columns


def is_truthy_cell(value):
    """Interpret exported hit/inside values."""
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
    """Extract AOI labels from an export row when AOI hits are already present."""
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
    """
    Find x/y gaze columns. Supports common mapped-gaze, snapshot, image,
    stimulus, fixation, and poster coordinate naming patterns.
    """
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

    normalized_to_original = {
        normalize_column_name(fieldname): fieldname
        for fieldname in fieldnames
    }

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

    x_candidates = [
        (coordinate_score(fieldname, "x"), fieldname)
        for fieldname in fieldnames
    ]
    y_candidates = [
        (coordinate_score(fieldname, "y"), fieldname)
        for fieldname in fieldnames
    ]

    x_candidates = [candidate for candidate in x_candidates if candidate[0] >= 3]
    y_candidates = [candidate for candidate in y_candidates if candidate[0] >= 3]

    if not x_candidates or not y_candidates:
        return None, None

    x_candidates.sort(reverse=True)
    y_candidates.sort(reverse=True)
    return x_candidates[0][1], y_candidates[0][1]


def find_stimulus_columns(fieldnames):
    """Find columns that can identify the poster/stimulus/recording."""
    stimulus_columns = []

    for fieldname in fieldnames:
        normalized = normalize_column_name(fieldname)

        if any(
            token in normalized
            for token in ("stimulus", "snapshot", "recording", "media", "poster")
        ):
            stimulus_columns.append(fieldname)

    return stimulus_columns


def row_matches_stimulus(row, stimulus_columns, stimulus_filter):
    """Filter a combined Tobii export down to one poster/session when requested."""
    if not stimulus_filter:
        return True

    normalized_filter = normalize_column_name(stimulus_filter)

    for column in stimulus_columns:
        if normalized_filter in normalize_column_name(row.get(column, "")):
            return True

    return False


def is_fixation_row(row, eye_movement_type_column):
    """Use fixation rows when Tobii export includes eye movement classification."""
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
    """Convert exported gaze coordinates into poster pixel coordinates."""
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


def get_matched_aoi_labels(x, y):
    """Return all AOI labels that contain a gaze coordinate."""
    if x is None or y is None:
        return []

    matched_labels = []

    for label, x1, y1, x2, y2 in aoi_boxes:
        if x1 <= x <= x2 and y1 <= y <= y2 and label not in matched_labels:
            matched_labels.append(label)

    return matched_labels


def add_attention(labels, duration_ms):
    """Accumulate attention for one raw/fixation event."""
    duration_ms = max(0, duration_ms)

    if labels:
        for label in labels:
            attention_scores[label] = attention_scores.get(label, 0) + duration_ms
        return

    attention_scores[BACKGROUND_LABEL] = (
        attention_scores.get(BACKGROUND_LABEL, 0) + duration_ms
    )


def apply_raw_event_attention(event, duration_ms, image_shape, coordinate_mode):
    """Apply one Tobii raw-data event to AOI/background attention."""
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
        labels = get_matched_aoi_labels(x, y)

    add_attention(labels, duration_ms)


def build_raw_event(row, labels, columns):
    """Create a normalized event object from one exported Tobii row."""
    aoi_labels = extract_aoi_labels_from_row(
        row,
        labels,
        columns["aoi_hit"],
        columns["aoi_indicators"],
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
        "movement_index": (
            row.get(columns["movement_index"]) if columns["movement_index"] else None
        ),
    }


def find_exact_normalized_column(fieldnames, target_name):
    """Find a column by normalized exact name."""
    normalized_target = normalize_column_name(target_name)

    for fieldname in fieldnames:
        if normalize_column_name(fieldname) == normalized_target:
            return fieldname

    return None


def find_iris_quality_columns(fieldnames):
    """Find optional WebGazer/Iris quality columns in exported CSVs."""
    columns = {
        "face_detected": find_exact_normalized_column(fieldnames, "face_detected"),
        "eyes_detected": find_exact_normalized_column(fieldnames, "eyes_detected"),
        "iris_tracking_ok": find_exact_normalized_column(fieldnames, "iris_tracking_ok"),
    }
    return {name: column for name, column in columns.items() if column}


def row_passes_iris_quality_filter(row, iris_quality_columns):
    """Reject WebGazer rows captured while face/iris tracking was unreliable."""
    if not iris_quality_columns:
        return True

    for column in iris_quality_columns.values():
        if not is_truthy_cell(row.get(column)):
            return False

    return True


def analyze_raw_gaze_file(
    file_path,
    image_shape,
    coordinate_mode="auto",
    stimulus_filter=None,
):
    """
    Convert exported Tobii Glasses/Pro Lab data into AOI attention time.
    Supports AOI-hit exports and mapped x/y gaze-coordinate exports.
    """
    delimiter = detect_delimiter(file_path)
    labels = get_present_aoi_labels(aoi_boxes)
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
            raise ValueError("The Tobii export file has no header row.")

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

        if not stats["aoi_hit_mode"] and not stats["coordinate_mode"]:
            raise ValueError(
                "Could not find AOI-hit columns or mapped gaze x/y columns in "
                "the Tobii export. Export AOI hit data, or export gaze points "
                "mapped to the poster/snapshot coordinate system."
            )

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
                duration_ms = previous_event["duration"]

                if duration_ms is None:
                    current_timestamp = event["timestamp"]
                    previous_timestamp = previous_event["timestamp"]

                    if current_timestamp is not None and previous_timestamp is not None:
                        duration_ms = current_timestamp - previous_timestamp

                if duration_ms is None or duration_ms <= 0:
                    duration_ms = RAW_DATA_DEFAULT_FIXATION_MS

                apply_raw_event_attention(
                    previous_event,
                    duration_ms,
                    image_shape,
                    coordinate_mode,
                )
                stats["events_used"] += 1

            previous_event = event

        if previous_event is not None:
            duration_ms = previous_event["duration"]
            if duration_ms is None or duration_ms <= 0:
                duration_ms = RAW_DATA_DEFAULT_FIXATION_MS

            apply_raw_event_attention(
                previous_event,
                duration_ms,
                image_shape,
                coordinate_mode,
            )
            stats["events_used"] += 1

    return stats


def clamp(value, minimum=0, maximum=100):
    """Keep a score inside the 0-100 range."""
    return max(minimum, min(maximum, value))


def score_for_ideal_range(value, low, high):
    """
    Score an attention percentage against a healthy marketing range.
    Values inside the range get 100; weak or excessive attention is penalized.
    """
    if low <= value <= high:
        return 100

    if value < low:
        return clamp((value / low) * 100) if low else 100

    remaining_space = 100 - high
    if remaining_space <= 0:
        return 0

    return clamp(100 - ((value - high) / remaining_space) * 100)


def get_present_aoi_labels(boxes):
    """Return AOI labels present on the selected poster in a stable order."""
    labels = {box[0] for box in boxes}
    ordered_labels = [label for label in CORE_AOI_ORDER if label in labels]
    extra_labels = sorted(labels - set(CORE_AOI_ORDER))
    return ordered_labels + extra_labels


def initialize_attention_scores(boxes):
    """Create a fresh per-poster attention store, including zero-attention AOIs."""
    scores = {label: 0 for label in get_present_aoi_labels(boxes)}
    scores[BACKGROUND_LABEL] = 0
    return scores


def calculate_attention_percentages(scores, boxes):
    """Convert accumulated fixation time into attention percentages."""
    labels = get_present_aoi_labels(boxes)
    report_labels = labels + [BACKGROUND_LABEL]
    total_attention = sum(scores.values())

    if total_attention == 0:
        return {label: 0 for label in report_labels}

    return {
        label: (scores.get(label, 0) / total_attention) * 100
        for label in report_labels
    }


def calculate_balance_score(percentages, boxes):
    """
    Reward posters where important AOIs receive some attention while background
    attention and one-element domination stay low.
    """
    labels = get_present_aoi_labels(boxes)

    if not labels:
        return 0

    meaningful_labels = [
        label
        for label in labels
        if percentages.get(label, 0) >= 5
    ]
    coverage_score = (len(meaningful_labels) / len(labels)) * 100

    dominant_attention = max(percentages.get(label, 0) for label in labels)
    dominance_score = (
        100
        if dominant_attention <= 55
        else clamp(100 - ((dominant_attention - 55) * 2))
    )

    background_attention = percentages.get(BACKGROUND_LABEL, 0)
    background_score = clamp(100 - (background_attention * 2))

    return (
        coverage_score * 0.45
        + dominance_score * 0.35
        + background_score * 0.20
    )


def calculate_hierarchy_score(percentages, boxes):
    """
    Reward a marketing-friendly visual hierarchy:
    Product should lead, while Headline and CTA should appear near the top.
    """
    labels = get_present_aoi_labels(boxes)
    ranked_labels = [
        label
        for label, _ in sorted(
            ((label, percentages.get(label, 0)) for label in labels),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    if not ranked_labels or max(percentages.get(label, 0) for label in labels) == 0:
        return 0

    rank_map = {
        label: rank
        for rank, label in enumerate(ranked_labels, start=1)
    }

    score = 0

    product_rank = rank_map.get("Product")
    if product_rank == 1:
        score += 35
    elif product_rank == 2:
        score += 25
    elif product_rank == 3:
        score += 15

    headline_rank = rank_map.get("Headline")
    if headline_rank and headline_rank <= 3:
        score += 25
    elif percentages.get("Headline", 0) > 0:
        score += 10

    cta_rank = rank_map.get("CTA")
    if cta_rank and cta_rank <= 3:
        score += 25
    elif percentages.get("CTA", 0) > 0:
        score += 10

    background_attention = percentages.get(BACKGROUND_LABEL, 0)
    if background_attention <= 10:
        score += 15
    elif background_attention <= 20:
        score += 8

    return clamp(score)


def get_pes_category(score):
    """Translate a numeric PES into a simple interpretation."""
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 50:
        return "Average"
    return "Needs Improvement"


def build_design_insights(percentages, boxes, component_scores):
    """Generate human-readable poster improvement notes from PES inputs."""
    labels = set(get_present_aoi_labels(boxes))
    insights = []

    if "Product" not in labels:
        insights.append("Product AOI is missing, so the main food item cannot be evaluated.")
    elif percentages.get("Product", 0) < 25:
        insights.append("Product attention is low; make the food item larger, clearer, or more central.")
    elif percentages.get("Product", 0) > 55:
        insights.append("Product dominates strongly; supporting text or CTA may need more visual weight.")
    else:
        insights.append("Product visibility is healthy.")

    if "CTA" not in labels:
        insights.append("CTA is missing; add a clear action such as order now, visit, or buy.")
    elif percentages.get("CTA", 0) < 8:
        insights.append("CTA is being ignored; improve contrast, size, or placement.")
    else:
        insights.append("CTA is receiving useful attention.")

    if "Headline" not in labels:
        insights.append("Headline is missing; add a short message to guide viewer understanding.")
    elif percentages.get("Headline", 0) < 10:
        insights.append("Headline engagement is weak; improve readability and position.")

    if percentages.get(BACKGROUND_LABEL, 0) > 25:
        insights.append("Too much attention is going outside AOIs; reduce clutter or mark important regions.")

    if component_scores["Attention Balance"] < 50:
        insights.append("Attention is concentrated on too few elements; improve balance across key AOIs.")

    if component_scores["Visual Hierarchy"] < 60:
        insights.append("Visual hierarchy is weak; Product, Headline, and CTA should guide the viewer in order.")

    return insights


def calculate_pes(percentages, boxes):
    """Calculate Poster Effectiveness Score from AOI attention analytics."""
    component_scores = {
        "Product Attention": score_for_ideal_range(
            percentages.get("Product", 0),
            *IDEAL_ATTENTION_RANGES["Product"],
        ),
        "CTA Visibility": score_for_ideal_range(
            percentages.get("CTA", 0),
            *IDEAL_ATTENTION_RANGES["CTA"],
        ),
        "Headline Engagement": score_for_ideal_range(
            percentages.get("Headline", 0),
            *IDEAL_ATTENTION_RANGES["Headline"],
        ),
        "Attention Balance": calculate_balance_score(percentages, boxes),
        "Visual Hierarchy": calculate_hierarchy_score(percentages, boxes),
    }

    score = sum(
        component_scores[name] * weight / 100
        for name, weight in PES_WEIGHTS.items()
    )

    return {
        "score": clamp(score),
        "category": get_pes_category(score),
        "components": component_scores,
        "insights": build_design_insights(percentages, boxes, component_scores),
    }


def print_raw_import_summary(stats):
    """Print how the Tobii export was interpreted."""
    print("\nRaw Data Import Summary:")
    print(f"Rows read: {stats['rows_read']}")
    print(f"Rows used: {stats['rows_used']}")
    print(f"Events applied: {stats['events_used']}")
    print(f"Iris quality filter used: {stats['iris_quality_filter']}")
    print(f"Rows rejected by Iris quality: {stats['rows_quality_rejected']}")
    print(f"AOI hit columns used: {stats['aoi_hit_mode']}")
    print(f"Gaze coordinate columns used: {stats['coordinate_mode']}")


def print_attention_report():
    """Print attention percentages, AOI ranking, PES, and design insights."""
    total_attention = sum(attention_scores.values())
    percentages = calculate_attention_percentages(attention_scores, aoi_boxes)
    labels = get_present_aoi_labels(aoi_boxes)

    print(f"\nOverall Attention: {total_attention:.0f} ms")

    print("\nAttention Percentages:")
    for label in labels:
        print(f"{label}: {percentages.get(label, 0):.2f}%")
    print(f"{BACKGROUND_LABEL}: {percentages.get(BACKGROUND_LABEL, 0):.2f}%")

    print("\nAOI Rankings:")
    sorted_aoi = sorted(
        ((label, percentages.get(label, 0)) for label in labels),
        key=lambda item: item[1],
        reverse=True,
    )

    for rank, (aoi, percent) in enumerate(sorted_aoi, start=1):
        print(f"{rank}. {aoi} -> {percent:.2f}%")

    pes = calculate_pes(percentages, aoi_boxes)
    print("\nPoster Effectiveness Score (PES):")
    print(f"{pes['score']:.2f}/100 - {pes['category']}")

    print("\nPES Components:")
    for component, score in pes["components"].items():
        weight = PES_WEIGHTS[component]
        print(f"{component} ({weight}%): {score:.2f}/100")

    print("\nDesign Insights:")
    for insight in pes["insights"]:
        print(f"- {insight}")


def show_aoi_preview(preview_image):
    """Show selected poster with AOI boxes after the report is printed."""
    plt.figure(figsize=(10, 10))
    plt.imshow(preview_image)
    plt.axis("off")
    plt.title("Poster AOI Preview")
    plt.show()


def get_image_data_uri(file_path):
    """Return a browser-friendly data URI for the selected poster."""
    extension = os.path.splitext(file_path)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".avif": "image/avif",
    }
    mime_type = mime_types.get(extension, "application/octet-stream")

    with open(file_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def build_webgazer_html(poster_name, poster_image_src, boxes, image_shape, poster_options=None):
    """Build a self-contained WebGazer page that exports analyzer-ready CSV."""
    height, width = image_shape[:2]
    if poster_options is None:
        poster_options = [
            {
                "name": poster_name,
                "imageSrc": poster_image_src,
                "width": width,
                "height": height,
                "aois": [
                    {"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    for label, x1, y1, x2, y2 in boxes
                ],
            }
        ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GazeLab — Poster Eye Tracking</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap" rel="stylesheet">
  <script src="https://webgazer.cs.brown.edu/webgazer.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #e2e8f0;
      --panel: #ffffff;
      --surface: #f8fafc;
      --accent: #6366f1;
      --accent-dark: #4f46e5;
      --accent-soft: rgba(99, 102, 241, 0.12);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --shadow: 0 4px 24px rgba(15, 23, 42, 0.08);
      --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "DM Sans", system-ui, -apple-system, sans-serif;
      color: var(--ink);
      background: linear-gradient(145deg, #eef2ff 0%, #f8fafc 45%, #f1f5f9 100%);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    header {{
      flex-shrink: 0;
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(12px);
      box-shadow: var(--shadow-sm);
    }}
    .app-brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }}
    .brand-icon {{
      width: 40px;
      height: 40px;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%);
      display: grid;
      place-items: center;
      color: #fff;
      font-size: 20px;
      flex-shrink: 0;
      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }}
    .brand-text h1 {{
      margin: 0;
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .brand-text p {{
      margin: 2px 0 0;
      font-size: 12px;
      color: var(--muted);
    }}
    .header-controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .poster-picker {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: min(42vw, 360px);
    }}
    .poster-search-status {{
      min-height: 16px;
      margin: -2px 2px 0;
      color: var(--muted);
      font-size: 11px;
    }}
    .poster-search-status.error {{
      color: var(--danger);
    }}
    #posterSearch {{
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0 12px;
      background: var(--surface);
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      transition: border-color 0.15s, box-shadow 0.15s;
    }}
    #posterSearch:focus {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }}
    #posterSelect {{
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0 12px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }}
    .header-actions {{
      display: flex;
      gap: 8px;
      flex-shrink: 0;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(280px, 1fr) 340px;
      gap: 20px;
      padding: 20px 24px;
      flex-grow: 1;
      overflow: hidden;
      min-height: 0;
    }}
    button {{
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0 16px;
      background: #ffffff;
      color: var(--ink);
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      transition: transform 0.12s, box-shadow 0.12s, background 0.12s;
    }}
    button:hover:not(:disabled) {{
      transform: translateY(-1px);
      box-shadow: var(--shadow-sm);
    }}
    button.primary {{
      border-color: transparent;
      background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%);
      color: #ffffff;
      box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }}
    button.danger {{
      border-color: rgba(239, 68, 68, 0.35);
      color: var(--danger);
      background: rgba(239, 68, 68, 0.06);
    }}
    button:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }}
    .stage {{
      display: grid;
      place-items: center center;
      height: 100%;
      width: 100%;
      overflow: hidden;
      background: var(--panel);
      border-radius: 16px;
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      position: relative;
    }}
    .stage-toolbar {{
      position: absolute;
      top: 12px;
      right: 12px;
      z-index: 5;
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .toggle-aoi {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.95);
      border: 1px solid var(--line);
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
      cursor: pointer;
      user-select: none;
      box-shadow: var(--shadow-sm);
    }}
    .toggle-aoi input {{
      accent-color: var(--accent);
    }}
    .poster-wrap {{
      position: relative;
      max-width: calc(100% - 32px);
      max-height: calc(100% - 32px);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(15, 23, 42, 0.12);
      display: inline-block;
    }}
    .poster-wrap.aoi-hidden .aoi {{
      opacity: 0;
      pointer-events: none;
    }}
    #poster {{
      display: block;
      max-width: 100%;
      max-height: calc(100vh - 140px);
      width: auto;
      height: auto;
      user-select: none;
    }}
    .aoi {{
      position: absolute;
      border: 2px solid var(--aoi-color, #6366f1);
      background: rgba(99, 102, 241, 0.12);
      background: color-mix(in srgb, var(--aoi-color, #6366f1) 12%, transparent);
      pointer-events: none;
      transition: opacity 0.2s;
    }}
    .aoi span {{
      position: absolute;
      left: -2px;
      top: -26px;
      min-width: 48px;
      padding: 4px 8px;
      background: var(--aoi-color, #6366f1);
      color: #ffffff;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      white-space: nowrap;
      border-radius: 6px 6px 6px 0;
    }}
    #gazeDot {{
      position: fixed;
      z-index: 30;
      width: 16px;
      height: 16px;
      margin: -8px 0 0 -8px;
      border-radius: 50%;
      background: radial-gradient(circle at 35% 35%, #fda4af, #e11d48);
      box-shadow: 0 0 0 4px rgba(225, 29, 72, 0.25), 0 0 20px rgba(225, 29, 72, 0.4);
      pointer-events: none;
      transform: translate(-100px, -100px);
      animation: gaze-pulse 1.6s ease-in-out infinite;
    }}
    @keyframes gaze-pulse {{
      0%, 100% {{ box-shadow: 0 0 0 4px rgba(225, 29, 72, 0.25), 0 0 20px rgba(225, 29, 72, 0.4); }}
      50% {{ box-shadow: 0 0 0 8px rgba(225, 29, 72, 0.15), 0 0 28px rgba(225, 29, 72, 0.5); }}
    }}
    aside {{
      overflow-y: auto;
      height: 100%;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .panel {{
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
      box-shadow: var(--shadow-sm);
    }}
    .panel-title {{
      margin: 0 0 12px;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .metric-grid {{
      display: grid;
      gap: 10px;
    }}
    .metric {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      background: var(--surface);
      border-radius: 10px;
      font-size: 14px;
    }}
    .metric strong {{
      color: var(--muted);
      font-weight: 600;
      font-size: 13px;
    }}
    .metric-value {{
      font-weight: 700;
      text-align: right;
    }}
    .status-badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .status-badge.ready {{ background: #ecfdf5; color: #059669; }}
    .status-badge.starting {{ background: #eff6ff; color: #2563eb; }}
    .status-badge.calibrating {{ background: #fffbeb; color: #d97706; }}
    .status-badge.recording {{ background: #fef2f2; color: #dc2626; animation: rec-blink 1.2s ease-in-out infinite; }}
    .status-badge.stopped {{ background: #f1f5f9; color: #475569; }}
    .status-badge.error {{ background: #fef2f2; color: #b91c1c; }}
    @keyframes rec-blink {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.65; }}
    }}
    .aoi-pill {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: var(--accent-soft);
      color: var(--accent-dark);
    }}
    .steps {{
      margin: 0;
      padding: 0;
      list-style: none;
      counter-reset: step;
    }}
    .steps li {{
      counter-increment: step;
      position: relative;
      padding: 10px 0 10px 36px;
      font-size: 13px;
      line-height: 1.5;
      color: var(--muted);
      border-bottom: 1px solid var(--line);
    }}
    .steps li:last-child {{
      border-bottom: 0;
    }}
    .steps li::before {{
      content: counter(step);
      position: absolute;
      left: 0;
      top: 10px;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: var(--accent-soft);
      color: var(--accent-dark);
      font-size: 12px;
      font-weight: 700;
      display: grid;
      place-items: center;
    }}
    .tip-box {{
      padding: 12px 14px;
      background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
      border-radius: 10px;
      font-size: 13px;
      line-height: 1.5;
      color: #4338ca;
    }}
    .tip-box code {{
      background: rgba(255, 255, 255, 0.7);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 12px;
    }}
    #cameraPanel {{
      overflow: hidden;
    }}
    #cameraMount {{
      display: grid;
      place-items: center;
      min-height: 180px;
      border: 1px solid #1e293b;
      border-radius: 12px;
      background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
      color: #94a3b8;
      font-size: 13px;
    }}
    #cameraMount #webgazerVideoContainer {{
      position: relative !important;
      inset: auto !important;
      width: 280px !important;
      height: 210px !important;
      max-width: 100% !important;
      overflow: hidden !important;
      border-radius: 10px;
    }}
    #cameraMount #webgazerVideoFeed,
    #cameraMount #webgazerFaceOverlay,
    #cameraMount #webgazerFaceFeedbackBox {{
      width: 280px !important;
      height: 210px !important;
      max-width: 100% !important;
    }}
    .calibration-layer {{
      position: fixed;
      inset: 0;
      z-index: 2147483647;
      cursor: crosshair;
      touch-action: manipulation;
      background: rgba(15, 23, 42, 0.55);
      backdrop-filter: blur(2px);
    }}
    .calibration-target {{
      position: absolute;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: #f59e0b;
      box-shadow: 0 0 0 12px rgba(245, 158, 11, 0.25), 0 0 40px rgba(245, 158, 11, 0.4);
      pointer-events: none;
      animation: cal-pulse 1.4s ease-in-out infinite;
    }}
    .calibration-target.accuracy {{
      background: #3b82f6;
      box-shadow: 0 0 0 12px rgba(59, 130, 246, 0.25), 0 0 40px rgba(59, 130, 246, 0.4);
    }}
    @keyframes cal-pulse {{
      0%, 100% {{ transform: scale(1); }}
      50% {{ transform: scale(1.08); }}
    }}
    @media (max-width: 960px) {{
      body {{
        height: auto;
        min-height: 100vh;
        overflow: auto;
      }}
      header {{
        flex-direction: column;
        align-items: stretch;
        padding: 14px 16px;
      }}
      .header-controls {{
        flex-direction: column;
        align-items: stretch;
      }}
      .poster-picker {{
        min-width: 100%;
      }}
      main {{
        grid-template-columns: 1fr;
        overflow-y: auto;
        padding: 16px;
      }}
      aside {{
        max-height: none;
      }}
    }}
    @media (max-width: 560px) {{
      .header-actions {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
      }}
      button {{
        min-width: 0;
        padding: 0 8px;
        font-size: 12px;
      }}
      .stage {{
        min-height: 52vh;
      }}
      .stage-toolbar {{
        top: 8px;
        right: 8px;
      }}
      .toggle-aoi {{
        padding: 7px 10px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="app-brand">
      <div class="brand-icon" aria-hidden="true">👁</div>
      <div class="brand-text">
        <h1>GazeLab</h1>
        <p id="posterTitle">{html.escape(poster_name)}</p>
      </div>
    </div>
    <div class="header-controls">
      <div class="poster-picker">
        <input id="posterSearch" type="search" placeholder="Search posters…" aria-label="Search posters">
        <select id="posterSelect" aria-label="Choose poster"></select>
        <p id="posterSearchStatus" class="poster-search-status" aria-live="polite"></p>
      </div>
      <div class="header-actions">
        <button id="startBtn" class="primary">▶ Start</button>
        <button id="stopBtn" class="danger" disabled>■ Stop</button>
        <button id="downloadBtn" disabled>↓ CSV</button>
      </div>
    </div>
  </header>
  <main>
    <section class="stage">
      <div class="stage-toolbar">
        <label class="toggle-aoi">
          <input id="aoiToggle" type="checkbox" checked>
          Show AOI regions
        </label>
      </div>
      <div id="posterWrap" class="poster-wrap">
        <img id="poster" src="{poster_image_src}" alt="{html.escape(poster_name)}">
      </div>
    </section>
    <aside>
      <div id="cameraPanel" class="panel">
        <h2 class="panel-title">Live Camera</h2>
        <div id="cameraMount">
          <span>Camera preview appears here after Start</span>
        </div>
      </div>
      <div class="panel">
        <h2 class="panel-title">Session Metrics</h2>
        <div class="metric-grid">
          <div class="metric"><strong>Status</strong><span id="status" class="metric-value status-badge ready">Ready</span></div>
          <div class="metric"><strong>Samples</strong><span id="sampleCount" class="metric-value">0</span></div>
          <div class="metric"><strong>Current AOI</strong><span id="currentAoi" class="metric-value">—</span></div>
          <div class="metric"><strong>Accuracy</strong><span id="accuracyStatus" class="metric-value">—</span></div>
          <div class="metric"><strong>Iris Quality</strong><span id="irisStatus" class="metric-value">—</span></div>
        </div>
      </div>
      <div class="panel">
        <h2 class="panel-title">How to Use</h2>
        <ol class="steps">
          <li>Pick a poster from the dropdown (or search by name).</li>
          <li>Click <strong>Start</strong> and allow webcam access.</li>
          <li>Click each <strong>orange</strong> calibration dot, then each <strong>blue</strong> accuracy dot.</li>
          <li>Look at the poster — recording begins automatically.</li>
          <li>Click <strong>Stop</strong>, then <strong>Download CSV</strong> for analysis.</li>
        </ol>
      </div>
      <div class="panel tip-box">
        Run the Python analyzer with <code>--gaze-file your_export.csv</code> to compute PES scores and attention heatmaps.
      </div>
    </aside>
  </main>
  <div id="gazeDot"></div>

  <script>
    const POSTERS = {json.dumps(poster_options)};
    const SAMPLE_INTERVAL_MS = 80;
    const GAZE_SMOOTHING_WINDOW = 6;
    const MAX_GAZE_JUMP_PX = 260;
    const CALIBRATION_POINTS = [
      [0.12, 0.16], [0.5, 0.16], [0.88, 0.16],
      [0.12, 0.5], [0.5, 0.5], [0.88, 0.5],
      [0.12, 0.84], [0.5, 0.84], [0.88, 0.84]
    ];
    const AOI_COLORS = {{
      Product: "#3b82f6",
      Headline: "#8b5cf6",
      CTA: "#f59e0b",
      Price: "#10b981",
      logo: "#ec4899"
    }};

    const poster = document.getElementById("poster");
    const posterWrap = document.getElementById("posterWrap");
    const posterTitle = document.getElementById("posterTitle");
    const posterSelect = document.getElementById("posterSelect");
    const posterSearch = document.getElementById("posterSearch");
    const posterSearchStatus = document.getElementById("posterSearchStatus");
    const aoiToggle = document.getElementById("aoiToggle");
    const cameraMount = document.getElementById("cameraMount");
    const gazeDot = document.getElementById("gazeDot");
    const statusEl = document.getElementById("status");
    const sampleCountEl = document.getElementById("sampleCount");
    const currentAoiEl = document.getElementById("currentAoi");
    const accuracyStatusEl = document.getElementById("accuracyStatus");
    const irisStatusEl = document.getElementById("irisStatus");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const downloadBtn = document.getElementById("downloadBtn");

    let samples = [];
    let currentPoster = POSTERS[0];
    let POSTER_NAME = currentPoster.name;
    let IMAGE_WIDTH = currentPoster.width;
    let IMAGE_HEIGHT = currentPoster.height;
    let AOIS = currentPoster.aois;
    let tracking = false;
    let lastSampleAt = 0;
    let calibrationIndex = 0;
    let accuracyIndex = 0;
    let calibrationLayer = null;
    let calibrationTarget = null;
    let calibrationPoint = null;
    let calibrationMode = "calibration";
    let lastCalibrationAdvanceAt = 0;
    let lastPrediction = null;
    let accuracyErrors = [];
    let gazeHistory = [];
    let lastStableGaze = null;
    let faceMesh = null;
    let irisRunning = false;
    let irisProcessing = false;
    let irisVideo = null;
    let irisState = createEmptyIrisState();
    let sessionActive = false;
    let sessionToken = 0;

    function createEmptyIrisState() {{
      return {{
        face_detected: 0,
        eyes_detected: 0,
        iris_tracking_ok: 0,
        iris_sample_age_ms: "",
        iris_left_x: "",
        iris_left_y: "",
        iris_right_x: "",
        iris_right_y: "",
        iris_diameter_px: ""
      }};
    }}

    function setStatus(text, tone = "ready") {{
      statusEl.textContent = text;
      statusEl.className = `metric-value status-badge ${{tone}}`;
    }}

    function formatPosterName(name) {{
      const base = name.replace(/\\.[^.]+$/, "").replace(/_jpg\\.rf\\.[^.]+$/i, "");
      return base.length > 48 ? `${{base.slice(0, 45)}}…` : base;
    }}

    function setCurrentAoi(text) {{
      if (!text || text === "—" || text === "-") {{
        currentAoiEl.textContent = "—";
        currentAoiEl.className = "metric-value";
        return;
      }}
      currentAoiEl.textContent = text;
      currentAoiEl.className = "metric-value aoi-pill";
    }}

    function setPickerEnabled(enabled) {{
      posterSelect.disabled = !enabled;
      posterSearch.disabled = !enabled;
    }}

    function resetSessionState() {{
      samples = [];
      tracking = false;
      calibrationIndex = 0;
      accuracyIndex = 0;
      accuracyErrors = [];
      gazeHistory = [];
      lastStableGaze = null;
      sampleCountEl.textContent = "0";
      setCurrentAoi("—");
      accuracyStatusEl.textContent = "—";
      irisStatusEl.textContent = "—";
      irisState = createEmptyIrisState();
      downloadBtn.disabled = true;
      finishCalibrationLayer();
      setStatus("Ready", "ready");
    }}

    function loadPosterByIndex(index) {{
      currentPoster = POSTERS[index];
      POSTER_NAME = currentPoster.name;
      IMAGE_WIDTH = currentPoster.width;
      IMAGE_HEIGHT = currentPoster.height;
      AOIS = currentPoster.aois;
      posterTitle.textContent = formatPosterName(POSTER_NAME);
      posterTitle.title = POSTER_NAME;
      poster.src = currentPoster.imageSrc;
      poster.alt = POSTER_NAME;
      resetSessionState();
      renderAois();
    }}

    function populatePosterSelect() {{
      posterSelect.replaceChildren();
      POSTERS.forEach((posterOption, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = formatPosterName(posterOption.name);
        option.title = posterOption.name;
        option.dataset.search = posterOption.name.toLowerCase();
        posterSelect.appendChild(option);
      }});
      posterSelect.value = "0";
      posterSearchStatus.textContent = `${{POSTERS.length}} posters available`;
    }}

    function filterPosterOptions() {{
      const query = posterSearch.value.trim().toLowerCase();
      let firstVisible = null;
      let visibleCount = 0;
      Array.from(posterSelect.options).forEach((option) => {{
        const visible = !query || option.dataset.search.includes(query);
        option.hidden = !visible;
        option.disabled = !visible;
        if (visible && firstVisible === null) {{
          firstVisible = option.value;
        }}
        if (visible) {{
          visibleCount += 1;
        }}
      }});
      posterSelect.disabled = visibleCount === 0;
      posterSearchStatus.textContent = visibleCount
        ? `${{visibleCount}} poster${{visibleCount === 1 ? "" : "s"}} found`
        : "No posters match this search";
      posterSearchStatus.classList.toggle("error", visibleCount === 0);
      if (firstVisible !== null && posterSelect.selectedOptions[0]?.hidden) {{
        posterSelect.value = firstVisible;
        loadPosterByIndex(Number(firstVisible));
      }}
    }}

    function renderAois() {{
      document.querySelectorAll(".aoi").forEach((node) => node.remove());
      posterWrap.classList.toggle("aoi-hidden", !aoiToggle.checked);
      const rect = poster.getBoundingClientRect();
      const scaleX = rect.width / IMAGE_WIDTH;
      const scaleY = rect.height / IMAGE_HEIGHT;

      AOIS.forEach((aoi) => {{
        const box = document.createElement("div");
        box.className = "aoi";
        const color = AOI_COLORS[aoi.label] || "#6366f1";
        box.style.setProperty("--aoi-color", color);
        box.style.left = `${{aoi.x1 * scaleX}}px`;
        box.style.top = `${{aoi.y1 * scaleY}}px`;
        box.style.width = `${{(aoi.x2 - aoi.x1) * scaleX}}px`;
        box.style.height = `${{(aoi.y2 - aoi.y1) * scaleY}}px`;
        box.innerHTML = `<span>${{aoi.label}}</span>`;
        posterWrap.appendChild(box);
      }});
    }}

    function attachCameraPreview() {{
      const container = document.getElementById("webgazerVideoContainer");
      if (!container || container.parentElement === cameraMount) {{
        return;
      }}

      cameraMount.replaceChildren(container);
      container.style.position = "relative";
      container.style.top = "auto";
      container.style.left = "auto";
      container.style.width = "260px";
      container.style.height = "195px";
      container.style.zIndex = "1";
      irisVideo = document.getElementById("webgazerVideoFeed");
    }}

    function averageLandmarks(landmarks, indexes) {{
      const total = indexes.reduce((point, landmarkIndex) => {{
        const landmark = landmarks[landmarkIndex];
        return {{
          x: point.x + landmark.x,
          y: point.y + landmark.y
        }};
      }}, {{ x: 0, y: 0 }});

      return {{
        x: total.x / indexes.length,
        y: total.y / indexes.length
      }};
    }}

    function landmarkDistancePx(first, second, width, height) {{
      return Math.hypot(
        (first.x - second.x) * width,
        (first.y - second.y) * height
      );
    }}

    function updateIrisStatus(text) {{
      irisStatusEl.textContent = text;
    }}

    function onFaceMeshResults(results) {{
      const landmarks = results.multiFaceLandmarks && results.multiFaceLandmarks[0];

      if (!landmarks || landmarks.length < 478) {{
        irisState = {{
          ...createEmptyIrisState(),
          face_detected: landmarks ? 1 : 0
        }};
        updateIrisStatus(landmarks ? "No iris" : "No face");
        return;
      }}

      const width = irisVideo ? irisVideo.videoWidth : 0;
      const height = irisVideo ? irisVideo.videoHeight : 0;
      const leftIris = averageLandmarks(landmarks, [468, 469, 470, 471, 472]);
      const rightIris = averageLandmarks(landmarks, [473, 474, 475, 476, 477]);
      const leftDiameter = width && height
        ? landmarkDistancePx(landmarks[469], landmarks[471], width, height)
        : 0;
      const rightDiameter = width && height
        ? landmarkDistancePx(landmarks[474], landmarks[476], width, height)
        : 0;
      const irisDiameter = leftDiameter && rightDiameter
        ? (leftDiameter + rightDiameter) / 2
        : Math.max(leftDiameter, rightDiameter);

      irisState = {{
        face_detected: 1,
        eyes_detected: 1,
        iris_tracking_ok: irisDiameter >= 3 ? 1 : 0,
        iris_sample_age_ms: 0,
        iris_left_x: Number(leftIris.x.toFixed(4)),
        iris_left_y: Number(leftIris.y.toFixed(4)),
        iris_right_x: Number(rightIris.x.toFixed(4)),
        iris_right_y: Number(rightIris.y.toFixed(4)),
        iris_diameter_px: irisDiameter ? Number(irisDiameter.toFixed(2)) : ""
      }};
      updateIrisStatus(irisState.iris_tracking_ok ? "Good" : "Weak");
    }}

    function getIrisSample() {{
      return {{
        ...irisState,
        iris_sample_age_ms: irisState.eyes_detected ? Math.round(irisState.iris_sample_age_ms || 0) : ""
      }};
    }}

    function resetGazeSmoothing() {{
      gazeHistory = [];
      lastStableGaze = null;
    }}

    function getSmoothedGaze(rawPoint) {{
      if (lastStableGaze) {{
        const jumpDistance = Math.hypot(rawPoint.x - lastStableGaze.x, rawPoint.y - lastStableGaze.y);
        if (jumpDistance > MAX_GAZE_JUMP_PX) {{
          resetGazeSmoothing();
          return null;
        }}
      }}

      gazeHistory.push(rawPoint);
      if (gazeHistory.length > GAZE_SMOOTHING_WINDOW) {{
        gazeHistory.shift();
      }}

      const smoothed = gazeHistory.reduce((total, point) => ({{
        x: total.x + point.x,
        y: total.y + point.y
      }}), {{ x: 0, y: 0 }});

      lastStableGaze = {{
        x: smoothed.x / gazeHistory.length,
        y: smoothed.y / gazeHistory.length
      }};
      return lastStableGaze;
    }}

    async function ensureFaceMesh() {{
      if (faceMesh || !window.FaceMesh) {{
        return faceMesh;
      }}

      faceMesh = new FaceMesh({{
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${{file}}`
      }});
      faceMesh.setOptions({{
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      }});
      faceMesh.onResults(onFaceMeshResults);
      return faceMesh;
    }}

    async function processIrisFrame() {{
      if (!irisRunning || irisProcessing) {{
        return;
      }}

      if (!irisVideo) {{
        irisVideo = document.getElementById("webgazerVideoFeed");
      }}

      if (!irisVideo || irisVideo.readyState < 2) {{
        window.requestAnimationFrame(processIrisFrame);
        return;
      }}

      const mesh = await ensureFaceMesh();
      if (!mesh) {{
        updateIrisStatus("Unavailable");
        return;
      }}

      irisProcessing = true;
      try {{
        await mesh.send({{ image: irisVideo }});
      }} catch (error) {{
        console.warn("Iris quality tracking failed", error);
        updateIrisStatus("Failed");
      }} finally {{
        irisProcessing = false;
      }}

      if (irisState.eyes_detected) {{
        irisState.iris_sample_age_ms = Number(irisState.iris_sample_age_ms || 0) + 80;
      }}
      window.setTimeout(processIrisFrame, 80);
    }}

    function startIrisTracking() {{
      irisState = createEmptyIrisState();
      irisRunning = true;
      updateIrisStatus("Starting");
      processIrisFrame();
    }}

    function stopIrisTracking() {{
      irisRunning = false;
      irisProcessing = false;
    }}

    function viewportToPosterPoint(x, y) {{
      const rect = poster.getBoundingClientRect();
      if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {{
        return null;
      }}

      return {{
        x: Math.round(((x - rect.left) / rect.width) * (IMAGE_WIDTH - 1)),
        y: Math.round(((y - rect.top) / rect.height) * (IMAGE_HEIGHT - 1))
      }};
    }}

    function getAoiHit(point) {{
      if (!point) {{
        return "";
      }}

      const hits = AOIS
        .filter((aoi) => (
          point.x >= aoi.x1 && point.x <= aoi.x2 &&
          point.y >= aoi.y1 && point.y <= aoi.y2
        ))
        .map((aoi) => aoi.label);

      return hits.join("|");
    }}

    function recordTrainingPoint(screenX, screenY, repeats = 6) {{
      if (!webgazer.recordScreenPosition) {{
        return;
      }}

      for (let index = 0; index < repeats; index += 1) {{
        webgazer.recordScreenPosition(screenX, screenY, "click");
      }}
    }}

    function finishCalibrationLayer() {{
      if (calibrationLayer) {{
        calibrationLayer.remove();
        calibrationLayer = null;
        calibrationTarget = null;
      }}
      calibrationPoint = null;
    }}

    function updateAccuracyStatus() {{
      if (!accuracyErrors.length) {{
        accuracyStatusEl.textContent = "—";
        return;
      }}

      const averageError = accuracyErrors.reduce((sum, value) => sum + value, 0) / accuracyErrors.length;
      accuracyStatusEl.textContent = `${{Math.round(averageError)}} px avg`;
    }}

    function recordAccuracyClick(screenX, screenY) {{
      if (lastPrediction) {{
        const error = Math.hypot(lastPrediction.x - screenX, lastPrediction.y - screenY);
        accuracyErrors.push(error);
        updateAccuracyStatus();
      }}
    }}

    function startRecording() {{
      if (!sessionActive) {{
        return;
      }}
      finishCalibrationLayer();
      tracking = true;
      setStatus("Recording", "recording");
    }}

    function advanceCalibration(event) {{
      event.preventDefault();
      event.stopPropagation();

      if (!calibrationPoint) {{
        return;
      }}

      const now = performance.now();
      if (now - lastCalibrationAdvanceAt < 250) {{
        return;
      }}
      lastCalibrationAdvanceAt = now;

      const [screenX, screenY] = calibrationPoint;
      recordTrainingPoint(screenX, screenY);

      if (calibrationMode === "accuracy") {{
        recordAccuracyClick(screenX, screenY);
        accuracyIndex += 1;
        showAccuracyPoint();
        return;
      }}

      calibrationIndex += 1;
      showCalibrationPoint();
    }}

    function ensureCalibrationLayer() {{
      if (calibrationLayer) {{
        return;
      }}

      calibrationLayer = document.createElement("div");
      calibrationLayer.className = "calibration-layer";
      calibrationLayer.setAttribute("aria-label", "Calibration area");

      calibrationTarget = document.createElement("div");
      calibrationTarget.className = "calibration-target";
      calibrationLayer.appendChild(calibrationTarget);

      ["pointerdown", "mousedown", "touchstart", "click"].forEach((eventName) => {{
        calibrationLayer.addEventListener(eventName, advanceCalibration, true);
      }});

      document.body.appendChild(calibrationLayer);
    }}

    function handleCalibrationKey(event) {{
      if (!calibrationLayer || ![" ", "Enter"].includes(event.key)) {{
        return;
      }}

      advanceCalibration(event);
    }}

    function moveCalibrationTarget(point, isAccuracy = false) {{
      ensureCalibrationLayer();
      calibrationPoint = point;
      calibrationTarget.className = isAccuracy
        ? "calibration-target accuracy"
        : "calibration-target";
      calibrationTarget.style.left = `${{point[0] - 17}}px`;
      calibrationTarget.style.top = `${{point[1] - 17}}px`;
    }}

    function showAccuracyPoint() {{
      if (!sessionActive) {{
        return;
      }}
      calibrationMode = "accuracy";

      if (accuracyIndex >= CALIBRATION_POINTS.length) {{
        startRecording();
        return;
      }}

      const [xRatio, yRatio] = CALIBRATION_POINTS[accuracyIndex];
      moveCalibrationTarget([
        Math.round(window.innerWidth * xRatio),
        Math.round(window.innerHeight * yRatio)
      ], true);
      setStatus(`Accuracy ${{accuracyIndex + 1}}/${{CALIBRATION_POINTS.length}}`, "calibrating");
    }}

    function showCalibrationPoint() {{
      if (!sessionActive) {{
        return;
      }}
      calibrationMode = "calibration";

      if (calibrationIndex >= CALIBRATION_POINTS.length) {{
        if (calibrationTarget) {{
          calibrationTarget.className = "calibration-target accuracy";
        }}
        accuracyIndex = 0;
        setStatus("Accuracy check", "calibrating");
        window.setTimeout(showAccuracyPoint, 300);
        return;
      }}

      const [xRatio, yRatio] = CALIBRATION_POINTS[calibrationIndex];
      moveCalibrationTarget([
        Math.round(window.innerWidth * xRatio),
        Math.round(window.innerHeight * yRatio)
      ]);
      setStatus(`Calibrating ${{calibrationIndex + 1}}/${{CALIBRATION_POINTS.length}}`, "calibrating");
    }}

    async function startTracking() {{
      const currentSessionToken = ++sessionToken;
      sessionActive = true;
      samples = [];
      calibrationIndex = 0;
      accuracyIndex = 0;
      accuracyErrors = [];
      resetGazeSmoothing();
      lastSampleAt = 0;
      sampleCountEl.textContent = "0";
      setCurrentAoi("—");
      accuracyStatusEl.textContent = "—";
      downloadBtn.disabled = true;
      startBtn.disabled = true;
      stopBtn.disabled = false;
      setPickerEnabled(false);
      setStatus("Starting camera", "starting");

      webgazer.params.faceMeshSolutionPath =
        "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh";

      await webgazer
        .setRegression("ridge")
        .setGazeListener((data, elapsedTime) => {{
          if (!data) {{
            return;
          }}

          const smoothedGaze = getSmoothedGaze({{ x: data.x, y: data.y }});
          if (!smoothedGaze) {{
            if (tracking) {{
              setCurrentAoi("Unstable gaze");
            }}
            return;
          }}

          lastPrediction = smoothedGaze;
          gazeDot.style.transform = `translate(${{smoothedGaze.x}}px, ${{smoothedGaze.y}}px)`;

          const now = performance.now();
          if (!tracking || now - lastSampleAt < SAMPLE_INTERVAL_MS) {{
            return;
          }}
          lastSampleAt = now;

          const point = viewportToPosterPoint(smoothedGaze.x, smoothedGaze.y);
          if (!point) {{
            setCurrentAoi("Outside poster");
            return;
          }}

          const aoiHit = getAoiHit(point);
          setCurrentAoi(aoiHit || "Background");
          samples.push({{
            timestamp_ms: Math.round(elapsedTime),
            gaze_x: point.x,
            gaze_y: point.y,
            duration_ms: SAMPLE_INTERVAL_MS,
            aoi_hit: aoiHit,
            poster: POSTER_NAME,
            ...getIrisSample()
          }});
          sampleCountEl.textContent = String(samples.length);
        }})
        .begin();

      if (!sessionActive || currentSessionToken !== sessionToken) {{
        webgazer.pause();
        return;
      }}

      webgazer.showVideoPreview(true)
        .showPredictionPoints(false)
        .applyKalmanFilter(true);
      attachCameraPreview();
      startIrisTracking();

      showCalibrationPoint();
    }}

    function stopTracking() {{
      sessionActive = false;
      sessionToken += 1;
      tracking = false;
      stopBtn.disabled = true;
      startBtn.disabled = false;
      setPickerEnabled(true);
      downloadBtn.disabled = samples.length === 0;
      setStatus(samples.length ? "Stopped" : "No samples", "stopped");
      finishCalibrationLayer();
      stopIrisTracking();
      webgazer.pause();
    }}

    function recoverFromStartFailure() {{
      sessionActive = false;
      sessionToken += 1;
      tracking = false;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      setPickerEnabled(true);
      finishCalibrationLayer();
      stopIrisTracking();
      webgazer.pause();
      setStatus("Camera blocked or WebGazer failed", "error");
    }}

    function csvEscape(value) {{
      const text = String(value ?? "");
      if (/[",\\n]/.test(text)) {{
        return `"${{text.replace(/"/g, '""')}}"`;
      }}
      return text;
    }}

    function downloadCsv() {{
      const headers = [
        "timestamp_ms", "gaze_x", "gaze_y", "duration_ms", "aoi_hit", "poster",
        "face_detected", "eyes_detected", "iris_tracking_ok", "iris_sample_age_ms",
        "iris_left_x", "iris_left_y", "iris_right_x", "iris_right_y", "iris_diameter_px"
      ];
      const rows = [headers.join(",")].concat(
        samples.map((sample) => headers.map((header) => csvEscape(sample[header])).join(","))
      );
      const blob = new Blob([rows.join("\\n")], {{ type: "text/csv;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const safePoster = POSTER_NAME.replace(/\\.[^.]+$/, "").replace(/[^a-z0-9_-]+/gi, "_");
      link.href = url;
      link.download = `${{safePoster}}_webgazer.csv`;
      link.click();
      URL.revokeObjectURL(url);
    }}

    startBtn.addEventListener("click", () => {{
      startTracking().catch((error) => {{
        console.error(error);
        recoverFromStartFailure();
      }});
    }});
    stopBtn.addEventListener("click", stopTracking);
    downloadBtn.addEventListener("click", downloadCsv);
    posterSelect.addEventListener("change", () => {{
      if (tracking) {{
        return;
      }}
      loadPosterByIndex(Number(posterSelect.value));
    }});
    posterSearch.addEventListener("input", filterPosterOptions);
    posterSearch.addEventListener("search", filterPosterOptions);
    aoiToggle.addEventListener("change", renderAois);
    document.addEventListener("keydown", handleCalibrationKey, true);
    window.addEventListener("resize", () => {{
      renderAois();
      if (calibrationLayer) {{
        if (calibrationMode === "accuracy") {{
          showAccuracyPoint();
        }} else {{
          showCalibrationPoint();
        }}
      }}
    }});
    poster.addEventListener("load", renderAois);
    populatePosterSelect();
    poster.src = currentPoster.imageSrc;
    poster.alt = currentPoster.name;
    renderAois();
  </script>
</body>
</html>
"""


def write_webgazer_session(output_path, poster_name, image_shape, boxes):
    """Write the WebGazer capture page for a selected poster."""
    poster_path = os.path.join(image_path, poster_name)
    poster_data_uri = get_image_data_uri(poster_path)
    page = build_webgazer_html(poster_name, poster_data_uri, boxes, image_shape)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(page)

    print(f"\nWebGazer session page created: {os.path.abspath(output_path)}")
    print("\nUse it like this:")
    print("1. Serve this folder locally: python3 -m http.server 8000")
    print(f"2. Open: http://localhost:8000/{os.path.basename(output_path)}")
    print("3. Click Start, allow camera access, calibrate, record, then Download CSV.")
    print(
        "4. Analyze it with: python3 aoi_visualizer.py "
        f"--poster \"{poster_name}\" --gaze-file DOWNLOADED_CSV --raw-coordinate-mode pixel"
    )


def get_poster_url(poster_name):
    """Return a URL path for a poster served from the project root."""
    return "Capstone.yolov8/train/images/" + quote(poster_name)


def build_poster_option(poster_name):
    """Build poster metadata for the browser poster chooser."""
    poster_path = os.path.join(image_path, poster_name)
    image = read_image(poster_path)
    if image is None:
        raise ValueError(f"Unable to read poster: {poster_name}")

    label_lines = load_labels(label_path, poster_name)
    _, boxes = draw_aoi_boxes(image, label_lines, classes)
    height, width = image.shape[:2]

    return {
        "name": poster_name,
        "imageSrc": get_poster_url(poster_name),
        "width": width,
        "height": height,
        "aois": [
            {"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            for label, x1, y1, x2, y2 in boxes
        ],
    }


def write_webgazer_gallery(output_path):
    """Write one WebGazer page with a dropdown for all labelled posters."""
    poster_names = list_available_posters(image_path)
    poster_options = []

    for poster_name in poster_names:
        poster_stem, _ = os.path.splitext(poster_name)
        label_file = os.path.join(label_path, f"{poster_stem}.txt")
        if os.path.isfile(label_file):
            poster_options.append(build_poster_option(poster_name))

    if not poster_options:
        raise FileNotFoundError("No posters with matching label files were found.")

    first = poster_options[0]
    boxes = [
        (aoi["label"], aoi["x1"], aoi["y1"], aoi["x2"], aoi["y2"])
        for aoi in first["aois"]
    ]
    image_shape = (first["height"], first["width"], 3)
    page = build_webgazer_html(
        first["name"],
        first["imageSrc"],
        boxes,
        image_shape,
        poster_options=poster_options,
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(page)

    print(f"\nWebGazer poster chooser created: {os.path.abspath(output_path)}")
    print(f"Included posters: {len(poster_options)}")
    print("\nUse it like this:")
    print("1. Serve this folder locally: python3 -m http.server 8001")
    print(f"2. Open: http://localhost:8001/{os.path.basename(output_path)}")
    print("3. Choose a poster, Start, calibrate, run accuracy check, record, Download CSV.")


def main():
    """Run one offline PES analysis session for one poster and Tobii export."""
    global aoi_boxes
    global attention_scores

    args = parse_args()

    print("Poster PES Analyzer")
    if args.webgazer_gallery:
        write_webgazer_gallery(args.webgazer_gallery)
        return

    selected_poster, image = load_poster(image_path, args.poster)
    lines = load_labels(label_path, selected_poster)
    preview_image, aoi_boxes = draw_aoi_boxes(image, lines, classes)
    attention_scores = initialize_attention_scores(aoi_boxes)

    if args.webgazer_session:
        write_webgazer_session(
            args.webgazer_session,
            selected_poster,
            image.shape,
            aoi_boxes,
        )
        return

    gaze_file = request_gaze_file(args.gaze_file)

    print(f"\nAnalyzing Tobii export: {gaze_file}")
    if args.stimulus_filter:
        print(f"Stimulus filter: {args.stimulus_filter}")

    stats = analyze_raw_gaze_file(
        gaze_file,
        image.shape,
        coordinate_mode=args.raw_coordinate_mode,
        stimulus_filter=args.stimulus_filter,
    )

    print_raw_import_summary(stats)
    print_attention_report()

    if not args.no_display:
        show_aoi_preview(preview_image)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}")
