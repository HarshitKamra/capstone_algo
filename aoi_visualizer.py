import argparse
import csv
import os
import re

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
        description="Offline Tobii Glasses poster PES analyzer"
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


def load_poster(folder_path):
    """Let the user select a poster and return its filename plus image array."""
    posters = list_available_posters(folder_path)

    if not posters:
        raise FileNotFoundError("No supported poster images found.")

    print("Available Posters:")
    for index, poster_name in enumerate(posters, start=1):
        print(f"{index}. {poster_name}")

    lower_to_original = {name.lower(): name for name in posters}

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
        "aoi_hit_mode": False,
        "coordinate_mode": False,
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
        stimulus_columns = find_stimulus_columns(fieldnames)

        stats["aoi_hit_mode"] = bool(columns["aoi_hit"] or columns["aoi_indicators"])
        stats["coordinate_mode"] = bool(columns["x"] and columns["y"])

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


def main():
    """Run one offline PES analysis session for one poster and Tobii export."""
    global aoi_boxes
    global attention_scores

    args = parse_args()

    print("Offline Tobii Glasses Poster PES Analyzer")
    gaze_file = request_gaze_file(args.gaze_file)

    selected_poster, image = load_poster(image_path)
    lines = load_labels(label_path, selected_poster)
    preview_image, aoi_boxes = draw_aoi_boxes(image, lines, classes)
    attention_scores = initialize_attention_scores(aoi_boxes)

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
