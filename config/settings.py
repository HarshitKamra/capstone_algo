"""Central configuration loaded from environment variables with sensible defaults."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(os.getenv("CAPSTONE_DATASET_DIR", PROJECT_ROOT / "Capstone.yolov8"))
DATASET_YAML = DATASET_DIR / "data.yaml"
DATASET_IMAGE_DIR = DATASET_DIR / "train" / "images"
DATASET_LABEL_DIR = DATASET_DIR / "train" / "labels"

DEFAULT_MODEL_WEIGHTS = Path(
    os.getenv(
        "MODEL_WEIGHTS",
        PROJECT_ROOT / "models" / "weights" / "best.pt",
    )
)

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".avif")

# Class order matches Capstone.yolov8/data.yaml
CLASS_NAMES: dict[int, str] = {
    0: "CTA",
    1: "Headline",
    2: "Price",
    3: "Product",
    4: "logo",
}

CLASS_NAME_TO_ID = {name: class_id for class_id, name in CLASS_NAMES.items()}

CORE_AOI_ORDER = ["Product", "Headline", "CTA", "Price", "logo"]
BACKGROUND_LABEL = "Background"

RAW_DATA_DEFAULT_FIXATION_MS = 300
