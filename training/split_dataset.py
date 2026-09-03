"""Create stratified train/valid split for the Capstone YOLO dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from config.settings import DATASET_DIR, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Split Capstone YOLO dataset into train/valid")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of images reserved for validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to Capstone.yolov8 dataset root",
    )
    return parser.parse_args()


def collect_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    image_dir = dataset_dir / "train" / "images"
    label_dir = dataset_dir / "train" / "labels"
    pairs: list[tuple[Path, Path]] = []

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            pairs.append((image_path, label_path))

    return pairs


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    random.seed(args.seed)

    pairs = collect_pairs(dataset_dir)
    if not pairs:
        raise FileNotFoundError(f"No labelled images found under {dataset_dir}")

    random.shuffle(pairs)
    val_count = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    valid_image_dir = dataset_dir / "valid" / "images"
    valid_label_dir = dataset_dir / "valid" / "labels"
    test_image_dir = dataset_dir / "test" / "images"
    test_label_dir = dataset_dir / "test" / "labels"

    for directory in (valid_image_dir, valid_label_dir, test_image_dir, test_label_dir):
        directory.mkdir(parents=True, exist_ok=True)

    def copy_pairs(target_image_dir: Path, target_label_dir: Path, items: list[tuple[Path, Path]]):
        for image_path, label_path in items:
            shutil.copy2(image_path, target_image_dir / image_path.name)
            shutil.copy2(label_path, target_label_dir / label_path.name)

    copy_pairs(valid_image_dir, valid_label_dir, val_pairs)

    # Use the same validation set as test holdout until a dedicated test set exists.
    for path in test_image_dir.glob("*"):
        path.unlink()
    for path in test_label_dir.glob("*"):
        path.unlink()
    copy_pairs(test_image_dir, test_label_dir, val_pairs)

    print(f"Dataset root: {dataset_dir}")
    print(f"Total labelled images: {len(pairs)}")
    print(f"Train (remaining in train/): {len(train_pairs)}")
    print(f"Valid: {len(val_pairs)}")
    print(f"Test (copy of valid): {len(val_pairs)}")
    print("\nNote: original train/images still contains all images.")
    print("Ultralytics uses train/ + valid/ paths from data.yaml.")
    print("Re-run with care — val files are overwritten each time.")


if __name__ == "__main__":
    main()
