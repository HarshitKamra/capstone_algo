"""Training wrapper for YOLOv8 using the ultralytics package.

This is a convenience script to start training using a `data.yaml` file such
as `Capstone.yolov8/data.yaml`. It requires `ultralytics` to be installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a dataset.")
    parser.add_argument("--data", help="Path to data.yaml", required=True)
    parser.add_argument("--model", help="Backbone model (e.g. yolov8n.pt)", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="capstone")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception:
        print("ultralytics is not installed. Install with: pip install ultralytics")
        return

    model = YOLO(args.model)
    model.train(data=str(Path(args.data)), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, project=args.project, name=args.name)


if __name__ == "__main__":
    main()
"""Train YOLOv8 poster element detector on Capstone.yolov8 dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config.settings import DATASET_DIR, DATASET_YAML, DEFAULT_MODEL_WEIGHTS, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Capstone poster dataset")
    parser.add_argument("--data", type=Path, default=DATASET_YAML, help="Path to data.yaml")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model (yolov8n.pt, yolov8s.pt, etc.)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="", help="CUDA device id or 'cpu'")
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "runs" / "detect",
        help="Ultralytics project directory",
    )
    parser.add_argument("--name", default="capstone_poster", help="Run name")
    parser.add_argument(
        "--copy-weights",
        action="store_true",
        help="Copy best.pt to models/weights/best.pt after training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    valid_images = DATASET_DIR / "valid" / "images"
    if not valid_images.is_dir() or not any(valid_images.iterdir()):
        raise FileNotFoundError(
            "Validation split not found. Run first:\n"
            "  python training/split_dataset.py"
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics: pip install ultralytics") from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else args.project / args.name
    best_weights = save_dir / "weights" / "best.pt"

    print(f"\nTraining complete. Run directory: {save_dir}")
    if best_weights.is_file():
        print(f"Best weights: {best_weights}")
        if args.copy_weights:
            DEFAULT_MODEL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_weights, DEFAULT_MODEL_WEIGHTS)
            print(f"Copied to: {DEFAULT_MODEL_WEIGHTS}")
    else:
        print("Warning: best.pt not found in run directory.")


if __name__ == "__main__":
    main()
