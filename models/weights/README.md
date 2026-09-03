# Model weights directory
#
# Trained YOLO weights are NOT committed to git.
#
# To obtain weights:
#   1. Train locally:
#        python training/train.py
#      Checkpoints are saved under runs/detect/ and copied to this folder.
#
#   2. Or set MODEL_WEIGHTS to any compatible YOLOv8 .pt file:
#        set MODEL_WEIGHTS=C:\path\to\best.pt   (Windows)
#        export MODEL_WEIGHTS=/path/to/best.pt (Linux/macOS)
#
# Expected default filename: best.pt
