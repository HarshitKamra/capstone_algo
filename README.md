# Capstone — Advertisement Poster Analysis

Quickstart and developer notes.

Prerequisites
- Python 3.10+
- Recommended: create a virtualenv and install `pip install -r requirements.txt`.

Key components
- `aoi_visualizer.py` — legacy CLI for poster AOI preview and gaze analysis.
- `analysis/` — modular analysis helpers (poster, gaze, attention, detection).
- `models/detector.py` — YOLO inference helpers.
- `training/` — scaffolds for `train.py` and `inference.py` that use Ultralytics YOLOv8.

Notes
- The repository does not include trained YOLO weights. Set the environment variable `MODEL_WEIGHTS` or pass `--weights` to `training/inference.py`.
- The WebGazer capture UI is in `webgazer_choose_poster.html` and exports CSVs compatible with the gaze parser.

Quick commands

Run the smoke test suite:
```
python -m pytest -q
```

Run inference (requires a `.pt` checkpoint):
```
python training/inference.py --weights path/to/best.pt --source path/to/poster.jpg --out inference_out
```

Start training (requires `ultralytics`):
```
python training/train.py --data Capstone.yolov8/data.yaml --epochs 50
```

Run the Streamlit app (local):
```
pip install -r requirements.txt
streamlit run app/app.py
```

Build Docker image:
```
docker build -t capstone-poster:latest .
```

Run tests locally:
```
python -m pytest -q
```

CI and deployment notes
-----------------------
- To enable the CI workflow to push Docker images, add the following repository secrets in GitHub:
	- `DOCKERHUB_USERNAME` — your Docker Hub user name
	- `DOCKERHUB_TOKEN` — a Docker Hub access token (not your password)

- To run production Compose, set `MODEL_WEIGHTS` and `DOCKER_IMAGE` env vars, for example:

```
export MODEL_WEIGHTS=/path/to/best.pt
export DOCKER_IMAGE=myuser/capstone-poster:latest
docker compose -f docker-compose.prod.yml up -d --build
```

Security note: Do not commit trained weights to the repository; reference them via environment variables or an external artifact store.
