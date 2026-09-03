Release notes and runbook
=========================

This file collects the key commands and steps to run, test, and deploy the project.

Run locally
-----------
- Install deps:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- Run tests:

```bash
python -m pytest -q
```

- Run the Streamlit app locally:

```bash
streamlit run app/app.py
```

Docker (local)
--------------
- Build image:

```bash
docker build -t capstone-poster:latest .
```

- Run with docker-compose (dev):

```bash
docker compose up --build
```

Production (example)
--------------------
- Use `docker-compose.prod.yml` with env vars. Example:

```bash
export MODEL_WEIGHTS=/path/to/best.pt
export DOCKER_IMAGE=myuser/capstone-poster:latest
docker compose -f docker-compose.prod.yml up -d --build
```

CI / Registry
-------------
- The GitHub Actions workflow runs tests and builds an image. To enable pushing images set these repo secrets:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN`

What remains (external inputs or optional work)
----------------------------------------------
- Provide real YOLOv8 weights (or path via `MODEL_WEIGHTS`) to run genuine inference.
- Provide registry credentials for CI image push.
- Configure monitoring and production logging (target platform specifics required).
- Optional: run full UI integration tests in CI using Playwright/Selenium for the Streamlit UI.

If you want, I can prepare a PR branch with these files committed and a short commit message. Tell me if you'd like that, or if I should proceed to add production monitoring scaffolding next.
