# National Wind Energy Production Forecast

Software engineering project for Portuguese wind-energy production forecasting.

This repository started as an academic Applied Artificial Intelligence project
and has been progressively refactored into a more maintainable Python
application: reusable package code, data validation, feature engineering,
automated tests, API serving, container support, CI, and local experiment
tracking around the existing forecasting workflow.

The project should be read as a software/backend/data engineering portfolio
project with AI/ML components. It does not currently claim cloud deployment,
workflow orchestration, or registry-based production serving.

## What this project includes

- `src/` Python package layout with reusable `wind_forecast` modules.
- Explicit WeatherAPI configuration through environment variables and `.env`.
- Historical and WeatherAPI data ingestion utilities.
- Data validation for raw production data, parsed weather data, and feature-ready
  v2 datasets.
- Feature engineering for calendar, cyclic, lag, and rolling-window features.
- Reproducible baseline training CLI for a lightweight historical holdout run.
- Backward-compatible batch scripts for API data processing and saved-model
  inference.
- FastAPI service for health checks, model artifact inspection, and prediction.
- Responsive React and TypeScript dashboard for historical forecast performance.
- Pytest coverage for schemas, configuration, features, validation, tracking,
  and API behavior.
- Ruff linting configured in `pyproject.toml`.
- GitHub Actions CI for tests, coverage, linting, Docker image builds, and
  container smoke checks.
- Dockerfile for running the FastAPI app in a non-root container with a health
  check.
- MLflow runs, artifacts, dataset lineage, and a local SQLite-backed Model
  Registry with explicit `candidate`/`champion` governance.

## Architecture

```text
wind-energy-forecast/
|-- src/wind_forecast/        # Reusable package code
|   |-- api.py                # FastAPI prediction service
|   |-- config.py             # Environment-based runtime configuration
|   |-- features.py           # Shared feature engineering
|   |-- ingestion.py          # WeatherAPI ingestion helpers
|   |-- inference.py          # Saved-model loading and prediction workflow
|   |-- training.py           # Lightweight baseline training helpers
|   |-- tracking.py           # MLflow server tracking helpers
|   |-- registry.py           # Candidate/champion governance
|   |-- artifacts.py          # Deterministic release bundles
|   |-- validation/           # Data validation modules
|   `-- data_sources/         # v2 source-specific ingestion helpers
|-- scripts/                  # Backward-compatible executable wrappers
|-- tests/                    # Pytest suite
|-- notebooks/                # Exploratory data prep, EDA, and training
|-- data/                     # Raw inputs and ignored generated outputs
|-- models/                   # Existing trained models and scalers
|-- frontend/                 # React dashboard and its container image
|-- docs/                     # Documentation index, roadmap, guides, and phase notes
|-- .github/workflows/ci.yml  # Matrix test/lint and Docker health CI
|-- Dockerfile                # FastAPI container image
|-- docker-compose.yml        # Local backend and frontend stack
`-- pyproject.toml            # Package, pytest, and Ruff configuration
```

The current tuned ANN/Optuna training workflow still lives in notebooks. A
lightweight baseline training CLI is available for reproducible historical
holdout runs, and scripts remain stable entry points for data processing,
training, and inference.

## Local setup

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pip install -e .
```

Optional notebook kernel:

```powershell
.\venv\Scripts\python.exe -m ipykernel install --user --name wind-energy-forecast --display-name "Python (wind-energy-forecast)"
```

Create a local environment file for WeatherAPI-backed workflows:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` locally:

```env
WEATHER_API_KEY=your_api_key_here
WEATHER_API_LOCATION=41.8345,-7.7889
WEATHER_API_DAYS=44
WEATHER_API_END_DATE=
WIND_FORECAST_CORS_ALLOW_ORIGINS=http://localhost:5173
```

Do not commit `.env` or real API keys. The included `.env.example` is safe to
commit because it contains placeholders only.

### Frontend setup

The frontend requires Node.js 22.12 or newer. Install its dependencies and
create a local environment file from the safe example:

```powershell
Set-Location .\frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

`VITE_API_BASE_URL` identifies the API endpoint exposed to the browser and
defaults to `http://localhost:8000` in the example file. Variables prefixed
with `VITE_` are bundled into browser code and must never contain secrets.

The development server is available at `http://localhost:5173`. The dashboard
loads historical model performance from `GET /api/v1/performance`. It provides
inclusive start/end date filters, MAE, RMSE, R² and observation-count cards,
actual-versus-predicted and signed-error charts, and the ten most recent
observations as an accessible table. Loading, empty, API error and invalid-date
states are reported explicitly, and each new request cancels the previous one.

The production values exposed by the historical artifact are daily sums of
15-minute MW readings. The dashboard uses that description for production,
MAE, RMSE and error instead of labelling the values as MWh.

Run the frontend checks from `frontend/`:

```powershell
npm run test
npm run lint
npm run build
```

### Docker Compose dashboard stack

Run the backend and frontend together from the repository root:

```powershell
docker compose up --build
```

Use `http://localhost:5173` to open the dashboard. Do not substitute
`127.0.0.1`: the backend CORS policy intentionally allows the exact documented
frontend origin. The API remains available at `http://localhost:8000`.

| Service | Published port | Purpose |
| --- | --- | --- |
| `frontend` | `5173` | Nginx serves the compiled React dashboard. |
| `backend` | `8000` | FastAPI serves `/health` and the dashboard API. |

The stack uses these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Public API URL compiled into the browser bundle. |
| `WIND_FORECAST_CORS_ALLOW_ORIGINS` | `http://localhost:5173` | Exact frontend origin accepted by FastAPI. |
| `WIND_FORECAST_PERFORMANCE_ARTIFACT_HOST_DIR` | `./outputs/training/baseline` | Host directory mounted at `/app/performance`. |
| `WIND_FORECAST_PERFORMANCE_ARTIFACT_DIR` | `/app/performance` | Fixed read-only artifact path inside the backend container. |

Variables prefixed with `VITE_` are public build-time configuration, never a
place for secrets. Rebuild the frontend image after changing
`VITE_API_BASE_URL`.

The Compose stack does not copy datasets, models, performance artifacts, local
environment files, or secrets into either image. It mounts `data/`, `models/`,
and the selected performance-artifact directory into the backend read-only.
The default performance directory must contain `predictions.csv`; optional
`metrics.json` and `run_summary.json` provide persisted metrics and provenance.

A fresh clone can start the services and pass `/health` without those generated
performance artifacts, but `/api/v1/performance` returns HTTP `503` and the
dashboard reports the API as unavailable until a valid local artifact set is
selected. For example, this checkout's smoke artifacts can be selected in
PowerShell before starting the stack:

```powershell
$env:WIND_FORECAST_PERFORMANCE_ARTIFACT_HOST_DIR=".\outputs\training\baseline_smoke"
docker compose up --build
```

Inspect or stop the stack from another terminal:

```powershell
docker compose ps
Invoke-RestMethod -Method Get -Uri http://localhost:8000/health
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:5173
docker compose down
```

## Common commands

Run tests with coverage:

```powershell
.\venv\Scripts\python.exe -m pytest
```

Run linting:

```powershell
.\venv\Scripts\python.exe -m ruff check .
```

Train a lightweight historical baseline:

```powershell
.\venv\Scripts\python.exe .\scripts\train_baseline.py --input data\processed\agg_data_ml.csv --output-dir outputs\training\baseline --overwrite
```

Tracking is enabled by default for this command. Start the local SQLite-backed
MLflow server first:

```powershell
.\venv\Scripts\python.exe -m mlflow server `
  --backend-store-uri sqlite:///var/mlflow/mlflow.db `
  --artifacts-destination ./var/mlflow/artifacts `
  --host 127.0.0.1 `
  --port 5000
```

Use `--tracking-mode off` only when an intentionally untracked run is needed.
After a clean tracked run, validate and register it as the candidate:

```powershell
.\venv\Scripts\python.exe .\scripts\register_candidate.py --run-id <RUN_ID>
```

Promotion is always explicit and optimistic-concurrency checked:

```powershell
.\venv\Scripts\python.exe .\scripts\promote_model.py promote `
  --expected-candidate-version <VERSION> `
  --expected-champion-version none `
  --approval-note "reviewed against the approved v1 contract"
```

Generate recent WeatherAPI feature data:

```powershell
.\venv\Scripts\python.exe .\scripts\process_api_data.py
```

Apply saved models to the latest generated API feature file:

```powershell
.\venv\Scripts\python.exe .\scripts\apply_models_to_api_data.py
```

Log that legacy evaluation run to the same MLflow server (still opt-in):

```powershell
.\venv\Scripts\python.exe .\scripts\apply_models_to_api_data.py --tracking-mode local
```

Start the FastAPI app locally:

```powershell
.\venv\Scripts\python.exe -m uvicorn --env-file .env wind_forecast.api:app --reload
```

`WIND_FORECAST_CORS_ALLOW_ORIGINS` accepts exact browser origins separated by
commas. If it is unset, only the Vite development origin
`http://localhost:5173` is allowed. For example, to add another local frontend:

```env
WIND_FORECAST_CORS_ALLOW_ORIGINS=http://localhost:5173,http://localhost:4173
```

Do not use `*`, URL paths, query strings, fragments, or credentials. The API
does not allow browser credentials; its CORS policy permits only `GET` and
`POST` requests from configured origins.

Build and run the API Docker image:

```powershell
docker build -t wind-energy-forecast-api:phase6a .
docker run --rm -d --name wind-energy-forecast-api-phase6a -p 8000:8000 wind-energy-forecast-api:phase6a
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
docker inspect --format '{{.State.Health.Status}}' wind-energy-forecast-api-phase6a
docker rm --force wind-energy-forecast-api-phase6a
```

Run the container with local data and model artifacts mounted read-only:

```powershell
docker run --rm -d --name wind-energy-forecast-api-phase6a -p 8000:8000 `
  --mount "type=bind,source=${PWD}\data,target=/app/data,readonly" `
  --mount "type=bind,source=${PWD}\models,target=/app/models,readonly" `
  wind-energy-forecast-api:phase6a
```

Regenerate historical training features through the notebook workflow:

```powershell
.\venv\Scripts\python.exe -m jupyter notebook .\notebooks\DataPreparation.ipynb
```

## API overview

The FastAPI app serves the existing saved model artifacts. It does not fetch
WeatherAPI data, generate datasets, train models, fit scalers, or replace the
batch scripts.

- `GET /health`: process health check.
- `GET /model-info`: model, scaler, and feature-reference artifact readiness.
- `POST /predict`: prediction endpoint for feature-ready records.

See `docs/PHASE_5.md` for request examples and Docker notes.

## Data and model workflow

The current v1 baseline uses tracked raw CSV files under `data/raw/`, existing
saved models and scalers under `models/`, and generated processed outputs under
`data/processed/`.

Important compatibility choices:

- New reusable code uses English column names such as `Date`,
  `Wind_Production`, `Average_Wind_Speed`, `Average_Temperature`, and
  `Average_Wind_Direction`.
- `wind_forecast.schemas` keeps compatibility with the original source headers
  and saved model/scaler feature order.
- Generated processed CSV files, MLflow SQLite/artifact state, receipts, and
  fetched release assets remain local and ignored by Git.
- Local baseline-training outputs under `outputs/training/` remain ignored by
  Git.
- V2 data-source work uses separate `data/raw/v2/` and `data/processed/v2/`
  paths so the v1 baseline is not overwritten.

## Continuous integration

GitHub Actions currently runs:

- `python -m pytest` with `pytest-cov` coverage reporting on Python 3.10 and
  3.11
- `python -m ruff check .` on Python 3.10 and 3.11
- `docker build --file Dockerfile --tag wind-energy-forecast-api:ci .`
- a container smoke test against `GET /health`
- a Docker health status check for the running API container
- `npm ci`, `npm run test`, `npm run lint`, and `npm run build` for the
  frontend on Node.js 22 LTS, with the npm cache keyed from
  `frontend/package-lock.json`
- `docker build --file frontend/Dockerfile --tag wind-energy-forecast-frontend:ci frontend`
- `docker compose config --quiet` to validate the Compose configuration
  without starting services

The standard CI jobs do not require real WeatherAPI credentials, local models,
local datasets, generated performance artifacts, or secrets. Compose validation
checks parsing, interpolation, and the service model; it does not start the
stack or verify runtime bind-mount contents. CI builds images for validation but
does not publish or deploy them.

## Current limitations

- Full tuned model training is still notebook-based in
  `notebooks/Modeling.ipynb`; the CLI covers a lightweight baseline.
- The MLflow Registry is local and is not consumed by the FastAPI service;
  serving continues to use the existing Keras/scaler paths.
- The first public artifact release remains blocked until source, licence,
  attribution, and redistribution permission are approved in
  `artifacts/catalog.json`.
- There is no cloud deployment.
- Airflow orchestration has not been implemented.
- PySpark processing has not been implemented.
- The FastAPI service is a local/container serving interface, not a deployed
  production service.
- The frontend integrates historical performance data only; it does not issue
  future prediction requests or provide live monitoring.

See `docs/README.md` for the documentation index,
`docs/ML_ENGINEERING_ROADMAP.md` for the longer engineering roadmap, and
`docs/PHASE_4.md` for the model lifecycle and `docs/REPRODUCIBILITY.md` for the
cross-machine artifact workflow.
