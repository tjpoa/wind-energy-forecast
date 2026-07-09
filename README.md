# National Wind Energy Production Forecast

Software engineering project for Portuguese wind-energy production forecasting.

This repository started as an academic Applied Artificial Intelligence project
and has been progressively refactored into a more maintainable Python
application: reusable package code, data validation, feature engineering,
automated tests, API serving, container support, CI, and local experiment
tracking around the existing forecasting workflow.

The project should be read as a software/backend/data engineering portfolio
project with AI/ML components. It does not currently claim cloud deployment,
workflow orchestration, or model-registry operations.

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
- Pytest coverage for schemas, configuration, features, validation, tracking,
  and API behavior.
- Ruff linting configured in `pyproject.toml`.
- GitHub Actions CI for tests, linting, and Docker image builds.
- Dockerfile for running the FastAPI app in a container.
- Optional local MLflow tracking for evaluation runs.

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
|   |-- tracking.py           # Local MLflow tracking helpers
|   |-- validation/           # Data validation modules
|   `-- data_sources/         # v2 source-specific ingestion helpers
|-- scripts/                  # Backward-compatible executable wrappers
|-- tests/                    # Pytest suite
|-- notebooks/                # Exploratory data prep, EDA, and training
|-- data/                     # Raw inputs and ignored generated outputs
|-- models/                   # Existing trained models and scalers
|-- docs/                     # Documentation index, roadmap, guides, and phase notes
|-- .github/workflows/ci.yml  # Test/lint CI and Docker build CI
|-- Dockerfile                # FastAPI container image
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
```

Do not commit `.env` or real API keys. The included `.env.example` is safe to
commit because it contains placeholders only.

## Common commands

Run tests:

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

Generate recent WeatherAPI feature data:

```powershell
.\venv\Scripts\python.exe .\scripts\process_api_data.py
```

Apply saved models to the latest generated API feature file:

```powershell
.\venv\Scripts\python.exe .\scripts\apply_models_to_api_data.py
```

Log that evaluation run to local MLflow tracking:

```powershell
.\venv\Scripts\python.exe .\scripts\apply_models_to_api_data.py --mlflow
```

Start the local MLflow UI:

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri .\mlruns
```

Start the FastAPI app locally:

```powershell
.\venv\Scripts\python.exe -m uvicorn wind_forecast.api:app --reload
```

Build and run the API Docker image:

```powershell
docker build -t wind-energy-forecast-api:phase6a .
docker run --rm -p 8000:8000 wind-energy-forecast-api:phase6a
```

Run the container with local data and model artifacts mounted read-only:

```powershell
docker run --rm -p 8000:8000 `
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
- Generated processed CSV files and MLflow runs remain local and ignored by Git.
- Local baseline-training outputs under `outputs/training/` remain ignored by
  Git.
- V2 data-source work uses separate `data/raw/v2/` and `data/processed/v2/`
  paths so the v1 baseline is not overwritten.

## Continuous integration

GitHub Actions currently runs:

- `python -m pytest`
- `python -m ruff check .`
- `docker build --file Dockerfile --tag wind-energy-forecast-api:ci .`

The standard CI jobs do not require real WeatherAPI credentials.

## Current limitations

- Full tuned model training is still notebook-based in
  `notebooks/Modeling.ipynb`; the CLI covers a lightweight baseline.
- MLflow support is local tracking only; MLflow model registry is not
  implemented.
- There is no cloud deployment.
- Airflow orchestration has not been implemented.
- PySpark processing has not been implemented.
- The FastAPI service is a local/container serving interface, not a deployed
  production service.

See `docs/README.md` for the documentation index,
`docs/ML_ENGINEERING_ROADMAP.md` for the longer engineering roadmap, and
`docs/PHASE_4.md` for the baseline training CLI, model card, and data card.
