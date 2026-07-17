# Prediction API

Phase 5A adds a minimal FastAPI app that serves the existing saved model
artifacts. It does not fetch WeatherAPI data, generate datasets, retrain
models, fit scalers, or replace the existing batch scripts.

## Run locally

Install runtime dependencies and the editable package, then start Uvicorn:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -e .
.\venv\Scripts\python.exe -m uvicorn wind_forecast.api:app --reload
```

## Run with Docker

Build the minimal API-serving image from the repository root:

```powershell
docker build -t wind-energy-forecast-api:phase6a .
```

Start the FastAPI app with Uvicorn:

```powershell
docker run --rm -p 8000:8000 wind-energy-forecast-api:phase6a
```

The container does not require WeatherAPI keys at startup. The image does not
include local notebooks, datasets, model artifacts, caches, or virtual
environments. To serve endpoints that need the current local data and model
artifacts, mount them read-only:

```powershell
docker run --rm -p 8000:8000 `
  --mount "type=bind,source=${PWD}\data,target=/app/data,readonly" `
  --mount "type=bind,source=${PWD}\models,target=/app/models,readonly" `
  wind-energy-forecast-api:phase6a
```

## Endpoints

- `GET /health`: returns API process health.
- `GET /model-info`: reports saved model, scaler, and feature-reference
  artifact readiness without loading TensorFlow models.
- `POST /predict`: predicts with the selected saved model.

## Phase 5B — Historical performance endpoint

`GET /api/v1/performance` exposes read-only historical prediction performance
from the explicitly selected artifact set. It does not alter model training,
prediction serving, CORS configuration, or any artifact.

Set `WIND_FORECAST_PERFORMANCE_ARTIFACT_DIR` to the directory that contains
the validated `predictions.csv` artifact. Optional `metrics.json` and
`run_summary.json` files supply persisted aggregate metrics and basic result
provenance. The endpoint never returns the configured directory or local paths.

Optional inclusive filters use ISO dates:

```text
GET /api/v1/performance?start_date=2026-01-01&end_date=2026-01-31
```

Example response:

```json
{
  "interval": {
    "requested_start_date": "2026-01-01",
    "requested_end_date": "2026-01-31",
    "available_start_date": "2025-01-01",
    "available_end_date": "2026-06-30",
    "returned_start_date": "2026-01-01",
    "returned_end_date": "2026-01-31"
  },
  "observation_count": 1,
  "metrics": {
    "r2": null,
    "mae": 50.0,
    "rmse": 50.0,
    "mape_percent": 4.17
  },
  "result": null,
  "observations": [
    {
      "date": "2026-01-01",
      "actual": 1200.0,
      "predicted": 1150.0,
      "error": -50.0,
      "absolute_error": 50.0
    }
  ]
}
```

The response contains the requested, available, and returned date bounds;
aggregate `r2`, `mae`, `rmse`, and `mape_percent`; optional non-sensitive
result provenance; and typed observations with `date`, `actual`, `predicted`,
`error`, and `absolute_error`.

Invalid date formats return `422`; an inverted interval returns `400`; an
interval with no observations returns `404`; and unavailable, missing, or
invalid artifacts return `503`.

The `/api/v1` contract evolves only through additive optional fields. Removing,
renaming, or changing the meaning or type of a published field requires a new
API version.

Example prediction request:

```json
{
  "target_type": "log",
  "records": [
    {
      "Date": "2026-01-01",
      "features": {
        "Average_Wind_Speed": 4.2,
        "Average_Temperature": 12.5,
        "Average_Wind_Direction": 180.0
      }
    }
  ]
}
```

`records[*].features` should contain feature-ready values using the project
English schema. The API aligns those values to the saved training feature order
from `data/processed/agg_data_ml.csv`; missing feature columns follow the
existing inference helper behavior.
