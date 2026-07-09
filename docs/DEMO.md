# Reproducible Demo

This guide shows a short local demo path for the wind-energy forecasting
project. It is intended for portfolio reviews and technical interviews.

The demo validates the package, tests, linting, API process, model-artifact
discovery, and Docker image build. It does not call WeatherAPI, retrieve REN or
ERA5-Land data, run notebooks, train models, fit scalers, or claim production
deployment.

## Demo Levels

| Level | What it proves | Requires local generated artifacts |
| --- | --- | --- |
| Quick validation | Tests and Ruff pass in the local Python environment. | No |
| API readiness | FastAPI starts and reports health/model artifact status. | No for `/health`; yes for full `/model-info` readiness |
| Prediction serving | Saved model artifacts can be loaded and called through `/predict`. | Yes |
| Docker smoke | The API image builds and the process starts in a container. | No for startup; yes for full model serving |

## Prerequisites

Install the project as described in `README.md`:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pip install -e .
```

No `.env` file or WeatherAPI key is required for the API demo.

For full prediction serving, the local workspace must include:

```text
models/best_model_log_target_ANN_Tuned.keras
models/scaler_X_log_ann.joblib
models/scaler_y_log_ann.joblib
data/processed/agg_data_ml.csv
```

The model files are tracked in Git. The processed feature table is generated
locally and intentionally ignored by Git.

Check whether the full-serving artifacts are available:

```powershell
Test-Path .\models\best_model_log_target_ANN_Tuned.keras
Test-Path .\models\scaler_X_log_ann.joblib
Test-Path .\models\scaler_y_log_ann.joblib
Test-Path .\data\processed\agg_data_ml.csv
```

## 1. Validate The Codebase

Run the same core checks used by CI:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m ruff check .
```

Expected result:

- Pytest completes with all tests passing.
- Ruff reports no lint failures.
- No live API credentials are needed.

## 2. Start The Local API

Start Uvicorn from the repository root:

```powershell
.\venv\Scripts\python.exe -m uvicorn wind_forecast.api:app --host 127.0.0.1 --port 8000
```

In a second PowerShell session, check process health:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

Check model and feature-reference readiness:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/model-info
```

If `data/processed/agg_data_ml.csv` is missing, `feature_reference_exists` will
be `false`. That is expected in a fresh clone until the local processed feature
table is regenerated.

## 3. Run A Prediction Request

Use this request only when `/model-info` shows that the selected model, scalers,
and feature reference are available.

```powershell
$body = @{
  target_type = "log"
  records = @(
    @{
      Date = "2026-01-01"
      features = @{
        Average_Wind_Speed = 4.2
        Average_Temperature = 12.5
        Average_Wind_Direction = 180.0
      }
    }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body $body
```

This is a serving smoke test. The API aligns provided feature values to the
saved training feature order and follows the current inference helper behavior
for missing feature columns. This payload is not a model-quality benchmark.

## 4. Build And Run The Docker Image

Build the API image:

```powershell
docker build -t wind-energy-forecast-api:demo .
```

Run the API without local artifact mounts:

```powershell
docker run --rm -p 8000:8000 wind-energy-forecast-api:demo
```

This is enough to test `/health`. For `/model-info` and `/predict` with local
artifacts, mount `data/` and `models/` read-only:

```powershell
docker run --rm -p 8000:8000 `
  --mount "type=bind,source=${PWD}\data,target=/app/data,readonly" `
  --mount "type=bind,source=${PWD}\models,target=/app/models,readonly" `
  wind-energy-forecast-api:demo
```

Then call the same endpoints:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/model-info
```

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Port `8000` is already in use | Another local API or container is running | Stop the other process or use another port. |
| `/model-info` reports missing feature reference | `data/processed/agg_data_ml.csv` is not present | Regenerate processed features through the existing notebook workflow before full serving. |
| `/predict` returns HTTP `503` | Required model, scaler, or feature-reference artifact is missing | Check `/model-info` and artifact paths. |
| TensorFlow model loading is slow | Saved Keras model load happens on first prediction | Wait for the first request or use `/model-info` for lightweight readiness. |
| Docker `/health` works but `/predict` fails | Container was started without mounted local artifacts | Run Docker with read-only `data/` and `models/` mounts. |

## What This Demo Proves

- The package imports and tests run without live external APIs.
- The FastAPI app starts and exposes stable health and metadata endpoints.
- Saved artifacts can be inspected without loading TensorFlow models.
- Full prediction serving is possible when local feature and model artifacts are
  present.
- The Docker image can run the API entry point.

## What This Demo Does Not Prove

- Production cloud deployment.
- Model registry operations or model promotion.
- Airflow orchestration.
- PySpark processing.
- Live WeatherAPI ingestion.
- V2 model/scaler validity.
- Future forecast quality on unseen production data.
