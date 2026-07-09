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

## Endpoints

- `GET /health`: returns API process health.
- `GET /model-info`: reports saved model, scaler, and feature-reference
  artifact readiness without loading TensorFlow models.
- `POST /predict`: predicts with the selected saved model.

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
