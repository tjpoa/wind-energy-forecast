# Full-Stack Forecasting Dashboard Demo

This guide runs the wind-energy forecasting project as a local full-stack ML
demonstration for portfolio reviews and technical interviews.

```text
React + TypeScript dashboard
        |
        | GET /api/v1/monitoring/* and /api/v1/performance
        v
FastAPI application
        |
        | read-only verification and projection
        v
Immutable Phase 9 evidence + local baseline evaluation artifacts
```

The dashboard presents retrospective historical batch monitoring as its
initial view and preserves historical holdout performance as a separate view.
It does not request future predictions, ingest live data, or monitor a deployed
model in real time. Screenshots are intentionally omitted because no maintained
dashboard screenshots exist in the repository.

## Prerequisites

- Python 3.10 or 3.11.
- Node.js 22.12 or newer.
- Docker with Docker Compose for the containerized path.
- A valid local performance-artifact directory for the historical-performance
  view. The monitoring view may start in a valid empty state.

Install the backend from the repository root:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pip install -e .
```

Install the frontend:

```powershell
Set-Location .\frontend
npm install
Copy-Item .env.example .env.local
Set-Location ..
```

No WeatherAPI key or `.env` file is required for the historical dashboard.
`VITE_API_BASE_URL` defaults to `http://localhost:8000` in the frontend example
configuration.

## 1. Prepare Local Evaluation Artifacts

The dashboard API reads one explicitly selected result set. `predictions.csv`
is required and must contain:

| Column | Meaning |
| --- | --- |
| `Date` | Observation date. |
| `Actual_Wind_Production` | Actual historical target value. |
| `Predicted_Wind_Production` | Model prediction for that date. |

`metrics.json` and `run_summary.json` are optional. When present, they provide
persisted aggregate metrics and basic run provenance; the API still validates
their consistency with the predictions. The endpoint never returns the
configured directory or another local filesystem path.

The project has no approved public demonstration bundle. The release entry in
`artifacts/catalog.json` remains blocked until provenance, licence, attribution,
and redistribution permission are resolved. Do not present synthetic data as
real observations.

### Generate the artifacts locally

If an authorized, feature-ready v1 table is available, confirm it first:

```powershell
Test-Path .\data\processed\agg_data_ml.csv
```

Then run the deterministic baseline without requiring an MLflow server:

```powershell
.\venv\Scripts\python.exe .\scripts\train_baseline.py `
  --input data\processed\agg_data_ml.csv `
  --output-dir outputs\training\baseline `
  --tracking-mode off
```

The command writes `predictions.csv`, `metrics.json`, `run_summary.json`, and
additional model, manifest, validation, and plot artifacts beneath the selected
output directory. It refuses to replace an existing result set unless
`--overwrite` is explicitly supplied. Files under `outputs/training/` are local
and ignored by Git.

If `data/processed/agg_data_ml.csv` is unavailable, regenerate it through the
existing data-preparation workflow using data you are authorized to use, or
obtain a compatible result directory separately. A fresh clone cannot assume
that the blocked public bundle is available.

Check the minimum artifact before continuing:

```powershell
Test-Path .\outputs\training\baseline\predictions.csv
```

## 2. Run The Backend And Frontend Locally

In the first PowerShell session, select the artifact directory before Uvicorn
creates the application:

```powershell
$env:WIND_FORECAST_PERFORMANCE_ARTIFACT_DIR=".\outputs\training\baseline"
.\venv\Scripts\python.exe -m uvicorn wind_forecast.api:app --host 127.0.0.1 --port 8000
```

In a second session, verify the process and dashboard data:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
$monitoring = Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/api/v1/monitoring/latest
$monitoring.state
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/api/v1/monitoring/history
$performance = Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/api/v1/performance
$performance.interval
$startDate = $performance.interval.available_start_date
$endDate = $performance.interval.available_end_date
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/performance?start_date=$startDate&end_date=$endDate"
```

The filtered check derives its dates from the local artifact bounds. The
unfiltered response is the reliable first check when local coverage is unknown.

In a third session, start the frontend:

```powershell
Set-Location .\frontend
npm run dev
```

Open `http://localhost:5173`. The exact hostname matters because the default
CORS policy permits `http://localhost:5173`, not `http://127.0.0.1:5173`.

The dashboard demonstrates:

- A default monitoring view permanently labelled as retrospective batch
  monitoring and not real time.
- D+5/D+7 freshness, verified model snapshot/version, last source/report run,
  moving 30/90-day metrics, sealed-test v2 thresholds, top feature drift,
  active alerts, and immutable run/alert history when evidence exists.
- Independent loading, empty, API-error, delayed-evidence, and partial-history
  states, plus explicit refresh without polling.
- A reporting-run selector whose details distinguish the reporting run,
  report ID, and source pipeline run without exposing local paths.
- An accessible `Historical performance` tab retaining the existing endpoint.
- Inclusive start and end date filters constrained to the available range.
- MAE, RMSE, R², and observation-count cards.
- An actual-versus-predicted line chart with tooltips.
- A signed-error bar chart distinguishing overprediction, underprediction, and
  exact predictions.
- An accessible table containing the ten most recent returned observations.
- Explicit loading, invalid-date, empty, and API-error states.
- Cancellation of an older request when a newer dashboard request begins.

The displayed production, MAE, RMSE, and error values are daily sums of
15-minute MW readings in the historical artifact. The UI deliberately does not
label them as MWh.

## 3. Dashboard API Contracts

| Endpoint | Dashboard use | Purpose |
| --- | --- | --- |
| `GET /health` | Operational check | Confirms that the FastAPI process is running. |
| `GET /api/v1/monitoring/latest` | Yes | Latest verified retrospective report, attempt, freshness, model, drift, performance, and active alerts. |
| `GET /api/v1/monitoring/history` | Yes | Paginated reporting attempts and causally ordered immutable alert events. |
| `GET /api/v1/monitoring/runs/{run_id}` | Yes | Sanitized detail for one reporting run and its report when successful. |
| `GET /api/v1/performance` | Yes | Returns historical evaluation results with optional inclusive ISO date filters. |
| `GET /model-info` | No | Reports saved model, scaler, and feature-reference readiness. |
| `POST /predict` | No | Runs saved-model inference for feature-ready records. |

`GET /api/v1/performance` returns `interval`, `observation_count`, `metrics`, an
optional `result`, and `observations`. Metrics are recalculated for the returned
interval and include `r2`, `mae`, `rmse`, and `mape_percent`. `r2` can be null
when it cannot be calculated. The current dashboard does not display MAPE.

Expected errors are:

| HTTP status | Condition |
| --- | --- |
| `400` | `start_date` is after `end_date`. |
| `404` | A valid requested interval contains no observations. |
| `422` | A query date does not use a valid ISO date. |
| `503` | Artifacts are not configured, missing, empty, invalid, or inconsistent. |

Monitoring endpoints return `200` with `state=empty` when no Phase 9 evidence
exists. An unknown monitoring `run_id` returns `404`, invalid pagination returns
`422`, and invalid schema/checksum/alert-chain evidence returns a sanitized
`503`. History defaults to 20 runs and 50 alerts and caps them at 100 and 200.
The monitoring API never returns configured artifact paths or raw failure text.

## 4. Run With Docker Compose

Compose mounts `data/`, `models/`, and the chosen performance-artifact
directory into the backend read-only. Datasets, models, results, local
environment files, and secrets are not copied into either image.
`WIND_FORECAST_MONITORING_STORE_ROOT` points to
`/app/data/processed/v2/monitoring` within the read-only data mount.

The default host result directory is `./outputs/training/baseline`. To select a
different compatible local directory, set it before building the stack:

```powershell
$env:WIND_FORECAST_PERFORMANCE_ARTIFACT_HOST_DIR=".\outputs\training\baseline"
docker compose up --build
```

Compose maps the selected directory to `/app/performance` and sets
`WIND_FORECAST_PERFORMANCE_ARTIFACT_DIR=/app/performance` inside the backend.
It publishes the API on port `8000` and Nginx-hosted frontend on port `5173`.

From another session:

```powershell
docker compose ps
Invoke-RestMethod -Method Get -Uri http://localhost:8000/health
Invoke-RestMethod -Method Get -Uri http://localhost:8000/api/v1/monitoring/latest
Invoke-RestMethod -Method Get -Uri http://localhost:8000/api/v1/performance
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:5173
```

Open `http://localhost:5173` in the browser. Stop the stack when the demo is
complete:

```powershell
docker compose down
```

The stack can start and `/health` can pass without valid evaluation artifacts,
but `/api/v1/performance` returns `503` until a compatible evaluation directory
is mounted. The monitoring view remains connected and empty when its store has
no reports or runs.

## 5. Optional Saved-Model Serving Check

The saved-model API is independent of the dashboard. Inspect readiness without
loading TensorFlow models:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/model-info
```

Full `/predict` serving requires the selected model and scaler files under
`models/` plus `data/processed/agg_data_ml.csv` for the saved training feature
order. See `docs/PHASE_5.md` for the request contract. The FastAPI service reads
the existing Keras/scaler paths; it does not serve the local MLflow Registry
`candidate` or `champion` aliases.

## 6. Run Final Validation

Run the backend checks from the repository root:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m ruff check .
```

Run the frontend checks from `frontend/`:

```powershell
npm run test
npm run lint
npm run build
```

Validate the Compose model without starting services:

```powershell
docker compose config --quiet
```

These checks do not require live API credentials or generated dashboard
artifacts.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Dashboard reports that the API is unavailable | Performance artifacts are not selected, readable, or valid | Check the backend response and `WIND_FORECAST_PERFORMANCE_ARTIFACT_DIR`; for Compose, check `WIND_FORECAST_PERFORMANCE_ARTIFACT_HOST_DIR`. |
| `/api/v1/performance` returns `503` | `predictions.csv` is missing, empty, invalid, or inconsistent with optional metadata | Verify the required columns and use one complete baseline output directory. |
| Date filter returns `400`, `404`, or `422` | The range is inverted, outside local coverage, or syntactically invalid | Start with the unfiltered endpoint and use its available bounds. |
| Browser reports a CORS failure | Frontend origin differs from the exact configured origin | Use `http://localhost:5173` or configure `WIND_FORECAST_CORS_ALLOW_ORIGINS` before API startup. |
| Port `5173` or `8000` is occupied | Another development server or container is running | Stop the conflicting process or deliberately reconfigure both the published port and client/API origin. |
| `/health` works but a dashboard view does not | Process health does not validate result artifacts | Call the corresponding `/api/v1/monitoring/*` or `/api/v1/performance` endpoint and inspect its status. |
| `/predict` returns `503` | Saved model, scaler, or feature-reference artifacts are missing | Check `/model-info`; this does not affect the historical dashboard contract. |

## What This Demo Proves

- React can consume a typed FastAPI contract and render historical evaluation
  results with responsive and tested UI states.
- The API validates one explicitly selected, read-only local artifact set.
- Backend and frontend can run separately or as a Docker Compose stack.
- The repository contains automated backend, frontend, image-build, and Compose
  validation in CI.
- Saved models can be inspected and served separately when all required local
  artifacts exist.

## What This Demo Does Not Prove

- Cloud deployment or a production environment.
- Real-time ingestion, live forecasting, orchestration, or external alert delivery.
- Enterprise scalability, availability, or operational support.
- Serving through MLflow Registry aliases.
- Public redistribution or clean-clone reproducibility before the artifact
  provenance and licence gate is resolved.
- Automatic model promotion, Airflow orchestration, or PySpark processing.
- Validity of the current v1 models and scalers for v2 REN + ERA5-Land data.
