# National Wind Energy Production Forecast

Project developed for the **Applied Artificial Intelligence Project** course.

The goal is to prepare weather and wind-production data, train forecasting models, and apply the saved models to recent WeatherAPI data.

The project now uses English column names in scripts and generated API outputs, such as `Date`, `Wind_Production`, `Average_Wind_Speed`, `Average_Temperature`, and `Average_Wind_Direction`. A small compatibility layer keeps the original source CSV headers and saved model/scaler feature order working behind the scenes.

## Project structure

- `data/raw/`: raw datasets.
- `data/processed/`: processed datasets and script-generated files.
- `models/`: trained models and scalers.
- `notebooks/`: exploratory analysis, data preparation, and modeling.
- `src/wind_forecast/`: reusable project package.
- `scripts/`: backward-compatible executable wrappers outside the notebooks.

The reusable package currently includes:

- `paths.py`: repository and artifact path resolution.
- `config.py`: explicit environment-based WeatherAPI configuration.
- `schemas.py`: canonical column names and legacy compatibility.
- `features.py`: shared feature-engineering logic.
- `ingestion.py`: WeatherAPI history ingestion.
- `inference.py`: model/scaler loading and prediction.
- `evaluation.py`: regression metric calculation.
- `tracking.py`: opt-in local MLflow experiment tracking helpers.

## Dataset versioning and v2 migration

The current tracked raw files remain the v1 baseline in `data/raw/*.csv`.
Generated v1-compatible outputs remain local under `data/processed/*.csv`, and
the saved models and scalers in `models/` are v1-only artifacts.

Future v2 data must use separate paths without moving or overwriting v1:

- `data/raw/v2/production/`: REN production source snapshots.
- `data/raw/v2/weather/`: ERA5-Land weather source snapshots.
- `data/processed/v2/daily_merged/`: daily merged production/weather tables.
- `data/processed/v2/ml_features/`: model-ready v2 feature tables.
- `data/manifests/historical_v2.json`: the planned v2 dataset manifest.
- `data/pilot/`: ignored temporary probe outputs only.

V2 manifests use `wind_forecast.manifests.DatasetManifest` with deterministic
JSON, schema version `wind_forecast.dataset_manifest.v1`, repository-relative
paths, SHA-256 checksums, retrieval metadata, coverage, units, timezone,
row/column counts, warnings, license and attribution notes, and no secrets.
Use `extra_metadata` for source-specific fields such as REN overlap/revision
notes, ERA5-Land product identifiers, station IDs, coordinates, UTC policy, and
aggregation formulas.

Future ingestion must write only to v2 paths and record raw-file checksums in
the manifest. REN historical revisions must be represented as explicit v2
revision metadata or overlap notes; never append to or silently mutate v1.
Rollback is done by removing/reverting v2 generated outputs and manifests while
leaving v1 usable. After v2 feature generation, scalers must be refit, models
must be retrained, and metrics must be re-baselined before making any v2 model
claim. The durable v2 decision record is
`docs/PHASE_2_V2_DATA_CONTRACT_DECISION.md`.

## Column naming

Use the English schema for new work:

- `Date`
- `Wind_Production`
- `Average_Wind_Speed`
- `Average_Temperature`
- `Average_Wind_Direction`

The `wind_forecast.schemas` module translates older source/training names at the import/model boundary only. The `scripts/schema.py` file remains as a legacy compatibility wrapper. This keeps the project readable in English without breaking the raw data import or the already-trained models.

## Environment setup

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -e .
.\venv\Scripts\python.exe -m ipykernel install --user --name wind-energy-forecast --display-name "Python (wind-energy-forecast)"
```

The editable install makes the local `wind_forecast` package importable while reflecting source-code changes immediately.

## API keys and secrets

The API key must not be written directly in the code, README, or notebooks.

1. Copy the example environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Edit the local `.env` file:

   ```env
   WEATHER_API_KEY=your_api_key_here
   WEATHER_API_LOCATION=41.8345,-7.7889
   WEATHER_API_DAYS=44
   WEATHER_API_END_DATE=
   ```

3. Confirm that Git is ignoring the local secrets file:

   ```powershell
   git check-ignore -v .env
   git status --short
   ```

The `.env.example` file can be committed because it does not contain a real key. The `.env` file stays only on your machine.

If a real key has already been committed, pushed to GitHub, or pasted into a chat, create a new key with the API provider.

Note: the raw production data included in this project ends on `2025-04-28`. If you want generated weather data to overlap with that historical production data, set `WEATHER_API_END_DATE` to a nearby date instead of using the current date.

## Running the pipeline

Processed CSV files in `data/processed/` are generated outputs and are not versioned by Git. Keep the raw input files in `data/raw/`, configure `.env`, and regenerate processed data when needed.

Regenerate the historical training features by running all cells in the data-preparation notebook:

```powershell
.\venv\Scripts\python.exe -m jupyter notebook .\notebooks\DataPreparation.ipynb
```

This writes `data/processed/agg_data_ml.csv`. If the legacy pre-feature aggregate `agg_data.csv` is needed, recreate it from the merged daily aggregate in the same notebook.

Generate recent WeatherAPI feature data:

```powershell
.\venv\Scripts\python.exe .\scripts\process_api_data.py
```

This script is a backward-compatible wrapper for WeatherAPI ingestion, feature engineering, and featured CSV generation.

Apply the trained models to generate prediction outputs:

```powershell
.\venv\Scripts\python.exe .\scripts\apply_models_to_api_data.py
```

This script is a backward-compatible wrapper for model inference, evaluation, plotting, and prediction CSV generation.

To log that evaluation run to local MLflow tracking, add `--mlflow`:

```powershell
.\venv\Scripts\python.exe .\scripts\apply_models_to_api_data.py --mlflow
```

Runs are written to `./mlruns` by default and are ignored by Git. To inspect
them locally:

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri .\mlruns
```

The current training workflow still lives in `notebooks/Modeling.ipynb`. For
manual training experiments, wrap only the approved training/evaluation cells
with `wind_forecast.tracking.start_local_run(...)` and
`wind_forecast.tracking.log_run_data(...)`; do not run the model-saving cells
unless you intend to overwrite the existing files in `models/`.

The scripts can be run from the project root. Internal paths are resolved automatically from each script location. API output files are date-stamped as `api_data_featured_YYYYMMDD.csv` and `api_data_predictions_YYYYMMDD.csv`.

## Notebooks

The notebooks remain exploratory, experimental, and historical workflows. Not all notebook logic has been migrated into the package.

- `notebooks/DataPreparation.ipynb`: historical data preparation and station-data preprocessing.
- `notebooks/EDA.ipynb`: exploratory data analysis.
- `notebooks/Modeling.ipynb`: model training, tuning, comparison, and artifact generation.
- `notebooks/WAPI.ipynb`: exploratory WeatherAPI collection example.

For new work, put reusable production logic under `src/wind_forecast/` and keep `scripts/` as thin executable wrappers. Keep notebooks for exploration and documentation.
