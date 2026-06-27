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

The scripts can be run from the project root. Internal paths are resolved automatically from each script location. API output files are date-stamped as `api_data_featured_YYYYMMDD.csv` and `api_data_predictions_YYYYMMDD.csv`.

## Notebooks

The notebooks remain exploratory, experimental, and historical workflows. Not all notebook logic has been migrated into the package.

- `notebooks/DataPreparation.ipynb`: historical data preparation and station-data preprocessing.
- `notebooks/EDA.ipynb`: exploratory data analysis.
- `notebooks/Modeling.ipynb`: model training, tuning, comparison, and artifact generation.
- `notebooks/WAPI.ipynb`: exploratory WeatherAPI collection example.

For new work, put reusable production logic under `src/wind_forecast/` and keep `scripts/` as thin executable wrappers. Keep notebooks for exploration and documentation.
