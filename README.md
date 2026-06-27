# National Wind Energy Production Forecast

Project developed for the **Applied Artificial Intelligence Project** course.

The goal is to prepare weather and wind-production data, train forecasting models, and apply the saved models to recent WeatherAPI data.

The project now uses English column names in scripts and generated API outputs, such as `Date`, `Wind_Production`, `Average_Wind_Speed`, `Average_Temperature`, and `Average_Wind_Direction`. A small compatibility layer keeps the original source CSV headers and saved model/scaler feature order working behind the scenes.

## Project structure

- `data/raw/`: raw datasets.
- `data/processed/`: processed datasets and script-generated files.
- `models/`: trained models and scalers.
- `notebooks/`: exploratory analysis, data preparation, and modeling.
- `scripts/`: reproducible entry points outside the notebooks.

## Column naming

Use the English schema for new work:

- `Date`
- `Wind_Production`
- `Average_Wind_Speed`
- `Average_Temperature`
- `Average_Wind_Direction`

The helper file `scripts/schema.py` translates older source/training names at the import/model boundary only. This keeps the project readable in English without breaking the raw data import or the already-trained models.

## Environment setup

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m ipykernel install --user --name wind-energy-forecast --display-name "Python (wind-energy-forecast)"
```

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

Generate recent weather-based features:

```powershell
.\venv\Scripts\python.exe .\scripts\process_api_data.py
```

Apply the trained models to the generated data:

```powershell
.\venv\Scripts\python.exe .\scripts\apply_models_to_api_data.py
```

The scripts can be run from the project root. Internal paths are resolved automatically from each script location.

## Notebooks

The notebooks document the exploration and training workflow:

- `notebooks/DataPreparation.ipynb`: data preparation.
- `notebooks/EDA.ipynb`: exploratory data analysis.
- `notebooks/Modeling.ipynb`: model training and comparison.
- `notebooks/WAPI.ipynb`: WeatherAPI collection example.

For new work, prefer moving reusable logic into `scripts/` and keeping notebooks for exploration and documentation.
