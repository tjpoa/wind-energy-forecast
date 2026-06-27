# Phase 0 Repository Audit, Security And Baseline

Audit date: 2026-06-27

## Table Of Contents

- [Objective And Scope](#objective-and-scope)
- [Repository And Asset Inventory](#repository-and-asset-inventory)
- [Current End-To-End Workflow](#current-end-to-end-workflow)
- [Environment And Dependency Management](#environment-and-dependency-management)
- [Security Status](#security-status)
- [Reproducibility Status](#reproducibility-status)
- [Baseline Periods And Dataset Dimensions](#baseline-periods-and-dataset-dimensions)
- [Evaluation Contexts](#evaluation-contexts)
- [Risk Register](#risk-register)
- [Completed Phase 0 Items](#completed-phase-0-items)
- [Remaining Phase 0 Items](#remaining-phase-0-items)
- [Safe Read-Only Commands](#safe-read-only-commands)
- [Commands That May Regenerate Or Overwrite Outputs](#commands-that-may-regenerate-or-overwrite-outputs)
- [Actions Requiring Explicit Approval](#actions-requiring-explicit-approval)
- [Phase 0 Acceptance-Criteria Checklist](#phase-0-acceptance-criteria-checklist)

## Objective And Scope

Phase 0 establishes a factual audit baseline for the current wind-energy forecasting repository. This document records the existing workflow, assets, security posture, reproducibility limits, baseline periods, available metrics, and risks before later engineering phases begin.

This phase is documentation-only. It does not refactor code, execute notebooks, regenerate datasets, retrain models, overwrite artifacts, or start Phase 1.

## Repository And Asset Inventory

### Tracked Files

- Project guidance and setup: `AGENTS.md`, `README.md`, `.env.example`, `.gitignore`, `requirements.txt`.
- Roadmap: `docs/ML_ENGINEERING_ROADMAP.md`.
- Raw datasets:
  - `data/raw/ReparticaoProducao.csv`
  - `data/raw/IntensidadeMediaVento10m.csv`
  - `data/raw/DirecaoMediaVento10m.csv`
  - `data/raw/TemperaturaMedia.csv`
- Processed-data placeholder: `data/processed/.gitkeep`.
- Scripts:
  - `scripts/schema.py`
  - `scripts/process_api_data.py`
  - `scripts/apply_models_to_api_data.py`
- Notebooks and notebook artifact:
  - `notebooks/DataPreparation.ipynb`
  - `notebooks/EDA.ipynb`
  - `notebooks/Modeling.ipynb`
  - `notebooks/WAPI.ipynb`
  - `notebooks/pico_anual_capacidade_eolica.png`
- Saved model and scaler artifacts:
  - `models/best_model_original_target_ANN_Tuned.keras`
  - `models/best_model_log_target_ANN_Tuned.keras`
  - `models/scaler_X_original_ann.joblib`
  - `models/scaler_X_log_ann.joblib`
  - `models/scaler_y_original_ann.joblib`
  - `models/scaler_y_log_ann.joblib`

### Ignored Local Artifacts

The following generated CSV files exist locally under `data/processed/` but are intentionally ignored by Git:

- `agg_data.csv`
- `agg_data_ml.csv`
- `api_data_featured_20250530.csv`
- `api_data_predictions_20250530.csv`
- `DirecaoMediaVento10m_processed.csv`
- `IntensidadeMediaVento10m_processed.csv`
- `TemperaturaMedia_processed.csv`

The local virtual environment `venv/` and `scripts/__pycache__/` are also ignored.

## Current End-To-End Workflow

### Raw Data Acquisition

- Historical production data comes from `data/raw/ReparticaoProducao.csv`, which has two metadata rows followed by 15-minute records.
- Historical weather inputs are daily station matrices in the three weather CSV files under `data/raw/`.
- Recent weather data is fetched from WeatherAPI by `scripts/process_api_data.py`; `notebooks/WAPI.ipynb` demonstrates the same API access pattern.

### Data Preparation

- `notebooks/DataPreparation.ipynb` loads the raw production file, converts `Data e Hora`, and aggregates wind production to daily totals.
- The notebook loads raw weather files, handles missing station values, computes average wind speed and temperature, and computes vector-averaged wind direction.
- Daily production and weather tables are merged on `Date`.

### Feature Engineering

- Date features: `Month`, `Day_Of_Week`, `Day_Of_Year`, `ISO_Week`, `Quarter`, `Is_Weekend`.
- Cyclical encodings: wind direction, day of week, month, and day of year sine/cosine features.
- Lag features: wind production lags 1, 2, 3, 7, and 14; weather lags 1, 2, 3, and 7.
- Rolling features: shifted rolling means and standard deviations over 3, 7, and 14 days.
- Initial lag and rolling nulls are currently backfilled.
- The notebook writes the local generated feature table to `data/processed/agg_data_ml.csv`.

### Model Training

- `notebooks/Modeling.ipynb` reads `data/processed/agg_data_ml.csv`, normalizes legacy column names, sorts rows by `Date`, and defines `X`, `y_orig`, and `y_log`.
- The notebook trains original-target and log-target variants across tree models, XGBoost, and ANN models.
- It includes randomized search, Optuna tuning, final comparisons, and model/scaler saving cells.

### Model Evaluation

- The modeling notebook uses a chronological final split: first 80 percent for train/validation and final 20 percent for holdout testing.
- Metrics used in the notebook include R2, MAE, RMSE, and MAPE.
- The tracked notebook source contains the evaluation logic, but the historical holdout metric outputs are not available as extractable text in the current tracked notebook state.

### Prediction On API Data

- `scripts/process_api_data.py` loads `.env`, fetches WeatherAPI history, merges API weather data with recent known production, engineers features, and writes `api_data_featured_YYYYMMDD.csv`.
- `scripts/apply_models_to_api_data.py` loads the latest API feature CSV, saved selected models, and scalers, then writes `api_data_predictions_YYYYMMDD.csv`.
- `scripts/schema.py` provides the English schema and legacy compatibility mappings used by both scripts.

## Environment And Dependency Management

- The README documents local setup with `venv`.
- Dependencies are declared in `requirements.txt`.
- No lockfile, Python version file, Dockerfile, or CI environment definition exists yet.
- `.env.example` documents required WeatherAPI-related variables:
  - `WEATHER_API_KEY`
  - `WEATHER_API_LOCATION`
  - `WEATHER_API_DAYS`
  - `WEATHER_API_END_DATE`

## Security Status

- `.env` is ignored by Git.
- `.env.example` is tracked and contains placeholders only.
- No local `.env` file was present during this audit.
- `.gitignore` excludes local virtual environments, caches, `.env`, `.env.*`, secret-like file extensions, and generated processed CSV files.
- Redacted current-file scans found only expected placeholder or environment-variable references, not concrete credentials.
- Redacted Git-history scans found environment-variable references and one irrelevant historical 32-character hex-like value inside a previously tracked `venv` path; no confirmed WeatherAPI credential was identified.

## Reproducibility Status

- Raw input data and saved model/scaler artifacts are tracked.
- Processed CSV files are generated outputs and are intentionally ignored.
- README documents the commands for environment setup, historical feature regeneration, API feature generation, and prediction.
- The project does not yet provide a single deterministic end-to-end reproduction command.
- TensorFlow and Optuna randomness is not fully controlled in the current notebook workflow.
- No automated test suite, quality gate, or CI workflow is currently present.
- Saved model/scaler compatibility depends on current file names, feature order, and mappings in `scripts/schema.py`.

## Baseline Periods And Dataset Dimensions

### Raw Inputs

| File | Rows | Columns | Period |
| --- | ---: | ---: | --- |
| `data/raw/ReparticaoProducao.csv` | 537,308 | 2 selected audit columns | 2010-01-01 00:00 to 2025-04-28 23:45 |
| `data/raw/IntensidadeMediaVento10m.csv` | 4,017 | 21 | 2013-01-01 to 2023-12-31 |
| `data/raw/DirecaoMediaVento10m.csv` | 4,017 | 21 | 2013-01-01 to 2023-12-31 |
| `data/raw/TemperaturaMedia.csv` | 4,017 | 21 | 2013-01-01 to 2023-12-31 |

### Local Generated Outputs

| File | Rows | Columns | Period |
| --- | ---: | ---: | --- |
| `data/processed/agg_data.csv` | 4,017 | 5 | 2013-01-01 to 2023-12-31 |
| `data/processed/agg_data_ml.csv` | 4,017 | 58 | 2013-01-01 to 2023-12-31 |
| `data/processed/api_data_featured_20250530.csv` | 13 | 58 | 2025-04-16 to 2025-04-28 |
| `data/processed/api_data_predictions_20250530.csv` | 13 | 4 | 2025-04-16 to 2025-04-28 |

### Historical Modeling Split

- Full local feature table: 4,017 rows, 2013-01-01 to 2023-12-31.
- Train/validation: first 3,213 rows, 2013-01-01 to 2021-10-18.
- Holdout test: final 804 rows, 2021-10-19 to 2023-12-31.
- ANN inner validation starts at 2020-06-24 and ends at 2021-10-18.

## Evaluation Contexts

### Historical Holdout-Test Metrics From `Modeling.ipynb`

- Context: historical model evaluation on the final chronological 20 percent of `agg_data_ml.csv`.
- Available evidence: the tracked notebook source defines metrics and selects minimum-MAE models, but the metric output tables are not available as extractable text in the current tracked notebook state.
- Limitation: historical holdout metrics cannot be safely extracted without executing the notebook, and notebook execution is outside Phase 0 because it can change notebooks, regenerate artifacts, or overwrite models/scalers.
- Wording rule: saved artifacts should be described as saved selected models, not as conclusively verified overall best models, unless historical ranking outputs are reproduced and captured in a later phase.

### Recent API-Period Metrics From `api_data_predictions_20250530.csv`

- Context: local backtest-style evaluation on known recent production dates, not genuine future forecasting.
- Period: 2025-04-16 to 2025-04-28.
- Rows: 13.
- Metrics from the local ignored prediction CSV:

| Prediction column | R2 | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: | ---: |
| `Pred_Best_Original_ANN_Tuned` | 0.499393 | 53,837.32 | 72,078.30 | 45.50% |
| `Pred_Best_Log_ANN_Tuned` | 0.479299 | 56,074.00 | 73,510.68 | 47.14% |

These metrics evaluate saved selected models on a recent known-production period. They should not be treated as historical holdout-test metrics or as proof of future forecasting performance.

## Risk Register

| Risk | Severity | Evidence From Repository | Possible Impact | Future Phase |
| --- | --- | --- | --- | --- |
| Chronological final split is correctly used, but should be preserved by tests. | Low | `Modeling.ipynb` sorts by `Date`, uses `split_index = int(len(df) * 0.8)`, and selects train/test by `.iloc`; ANN validation uses `shuffle=False`. | Reduces leakage risk now, but future refactors could accidentally shuffle or mix time periods. | Phase 3 |
| Initial lag/rolling nulls are backfilled using later rows. | Medium | `DataPreparation.ipynb` backfills columns containing `_Lag` or `_Rolling_`; `scripts/process_api_data.py` uses backfill in `handle_final_nans`. | Early records receive information derived from later observations, which can make features less realistic for strict forecasting. | Phase 2 |
| Scaling is fit before inner ANN validation/CV splits. | Medium | `Modeling.ipynb` fits `MinMaxScaler` on `X_train_val_orig` and target train/validation data before `train_test_split`; Optuna sections use already-scaled train/validation arrays. | Inner validation and tuning scores may be optimistic even though the final holdout test is not directly fit by the scalers. | Phase 3 |
| Recent API metrics evaluate known production, not unseen future production. | Medium | `process_api_data.py` merges API weather data with production data; `apply_models_to_api_data.py` calculates metrics against `Actual_Wind_Production`. | Recent API-period metrics are useful backtests but do not represent live future forecasting where actual production is unavailable. | Phase 8 or Phase 9 |
| Historical holdout metrics are not captured in a reproducible artifact. | Medium | `Modeling.ipynb` contains metric code but current tracked notebook outputs do not expose the final tables as text; no metrics file is tracked. | Baseline model ranking cannot be independently verified without rerunning the notebook. | Phase 4 |
| TensorFlow and Optuna randomness are not fully controlled. | Medium | Some sklearn/XGBoost estimators use `random_state=42`, but no complete TensorFlow, NumPy, Python, or Optuna seed policy is documented. | Retraining may produce different selected models or metrics across runs. | Phase 3 or Phase 4 |
| No lockfile or Python version pin exists. | Low | `requirements.txt` pins some ranges/versions, but there is no `requirements.lock`, `pyproject.toml`, `.python-version`, or container. | Dependency resolution may drift between machines or dates. | Phase 6 |
| Generated processed CSVs are ignored and must be regenerated per clone. | Low | `.gitignore` ignores `data/processed/*.csv`; README documents regeneration commands. | New users need local raw data, environment setup, and notebook/script execution before modeling or prediction scripts have all inputs. | Phase 8 |
| WeatherAPI calls depend on secrets, network, quota, and provider availability. | Medium | `process_api_data.py` requires `WEATHER_API_KEY` and calls `https://api.weatherapi.com/v1/history.json`. | Pipeline runs may fail or vary outside local control; accidental repeated calls can consume quota. | Phase 7 or Phase 10 |
| No automated tests or CI protect the baseline. | Medium | No `tests/` directory, pytest config, or GitHub Actions workflow is tracked. | Future changes may break schema compatibility, model loading, or data processing without immediate feedback. | Phase 3 |

No risk in this table should be fixed during Phase 0. Phase 0 only records the baseline and defers remediation to the listed future phases.

## Completed Phase 0 Items

- Secure environment-variable handling for WeatherAPI access.
- `.env.example` exists and contains no real key.
- `.gitignore` excludes `.env`, secret-like files, virtual environments, caches, and generated processed CSVs.
- README includes setup, secret handling, and regeneration instructions.
- Dependency list has been cleaned up for the current workflow.
- Schema compatibility helpers exist in `scripts/schema.py`.
- Generated processed CSV files have been removed from Git tracking and remain local/ignored.

## Remaining Phase 0 Items

- Review and commit this audit document.
- Do not add code, tests, modules, CI, MLflow, APIs, containers, orchestration, or cloud design during Phase 0.
- Optional future documentation-only follow-up: link this document from `README.md`.

## Safe Read-Only Commands

```powershell
git status --short --ignored
git ls-files
git check-ignore -v .env data/processed/agg_data_ml.csv
rg -n -i "api[_-]?key|token|secret|password|authorization|bearer" README.md .env.example scripts notebooks docs\PHASE_0_AUDIT_BASELINE.md
rg -n "\b[A-Fa-f0-9]{32}\b|Bearer\s+[A-Za-z0-9._\-]{12,}" README.md .env.example scripts notebooks docs\PHASE_0_AUDIT_BASELINE.md
.\venv\Scripts\python.exe -c "<read-only syntax, notebook-output, and CSV shape/date checks>"
```

Read-only file listing and CSV metadata checks are safe. They must not write notebooks, regenerate processed outputs, call WeatherAPI, retrain models, or overwrite artifacts.

## Commands That May Regenerate Or Overwrite Outputs

```powershell
.\venv\Scripts\python.exe -m jupyter notebook .\notebooks\DataPreparation.ipynb
.\venv\Scripts\python.exe -m jupyter notebook .\notebooks\Modeling.ipynb
.\venv\Scripts\python.exe .\scripts\process_api_data.py
.\venv\Scripts\python.exe .\scripts\apply_models_to_api_data.py
```

- Notebook execution can modify notebook outputs and execution metadata.
- `DataPreparation.ipynb` can regenerate processed training features.
- `Modeling.ipynb` can retrain models and overwrite saved models/scalers.
- `process_api_data.py` calls WeatherAPI and writes `api_data_featured_YYYYMMDD.csv`.
- `apply_models_to_api_data.py` writes `api_data_predictions_YYYYMMDD.csv`.

## Actions Requiring Explicit Approval

- Deleting data, notebooks, models, scalers, or generated outputs.
- Rewriting Git history, force-pushing, or removing artifacts from history.
- Exposing, validating, rotating, or testing real API keys.
- Running WeatherAPI requests that consume quota.
- Retraining models or overwriting saved model artifacts.
- Starting Phase 1 or any later roadmap phase.

## Phase 0 Acceptance-Criteria Checklist

- [x] Repository and asset inventory documented.
- [x] Current end-to-end workflow documented.
- [x] Tracked versus ignored artifacts documented.
- [x] Security status documented without exposing secrets.
- [x] Reproducibility status documented.
- [x] Baseline periods and dataset dimensions documented.
- [x] Historical holdout-test context separated from recent API-period metrics.
- [x] Saved artifacts described as saved selected models unless ranking is verifiable.
- [x] Data-leakage and evaluation risks listed individually with severity, evidence, impact, and future phase.
- [x] Safe read-only commands documented.
- [x] Commands that may regenerate or overwrite outputs documented.
- [x] No application code, scripts, notebooks, datasets, models, scalers, dependencies, configuration files, or roadmap content modified by this Phase 0 implementation.

## Commands For Reproducing Current Outputs

```powershell
.\venv\Scripts\python.exe -m jupyter notebook .\notebooks\DataPreparation.ipynb
.\venv\Scripts\python.exe .\scripts\process_api_data.py
.\venv\Scripts\python.exe .\scripts\apply_models_to_api_data.py
```

Model retraining and scaler regeneration are performed from `notebooks/Modeling.ipynb` and should only be run with explicit approval because they can overwrite artifacts in `models/`.
