# Project Status

Last reviewed: 2026-07-10.

This document summarizes the current state of the wind-energy forecasting
repository for portfolio and hiring-review purposes. It is a factual status
overview, not a replacement for the roadmap or phase decision records.

## Executive Summary

This project started as an academic Applied Artificial Intelligence project for
Portuguese wind-energy production forecasting and has been refactored into a
Data/ML Engineering portfolio project.

The repository now demonstrates reusable Python packaging, data validation,
feature engineering, saved-model inference, a tested FastAPI service, Docker
container support, GitHub Actions CI, and local MLflow tracking around the
existing forecasting workflow.

The project should not be presented as a deployed production system. Cloud
deployment, orchestration, model registry operations, PySpark processing, and
full monitoring are future roadmap items.

Historical phase documents preserve the stop-gate wording that was true when
each checkpoint was written. This file reflects the latest repository state.

## Current Capabilities

| Area | Status | Evidence | Current limitation |
| --- | --- | --- | --- |
| Python package structure | Implemented | `src/wind_forecast/`, `pyproject.toml` | Full tuned training still lives in notebooks. |
| Configuration and paths | Implemented | `wind_forecast.config`, `wind_forecast.paths`, `.env.example` | WeatherAPI workflows still require local credentials and network access. |
| Schema compatibility | Implemented | `wind_forecast.schemas`, schema tests | Existing saved artifacts still depend on the recovered v1 feature order. |
| Feature engineering | Implemented | `wind_forecast.features`, `wind_forecast.v2_features` | V2 features are built locally but v2 scalers and models are not promoted. |
| Data validation | Implemented and active | `wind_forecast.validation`, `scripts/validate_feature_ready_v2_dataset.py`, Phase 2 docs | Validation is strong for current contracts, but future data-source changes still require explicit decisions. |
| V2 data-source work | Substantial local progress | REN and ERA5-Land source modules, Phase 2 acceptance docs | V2 data does not replace v1; v2 model/scaler validity is not claimed. |
| Automated tests | Implemented | `tests/`, pytest configuration | Coverage reporting is not yet configured. |
| Code quality | Implemented baseline | Ruff configuration in `pyproject.toml` | Ruff rule set is intentionally minimal. |
| Prediction API | Implemented for local/container use | `wind_forecast.api`, `docs/PHASE_5.md`, API tests | The API is not deployed and depends on local mounted artifacts for full serving. |
| Baseline training CLI | Implemented baseline | `wind_forecast.training`, `scripts/train_baseline.py`, `docs/PHASE_4.md` | Lightweight tree baseline only; tuned ANN/Optuna workflow remains notebook-based. |
| Docker support | Implemented baseline | `Dockerfile`, `.dockerignore`, Docker CI build | No production hardening, image publishing, or runtime healthcheck yet. |
| CI | Implemented baseline | `.github/workflows/ci.yml` | CI runs tests, Ruff, and Docker build; it does not yet publish coverage or run container smoke tests. |
| MLflow tracking | Partial | `wind_forecast.tracking`, optional `--mlflow` evaluation logging | Local tracking only; model registry, aliases, and promotion workflow are not implemented. |
| Documentation | Strong | `README.md`, `docs/README.md`, `docs/DEMO.md`, phase docs, roadmap | Model/data cards are baseline-level and not full registry artifacts. |

## Roadmap Status

| Phase | Roadmap focus | Current status |
| --- | --- | --- |
| 0 | Repository audit, security, and baseline | Completed. Baseline and risks are documented in `docs/PHASE_0.md`. |
| 1 | Modular project structure and configuration | Completed. Reusable package modules and configuration helpers exist. |
| 2 | Data validation and sanity checks | Substantially implemented. V1 and v2 validation work exists, with documented v2 contracts and acceptance checks. |
| 3 | Automated testing and code quality | Implemented baseline. Pytest and Ruff are configured and covered by CI. |
| 4 | MLflow experiment tracking and model registry | Partial. Local MLflow tracking and a lightweight baseline-training CLI exist; registry and promotion are not implemented. |
| 5 | Prediction API with FastAPI | Implemented for local/container inference over saved artifacts. |
| 6 | Docker containerization | Implemented baseline Dockerfile and Docker build CI. |
| 7 | GitHub Actions continuous integration | Implemented baseline CI for tests, linting, and Docker build. |
| 8 | Idempotency, safe reruns, and observability | Not implemented as a roadmap phase. Some v2 builders already use explicit overwrite and checksum patterns. |
| 9 | Data drift and model-performance monitoring | Not implemented. |
| 10 | Batch orchestration with Apache Airflow | Not implemented. |
| 11 | PySpark data-processing implementation | Not implemented. |
| 12 | Azure and Databricks deployment design | Not implemented. |

## Validation Snapshot

The standard local validation commands are:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m ruff check .
```

The standard CI checks are:

```text
python -m pytest
python -m ruff check .
docker build --file Dockerfile --tag wind-energy-forecast-api:ci .
```

Docker can also be checked locally with:

```powershell
docker build -t wind-energy-forecast-api:status .
```

Generated processed CSV files, local MLflow runs, temporary pilot outputs, and
v2 raw/processed artifacts are intentionally ignored by Git unless a future
approved versioning policy changes that.

## Portfolio Value

This repository demonstrates:

- Backward-compatible refactoring from notebooks and scripts into package code.
- Data-contract design across legacy v1 data and isolated v2 data-source work.
- Defensive validation for raw, processed, and feature-ready datasets.
- Time-series feature engineering with lags, rolling windows, and cyclic terms.
- Saved-model serving through a typed FastAPI interface.
- Automated testing with mocked or synthetic data instead of live API calls.
- CI, Docker, and local experiment-tracking foundations.
- Clear documentation of assumptions, risks, source decisions, and non-goals.

## Known Limitations

- The full tuned modelling workflow remains in `notebooks/Modeling.ipynb`;
  the CLI covers only a lightweight baseline.
- No production cloud deployment exists.
- No Airflow orchestration exists.
- No PySpark implementation exists.
- No model registry or model-promotion workflow exists.
- No drift or live model-performance monitoring exists.
- V2 REN + ERA5-Land data work does not validate current v1 scalers or models.
- A fresh clone may need local generated artifacts before the full prediction
  workflow can be demonstrated end to end.

## Recommended Next Steps

1. Extend the baseline training CLI toward the tuned notebook workflow.
2. Add test coverage reporting and a container health smoke test to CI.
3. Define a local model registry or promotion convention before claiming model
   lifecycle operations.
4. Decide whether generated data and model artifacts should use DVC, GitHub
   Releases, or another explicit artifact versioning mechanism.
