# Project Status

Last reviewed: 2026-07-12.

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
existing forecasting workflow, including a local MLflow Registry and
reproducibility bundle tooling.

The project should not be presented as a deployed production system. Cloud
deployment, orchestration, registry-based serving, PySpark processing, and
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
| Automated tests | Implemented | `tests/`, pytest and pytest-cov configuration | Coverage has a conservative baseline gate; source-ingestion and v2 modules still need deeper tests. |
| Code quality | Implemented baseline | Ruff configuration in `pyproject.toml` | Ruff rule set is intentionally minimal. |
| Prediction API | Implemented for local/container use | `wind_forecast.api`, `docs/PHASE_5.md`, API tests | The API is not deployed and depends on local mounted artifacts for full serving. |
| Baseline training CLI | Implemented baseline | `wind_forecast.training`, `scripts/train_baseline.py`, `docs/PHASE_4.md` | Lightweight tree baseline only; tuned ANN/Optuna workflow remains notebook-based. |
| Docker support | Implemented baseline with runtime hardening | Backend and frontend Dockerfiles, Compose stack, CI image builds, and backend smoke test | No image publishing, digest pinning, or production deployment workflow yet. |
| CI | Implemented for backend and frontend validation | `.github/workflows/ci.yml` | CI covers Python tests and Ruff, frontend tests/lint/build, both Docker image builds, Compose validation, and backend health checks; it does not deploy artifacts. |
| MLflow lifecycle | Implemented; local integration smoke completed | `wind_forecast.tracking`, `wind_forecast.registry`, baseline CLI, synthetic registry tests, and real local SQLite smoke | A clean v1 run, candidate registration, manual promotion, and rollback succeeded locally; Registry state remains local and serving does not consume aliases. |
| Artifact versioning | Local bundle validation completed; publication blocked | Deterministic builder/fetcher/verifier, `artifacts/catalog.json`, two matching local bundle hashes, and local verify/retrain evidence | No public v1 data bundle until provenance/licence/redistribution approval and a catalog SHA-256; no cross-machine claim until a release round-trip runs. |
| Documentation | Strong | `README.md`, `docs/README.md`, `docs/DEMO.md`, phase docs, roadmap | Model/data cards are baseline-level and not full registry artifacts. |

## Roadmap Status

| Phase | Roadmap focus | Current status |
| --- | --- | --- |
| 0 | Repository audit, security, and baseline | Completed. Baseline and risks are documented in `docs/PHASE_0.md`. |
| 1 | Modular project structure and configuration | Completed. Reusable package modules and configuration helpers exist. |
| 2 | Data validation and sanity checks | Substantially implemented. V1 and v2 validation work exists, with documented v2 contracts and acceptance checks. |
| 3 | Automated testing and code quality | Implemented with a 30% coverage gate, Pytest, and Ruff in CI. |
| 4 | MLflow experiment tracking and model registry | Phase 4B code, synthetic tests, and the local SQLite-backed end-to-end smoke are complete; public release and clean-clone round-trip remain blocked by provenance/licence approval and explicit publication authorization. |
| 5 | Prediction API with FastAPI | Implemented for local/container inference over saved artifacts. |
| 6 | Docker containerization | Implemented non-root Dockerfile with a runtime health check and CI smoke test. |
| 7 | GitHub Actions continuous integration | Implemented matrix backend CI plus frontend tests, linting, build, container build, Compose validation, and backend health checks. |
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
docker run --detach --name wind-energy-forecast-api-ci --publish 8000:8000 wind-energy-forecast-api:ci
curl --fail --silent http://127.0.0.1:8000/health
docker inspect --format='{{.State.Health.Status}}' wind-energy-forecast-api-ci
cd frontend
npm ci
npm run test
npm run lint
npm run build
cd ..
docker build --file frontend/Dockerfile --tag wind-energy-forecast-frontend:ci frontend
docker compose config --quiet
```

The frontend quality job uses Node.js 22 LTS and caches npm downloads using
`frontend/package-lock.json`. These checks require no WeatherAPI credentials,
local models, datasets, generated performance artifacts, or secrets. Compose
validation is structural and does not start services or inspect bind-mount
contents; the workflow builds images for validation but does not publish or
deploy them.

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
- Registry state is local-only and is not yet part of API serving.
- Public artifact distribution and a clean-clone round-trip remain gated by
  provenance/licence approval and explicit network authorization.
- No drift or live model-performance monitoring exists.
- V2 REN + ERA5-Land data work does not validate current v1 scalers or models.
- A fresh clone may need local generated artifacts before the full prediction
  workflow can be demonstrated end to end.

## Recommended Next Steps

1. Extend the baseline training CLI toward the tuned notebook workflow.
2. Raise the coverage threshold as source-ingestion and v2 module tests mature.
3. Resolve v1 redistribution provenance/licence and publish the first immutable
   GitHub Release bundle.
4. Prove the documented clean-clone fetch/retrain round-trip before claiming
   cross-machine reproducibility.
