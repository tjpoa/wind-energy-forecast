# Project Status

Last reviewed: 2026-07-31.

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
reproducibility bundle tooling. A responsive React and TypeScript dashboard now
opens on a read-only retrospective monitoring projection and retains the typed
historical-performance view, completing a local frontend-to-API-to-verified
artifact demonstration.

The accepted v2 historical hindcast is also operational in one governed local
environment. Its checksum-pinned Phase 8/9 evidence, generation-one deployment,
MLflow aliases, and Windows Task Scheduler ownership were verified on
2026-07-30. This does not change the v1 API or turn the hindcast into a live or
day-ahead forecast.

The project should not be presented as a deployed production system. Cloud
deployment, production operation, real-time data, enterprise scalability,
external alert delivery, production orchestration, registry-based serving, and
PySpark processing are not current capabilities.

Historical phase documents preserve the stop-gate wording that was true when
each checkpoint was written. This file reflects the latest repository state.

## Current Capabilities

| Area | Status | Evidence | Current limitation |
| --- | --- | --- | --- |
| Python package structure | Implemented | `src/wind_forecast/`, `pyproject.toml` | Full tuned training still lives in notebooks. |
| Configuration and paths | Implemented | `wind_forecast.config`, `wind_forecast.paths`, `.env.example` | WeatherAPI workflows still require local credentials and network access. |
| Schema compatibility | Implemented | `wind_forecast.schemas`, schema tests | Existing saved artifacts still depend on the recovered v1 feature order. |
| Feature engineering | Implemented | `wind_forecast.features`, `wind_forecast.v2_features` | V2 features are built locally; the governed v2 hindcast has no scaler and remains separate from v1 serving. |
| Data validation | Implemented and active | `wind_forecast.validation`, `scripts/validate_feature_ready_v2_dataset.py`, Phase 2 docs | Validation is strong for current contracts, but future data-source changes still require explicit decisions. |
| V2 data-source work | Substantial local progress | REN and ERA5-Land source modules, Phase 2 acceptance docs | V2 does not replace v1; the accepted model is valid only for the documented historical-hindcast contract and has no scaler. |
| V2 incremental updates | Implemented and synthetically validated | `wind_forecast.incremental`, batch-quality sidecars, Phase 8 docs and tests | Live REN/CDS refresh was not exercised by the repository test suite. |
| Historical monitoring | Immutable quality, prediction evidence, drift, performance, reports, and local alerts implemented | `wind_forecast.monitoring`, `wind_forecast.monitoring_reporting`, Phase 9 CLIs/docs/tests | It is delayed historical hindcast monitoring, not live forecasting or external notification. |
| Batch orchestration | Local CLI, Windows Task Scheduler, and Airflow 3.3.0 local stack implemented | `wind_forecast.orchestration`, `wind_forecast.airflow_orchestration`, `airflow/`, Phase 10 docs/tests | Airflow uses temporary synthetic fixtures for offline validation; a shared owner pointer and lease prevent both schedulers from executing concurrently. |
| Controlled retraining | Contract, eligibility, backtesting, candidate registration, bootstrap, model-era monitoring, manual lifecycle, recommendation-only monthly scheduling, and final synthetic lifecycle acceptance implemented | Controlled-retraining modules, manual/scheduled CLIs, Airflow DAGs, `tests/test_controlled_retraining_acceptance.py`, and `docs/CONTROLLED_RETRAINING.md` | The one-time local bootstrap completed from the accepted Phase 9 ledger. Training and every later lifecycle transition still require explicit operator action and exact evidence. The final lifecycle acceptance remains synthetic and uses an in-memory Registry boundary. |
| Automated tests | Implemented | `tests/`, pytest and pytest-cov configuration | Coverage has a conservative baseline gate; source-ingestion and v2 modules still need deeper tests. |
| Code quality | Implemented baseline | Ruff configuration in `pyproject.toml` | Ruff rule set is intentionally minimal. |
| Prediction API | Implemented for local/container use | `wind_forecast.api`, `docs/PHASE_5.md`, API tests | The API is not deployed and depends on local mounted artifacts for full serving. |
| Historical performance API | Implemented and consumed by the dashboard | `GET /api/v1/performance`, `wind_forecast.performance`, backend contract tests | It reads explicitly selected local evaluation artifacts; it is not live monitoring. |
| Historical monitoring API | Implemented as a read-only verified projection | `GET /api/v1/monitoring/latest`, `/history`, `/runs/{run_id}`, `wind_forecast.monitoring_projection` | It projects immutable local batch evidence; it is not real time and performs no writes. |
| Operational Read-only Copilot | Typed deterministic query layer and local-only read-only API implemented; no Copilot | `POST /api/v1/operational-query`, `wind_forecast.operational_query`, `wind_forecast.operational_api`, dedicated tests, `docs/OPERATIONAL_COPILOT.md` | Evaluation harness, Copilot, MCP, RAG, relational projection, observability, production authentication, remote exposure, and cloud design remain future work. |
| React dashboard | Implemented for local/container demonstration | `frontend/`, frontend tests, `docs/DEMO.md` | It shows retrospective monitoring and historical holdout performance; it does not call `/predict`. |
| Training CLIs | V1 baseline preserved; first v2 reference accepted and bootstrapped locally | `wind_forecast.training`, `wind_forecast.v2_training`, dedicated scripts, and `docs/PHASE_4.md` | The v2 result is a historical hindcast used by the local batch, not by API serving; tuned ANN/Optuna remains notebook-based. |
| Docker support | Implemented baseline with runtime hardening | Backend and frontend Dockerfiles, Compose stack, CI image builds, and backend smoke test | No image publishing, digest pinning, or production deployment workflow yet. |
| CI | Implemented for backend and frontend validation | `.github/workflows/ci.yml` | CI covers Python tests and Ruff, frontend tests/lint/build, both Docker image builds, Compose validation, and backend health checks; it does not deploy artifacts. |
| MLflow lifecycle | Implemented; v1 and v2 local integration smokes plus v2 bootstrap completed | Tracking/Registry modules, both training CLIs, tests, and real local SQLite runs | `wind-forecast-v2-hindcast` version 1 is the local `champion` and `stable`; Registry state remains local and serving does not consume aliases. |
| Artifact versioning | Local bundle validation completed; publication blocked | Deterministic builder/fetcher/verifier, `artifacts/catalog.json`, two matching local bundle hashes, and local verify/retrain evidence | No public v1 data bundle until provenance/licence/redistribution approval and a catalog SHA-256; no cross-machine claim until a release round-trip runs. |
| Documentation | Strong | `README.md`, `docs/README.md`, `docs/DEMO.md`, phase docs, roadmap | Model/data cards are baseline-level and not full registry artifacts. |

## Roadmap Status

| Phase | Roadmap focus | Current status |
| --- | --- | --- |
| 0 | Repository audit, security, and baseline | Completed. Baseline and risks are documented in `docs/PHASE_0.md`. |
| 1 | Modular project structure and configuration | Completed. Reusable package modules and configuration helpers exist. |
| 2 | Data validation and sanity checks | Substantially implemented. V1 and v2 validation work exists, with documented v2 contracts and acceptance checks. |
| 3 | Automated testing and code quality | Implemented with a 30% coverage gate, Pytest, and Ruff in CI. |
| 4 | MLflow experiment tracking and model registry | Phase 4B v1 lifecycle is complete locally. The v2 reference pipeline, manifests, tests, real MLflow run, logged-model reload validation, and generation-one local bootstrap are also complete. |
| 5 | Prediction API with FastAPI | Implemented for local/container inference over saved artifacts, with an additional historical-performance endpoint for the dashboard. |
| 6 | Docker containerization | Implemented non-root Dockerfile with a runtime health check and CI smoke test. |
| 7 | GitHub Actions continuous integration | Implemented matrix backend CI plus frontend tests, linting, build, container build, Compose validation, and backend health checks. |
| 8 | Idempotency, safe reruns, and observability | Implemented for the accepted v2 dataset with dry-run planning, immutable revisions, atomic publication, structured run evidence, and failure recovery tests. |
| 9 | Data drift and model-performance monitoring | Completed locally for the accepted historical-batch contract: quality evidence, calibrated 30/90-day drift, as-issued performance, immutable JSON/Markdown reports, and persistent local alerts. |
| 10 | Batch orchestration with Apache Airflow | Completed locally: Airflow 3.3.0 build/import checks and a serial real-CLI three-date synthetic backfill passed; no live provider refresh or production deployment is claimed. |
| 11 | PySpark data-processing implementation | Not implemented. |
| 12 | Azure and Databricks deployment design | Not implemented. |

## Validation Snapshot

The standard local validation commands are:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m ruff check .
```

The 2026-07-28 final controlled-retraining acceptance ran both complete
bootstrap-to-stability and bootstrap-to-rollback paths. The dedicated suite
passed `7` tests; the full backend suite passed `388` tests with `4` skipped
and `70.74%` total coverage. It also pinned the tracked v1 raw data,
model/scaler artifacts, and modelling notebook by SHA-256.

### Governed local v2 snapshot

At 2026-07-30 00:25 `Europe/Lisbon`, read-only verification recorded:

- Phase 8 generation 1 pinned by manifest SHA-256
  `ddb7f18098434f29ed3402a752f1edbfa10332a513d1bd880291fe78070181e2`
  and Phase 9 generation 1 pinned by ledger SHA-256
  `6471f9fd220cecaf5eb2d8a5bc1a2062e8d8160777fd9620b32f8b23be8ba10d`,
  through 2026-06-27;
- dataset SHA-256
  `d0d073748c5d963cba30212e6b0ab666ec2000197b8f61a5c439b4aaf786b2a6`
  and calibration ID
  `ff56dd507607a95aea81f76ab6ce694f1fd8eb51a97175f834bdb83c16b2fe58`;
- active deployment generation 1 with status `verified`;
- registered model `wind-forecast-v2-hindcast`, version 1, status `READY`;
- `champion=1`, `stable=1`, and no `candidate`;
- run `aaedd79348ee404880a4608760cebafd` in `FINISHED` state. Under explicit
  operator authorization, the supported MLflow API set that administrative
  terminal state only after the bundle, signature, receipt, and reload evidence
  had been verified;
- scheduler owner `local` set to `windows_task_scheduler`, generation 1;
- daily `WindForecastHistoricalBatch` and monthly
  `WindForecastMonthlyGovernance` tasks enabled and `Ready`, with no prior
  executions. Their next starts were 2026-07-30 12:00 and 2026-08-08 13:00
  local time.

The Airflow implementation has passed its synthetic three-date real-CLI
acceptance, but its DAGs are inactive in this environment. The owner/lease
contract prevents it from competing with Windows Task Scheduler.

Operational cautions remain: the v2 evidence ends on 2026-06-27; the 2026-07
governance dry-run could not find a report dated 2026-06-30; the monthly task
required an equivalent Windows Task Scheduler COM registration fallback
because the script's CIM path was incompatible on this machine; and MLflow
must be reachable during verification and scheduled runs. All of this local
state is ignored by Git and is absent from a fresh clone.

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
- A reproducible v2 hindcast reference model selected chronologically against
  one-day persistence, with explicit checksum-pinned deployment governance.
- Saved-model serving through a typed FastAPI interface.
- A full-stack React-to-FastAPI demonstration over read-only historical
  evaluation artifacts.
- Tested date filtering, performance metrics, actual-versus-predicted charts,
  signed-error visualization, and explicit frontend failure states.
- Automated testing with mocked or synthetic data instead of live API calls.
- CI, Docker, and local experiment-tracking foundations.
- Clear documentation of assumptions, risks, source decisions, and non-goals.

## Known Limitations

- The full tuned modelling workflow remains in `notebooks/Modeling.ipynb`;
  the CLIs cover deterministic tree baselines rather than that tuned workflow.
- The accepted v2 RandomForest result is a historical hindcast using
  contemporaneous ERA5-Land weather, not an operational day-ahead forecast.
- No production cloud deployment exists.
- No production environment, real-time data path, enterprise-scalability
  guarantee, or complete monitoring system exists.
- Airflow orchestration is a local Docker Compose workflow, not a production deployment.
- No PySpark implementation exists.
- Registry and deployment-pointer state are local-only and are not consumed by
  API serving.
- The dashboard displays retrospective historical batch monitoring and
  historical evaluation results. It does not call `/predict`, generate future
  forecasts, poll continuously, or provide live model monitoring.
- Public artifact distribution and a clean-clone round-trip remain gated by
  provenance/licence approval and explicit network authorization.
- Monitoring is local and retrospective. Its schedulers are local-only; it has
  no external alert delivery, live forecasting, automatic retraining, or model
  promotion.
- Controlled retraining remains offline. A scheduler may seal monthly
  retraining/stability recommendations over verified Phase 9 evidence; operators
  can then seal a
  fail-closed temporal backtest, and can register an accepted version under
  only the v2 `candidate` alias. The one-time, manually approved local
  bootstrap initialized generation-one `stable` and `champion` plus an
  immutable checksum-pinned deployment pointer. Normal V2 promotion, probation,
  stability review over the first 90 eligible same-era observations plus current
  health, and rollback to
  the promotion-fixed last stable are explicit approval-gated commands with
  immutable evidence and atomic pointer publication. Scheduling is
  recommendation-only; no lifecycle transition is automatic.
- The v2 reference is independent of v1 scalers/models and is active only in
  the local governed historical batch; it is not connected to API serving.
- The Phase 8 live provider refresh path requires approved credentials/network
  access and has only synthetic, offline test coverage in this repository.
- A fresh clone may need local generated artifacts before the full prediction
  workflow can be demonstrated end to end.

## Recommended Next Steps

1. Resolve v1 provenance, licence, attribution, and redistribution approval,
   then publish the first checksum-pinned immutable artifact bundle.
2. Prove the documented fetch, verification, and retraining round-trip from a
   clean clone before claiming cross-machine reproducibility.
3. Extend the baseline training CLI toward the tuned notebook workflow while
   preserving the current contracts.
4. Raise test coverage as source-ingestion and v2 modules mature.
5. Build the versioned evaluation harness for the implemented typed
   operational query layer and local-only read-only API. Copilot, MCP, RAG,
   relational projection, observability, production authentication, remote
   exposure, and cloud stages are not implemented.
6. Define and review an interactive-prediction UI/API contract separately,
   then address production deployment as a later roadmap phase.
