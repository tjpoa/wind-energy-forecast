# Project Status

Last reviewed: 2026-08-16.

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

The accepted v2 historical hindcast exists in one governed local environment.
On 2026-08-10, PR #50 corrected the observed REN-ahead/ERA5-pending
integration defect, a provider-backed update published generation 3, and a
Windows Task Scheduler cycle completed with alerts and `LastTaskResult=0`.
MLflow and the read-only operational API were also healthy on exclusive
loopback listeners and answered the three typed deployment/model queries. The
unattended 2026-08-11 service and daily-task failures subsequently moved
automatic operation to **NO-GO**. The daily task remains disabled pending live
persistence verification of the single-worker MLflow correction, one
manual success, and one later automatic success. The historical hindcast
artifacts remain accepted conditionally for local retrospective use only. The
active source-lateness alert, one-day miss against authoritative D+5, and six
historical REN gaps remain visible; there is still explicit NO-GO for real-time,
D+1, or production
claims.

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
| Batch orchestration | Local CLI, Windows Task Scheduler, and Airflow 3.3.0 local stack implemented | `wind_forecast.orchestration`, `wind_forecast.airflow_orchestration`, `airflow/`, Phase 10 docs/tests | The 2026-08-11 unattended cycle failed before a new manifest; the daily task is disabled and automatic operation is NO-GO. Airflow remains inactive, and the shared owner pointer and lease prevent concurrent schedulers. |
| Controlled retraining | Contract, eligibility, backtesting, candidate registration, bootstrap, model-era monitoring, manual lifecycle, recommendation-only monthly scheduling, and final synthetic lifecycle acceptance implemented | Controlled-retraining modules, manual/scheduled CLIs, Airflow DAGs, `tests/test_controlled_retraining_acceptance.py`, and `docs/CONTROLLED_RETRAINING.md` | The one-time local bootstrap completed from the accepted Phase 9 ledger. Training and every later lifecycle transition still require explicit operator action and exact evidence. The final lifecycle acceptance remains synthetic and uses an in-memory Registry boundary. |
| Automated tests | Implemented | `tests/`, pytest and pytest-cov configuration | Coverage has a conservative baseline gate; source-ingestion and v2 modules still need deeper tests. |
| Code quality | Implemented baseline | Ruff configuration in `pyproject.toml` | Ruff rule set is intentionally minimal. |
| Prediction API | Implemented for local/container use | `wind_forecast.api`, `docs/PHASE_5.md`, API tests | The API is not deployed and depends on local mounted artifacts for full serving. |
| Historical performance API | Implemented and consumed by the dashboard | `GET /api/v1/performance`, `wind_forecast.performance`, backend contract tests | It reads explicitly selected local evaluation artifacts; it is not live monitoring. |
| Historical monitoring API | Implemented as a read-only verified projection | `GET /api/v1/monitoring/latest`, `/history`, `/runs/{run_id}`, `wind_forecast.monitoring_projection` | It projects immutable local batch evidence; it is not real time and performs no writes. |
| Operational Read-only Copilot | Typed deterministic query layer, persistent local-only read-only API, versioned offline evaluation harness, optional default-disabled PostgreSQL query projection, and sanitized local operational observability implemented; no Copilot evaluated | `POST /api/v1/operational-query`, `GET /api/v1/operational-observability/health`, `GET /api/v1/operational-observability/metrics`, `wind_forecast.operational_query`, `wind_forecast.operational_observability`, `wind_forecast.operational_projection_reader`, `wind_forecast.operational_evaluation`, 88-case synthetic dataset, dedicated tests, `docs/OPERATIONAL_COPILOT.md`, `docs/OPERATIONAL_POSTGRES_PROJECTION.md` | Task Scheduler restart and three deployment/model queries were verified locally on loopback. Files/loaders remain authoritative and PostgreSQL is never cited. The observability writer is separate, lazy, sanitized, and process-local; Copilot, MCP, RAG, production authentication, remote exposure, and cloud design remain future work. |
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
| 10 | Batch orchestration with Apache Airflow | Completed locally: Airflow 3.3.0 build/import checks and a serial real-CLI three-date synthetic backfill passed. On 2026-08-10, Task Scheduler also completed an end-to-end local delayed-hindcast cycle with alerts and exit code 0. This is conditional local acceptance, not production deployment. |
| 11 | PySpark data-processing implementation | Not implemented. |
| 12 | Azure and Databricks deployment design | Not implemented. |

## Validation Snapshot

The standard local validation commands are:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m ruff check .
```

The current backend validation snapshot is pinned to merged PR #42 and master
commit `271ed3a09cdabb3fbcec756f2a3121c642f07921`. The PR validation and a local
Windows rerun on 2026-08-08 both reported `624` passed tests, `17` skipped
tests, and `71.41%` total coverage; the PR's GitHub Actions CI workflow also
completed successfully.

The earlier 2026-07-28 final controlled-retraining acceptance remains a
separate historical checkpoint. It ran both complete bootstrap-to-stability
and bootstrap-to-rollback paths: the dedicated suite passed `7` tests, while
the then-current full backend suite passed `388` tests with `4` skipped and
`70.74%` total coverage. It also pinned the tracked v1 raw data, model/scaler
artifacts, and modelling notebook by SHA-256. The complete dated evidence
remains in `docs/CONTROLLED_RETRAINING.md`.

### Governed local v2 snapshots

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

This is retained as the historical pre-operation snapshot. The subsequent
assisted local cycle produced the following ignored, checksum-pinned evidence:

- audited scheduler-lease recovery
  `0ec0e5ddb73f3b6d8fe333b1a0255d5e5ac7c7213d45d67e1d01548336fbbbb9`
  on 2026-08-08; its record states that both configured Windows scheduler
  tasks were absent, so no successful scheduled execution is claimed;
- real REN + ERA5-Land source run `20260808T182255Z-cdd58769f31d`, manifest
  SHA-256
  `afbd58f7f184dbccdfa05af3a1568ad70dc877f31b7095d75ea5d15da6d0505b`,
  status `succeeded`, generation 2, and common validated watermark advanced
  from 2026-06-27 to 2026-06-28;
- quality verdict `FAIL`, with six critical `source_late` findings, six
  `incomplete_source_coverage` warnings, and one active
  `quality:source_late` alert;
- monitoring run `20260808T212647Z-fe38fb4105ac`, result SHA-256
  `c6df214c032a726d3289c3b560bcddd36ed49be862486b44af1967934179a977`,
  which succeeded with one prediction, one actual, and one metric for
  2026-06-28;
- reporting run `20260808T214827096913Z-86c1e2ada396`, report ID
  `240ff4039c3f55045420b2ee4db47305c8415824b26b40c3e99c616d804687f4`,
  and result SHA-256
  `3c58000997ff5e61df1a79b231e3583e4dfb1f764c233b94deeb59f2a8c1731a`,
  which succeeded with the active quality alert; and
- idempotency rerun `20260809T102023Z-2239ea7854eb`, status `no_op`, no source
  refresh, and manifest SHA-256
  `095128609373facca71d4ffa50a43eb33f50a8bb444879c2ee55585bfe27a198`.

The Airflow implementation has passed its synthetic three-date real-CLI
acceptance, but its DAGs are inactive in this environment. The owner/lease
contract prevents it from competing with Windows Task Scheduler.

### 2026-08-10 local scheduler runtime

- `WindForecastMlflow` and `WindForecastOperationalApi` were registered under
  the interactive operator, `RunLevel Limited`, and loopback-only bindings.
  Final verification found both tasks `Running`, one listener each on
  `127.0.0.1:5000` and `127.0.0.1:8000`, and HTTP 200 health responses.
- `operational_summary`, `active_deployment`, and `active_model_metadata`
  returned HTTP 200 with typed status `answered`. Deployment generation 1,
  model `wind-forecast-v2-hindcast` version 1, `champion=1`, `stable=1`, and no
  `candidate` remained unchanged.
- The first batch evidence, coordinator
  `20260810T094943124488Z-3e7965d1`, is preserved as a fail-closed attempt. Its
  cause was asymmetric source lag: REN had advanced while ERA5-Land for
  2026-08-05 was pending. PR #50 now limits downstream integration to dates
  complete in both sources.
- Provider-backed source run `20260810T160650Z-4b94be9391ff` succeeded,
  published generation 3, and validated 141 rows. REN is validated through
  2026-08-09; ERA5-Land and the common watermark are 2026-08-04. Its manifest
  SHA-256 prefix is `d9c3d39`. The enclosing batch then exposed a separate
  stdout-contract defect: event JSONL preceded the child's final JSON. The
  working-tree recovery fix retains file JSONL and routes events to stderr.
- The Scheduler cycle started at 2026-08-10 17:24:37 `Europe/Lisbon`.
  Coordinator `20260810T162437840444Z-e84c8f23` completed with alerts,
  manifest SHA-256 prefix `cba08ab`, and `LastTaskResult=0`. The enabled daily
  task returned to `Ready`, next start 2026-08-11 12:00. Source child
  `20260810T162529Z-62bb6c3c3135` was `no_op` (manifest prefix `0bd1`);
  monitoring `20260810T162550Z-25ba7c6af311` succeeded with 36 predictions,
  37 actuals, and 37 metrics (result prefix `11655`); reporting
  `20260810T164401115962Z-008883b7813d` succeeded with report ID prefix
  `6f8ca` and result prefix `34a3`. The active `quality:source_late` alert was
  preserved.
- `WindForecastMonthlyGovernance` remains enabled and `Ready` but was not
  manually triggered. Its pre-existing `LastTaskResult=1` is not successful
  evidence and its impact must be re-evaluated separately against the advanced
  horizon. Airflow remains inactive.

Task Scheduler's Operational event channel was disabled, so the successful
cycle is correlated through `LastRunTime`, scheduler lease, and immutable
manifests. `Stop-ScheduledTask` did not terminate the native MLflow/API child
processes; exact PID-tree verification and termination were required before a
clean restart. The service evidence must not be represented as a clean
scheduler stop.

The 2026-08-10 decision was **CONDITIONAL GO** for the local delayed historical
hindcast, Windows orchestration, and read-only API only, and **NO-GO** for
real-time, D+1, or production. It was superseded for automatic operation by the
2026-08-11 incident described below. Phase 9 D+5 is the authoritative SLO and
required data through 2026-08-05; the common watermark of 2026-08-04 misses it
by one day. Phase 8 D-6 remains only the conservative recovery-time gate.
Six historical REN dates remain unavailable:
2014-05-03, 2016-02-03, 2016-02-04, 2021-10-03, 2023-08-30, and 2025-08-02.
The quality verdict remains `FAIL` as an accepted limitation only; gaps and
alerts are not suppressed. All local evidence is ignored by Git and absent
from a fresh clone.

### 2026-08-11 automatic-operation incident

- Both local service tasks stopped at 10:24 with `0xC000013A`; no listener on
  `127.0.0.1:5000` or `127.0.0.1:8000` and no same-day service log remained.
- `WindForecastHistoricalBatch` started at 12:00, returned
  `LastTaskResult=1`, and created no coordinator manifest. Its lease was
  released and the current pointer remained on the 2026-08-10 manual success.
- On 2026-08-12 the four pre-containment task definitions and a checksum
  inventory of local evidence were exported to an ignored recovery snapshot.
  The live daily registration then changed only `Settings.Enabled`; its action,
  principal, triggers, and other settings were preserved. Airflow remained
  inactive and no data, model, Registry, deployment, pointer, or manifest was
  changed.
- Enabling `Microsoft-Windows-TaskScheduler/Operational` initially failed with
  `Access is denied`. An administrator enabled it without clearing events on
  2026-08-16.
- The 2026-08-16 recovery attempt briefly reached one MLflow listener and HTTP
  200, then ended again with `0xC000013A`. Windows recorded a Uvicorn worker
  `python.exe` access violation in the Conda `MSVCP140.dll` (`0xc0000005`), the
  same native signature seen five times since 2026-08-10, followed by
  multiprocess socket failures (`WinError 10022`). Isolated copied-store probes
  stayed healthy with one worker and also with four workers interactively,
  narrowing the fault to the contextual scheduled multiprocess path. The
  runner now pins one worker; this has not yet passed the live persistence gate.
- A later one-worker probe kept MLflow healthy beyond the restart window and
  kept the API healthy with repeated HTTP 200 responses while the PowerShell
  session that started its task remained open. The API ended with `0xC000013A`
  exactly when that session closed. MLflow later ended with the same result,
  but no equivalent initiating-session correlation was captured. Neither
  output had an application traceback or graceful shutdown, and their JSONL
  evidence stopped at `child:started`. The interactive execution context is
  therefore the leading hypothesis, not a proven common cause. The reviewed
  service registration change selected the same user with non-interactive
  `LogonType S4U`; actions, triggers, settings, runners, loopback bindings,
  logs, and exit-code propagation are unchanged. S4U stores no password,
  cannot access network resources or encrypted files, and requires the
  effective `Log on as a batch job` right. At that checkpoint it was definition
  only; the next bullet records its post-merge live application.
- After merge, the effective logon right was verified and only the two service
  principals were replaced. MLflow and the API then stayed `Running` for a
  five-minute observation with stable PIDs, one listener each, HTTP 200, and all
  three typed queries `answered`.
- The daily task remained `InteractiveToken`. Its 2026-08-16 read-only plan
  succeeded, but its single authorized manual start ended after 5m58s with
  `0xC000013A`/`STATUS_CONTROL_C_EXIT`. Runner PID 25588 stopped without a
  graceful PowerShell event, coordinator `20260816T205748784400Z-7b0bee70`
  retained only the verified deployment preflight, both locks were preserved,
  and no current pointer changed. The monitoring launcher ended only after the
  task action, and no matching session, power, Application Error, or System
  event identified the control-signal sender.
- The batch registration script now changes only its principal from
  `Interactive` to the same user with `LogonType S4U` and `RunLevel Limited`.
  Its action, arguments, 12:00 trigger, six-hour limit, retries, `IgnoreNew`,
  lease, runner, and exit semantics are unchanged. This definition has not been
  applied or validated live; provider access under the exact S4U principal is a
  mandatory post-merge gate.
- Automatic operation is **NO-GO** until persistent MLflow/API health, one
  successful manual daily batch, and one later successful automatic batch are
  all verified without locks, leases, or child processes left behind.

The D+5 objective remains authoritative and D-6 remains a provisional
conservative eligibility gate during recovery. The owner, final decision, and
review deadline will be recorded with the live recovery evidence; any D-5 code
change remains a separate evidence-gated task.

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
- The Phase 8 live-provider path requires approved credentials and network
  access. Local provider-backed and Task Scheduler cycles have exercised it,
  but their ignored machine-local evidence retains a critical source-lateness
  alert and supports only the conditional delayed-hindcast decision.
- A fresh clone may need local generated artifacts before the full prediction
  workflow can be demonstrated end to end.
- The 2026-08-10 Scheduler recovery reached exit code 0, but the common
  watermark was one day behind the authoritative D+5 objective and six
  historical REN gaps remain. Successful local execution is not evidence of
  real-time, D+1, or production readiness.

## Recommended Next Steps

1. Resolve v1 provenance, licence, attribution, and redistribution approval,
   then publish the first checksum-pinned immutable artifact bundle.
2. Prove the documented fetch, verification, and retraining round-trip from a
   clean clone before claiming cross-machine reproducibility.
3. Extend the baseline training CLI toward the tuned notebook workflow while
   preserving the current contracts.
4. Raise test coverage as source-ingestion and v2 modules mature.
5. Use the accepted offline evaluation harness as a mandatory gate for any
   separately approved future Copilot candidate. No candidate has been
   evaluated; Copilot, MCP, RAG, production authentication, remote exposure,
   and cloud stages are not implemented.
6. Define and review an interactive-prediction UI/API contract separately,
   then address production deployment as a later roadmap phase.
