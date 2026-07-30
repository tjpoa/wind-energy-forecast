# Phase 10 — Local Batch and Airflow Orchestration

## Status

Phase 10 Part 1 implements the approved local-first batch contract. Part 2
implements a separate local Apache Airflow 3.3.0 stack after Part 1 was
reviewed and merged as PR #19. Its required synthetic three-date real-CLI
offline backfill gate passed. Both operating modes call the same stable CLIs,
but only one scheduler may own an environment.

The operating mode remains the Phase 9 delayed historical hindcast. This work
does not introduce D+1 forecasting, retraining, model promotion, external
notifications, or notebook execution.

## Stable batch interface

The supported coordinator is:

```powershell
.\venv\Scripts\python.exe .\scripts\run_batch_pipeline.py plan `
  --through-date YYYY-MM-DD `
  --model-bundle outputs\training\v2_reference_mlflow `
  --calibration-dir data\processed\v2\monitoring\reporting\calibrations\<ID> `
  --deployment-root data\processed\v2\deployment
```

After reviewing the plan, replace `plan` with `run`. `status` verifies the
latest coordinator manifest and its checksum-pinned pointer. An installed
editable package also exposes the equivalent `wind-forecast-batch` command.

The coordinator preserves the existing atomic boundaries:

1. fail-closed active-deployment preflight;
2. read-only source availability plan;
3. transactional Phase 8 ingestion, validation, integration, and features;
4. Phase 9 hindcast issuance and actual reconciliation;
5. immutable era-scoped report and active-deployment postcheck.

Every real attempt writes an append-only
`wind_forecast.batch_run.v2` manifest below
`data/processed/v2/orchestration/runs/`. The only mutable coordinator artifact
is the atomic `state/current.json` pointer. A failed stage blocks all downstream
stages. Recovery is an identical rerun after correcting the cause.

Deployment verification and batch execution require the tracking URI sealed in
the active deployment to be reachable. For the governed local environment this
means keeping the local MLflow service available; do not silently substitute a
different Registry or tracking backend.

## Local schedule

Configure the environment owner first, then register the task only after
validating all selected paths:

```powershell
.\venv\Scripts\python.exe .\scripts\manage_scheduler_owner.py configure `
  --scheduler-root .\data\processed\v2\orchestration\scheduler `
  --environment-id local `
  --owner windows_task_scheduler `
  --expected-generation 0 `
  --expect-no-owner
```

```powershell
.\scripts\register_local_batch_task.ps1 `
  -PythonExecutable .\venv\Scripts\python.exe `
  -RepositoryRoot $PWD `
  -ModelBundle .\outputs\training\v2_reference_mlflow `
  -CalibrationDirectory .\data\processed\v2\monitoring\reporting\calibrations\<ID> `
  -DeploymentRoot .\data\processed\v2\deployment `
  -SchedulerStateRoot .\data\processed\v2\orchestration\scheduler `
  -EnvironmentId local `
  -ActivationDate YYYY-MM-DD `
  -WhatIf
```

Remove `-WhatIf` only after reviewing the generated action. The task runs daily
at local 12:00, never overlaps itself, has a six-hour execution limit, and is
retried twice at 30-minute intervals. It uses the current interactive Windows
identity and stores no credential in the repository.

Register the separate recommendation-only monthly task against the same owner
and lease:

```powershell
.\scripts\register_local_monthly_governance_task.ps1 `
  -PythonExecutable .\venv\Scripts\python.exe `
  -RepositoryRoot $PWD `
  -MonitoringStoreRoot .\data\processed\v2\monitoring `
  -DeploymentRoot .\data\processed\v2\deployment `
  -SchedulerStateRoot .\data\processed\v2\orchestration\scheduler `
  -EnvironmentId local `
  -WhatIf
```

It runs on day 8 at 13:00 local time and invokes only
`run_monthly_governance.py`. Training, backtesting, Registry operations and
deployment transitions are not scheduled.

### Verified local scheduler snapshot

At 2026-07-30 00:25 `Europe/Lisbon`, the ignored `local` scheduler state was
verified at generation 1 with owner `windows_task_scheduler`. The tasks
`WindForecastHistoricalBatch` and `WindForecastMonthlyGovernance` were enabled
and `Ready`, but neither had run. Their next scheduled starts were
2026-07-30 12:00 and 2026-08-08 13:00 local time.

The monthly registration script's CIM path was incompatible with this Windows
installation. The monthly task was therefore registered through the equivalent
Task Scheduler COM definition and then inspected against the intended action,
trigger, account, overlap, retry, and execution-limit contract. A future
re-registration on this machine must either use the same reviewed fallback or
first correct and validate the CIM incompatibility.

The v2 store currently ends on 2026-06-27. Consequently, the governance
dry-run for evaluation period 2026-07 could not select the required report
dated 2026-06-30. This is an expected data-availability warning, not evidence
that the monthly task has run successfully.

Model, calibration, and deployment paths may alternatively be supplied through
`WIND_FORECAST_BATCH_MODEL_BUNDLE` and
`WIND_FORECAST_BATCH_CALIBRATION_DIR`, and
`WIND_FORECAST_DEPLOYMENT_ROOT`. CDS credentials remain in the scheduled
user's environment, an explicitly selected ignored `.env`, or `.cdsapirc`.
Persisted evidence records no credential values.

## Apache Airflow operation

Airflow is implemented and synthetically validated, but it is inactive in the
governed `local` environment above. Do not start its DAGs while the owner is
`windows_task_scheduler`. Switching owners is an explicit operational change,
not a second concurrent schedule.

Copy `airflow/.env.example` to the ignored `airflow/.env` and replace every
explicit artifact selection and blank local credential/database field. The
example contains no functional passwords or connection string. Before starting
Airflow, atomically change the same environment owner to `airflow`. The daily
and monthly DAGs both fail closed if the owner or shared execution lease does
not permit Airflow.

```powershell
.\venv\Scripts\python.exe .\scripts\manage_scheduler_owner.py configure `
  --scheduler-root .\data\processed\v2\orchestration\scheduler `
  --environment-id local `
  --owner airflow `
  --expected-generation <CURRENT_GENERATION> `
  --expected-owner windows_task_scheduler

docker compose -f airflow/docker-compose.yml config --quiet
docker compose -f airflow/docker-compose.yml build
docker compose -f airflow/docker-compose.yml up airflow-init
docker compose -f airflow/docker-compose.yml up -d
```

The local UI/API server is exposed on `http://localhost:8080`. Airflow's local
Simple Auth Manager generates the selected admin user's password in
`airflow/logs/simple_auth_manager_passwords.json.generated`. The stack uses
Linux containers, PostgreSQL and `LocalExecutor`; it is deliberately separate
from the API/dashboard Compose stack and is not a production deployment.

`wind_forecast_historical_batch_v1` runs at 12:00 in `Europe/Lisbon`, derives
`through_date` from `data_interval_end` converted to Lisbon, and has the linear
graph:

```text
scheduler_lease -> deployment_preflight -> availability_plan
  -> dataset_update -> predict_reconcile -> drift_publish
  -> deployment_postcheck -> scheduler_release
```

Only compact status, paths and checksums cross task boundaries. Data, reports
and evidence stay in the existing versioned stores.

If a process terminates while holding the lease, inspect its immutable JSON
first and recover only that exact `lease_id` with an operator identity and
audit note:

```powershell
.\venv\Scripts\python.exe .\scripts\manage_scheduler_owner.py recover `
  --scheduler-root .\data\processed\v2\orchestration\scheduler `
  --environment-id local `
  --lease-id <ABANDONED_LEASE_ID> `
  --recovered-by <OPERATOR> `
  --note "<WHY THE RUN IS CONFIRMED ABANDONED>"
```

For the required limited offline backfill, generate the temporary synthetic
REN/ERA5 fixture inside the Airflow image, point the ignored `airflow/.env` at
that fixture, and select exactly three consecutive dates. The fixture builder
does not contact a provider and is never committed:

```powershell
docker compose -f airflow/docker-compose.yml exec airflow-scheduler `
  python airflow/tests/build_real_fixture.py `
  --root /opt/wind-energy-forecast/data/processed/v2/airflow_smoke_fixture
```

Keep source refresh disabled and use `failed` reprocessing:

```powershell
docker compose -f airflow/docker-compose.yml run --rm airflow-cli backfill create `
  --dag-id wind_forecast_historical_batch_v1 `
  --from-date YYYY-MM-DD `
  --to-date YYYY-MM-DD `
  --reprocess-behavior failed `
  --max-active-runs 1 `
  --dry-run
```

Remove `--dry-run` only after confirming three intervals and verified local
fixtures. Never use this validation command with live REN/CDS access.

## Part 2 acceptance evidence

- the Airflow 3.3.0 image builds with Python 3.11;
- Compose validates independently from the API stack;
- `airflow dags list-import-errors --output=json` returns `[]`;
- graph, schedule, retries and timeouts are inspectable without provider calls;
- the bridge verifies the Phase 8 manifest checksum before downstream work;
- a synthetic three-date real-CLI backfill (2026-01-15 through 2026-01-17)
  completed serially with all four tasks successful;
- rerunning the same `failed` backfill left the immutable counts unchanged
  (3 prediction records, 3 actual records, 3 metric records, 6 report files,
  and 6 source-run files); and
- setting `WIND_FORECAST_FAIL_ON_ACTIVE_ALERT=true` caused only
  `drift_publish` to enter `up_for_retry`; restoring the setting and clearing
  that task recovered it to `success` while the three upstream tasks remained
  successful and their checksums were unchanged;
- the local `GET /api/v1/monitoring/latest` projection returned HTTP 200 for
  the resulting immutable evidence.

No live provider refresh is part of this evidence. The accepted three-date
backfill exercised the real CLIs over a generated synthetic fixture; its
observations and evidence were temporary ignored artifacts, not committed
demonstration data. A live-provider backfill remains a separate, explicitly
authorized operation and is not required to close the synthetic Phase 10
acceptance gate.
