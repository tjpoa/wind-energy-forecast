# Phase 10 — Local Batch and Airflow Orchestration

## Status

Phase 10 Part 1 implements the approved local-first batch contract. Part 2
implements a separate local Apache Airflow 3.3.0 stack after Part 1 was
reviewed and merged as PR #19. Its required synthetic three-date real-CLI
offline backfill gate passed. Both operating modes call the same stable CLIs,
but only one scheduler may own an environment.

An assisted local cycle on 2026-08-08 subsequently exercised the real REN and
ERA5-Land refresh, monitoring, and reporting paths, followed by an idempotent
`no_op` rerun on 2026-08-09. It followed an audited abandoned-lease recovery
and is not evidence of a successful Windows Task Scheduler or Airflow run.

On 2026-08-10 the four `WindForecast*` definitions were audited in the
interactive operator session. MLflow and the operational API
were started, stopped, and restarted through Task Scheduler, with exclusive
loopback listeners and healthy read-only responses. A real scheduled batch was
also started exactly once, but failed closed while validating the integrated
ERA5-Land partition for 2026-08-05. The daily task was disabled to prevent its
configured retries. This proves the service runtime and the scheduler's batch
invocation boundary, but does not satisfy successful batch end-to-end
acceptance.

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

### Historical local scheduler snapshot

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

### Assisted real-provider cycle evidence

On 2026-08-08, audited recovery
`0ec0e5ddb73f3b6d8fe333b1a0255d5e5ac7c7213d45d67e1d01548336fbbbb9`
released an abandoned scheduler lease. Its sanitized record states that both
configured Windows scheduler tasks were absent. The following steps were then
run with operator assistance; they do not establish unattended scheduler
operation:

- real REN + ERA5-Land refresh run `20260808T182255Z-cdd58769f31d` succeeded,
  published generation 2, and advanced the common validated watermark from
  2026-06-27 to 2026-06-28; its manifest SHA-256 is
  `afbd58f7f184dbccdfa05af3a1568ad70dc877f31b7095d75ea5d15da6d0505b`;
- the source quality sidecar returned `FAIL`, with six critical `source_late`
  findings and six `incomplete_source_coverage` warnings;
- monitoring run `20260808T212647Z-fe38fb4105ac` succeeded with one prediction,
  one actual, and one metric for 2026-06-28; its result SHA-256 is
  `c6df214c032a726d3289c3b560bcddd36ed49be862486b44af1967934179a977`;
- reporting run `20260808T214827096913Z-86c1e2ada396` succeeded with report ID
  `240ff4039c3f55045420b2ee4db47305c8415824b26b40c3e99c616d804687f4`
  and one active `quality:source_late` alert; its result SHA-256 is
  `3c58000997ff5e61df1a79b231e3583e4dfb1f764c233b94deeb59f2a8c1731a`;
  and
- the 2026-08-09 rerun `20260809T102023Z-2239ea7854eb` performed no source
  refresh and converged to `no_op`; its manifest SHA-256 is
  `095128609373facca71d4ffa50a43eb33f50a8bb444879c2ee55585bfe27a198`.

The technical update and monitoring steps therefore completed while their
quality contract correctly preserved a critical source-lateness signal. All
referenced operational evidence is local, ignored by Git, and absent from a
fresh clone. It neither changes the retrospective-hindcast contract nor
constitutes a production claim.

### 2026-08-10 Task Scheduler runtime validation

All registration commands were reviewed with `-WhatIf` using model bundle
SHA-256 `0cf133d73b2c9c949899bc3bc89c7ab4c76c8c641246f7cc157d88fed596a96d`
and calibration ID
`ff56dd507607a95aea81f76ab6ce694f1fd8eb51a97175f834bdb83c16b2fe58`.
The two logon tasks use the interactive operator, `RunLevel Limited`,
`IgnoreNew`, and loopback-only bindings. The daily task retains its 12:00
trigger, six-hour limit, two 30-minute retries, and owner
`windows_task_scheduler`; the monthly recommendation-only task retains day 8
at 13:00, a two-hour limit, two 15-minute retries, and its reviewed COM
definition. Monthly governance was not manually triggered during this
validation because the verified reporting horizon remains 2026-06-28 and the
required month-close report is absent. Its pre-existing 2026-08-08 execution
record remains `LastTaskResult=1`; this validation does not claim it succeeded.

The local service runners now preserve setup-time fail-fast handling while
allowing PowerShell 5.1 to append native-process stderr to the ignored service
log without converting normal MLflow/Uvicorn diagnostics into terminating
errors. The operational API task passes an explicit, resolved model bundle and
calibration directory through process-local environment variables; missing
values fail closed. No user-level logging or service environment override
remains configured.

At `2026-08-10T09:36:10.9603473Z`, both service tasks were stopped, ports 5000
and 8000 were confirmed closed, and both were restarted with
`Start-ScheduledTask`. Task Scheduler reported both `Running`; the only
listeners were `127.0.0.1:5000` and `127.0.0.1:8000`. MLflow `/health`, API
`/health`, and `/api/v1/monitoring/latest` returned HTTP 200. The monitoring
response selected report
`240ff4039c3f55045420b2ee4db47305c8415824b26b40c3e99c616d804687f4`.
The `operational_summary`, `active_deployment`, and `active_model_metadata`
queries each returned HTTP 200 with status `answered`. Live Registry
observation remained `wind-forecast-v2-hindcast` version 1 with
`champion=1`, `stable=1`, no `candidate`, and run
`aaedd79348ee404880a4608760cebafd`, consistent with verified deployment ID
`87c25b9b9cf23cb85799ce23f7306c40399fbb4c14dec5a5ac9a2136614d4159`.

The daily task was started exactly once at
`2026-08-10T09:49:42.1669194Z`. Scheduler run
`windows-daily-0108a6d735db45d39f07fb42ce660447` acquired lease
`90d1c5365126ae22ee695af6250abe1bd5fca3a88e8ddb9132d488c4d5896865`;
coordinator run `20260810T094943124488Z-3e7965d1` invoked source run
`20260810T095059Z-8699026fcbce`. Provider retrieval completed and base
validation recorded 141 rows, but integrated validation rejected 2026-08-05
because its ERA5 station-day/hour coverage was incomplete. The coordinator
manifest is checksum-pinned by
`d296b60868eef4ffe2af16a1d542fe28712f2bba9db061b05286a5d7e7976427`
and the failed source manifest by
`89c1549aa5d83d228f58fbddeb0fa20685f349dfead2a9606d0f086ea1e67922`.
Task Scheduler returned `LastTaskResult=1`; the coordinator status is
`failed` at `dataset_update`. The coordinator `state/current.json` advanced to
this immutable failed manifest, run
`20260810T094943124488Z-3e7965d1`, whose recorded checksum is
`d296b60868eef4ffe2af16a1d542fe28712f2bba9db061b05286a5d7e7976427`.
The source dataset remained on generation 2, and the deployment, monitoring,
and Registry-alias pointers remained unchanged. No lease, lock, or child
process remains. The daily task was disabled immediately to block automatic
retry and was not rerun. Final runtime state is therefore:
MLflow and API enabled and `Running`; monthly governance enabled and `Ready`;
historical batch disabled and `Ready`. The requested four-enabled final state
and accepted successful scheduled batch were not achieved. Airflow remains
inactive.

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

No live provider refresh is part of this Airflow acceptance evidence. The
accepted three-date backfill exercised the real CLIs over a generated synthetic
fixture; its observations and evidence were temporary ignored artifacts, not
committed demonstration data. The later assisted real-provider cycle documented
above was not an Airflow backfill and does not alter the synthetic Phase 10
acceptance gate.
