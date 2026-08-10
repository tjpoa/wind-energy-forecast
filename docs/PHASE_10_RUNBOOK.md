# Phase 10 Local Batch Runbook

## Safety rules

- Never edit, overwrite, or delete raw observations, release partitions,
  predictions, actuals, metrics, reports, or their manifests during recovery.
- Never bypass checksum, schema, activation, or lock validation.
- Use the same `through-date`, model, calibration, activation, and backfill
  arguments when retrying an interrupted run.
- Use `--no-source-refresh` only when recovering from verified local inputs; it
  is not evidence that providers were checked.
- Do not run notebooks, training, promotion, or live downloads as a workaround.

## First response

1. Inspect Task Scheduler `LastTaskResult` and task history.
2. Run `run_batch_pipeline.py status` and identify `failed_stage`.
3. Inspect the referenced child manifest and its events without modifying it.
4. Verify the Phase 8, Phase 9, and reporting current pointers with their
   supported readers or dry-run commands.
5. Correct only the external/configuration cause and rerun the identical batch.

## Failure recovery

| Failure | Evidence | Recovery |
| --- | --- | --- |
| Missing credentials | CLI stderr names missing configuration without values | Configure the scheduled user's environment, ignored `.env`, or `.cdsapirc`; rerun |
| Provider unavailable or quota limited | Dataset-update child events and failed coordinator stage | Wait for provider recovery; rerun the identical batch |
| Source remains unavailable | Successful update with explicit pending/gap state | Take no destructive action; later daily runs retry according to source policy |
| Validation/schema/checksum failure | Failed Phase 8 manifest and unchanged verified current pointer | Correct or reacquire the source in a new immutable revision; rerun |
| Timeout | Task Scheduler result plus coordinator stage | Confirm no child process remains and pointers verify; rerun |
| Live lock | Lock names a live local PID | Do not start another run; inspect or wait for the owner |
| Stale same-host lock | PID no longer exists | A new coordinator run recovers the lock automatically |
| Other-host lock | Lock host differs | Do not guess staleness; coordinate with that host |
| Failure after dataset publication | Phase 8 pointer verifies but downstream stage failed | Rerun; Phase 8 converges to no-op and downstream work resumes |
| Failure after prediction | Immutable prediction exists without later evidence | Rerun; reconciliation/reporting continue without duplicate issuance |
| Active drift alert | Batch is `completed_with_alerts` by default | Inspect the report and alert history; do not retrain or promote automatically |
| Corrupt current pointer | Reader rejects path/checksum/schema | Stop. Preserve all files and escalate for an evidence-based repair plan |
| Deployment/bundle/alias divergence | Preflight fails before source update, or a later postcheck blocks publication | Preserve evidence; restore pointer, bundle selection, and aliases to the same approved state, then rerun identically |

## Scheduler operations

The task is registered as `WindForecastHistoricalBatch` unless overridden.

```powershell
Get-ScheduledTask -TaskName WindForecastHistoricalBatch
Get-ScheduledTaskInfo -TaskName WindForecastHistoricalBatch
Start-ScheduledTask -TaskName WindForecastHistoricalBatch
Disable-ScheduledTask -TaskName WindForecastHistoricalBatch
```

Disabling the task is the safe operational rollback. Do not unregister it or
remove data during incident response. Re-registration replaces only the task
definition and must first be reviewed with `-WhatIf`.

## Persistent local MLflow and operational API

MLflow and the read-only operational API are separate logon-triggered tasks.
Both run under the current interactive identity with `RunLevel Limited`, bind
only to IPv4 loopback, never overlap a running instance, and have no execution
time limit. They do not run before that user logs on. The registration scripts
create or replace task definitions but do not start them.

The MLflow runner requires the existing ignored SQLite database and artifact
directory below `var/mlflow/`. The API runner sets only its process-local
deployment, monitoring, tracking, and projection-mode variables. It does not
load a general `.env`; PostgreSQL projection consumption remains disabled.
Each process appends to a uniquely named log below ignored
`var/local_services/`.

Review both exact task definitions before registration:

```powershell
.\scripts\register_local_mlflow_task.ps1 `
  -PythonExecutable .\venv\Scripts\python.exe `
  -RepositoryRoot $PWD `
  -WhatIf

.\scripts\register_local_operational_api_task.ps1 `
  -PythonExecutable .\venv\Scripts\python.exe `
  -RepositoryRoot $PWD `
  -DeploymentRoot .\data\processed\v2\deployment `
  -MonitoringStoreRoot .\data\processed\v2\monitoring `
  -ModelBundle .\outputs\training\v2_reference_mlflow `
  -CalibrationDirectory `
    .\data\processed\v2\monitoring\reporting\calibrations\<CALIBRATION_ID> `
  -WhatIf
```

After reviewing the resolved executable, arguments, identity, logon trigger,
restart policy, zero execution limit, and `IgnoreNew`, repeat each command
without `-WhatIf`. Stop if the displayed identity is a sandbox, automation,
service, or different Windows account instead of the intended interactive
operator; registration must be performed from that operator's own session.
Registration is not a health check. Start and verify MLflow first, then start
the API, whose runner waits at most 120 seconds for MLflow:

```powershell
Start-ScheduledTask -TaskName WindForecastMlflow
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:5000/health

$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
.\venv\Scripts\python.exe .\scripts\verify_active_deployment.py `
  --deployment-root .\data\processed\v2\deployment `
  --model-bundle .\outputs\training\v2_reference_mlflow `
  --calibration-dir `
    .\data\processed\v2\monitoring\reporting\calibrations\<CALIBRATION_ID>

Start-ScheduledTask -TaskName WindForecastOperationalApi
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
Invoke-RestMethod -Method Get `
  -Uri http://127.0.0.1:8000/api/v1/monitoring/latest
```

Use `Get-ScheduledTask`, `Get-ScheduledTaskInfo`, the ignored service logs, and
the health responses together when diagnosing startup. A configuration change
does not alter an already running API process: stop it and start its task again
so Uvicorn inherits the reviewed environment.

An `empty` operational answer is valid only when the configured, verified store
has no matching accepted evidence. Compare the API task's resolved
`MonitoringStoreRoot` and `DeploymentRoot` with the in-process configuration,
and confirm that its resolved model bundle and calibration directory match the
active deployment. These two selections are mandatory and are passed only to
the API process as `WIND_FORECAST_OPERATIONAL_MODEL_BUNDLE` and
`WIND_FORECAST_OPERATIONAL_CALIBRATION_DIR`; absence or a blank value is a
lazy operational-query service configuration failure, surfaced as HTTP 503
when that service is first required. The API process and `/health` endpoint can
already be running because configuration is loaded lazily. Then use the
supported Phase 9 loader or monitoring endpoint to verify the same
current pointer and report. Also compare the deployment loader with live MLflow
aliases. Do not infer a cause from an older `empty`: if the original process
configuration cannot be reproduced, record only that the current store has
evidence while the earlier process was stopped, stale, or selected another
root. Restart the API after every environment or task-action change.

Operational rollback is limited to stopping and disabling the two tasks:

```powershell
Stop-ScheduledTask -TaskName WindForecastOperationalApi
Disable-ScheduledTask -TaskName WindForecastOperationalApi
Stop-ScheduledTask -TaskName WindForecastMlflow
Disable-ScheduledTask -TaskName WindForecastMlflow
```

Do not unregister tasks or delete SQLite, artifacts, reports, pointers,
manifests, aliases, or append-only evidence during rollback. The monthly
governance task is outside this service-restoration scope.

## Airflow operations

Use exactly one scheduler. Disable the Windows task before unpausing
`wind_forecast_historical_batch_v1`. If Airflow must be rolled back, pause the
DAG, let any running task finish, stop the stack without `-v`, and re-enable the
Windows task only after all pointers and locks verify.

Inspect task logs and immutable manifests together. When an upstream task
remains successful and its checksum-pinned manifest verifies, clear only the
failed task and its downstream tasks. Otherwise rerun the entire DAG; existing
contracts converge idempotently.

```powershell
docker compose -f airflow/docker-compose.yml ps
docker compose -f airflow/docker-compose.yml logs airflow-scheduler
docker compose -f airflow/docker-compose.yml stop
```

Never use `docker compose down -v` during recovery. Metadata or volume removal
requires separate authorization. Active drift alerts complete the final task
with logical status `completed_with_alerts` unless
`WIND_FORECAST_FAIL_ON_ACTIVE_ALERT=true`, in which case exit code 2 fails only
`drift_publish`.

## Recovery completion

Recovery is complete only when the coordinator returns `succeeded`,
`completed_with_alerts`, or a verified no-op child update; all current pointers
verify; no lock remains; and Task Scheduler records exit code `0` unless
alert-failure behavior was explicitly requested.

### Recorded fail-closed example: 2026-08-10

A real `WindForecastHistoricalBatch` start reached provider retrieval and base
validation, then rejected the integrated 2026-08-05 ERA5-Land partition for
incomplete station-day/hour coverage. It ended with coordinator status
`failed`, stage `dataset_update`, and `LastTaskResult=1`. No current pointer
for the source dataset, deployment, monitoring, or Registry aliases changed.
The coordinator current pointer did advance to immutable failed run
`20260810T094943124488Z-3e7965d1`, recorded with manifest checksum
`d296b60868eef4ffe2af16a1d542fe28712f2bba9db061b05286a5d7e7976427`.
The scheduler lease, batch lock, and child processes were absent after exit.
The task was disabled to prevent its configured retry and was not rerun.
Recovery is not complete, and this event must not be represented as a
successful scheduler batch. Preserve its immutable manifests and reacquire or
correct the source in a new revision before an explicitly authorized identical
retry. Monthly governance was not manually triggered during this validation;
its pre-existing 2026-08-08 run remains `LastTaskResult=1` and is not evidence
of successful governance.
