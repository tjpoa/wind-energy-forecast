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
