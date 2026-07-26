# Phase 10 — Local Batch Orchestration

## Status

Phase 10 Part 1 implements the approved local-first batch contract. Apache
Airflow remains gated until this workflow passes local review and recovery
evidence.

The operating mode remains the Phase 9 delayed historical hindcast. This work
does not introduce D+1 forecasting, retraining, model promotion, external
notifications, or notebook execution.

## Stable batch interface

The supported coordinator is:

```powershell
.\venv\Scripts\python.exe .\scripts\run_batch_pipeline.py plan `
  --through-date YYYY-MM-DD `
  --model-bundle outputs\training\v2_reference_mlflow `
  --calibration-dir data\processed\v2\monitoring\reporting\calibrations\<ID>
```

After reviewing the plan, replace `plan` with `run`. `status` verifies the
latest coordinator manifest and its checksum-pinned pointer. An installed
editable package also exposes the equivalent `wind-forecast-batch` command.

The coordinator preserves the existing atomic boundaries:

1. read-only source availability plan;
2. transactional Phase 8 ingestion, validation, integration, and features;
3. Phase 9 hindcast issuance and actual reconciliation;
4. immutable drift/performance calculation and report publication.

Every real attempt writes an append-only
`wind_forecast.batch_run.v1` manifest below
`data/processed/v2/orchestration/runs/`. The only mutable coordinator artifact
is the atomic `state/current.json` pointer. A failed stage blocks all downstream
stages. Recovery is an identical rerun after correcting the cause.

## Local schedule

Register the task only after validating all selected paths:

```powershell
.\scripts\register_local_batch_task.ps1 `
  -PythonExecutable .\venv\Scripts\python.exe `
  -RepositoryRoot $PWD `
  -ModelBundle .\outputs\training\v2_reference_mlflow `
  -CalibrationDirectory .\data\processed\v2\monitoring\reporting\calibrations\<ID> `
  -ActivationDate YYYY-MM-DD `
  -WhatIf
```

Remove `-WhatIf` only after reviewing the generated action. The task runs daily
at local 12:00, never overlaps itself, has a six-hour execution limit, and is
retried twice at 30-minute intervals. It uses the current interactive Windows
identity and stores no credential in the repository.

Model and calibration paths may alternatively be supplied through
`WIND_FORECAST_BATCH_MODEL_BUNDLE` and
`WIND_FORECAST_BATCH_CALIBRATION_DIR`. CDS credentials remain in the scheduled
user's environment, an explicitly selected ignored `.env`, or `.cdsapirc`.
Persisted evidence records no credential values.

## Acceptance gate for Airflow

Airflow must not start until the local CLI and scheduler definition pass:

- read-only planning;
- synthetic end-to-end execution;
- no-op/idempotent rerun;
- injected failure and recovery;
- local artifact dry-run;
- full tests, lint, diff review, and user review of the draft PR.

No live provider refresh is required by this gate. Any live REN/CDS request
requires separate authorization.
