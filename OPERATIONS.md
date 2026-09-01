# Operations

This is the short human runbook. Runtime contracts live in Python, JSON
policies, schemas, manifests, migrations, and tests.

## Safe local demo

The tracked `demo/v1` bundle is deterministic synthetic evidence. It needs no
credentials or provider access.

```powershell
docker compose up --build
Invoke-RestMethod http://localhost:8000/health
Invoke-WebRequest -UseBasicParsing http://localhost:5173
docker compose down
```

Run the smoke check from the repository root:

```powershell
.\venv\Scripts\python.exe .\scripts\smoke_demo_stack.py
```

## Local API

```powershell
.\venv\Scripts\python.exe -m uvicorn --env-file .env wind_forecast.api:app --reload
```

The operational-query route is local-only and typed. Keep it on numeric
loopback listeners; do not expose it through a proxy or remote port mapping.

## Batch lifecycle

Always plan before running. Use explicit model, calibration, deployment, and
through-date inputs; review the plan before any provider refresh or write.

```powershell
.\venv\Scripts\python.exe .\scripts\run_batch_pipeline.py plan `
  --through-date YYYY-MM-DD `
  --model-bundle <bundle> `
  --calibration-dir <calibration> `
  --deployment-root <deployment>

.\venv\Scripts\python.exe .\scripts\run_batch_pipeline.py run `
  --through-date YYYY-MM-DD `
  --model-bundle <bundle> `
  --calibration-dir <calibration> `
  --deployment-root <deployment>
```

The coordinator owns the lease, validates the deployment before source
mutation, publishes immutable evidence, and releases the lease. If a lease is
present, inspect it and use the explicit recovery command; never delete the
lock manually.

```powershell
.\venv\Scripts\python.exe .\scripts\manage_scheduler_owner.py --help
.\venv\Scripts\python.exe .\scripts\run_batch_pipeline.py --help
```

## Readiness gate

`config/local_automation_readiness_v1.json` is the executable local
automation decision. It is currently `NO-GO`, so the daily Task Scheduler
registration and runner must fail closed.

```powershell
.\venv\Scripts\python.exe .\scripts\verify_local_automation_readiness.py `
  --workflow historical_daily_batch
```

Only a reviewed, evidence-backed receipt may change the status to `GO`; a GO
receipt must name the workflow and reference immutable evidence. Do not enable
the daily task while the receipt is `NO-GO`.

The registration script checks scheduler ownership, the Europe/Lisbon
timezone, and readiness. It does not start the task:

```powershell
powershell.exe -File .\scripts\register_local_batch_task.ps1 `
  -PythonExecutable .\venv\Scripts\python.exe `
  -RepositoryRoot . `
  -ModelBundle <bundle> `
  -CalibrationDirectory <calibration> `
  -DeploymentRoot <deployment> `
  -SchedulerStateRoot <scheduler-state> `
  -EnvironmentId local `
  -WhatIf
```

Use `-WhatIf` first. A real registration is an external mutation and requires
explicit approval plus a GO receipt. Keep exactly one scheduler owner between
Windows Task Scheduler and Airflow.

## Monitoring and retraining

Monitoring is delayed historical evidence, not live alerting. Use the command
help and explicit policy paths for report generation, calibration, and monthly
governance:

```powershell
.\venv\Scripts\python.exe .\scripts\run_monitoring_report.py --help
.\venv\Scripts\python.exe .\scripts\run_monthly_governance.py --help
.\venv\Scripts\python.exe .\scripts\manage_v2_deployment.py --help
```

Retraining, candidate registration, promotion, stabilization, and rollback are
manual approval-gated operations. Automatic training and promotion remain
disabled.

## Provider and artifact safety

- Do not overwrite v1 raw data, models, scalers, manifests, or reports.
- The v1 CSVs are internal material. Keep authorized local copies only at the
  paths in `data/manifests/v1_source_contract.json`; they are intentionally not
  tracked in the public repository. A clean clone must use metadata validation
  and must fail closed when a reader needs the missing files.
- Use `--dry-run` for ingestion, monitoring, deployment, and recovery tools
  before any write or provider call.
- Keep provider credentials in `.env` or the approved user credential store;
  never commit them.
- A source, feature, spatial, or distribution change requires a new manifest,
  refitted scalers, retraining, and a new baseline.
- The real-data release remains blocked while provenance/licence/
  redistribution approval is unresolved. `approved: true` requires explicit
  evidence for both production and weather; absence of a prohibition is not
  permission.
- Earlier Git commits still contain the v1 CSV blobs. History purging or
  provider takedown is a separate owner-authorized remediation and is not
  performed by the provenance gate.

## Static validation

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m ruff check .
docker compose config --quiet
```

The Azure workflows additionally validate Terraform formatting, plans, image
digests, smoke checks, and post-deployment drift. Do not run `terraform plan`,
`terraform apply`, Azure CLI, or provider-backed ingestion unless the relevant
external authorization and environment readiness have been reviewed.
