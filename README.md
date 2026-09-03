# National Wind Energy Production Forecast

Local, reproducible engineering demonstration for Portuguese wind-energy
production forecasting. The repository contains the implementation, tests,
machine-readable contracts, and deterministic evidence used by the supported
workflows. It is not a production or real-time forecasting service.

## Quick start

The tracked dashboard bundle is synthetic and requires no credentials or
network access:

```powershell
docker compose up --build
Invoke-RestMethod http://localhost:8000/health
Invoke-WebRequest -UseBasicParsing http://localhost:5173
docker compose down
```

For a local Python setup:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pip install -e .
```

Run the supported checks with:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m ruff check .
```

Use each executable's `--help` as the command contract. The stable batch
interface is available through `wind-forecast-batch`; the compatibility
wrappers remain under `scripts/`.

## Architecture

```text
REN + ERA5-Land sources
          |
  validation + feature engineering
          |
  versioned v2 historical hindcast
          |
  immutable batch monitoring evidence
          |
React dashboard <-- FastAPI <-- verified local artifacts
```

The FastAPI service also exposes the legacy saved-model interface. The
dashboard is intentionally read-only and currently consumes historical
performance and monitoring evidence; it does not call `/predict` or create
future forecasts.

## Repository sources of truth

- Runtime behaviour: `src/wind_forecast/` and the tested CLI wrappers.
- API contracts: Pydantic models and FastAPI's generated OpenAPI schema.
- Policies: `config/monitoring_policy_v1.json` and
  `config/retraining_policy_v1.json`.
- Data provenance: versioned manifests under `data/manifests/` and generated
  v2 source manifests.
- Model and scaler lineage: MLflow manifests and the versioned scaler manifest
  under `models/v2/scalers/`.
- Infrastructure: `.github/workflows/`, `infra/azure/`, and Terraform state
  external to this repository.
- Synthetic demo claims: `demo/v1/manifest.json` and its checksums.

Do not edit generated data, models, receipts, or manifests in place. Create a
new version or an explicit immutable evidence record.

## Supported interfaces

| Interface | Purpose | Boundary |
| --- | --- | --- |
| `GET /health` | Process health | Local/container use |
| `GET /api/v1/performance` | Historical evaluation | Selected local artifacts |
| `GET /api/v1/monitoring/*` | Verified monitoring projection | Delayed historical evidence |
| `POST /predict` | Saved-model inference | Legacy v1 artifacts |
| `wind-forecast-build-v1` | Rebuild the accepted v1 raw-to-features table | Authorized raw snapshot and immutable contracts |
| `POST /api/v1/operational-query` | Typed operational questions | Numeric loopback only |
| `POST /api/v1/copilot` | Read-only Portuguese Copilot | Local loopback, verified evidence |
| `scripts/run_batch_pipeline.py` | Plan/run historical batch | Explicit artifacts and leases |
| `wind-forecast-validate-manifest` | Validate manifest paths, hashes, or release provenance | Metadata, local integrity, or release provenance |

The operational copilot and PostgreSQL projection are implemented as
default-disabled, read-only extensions. Candidate adapters are not activated.

## Data and model boundaries

The v1 source contract and saved models are preserved for compatibility, but
the four v1 raw CSVs are not tracked in the public repository. An authorized
local copy must be placed at the paths declared by the contract before any v1
reader is used. The v2 REN/ERA5-Land path uses separate versioned directories
and does not silently replace v1 data, scalers, or serving behaviour.
ERA5-Land is reanalysis data, so the accepted v2 result is a retrospective
hindcast, not a day-ahead forecast.

The four v1 raw CSV hashes are owned exclusively by
`data/manifests/v1_source_contract.json`. The immutable review record at
`data/manifests/v1_provenance_review_2026-09-01.json` records the evidence
decision and unresolved fields without inventing a provider or licence.
The unsent REN permission request is templated at
`data/manifests/ren_redistribution_authorization_request_template.md`.
Supported Python readers validate the complete v1 snapshot before reading it.
`metadata` validation checks declarations without local files; `integrity`
checks paths and bytes; `release` additionally requires complete source
provenance and is intentionally blocked while the v1 contract remains
`provenance_incomplete`.

The real-data release catalog marks v1 as `classification: internal` and
remains blocked until every component has explicit, snapshot-scoped
redistribution authorization. No historical v1 data or real-data bundle is a
public artefact. The tracked `demo/v1` bundle is synthetic-original evidence
and makes no historical-production, live-monitoring, production-model, or
cloud-deployment claim.

## Local Copilot (portfolio priority)

The current product priority is a small, local, single-user, read-only
Copilot over verified `demo/v1` evidence. It exposes a deterministic
Portuguese `/copilot` experience over the existing typed operational query
service, with short answers, evidence, freshness, limitations, and explicit
refusals. The existing Forecast Replay page remains responsible for historical
prediction/observation inspection.

The app requires no external key. Its documentary route searches only the
hash-pinned `README.md`, `OPERATIONS.md`, and `APP_COPILOT_ROADMAP.md` corpus,
using deterministic Markdown sections and lexical retrieval. Optional OpenAI
synthesis is enabled only when the backend, model, and backend-only key are
configured; any provider failure falls back visibly to local extraction.
MCP, agents, vector stores, embeddings, and persistent conversations are not
part of this portfolio MVP. New ANN v2 experimentation and promotion, MLflow
changes, Azure work, and remote exposure are paused. See
[APP_COPILOT_ROADMAP.md](APP_COPILOT_ROADMAP.md) for the closed question
catalogue, delivery gates, and deliberate limits.

### Copilot demonstration (60–90 seconds)

1. Open `/copilot`.
2. Ask which deployment is active and show the evidence and freshness details.
3. Ask for the methodology and show the `rag_local` mode and documentary source.
4. Ask a historical question and open Forecast Replay from the response.
5. Ask for a prohibited operation and show the explicit refusal.

## Local operations

Human-only setup, scheduler recovery, readiness gates, monitoring, rollback,
and destructive-action guidance are in [OPERATIONS.md](OPERATIONS.md). Azure
bootstrap, OIDC, promotion, and rollback are in
[infra/azure/README.md](infra/azure/README.md). Those files describe actions
that cannot be inferred safely from Python alone; they do not redefine runtime
contracts.

## Current non-goals

- Production cloud operation or external alert delivery.
- Real-time or D+1 forecasting.
- Automatic retraining or model promotion.
- PySpark processing.
- Registry-based API serving.
- Remote or authenticated exposure of the local operational-query route.

The tuned v1 models are supported for inference only; v1 retraining is
deprecated. Interactive artefacts are not part of the supported workflow.
Future training must use tested code paths with versioned contracts.
