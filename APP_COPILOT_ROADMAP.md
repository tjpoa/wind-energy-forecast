# Local Copilot: portfolio roadmap

This document defines a deliberately small, local-first portfolio demo. The
executable Python contracts, manifests, and tests remain the sources of truth.

## Product goal

Demonstrate a useful Portuguese Copilot over verified local evidence while
showing React, FastAPI, typed contracts, testing, observability, and a small
document-retrieval integration. The app is for a single local user and is not
a production, real-time, or cloud service.

Every answer shown by the app must provide a short response, its evidence,
the evidence date/freshness, and visible limitations. Operational facts and
documentary explanations are separate routes.

## Deliberate boundaries

- The Copilot is read-only. It never trains, promotes, deploys, writes data,
  generates SQL, or reads a database directly.
- The deterministic path works without credentials or network access.
- The existing Forecast Replay page remains the UI for historical
  prediction/observation inspection; the Copilot does not duplicate it.
- The corpus is limited to approved project documentation. Raw datasets,
  logs, secrets, model binaries, and the whole repository are not indexed.
- No MCP, agents, streaming, persistent chat, multi-user auth, vector store,
  embeddings, cloud exposure, or new ML pipeline is included in this MVP.

## Closed question catalogue

The first app release accepts only these operational families and delegates
calculation and evidence selection to the existing `OperationalCopilot` and
`OperationalQueryService`:

| Portuguese question family | Existing query kind | MVP state |
| --- | --- | --- |
| Qual é o estado operacional verificado? | `operational_summary` | Supported |
| Que deployment está ativo? | `active_deployment` | Supported |
| Como estão a qualidade e a freshness dos dados? | `data_quality` | Supported |
| Qual foi a performance/MAE dos últimos 30 ou 90 dias? | `monitoring_performance` | Supported |
| Há drift nos últimos 30 ou 90 dias? | `monitoring_drift` | Supported |
| Há alertas ativos? | `monitoring_alerts` | Supported |
| Que modelo está ativo e que metadados o identificam? | `active_model_metadata` | Supported |

The deterministic selector uses only bounded normalization and an explicit
Portuguese synonym table. It does not parse arbitrary dates, report IDs,
reporting-run IDs, or free-form metric expressions. Unsupported questions are
refused instead of being approximated.

Questions about methodology, limitations, decisions, local operation, and the
meaning of the current roadmap are documentary questions. They are answered
only from the approved corpus and are never evidence for an operational fact.

The following remain backlog, not hidden MVP capabilities:

- `prediction_observations` for date/interval comparisons;
- `dataset_coverage` for version, bounds, counts, freshness, and gaps;
- future forecasts, causal explanations, retraining, promotion, rollback,
  writes, remote data, and general-knowledge answers.

## Three small increments

### 0. Contract and roadmap

Create this document and update the repository status. Record ANN v2
experimentation, promotion, MLflow changes, Azure work, and remote exposure
as paused. No runtime or artifact changes are allowed.

Gate: scope, boundaries, catalogue, and review process are explicit.

### 1. Deterministic local app

Add `POST /api/v1/copilot` and a React `/copilot` page. The endpoint accepts a
single bounded `question` and returns a typed union with `route`, `mode`, the
existing operational answer when applicable, visible limitations, and a
structured refusal when unsupported.

The page provides suggested questions, a concise answer card, evidence and
freshness details, a data-unavailable state, an explicit mode badge, in-memory
history, and a clear button. It includes a link to Forecast Replay for
historical prediction/observation questions.

Gate: the complete demo works without an API key; every operational fact is
evidence-backed; unsupported and unavailable states are visible; no write or
second business-logic path is introduced.

### 2. Documentary retrieval and optional AI

Add a versioned manifest for `README.md`, `OPERATIONS.md`, and this roadmap.
Create deterministic section chunks and a small lexical retriever with logical
document URIs, chunk IDs, versions, and SHA-256 values. No embeddings or
vector database are needed.

Without a key, documentary answers are short extractive responses. With an
explicit backend setting and key, a provider-neutral document synthesizer may
use at most three retrieved chunks in one bounded call, validate the returned
chunk IDs, use no retries, and retain no conversation by default. Provider
failure must be visible and must fall back to the local extractive response.

Gate: retrieval is deterministic and hash-verified; no physical paths or
secrets reach the browser; provider tests use a fake transport; the app still
works with the provider disabled.

## Delivery and review

Use one short branch and draft PR per increment:

1. `codex/app-copilot-roadmap`
2. `codex/app-copilot-mvp`
3. `codex/app-copilot-docs-ai`

Each PR is reviewed and paused before the next one starts. The MVP is already
portfolio-complete after increment 1; increment 2 is an optional depth layer,
not a prerequisite for a useful demo.

## Acceptance checklist

- The local stack starts with `docker compose up --build` without credentials.
- Backend tests cover canonical Portuguese questions, refusals, evidence,
  unavailable states, zero writes, and one query per request.
- Frontend tests cover navigation, suggestions, loading, errors, evidence,
  limitations, and clearing in-memory history without `localStorage`.
- Retrieval tests cover manifest hashes, deterministic ranking, missing
  documents, invalid citations, provider timeout, and explicit fallback.
- Python tests/Ruff, frontend tests/lint/build, one keyless Playwright smoke
  path, `git diff --check`, and the complete diff review pass before each PR.
