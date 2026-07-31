# Operational Read-only Copilot Product Contract

## Decision Record

| Field | Value |
| --- | --- |
| Decision | Product and architecture contract for the Operational Read-only Copilot |
| Status | `Accepted` |
| Decision date | `2026-07-30` |
| Contract version | `operational_read_only_copilot_v1` |
| Audience | One authorized operator in a trusted local environment |
| Operating mode | `retrospective_historical_batch_not_real_time` |
| Implementation status | Typed read-only Python query layer, local-only HTTP adapter, and versioned offline evaluation dataset/harness implemented and locally validated; no Copilot or candidate has been evaluated, and no MCP, RAG, database, production authentication, observability, or deployment exists |

## Objective

Define the bounded product behavior and architecture boundary for a future
operator-facing Copilot over verified v2 operational evidence. The future
system may answer deterministic questions about:

- the active v2 deployment and its model era;
- historical data quality and source freshness;
- calibrated 30- and 90-day drift and performance;
- active alerts and immutable alert transitions;
- checksum-pinned model, dataset, transformation, and calibration metadata;
- identified monitoring reports and reporting attempts.

This record was the first item in the roadmap delivery sequence. The separately
reviewed second and third increments implement the typed query layer and its
local-only read-only API while preserving this contract. They do not authorize
any later delivery item.

## Preserved Behavior

This contract does not change any current runtime or data contract.

- Phase 8 and Phase 9 stores remain authoritative, immutable or append-only as
  already defined, with only their existing atomic pointers allowed to advance
  through their existing workflows.
- The checksum-pinned deployment pointer, immutable deployment state and
  authorizing receipt remain the runtime source of truth. The active MLflow
  aliases remain its required governance mirror.
- `candidate` is never a runtime selection. `champion` and `stable` keep their
  accepted controlled-retraining meanings.
- The v2 result remains a delayed historical hindcast. It is never represented
  as real-time, same-day, D+1, multi-day, or production forecasting.
- The target scale remains `sum_of_15_minute_MW_observations`. It is not MW,
  MWh, daily energy, or another physical unit.
- The legacy v1 API, models, scalers, artifacts, notebooks, and serving path
  remain outside this Copilot contract.
- Existing `/api/v1/performance`, `/api/v1/monitoring/latest`,
  `/api/v1/monitoring/history`, `/api/v1/monitoring/runs/{run_id}`,
  `/model-info`, and `/predict` behavior remains unchanged.
- Dashboard, CORS, batch, monitoring, controlled retraining, scheduler
  ownership, Airflow, and Windows Task Scheduler behavior remains unchanged.

## Product Scope

### Intended User And Environment

Version 1 is for one authorized operator in a trusted local environment.
Authorization is inherited from the local process and operating-system context;
this is an environment assumption, not a production authentication control.

Anonymous or remote exposure is not authorized. Any non-loopback exposure must
first define and implement explicit identity, authentication, authorization,
audit, secret management, and network controls in a separate decision and
reviewed increment. User-supplied request fields must never be treated as proof
of identity.

### Closed Question Allowlist

Natural language may later select one of the following question kinds. It does
not expand their inputs, evidence, or permitted output.

| Conceptual `query_kind` | Permitted question | Required selector | Permitted output | Authoritative evidence |
| --- | --- | --- | --- | --- |
| `operational_summary` | What is the latest verified local v2 operational state? | `latest` | Active deployment/model-era identity, latest historical monitoring state, freshness, active alerts, explicit limitations | Verified deployment runtime binding plus verified Phase 9 report state/report and alert state |
| `active_deployment` | Which v2 deployment and model era are active? | `latest` | Deployment/model-era IDs, generation, model version, expected active aliases, cutoffs, and checksum pins | Verified deployment pointer, state, receipt, bundle, calibration, and live Registry alias verification |
| `data_quality` | What verified quality or freshness evidence exists in a monitoring report? | Exact report ID, reporting-run ID, or `latest` | Source watermark/freshness, completeness state, validated quality status, and recorded issues | Verified Phase 9 report, its source-batch lineage, and report-scoped model era |
| `monitoring_performance` | What performance was measured in an existing 30- or 90-day window? | Exact report ID or `latest`, plus window `30` or `90` | Sample count, coverage, MAE, RMSE, signed bias, protected MAPE, sample-gated R2, thresholds, and severity | Verified report, calibration, and report-scoped model era |
| `monitoring_drift` | What drift was measured in an existing 30- or 90-day window? | Exact report ID or `latest`, plus window `30` or `90` | Feature/comparator/detector, observed value, calibrated thresholds, and severity | Verified report and calibration/reference |
| `monitoring_alerts` | Which alerts are active, or which immutable transitions occurred? | `latest`, exact alert-event ID, or bounded history selector | Rule, event type, severity, through date, predecessor, and active/resolved status derivable from the chain | Verified alert state, alert history, and referenced report evidence |
| `active_model_metadata` | What checksum-pinned metadata describes the active v2 model? | `latest` | Registered model/version, model and bundle checksums, dataset version/checksum, ordered feature-schema checksum, transformation version, cutoffs, and calibration/reference IDs | Verified deployment runtime binding, exact v2 bundle, calibration, and model-era evidence |
| `reporting_run` | What happened in one identified reporting attempt or verified report? | Exact reporting-run ID or report ID | Sanitized run status, source-run identity/status, through date, report identity, alert count, and verified report detail when present | Verified reporting request/result/failure and report loaders |

The terminal-result mapping is also closed:

| `query_kind` | Successful/absence states | Evidence-failure states |
| --- | --- | --- |
| `operational_summary` | `answered`, `empty` | `unauthorized`, `unavailable`, `corrupt`, `conflict`, `timeout`, `refused` |
| `active_deployment` | `answered`, `empty` | `unauthorized`, `unavailable`, `corrupt`, `conflict`, `timeout`, `refused` |
| `data_quality` | `answered`, `empty`, `not_found` | `unauthorized`, `unavailable`, `corrupt`, `conflict`, `timeout`, `refused` |
| `monitoring_performance` | `answered`, `empty`, `not_found` | `unauthorized`, `unavailable`, `corrupt`, `conflict`, `timeout`, `refused` |
| `monitoring_drift` | `answered`, `empty`, `not_found` | `unauthorized`, `unavailable`, `corrupt`, `conflict`, `timeout`, `refused` |
| `monitoring_alerts` | `answered`, `empty`, `not_found` | `unauthorized`, `unavailable`, `corrupt`, `conflict`, `timeout`, `refused` |
| `active_model_metadata` | `answered`, `empty` | `unauthorized`, `unavailable`, `corrupt`, `conflict`, `timeout`, `refused` |
| `reporting_run` | `answered`, `not_found` | `unauthorized`, `unavailable`, `corrupt`, `conflict`, `timeout`, `refused` |

Selectors are explicit and bounded:

- `latest` means evidence verified at query time; it is not a reconstructed
  historical deployment state.
- Identifiers are exact opaque IDs. Empty, path-like, traversal, or malformed
  identifiers are invalid.
- Date intervals, where accepted for alert history, are inclusive ISO calendar
  dates and must have start less than or equal to end.
- The existing monitoring projection retains reporting-run pagination with
  default `20` and maximum `100`. The operational query layer exposes no
  reporting-run listing: `reporting_run` accepts only an exact run or report
  identifier.
- Alert pagination retains the current default `50` and maximum `200`.
- Every query has a finite caller-supplied or service-configured deadline.

Questions outside this table are refused. In particular, the product cannot:

- infer an unrecorded root cause or claim causality from correlation;
- recommend or execute remediation, retraining, promotion, stabilization,
  rollback, alias changes, scheduling, notification, or provider refresh;
- predict future production or answer D+1 or multi-day forecast questions;
- compare arbitrary models, search model families, or inspect staged
  candidates;
- query v1 serving artifacts, notebooks, scheduler ownership, scheduler
  leases, Airflow state, Windows Task Scheduler state, or monthly retraining
  recommendations;
- answer from model memory, general world knowledge, raw filesystem parsing,
  or an unapproved documentary corpus.

## Architecture Decision

### Decision

Use one typed, deterministic, read-only operational query layer as the sole
business-logic boundary for all future interfaces.

The query layer must call existing verified loaders. A later API, Copilot, or
MCP adapter may translate requests to that layer and render its result, but may
not read operational stores directly, weaken validation, implement a separate
authorization path, or add business logic.

Immutable local files remain authoritative. A future relational store, if
approved because measured query requirements justify it, is a rebuildable
derived projection and never replaces the immutable evidence.

### Authorized Loader Boundary

The query layer composes only verified read interfaces, including:

- `load_verified_deployment_pointer` and `verify_active_model_era` for the
  active v2 deployment, exact artifacts, calibration, and live aliases;
- `load_prediction_evidence`, `load_model_era`, and `list_model_eras` for
  verified Phase 9 lineage;
- `load_monitoring_report_state`, `load_monitoring_report`,
  `load_monitoring_calibration`, `resolve_report_model_era`,
  `load_active_alerts`, and `load_alert_history` for reporting evidence;
- `load_reporting_attempt` and `load_reporting_attempts` for verified,
  sanitized reporting-attempt evidence;
- the existing sanitized monitoring projection only where its output contract
  fully satisfies the requested product fact.

The implementation must not introduce an alternative JSON parser or partial
verification path for these stores. Relative-path relocation behavior already
accepted by verified loaders may be preserved; raw paths must not cross the
product boundary.

Deployment answers require the complete current verification chain. If MLflow
is required to check the active aliases and is unavailable, the result is
`unavailable`. The system must not answer from the pointer alone. A pointer,
state, receipt, bundle, calibration, or alias disagreement is `conflict` or
`corrupt` as defined below and always fails closed.

### Read-only Boundary

One query may:

- read configured local evidence;
- perform checksum, schema, identity, lineage, and Registry-alias verification;
- calculate deterministic projections from already verified values;
- return sanitized facts, citations, limitations, or failures.

One query must not:

- create or modify a file, directory, pointer, lock, receipt, report, Registry
  version, alias, model, scaler, dataset, task, lease, or scheduler state;
- call ingestion, training, lifecycle, deployment, orchestration, notification,
  or write APIs;
- update access times or caches where the application controls that behavior;
- perform provider or general internet calls;
- fall back from failed verification to raw or stale evidence.

Future sanitized request telemetry may write only to a separately approved
observability store. It must never write to Phase 8/9, deployment, Registry,
artifact, scheduler, or serving stores.

## Conceptual Product Schemas

These schemas fix product semantics and invariants. They do not select Python
classes, HTTP paths, JSON wire encoding, an LLM vendor, or an adapter protocol.
The typed-query-layer increment must publish exact executable schemas that
preserve these fields and rules.

### `OperationalQuery`

| Field | Contract |
| --- | --- |
| `contract_version` | Exactly `operational_read_only_copilot_v1` |
| `query_kind` | One value from the closed allowlist |
| `selector` | Typed `latest`, exact identifier, or bounded inclusive date interval accepted by that query kind |
| `window_days` | Nullable; when present, exactly `30` or `90` |
| `pagination` | Nullable typed limit/offset with the accepted bounds |
| `requested_at_utc` | Required timezone-aware UTC instant |
| `correlation_id` | Required opaque boundary-generated identifier; not evidence and not user identity |
| `deadline` | Required finite deadline propagated to every dependency |

Unknown query kinds return an `OperationalAnswer` with status `refused` and
`query_kind=null` before an `OperationalQuery` is formed. Malformed
identifiers, invalid intervals, invalid pagination, and unsupported selector
combinations for a known query kind return `refused` with that validated
`query_kind`. Neither path performs an operational read.

### `OperationalAnswer`

| Field | Contract |
| --- | --- |
| `contract_version` | Exactly the accepted contract version |
| `query_kind` | The validated requested question kind; null only when an unknown question kind is refused before `OperationalQuery` formation |
| `status` | One terminal status from the status contract |
| `mode` | Always `retrospective_historical_batch_not_real_time` |
| `summary` | Sanitized deterministic rendering derived only from returned facts; every factual clause carries its supporting inline `evidence_id` references; nullable for non-answered states |
| `facts` | Ordered `GroundedFact` values; never contains an uncited factual claim |
| `evidence` | Deduplicated `EvidenceCitation` values referenced by the facts |
| `limitations` | Explicit missing domains, staleness, partial-result boundaries, and non-causal interpretation |
| `failure` | Nullable `OperationalFailure`; required for failure states |
| `served_at_utc` | Timezone-aware UTC verification/response instant |
| `correlation_id` | Exact correlation ID from the validated boundary request |

For a multi-source question, one failed required source determines the terminal
failure status. Independently verified facts may be retained only when their
independence is explicit in `limitations`; no overall-health conclusion may be
synthesized from a partial result.

### `GroundedFact`

| Field | Contract |
| --- | --- |
| `fact_id` | Stable identifier within one answer |
| `name` | Allowlisted semantic fact name |
| `value` | Typed scalar or bounded structured value copied or deterministically derived from verified evidence |
| `unit_or_scale` | Explicit unit, legacy target scale, or `not_applicable` |
| `as_of` | Relevant report date, cutoff, generation, or UTC observation instant |
| `evidence_ids` | Non-empty list of citations supporting the complete fact |

The summary may paraphrase these facts but cannot add new claims, explanations,
causes, recommendations, or confidence not represented in them. Every factual
clause in the summary appends the same inline `evidence_id` references that
support the underlying facts.

### `EvidenceCitation`

| Field | Contract |
| --- | --- |
| `evidence_id` | Answer-local stable reference used by facts |
| `domain` | Deployment, monitoring report, alert, prediction/model era, model bundle, calibration/reference, or verified Registry binding |
| `source_kind` | Verified loader/result type, not a filesystem location |
| `schema_version` | Original persisted schema or verified projection contract |
| `record_id` | Opaque content, report, run, deployment, model-era, or calibration identity |
| `sha256` | SHA-256 of the cited immutable record or its verified manifest/binding |
| `effective_at` | Relevant date, cutoff, generation, or period |
| `observed_at_utc` | UTC instant at which mutable external state, including aliases, was verified |

One citation may expose only identities and digests already safe for the
operator. It never exposes absolute or host-specific paths, raw payloads,
credentials, secrets, connection strings, stack traces, or internal exception
messages.

### `OperationalFailure`

| Field | Contract |
| --- | --- |
| `code` | Stable sanitized domain code |
| `message` | Operator-safe message without internal details |
| `retryable` | Whether an unchanged request may reasonably succeed later |
| `evidence_state` | `empty`, `not_found`, `unavailable`, `corrupt`, `conflict`, `timeout`, `unauthorized`, or `unsupported` |

Adapters may later map these failures to protocol-specific status codes only in
the separately reviewed API or MCP contract.

## Result And Failure Semantics

| `OperationalAnswer.status` | Meaning | Facts allowed |
| --- | --- | --- |
| `answered` | All required evidence was verified and the permitted question was answered | Yes, with complete citations |
| `empty` | The configured store is valid but contains no matching accepted evidence | Normally none; existence/state facts only when cited |
| `not_found` | A valid exact identifier or bounded selector has no matching record | No substantive operational facts |
| `refused` | The request is malformed, uses an invalid selector, or asks an unsupported, prescriptive, mutating, unbounded, or out-of-allowlist question | No |
| `unauthorized` | The caller is not authorized by the enclosing trusted boundary | No |
| `unavailable` | Required local evidence or a required dependency such as MLflow cannot be read or verified | Only explicitly independent verified facts, with limitations |
| `corrupt` | Schema, checksum, identity, lineage, or alert-chain verification failed within one source | Only facts independent of the corrupt source, with limitations |
| `conflict` | Individually readable authoritative sources disagree, including deployment and Registry aliases | Only facts independent of the conflict; no active-state conclusion |
| `timeout` | The finite query deadline expired | Only facts completed and verified before expiry, with limitations |

`empty` and `not_found` are valid evidence states, not service failures.
`unavailable`, `corrupt`, and `conflict` must never be silently converted to
`empty`, a cached answer, or an answer from a weaker source. Retries must
re-verify evidence rather than reuse an unverified model-generated statement.

## Grounding, Citation, And Sanitization Rules

- Every factual value and every factual clause in the summary has at least one
  inline `evidence_id`.
- A citation identifies immutable evidence by record/schema identity and
  digest, and gives the relevant report date, cutoff, generation, or observed
  time.
- Current alias claims include the observation time and the verified
  deployment/model-era binding; they are not described as immutable forever.
- `latest` answers always disclose `served_at_utc`, evidence dates, and
  freshness. They never imply that local historical data is real time.
- Threshold interpretations use the accepted calibration and existing
  direction semantics. The Copilot does not invent severity or recalculate a
  new policy.
- A reporting failure returns the existing sanitized operator message. Raw
  failure text and host paths stay private.
- Lack of evidence produces a limitation, empty state, failure, or refusal. It
  is never filled from model memory or general knowledge.
- The system may explain a recorded rule, threshold, transition, or lineage.
  It may not claim an unrecorded causal root cause.

## Compatibility And Versioning

This contract is additive. It creates no current public API or persisted schema.

- Existing HTTP contracts and error mappings remain unchanged.
- Existing Phase 8, Phase 9, deployment, model-era, alert, calibration,
  controlled-retraining, scheduler, and v1 schemas remain unchanged.
- The additive local HTTP contract is `POST /api/v1/operational-query`; all
  other HTTP contracts remain unchanged.
- An MCP adapter must later reuse the same query-layer schemas, authorization,
  failures, and citations without a second business-logic path.
- Optional response facts may be added compatibly only when they use an
  already-permitted question kind and authoritative evidence.
- Removing a field, weakening evidence, changing question semantics, changing
  the operating mode, adding a mutating capability, or expanding to a new
  authority domain requires a new major contract version and reviewed ADR.
- New question kinds require explicit product review even if they can be added
  to an executable schema without breaking serialization.

## Explicit Non-goals

The implemented query/API increments remain intentionally limited. They do not
authorize or implement:

- TypeScript code, new dependencies, frontend changes, additional endpoints,
  tools, prompts, or an LLM;
- Copilot, MCP, RAG, embeddings, a document corpus, PostgreSQL, `pgvector`, a
  relational projection, observability, staging, cloud, or CI/CD;
- authentication beyond the accepted trusted-local expectation;
- data/model/scaler changes, artifact generation, notebook execution,
  ingestion, provider calls, training, retraining, lifecycle transitions,
  deployment changes, alias changes, rollback, or notifications;
- scheduler ownership, lease, Airflow, Windows Task Scheduler, or monthly
  governance queries;
- v1 artifact inspection, interactive `/predict`, live forecasting, D+1, or
  multi-day forecasting;
- a claim that the project is a production system or that the Copilot exists.

## Typed Query Layer Implementation

`wind_forecast.operational_query_models` publishes the strict, frozen,
extra-forbid executable schemas. `wind_forecast.operational_query` publishes
`OperationalQueryService` as the only business-logic entry point for the eight
accepted question kinds.

The service is a Python library only. It requires an injected trusted-local
authorization policy and denies by default. Deployment questions additionally
require an injected Registry client with a declared finite timeout no greater
than the remaining cooperative deadline.
`TimeoutAwareMlflowRegistryClient` adapts the existing MLflow REST client to
GET-only alias reads with per-call request/retry timeouts and zero retries;
non-REST Registry backends are refused as unavailable because their deadline
cannot be bounded by this increment.

## Local Read-only HTTP API

`POST /api/v1/operational-query` accepts only the five public query fields:
`contract_version`, `query_kind`, `selector`, nullable `window_days`, and
nullable `pagination`. The adapter rejects unknown fields, invalid JSON,
incompatible bodies, and bodies above 64 KiB before any operational read.
`requested_at_utc`, a UUID correlation ID, and a deadline of at most five
seconds are generated by the server and cannot be supplied by the client.

The response body is always the existing `OperationalAnswer` contract. HTTP
status mapping is:

| Answer status | HTTP status |
| --- | --- |
| `answered`, `empty` | `200` |
| `refused` | `400` |
| `unauthorized` | `403` |
| `not_found` | `404` |
| `unavailable`, `corrupt`, `conflict` | `503` |
| `timeout` | `504` |

The adapter derives trust only from `Request.client`: exactly `127.0.0.1` and
`::1` are trusted. Hostnames, missing client data, proxies, and non-loopback
addresses are untrusted; `Forwarded`, `X-Forwarded-For`, and `X-Real-IP` do not
participate in authorization. The injected query-layer policy remains
deny-by-default and permits the eight query kinds only for the process-local
operator principal. CORS is unchanged and is not authentication.

Configuration is read lazily. `WIND_FORECAST_DEPLOYMENT_ROOT` defaults to
`data/processed/v2/deployment`,
`WIND_FORECAST_MONITORING_STORE_ROOT` retains its existing default, and
`WIND_FORECAST_OPERATIONAL_QUERY_TIMEOUT_SECONDS` defaults to `5` and must be
finite, positive, and no greater than five. `MLFLOW_TRACKING_URI` is eligible
only when it is an HTTP(S) URI with the exact numeric host `127.0.0.1` or
`::1`; the local API adapter also disables redirects and environment proxies.
Other Registry forms are disabled and deployment questions fail closed as
`unavailable`. Configuration and application creation do not require stores to
exist and do not create them. The five-second query deadline is cooperative:
each dependency receives a timeout no greater than the service budget, but
this increment does not add a pre-emptive wall-clock cancellation mechanism
around the existing verified loaders.

`wind_forecast.operational_api` calls only
`OperationalQueryService.answer()`. It creates no CLI, worker, cache,
telemetry, persisted request/response record, direct store parser, or second
authorization path.

Reporting-attempt verification now lives in the reporting domain and the
existing monitoring projection delegates to that public loader without
changing its API response contract. Deployment verification exposes additive
read-only error subclasses for uninitialized, unavailable, and conflicting
evidence while preserving the existing base exceptions.

`tests/test_operational_query.py` uses synthetic evidence and dependency
doubles to cover the closed allowlist, strict selectors, authorization,
deadlines, all terminal states, citations, sanitization, alert-state and
model-era consistency, zero-write snapshots, and import-time side effects.
`tests/test_operational_api.py` covers all eight HTTP request shapes, the full
status mapping, body limits, server metadata, socket-derived authorization,
proxy-header spoofing, configuration/Registry gating, OpenAPI, sanitization,
and startup zero-write behavior.
Existing deployment, monitoring projection, and API tests remain the
compatibility boundary.

## Acceptance Tests And Evaluation

The implemented typed query layer uses synthetic stores and controlled
dependency doubles. Its acceptance suite covers:

- a valid complete deployment/report/alert chain and correctly cited facts;
- a valid empty store distinct from unavailable or corrupt state;
- unknown, malformed, and path-like identifiers;
- inverted/unbounded date intervals and pagination limits;
- checksum, schema, identity, lineage, report-state, and alert-chain corruption;
- pointer/state/receipt/bundle/calibration mismatch;
- deployment/Registry alias conflict and MLflow unavailability;
- deterministic timeout and cancellation propagation;
- refusal of unsupported, predictive, causal, prescriptive, and mutating
  questions before operational reads;
- every returned fact referencing present evidence and every summary claim
  being derivable from those facts;
- sanitization of paths, raw errors, secrets, and connection details;
- partial multi-source failure without an overall-health inference;
- filesystem, Registry, alias, scheduler, and artifact snapshots proving zero
  operational writes;
- regression tests for all existing API endpoints and monitoring projection
  behavior.

### Versioned offline evaluation harness

The fourth delivery increment adds the immutable English dataset
`operational_read_only_copilot_eval_en_v1` under
`evaluation/operational_read_only_copilot/v1/`. Its manifest pins the accepted
contract, source commit, case count, case-file SHA-256, distribution, expected
facts, evidence source kinds, failure semantics, and gate policy.

The 88 cases comprise 20 canonical supported request shapes, 20 benign English
paraphrases, 16 absence/evidence-failure scenarios, 24 refusals, and 8
adversarial authorization, tool-substitution, privacy, traversal, citation,
and stale-fallback cases. All identifiers and evidence values are synthetic.
The symbolic expected tool is always `operational_query`, with exactly the
five public fields accepted by `OperationalHttpRequest`; it does not represent
a new tool or a Copilot implementation.

`wind_forecast.operational_evaluation` loads and checksum-verifies the dataset
and scores externally produced `CandidateTrace` JSONL. It never invokes an
LLM, the API, the query service, verified loaders, Registry, network, or an
operational store. The runner is stdout-only:

```powershell
.\venv\Scripts\python.exe .\scripts\evaluate_operational_copilot.py `
  --dataset .\evaluation\operational_read_only_copilot\v1\manifest.json `
  --responses <candidate-results.jsonl>
```

Exit `0` means every critical gate passed, exit `1` means a schema-valid
candidate failed a gate, and exit `2` means the dataset or response set was
invalid. Reports contain only dataset/response digests, metrics, case IDs, and
sanitized failure codes. Candidate payloads and validation details are never
echoed.

Authorization, refusal, canonical tool selection, factual correctness,
grounding, citations, privacy, evidence-state distinctions, and zero-write
checks require 100%. Benign paraphrase recognition requires 95%; the only
permitted miss is one safe refusal with no tool, facts, evidence, or summary.
A wrong tool, wrong arguments, or an inaccurate/ungrounded answer always fails
a critical gate.

The acceptance tests bind all 20 canonical goldens back to
`OperationalQueryService.answer()` using temporary synthetic evidence, a fixed
clock, and a controlled Registry boundary. Zero-write is proved separately by
instrumented dispatch checks and byte/size/mtime snapshots; it is not inferred
from candidate JSONL.

The current state is exactly `harness accepted; no Copilot evaluated`. No
Copilot may pass its introduction gate until it supplies one complete,
schema-valid response set and passes these evaluation gates.

## Acceptance Criteria

- The audience, question allowlist, selectors, evidence sources,
  authentication expectation, read-only boundary, failures, and non-goals are
  explicit.
- Every permitted question maps to verified evidence and bounded output.
- Every factual answer requires checksum-pinned citations and temporal context.
- Partial, empty, unavailable, corrupt, conflict, timeout, and refusal behavior
  is deterministic and distinct.
- Historical hindcast, target scale, model-era, and alias semantics are
  preserved.
- No current API, store, model, scheduler, or artifact contract changes.
- The roadmap shows the query layer, local-only API, and offline evaluation
  harness as implemented and keeps relational projection, observability,
  Copilot, MCP, RAG, and cloud as separate future increments.
- Repository status does not describe any Copilot implementation as current.

## Risks And Controls

| Risk | Control |
| --- | --- |
| A generated answer invents or combines unsupported facts | Closed question allowlist, typed facts, evidence required per claim, and deterministic refusal |
| Historical hindcast is presented as a live forecast | Mandatory operating-mode field, evidence dates, freshness, and explicit forecast non-goals |
| A stale pointer is trusted while Registry aliases differ or MLflow is unavailable | Complete verified runtime binding and fail-closed `conflict`/`unavailable` results |
| Paths, secrets, or raw failures leave the local boundary | Allowlisted projections, sanitized failures, and citation identities instead of locations |
| Natural language expands authority | Natural language selects only a typed query kind; the query layer owns validation and authorization |
| A future adapter creates another business-logic or authorization path | One query layer is mandatory for API, Copilot, and MCP |
| A relational projection becomes authoritative | Immutable checksum-pinned files remain the source of truth; projections are rebuildable |
| A read-only query changes operational state | No write-capable dependencies and explicit zero-write acceptance tests |
| Query/API acceptance is mistaken for a Copilot or production service | Status and roadmap distinguish the implemented local HTTP adapter from every later product and deployment increment |
| Harness acceptance is mistaken for candidate acceptance | Reports state `harness accepted; no Copilot evaluated` until an external candidate supplies a complete response set and passes every gate |

## Rollback

Before merge, this decision can be abandoned by closing the draft change
without touching operational state. After acceptance, a changed decision must
be recorded by a later reviewed ADR that marks this record `Superseded` and
updates navigation and roadmap status. Historical evidence and this decision
must not be deleted or silently rewritten.

Documentation rollback never means changing Phase 8/9 evidence, deployment,
Registry aliases, models, artifacts, schedulers, or the operational checkout.

## Delivery Gates

The remaining delivery sequence is unchanged and each item requires a separate
reviewed increment:

1. Product contract and ADR: accepted by this record.
2. Typed operational query layer: implemented and locally validated.
3. Separately reviewed read-only API endpoint: implemented and locally
   validated.
4. Versioned evaluation dataset and harness: implemented and locally
   validated; no Copilot candidate evaluated.
5. Relational projection only if requirements justify it.
6. Local observability with sanitization.
7. Copilot restricted to accepted deterministic tools.
8. MCP adapter over the same contracts.
9. Document-only RAG only for questions deterministic tools cannot answer.
10. Separate staging and cloud design.

No later item may be started implicitly while delivering an earlier one.

## Stop Gate

This increment stops at the versioned offline evaluation dataset and harness.
It does not authorize a relational projection, observability, Copilot, MCP,
RAG, staging, cloud work, or any other later delivery item.

Future work must stop and return for review if it needs:

- a source, question kind, authority, mutation, or user population outside this
  contract;
- changes to existing API, persisted schemas, target semantics, model lifecycle,
  scheduler ownership, or operational stores;
- remote exposure without an approved identity and authorization design;
- weaker validation, parsing around a verified loader, uncited facts, or a
  fallback from corrupt/conflicting evidence;
- a second business-logic or authorization path.
