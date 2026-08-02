# PostgreSQL Operational Projection Contract

## Decision Record

| Field | Value |
| --- | --- |
| Decision | Contract for a derived PostgreSQL projection of verified operational evidence |
| Status | `Accepted` |
| Decision date | `2026-07-31` |
| Contract version | `operational_postgres_projection_v1` |
| Audience | One authorized operator in a trusted local environment |
| Operating mode | `retrospective_historical_batch_not_real_time` |
| Implementation status | Dedicated foundation, migrations, manual projector, verifier, and deterministic benchmark harness implemented; the mandatory full benchmark recorded `NO-GO`, so query integration is blocked and not implemented |

## Objective And Justification Gate

Define a local, normalized, indexed, disposable PostgreSQL projection of
verified operational artifacts. The projection may improve bounded historical
selection and pagination, but it never replaces the immutable files or their
verified loaders as evidence.

The accepted implementation sequence is deliberately split into four later
plans. Schema and projector work may establish and measure the design, but the
operational query layer must not consume PostgreSQL unless the separately
reviewed benchmark plan records `GO`. A failed benchmark records `NO-GO` and
stops query integration; its thresholds must not be weakened to obtain a pass.

This record approves only the contract. It creates no database, dependency,
schema, migration, service, projector, configuration, or runtime behavior.

## Preserved Behavior

The following contracts remain unchanged:

- Phase 8 and Phase 9 files, immutable records, append-only histories, and
  atomic pointers remain authoritative.
- The active deployment pointer, immutable state, authorizing receipt, and
  required live MLflow Registry alias verification remain the runtime binding.
- Existing monitoring, controlled-retraining, deployment, and lifecycle
  workflows retain their current ownership and mutation boundaries.
- Airflow and Windows Task Scheduler remain governed by the existing exclusive
  scheduler-owner and lease contracts. The projection is not scheduled.
- The Airflow metadata PostgreSQL database, roles, volume, migrations, and
  lifecycle remain completely separate.
- The legacy v1 API, models, scalers, artifacts, notebooks, and serving path
  remain unchanged and outside the projection.
- Existing HTTP endpoints, OpenAPI schemas, status mappings, CORS, dashboard,
  frontend, model files, scalers, datasets, and notebooks remain unchanged.
- The v2 result remains a delayed historical hindcast on the legacy
  `sum_of_15_minute_MW_observations` scale, not a live or future forecast.

## Initially Projected Questions

PostgreSQL may initially index only these existing query kinds:

| `query_kind` | Permitted projection use |
| --- | --- |
| `data_quality` | Resolve an exact report or reporting attempt and index its verified quality, freshness, and issue fields. |
| `monitoring_performance` | Resolve an exact report and 30- or 90-day window and index its verified metrics, severities, and accepted thresholds. |
| `monitoring_drift` | Resolve an exact report and 30- or 90-day window and index verified feature, comparator, detector, value, severity, and threshold fields. |
| `monitoring_alerts` | Select verified alert-event identities by exact ID, active state, bounded inclusive date interval, and existing pagination. |
| `reporting_run` | Resolve one verified reporting attempt by exact reporting-run ID or report ID. |

PostgreSQL selects candidate identities, ordering, and pagination only. Before
an answer is formed, every selected record is loaded and verified again from
the authoritative files. The current operational query layer remains the sole
business-logic and authorization boundary.

`operational_summary`, `active_deployment`, and `active_model_metadata` remain
on the current direct loader path. Their active-deployment claims require live
Registry verification and must never be answered from a database snapshot.

No new query kind, selector, fact, response field, endpoint, or authority is
approved by this record.

## Authorized Evidence Boundary

The projector and any later reader may compose only these existing verified
interfaces:

- `load_monitoring_report_state`, `load_monitoring_report`, and
  `load_monitoring_calibration`;
- `resolve_report_model_era`, `load_model_era`, and `list_model_eras`;
- `load_active_alerts` and `load_alert_history`;
- `load_reporting_attempt` and `load_reporting_attempts`;
- `load_prediction_evidence` only when needed to verify already-referenced
  lineage; individual prediction values are not projected.

The direct deployment questions continue to use
`load_verified_deployment_pointer` and `verify_active_model_era` outside the
PostgreSQL projection.

PostgreSQL rows, documentation, logs, captured HTTP responses, model memory,
general knowledge, raw filesystem parsing, and alternative JSON parsers are
not evidence. A projector must not introduce a partial or weaker verification
path around an existing loader.

## Prohibited Actions And Non-goals

The projection must not:

- create, edit, repair, reformat, delete, or advance any operational file,
  pointer, lock, receipt, report, alert, model, scaler, dataset, or artifact;
- call ingestion, provider, training, retraining, lifecycle, deployment,
  notification, Registry-mutation, orchestration, or scheduler APIs;
- run on Python import, API startup, a health check, or a user query;
- integrate with Airflow, Windows Task Scheduler, or another scheduler;
- connect to or reuse Airflow's PostgreSQL database, roles, migrations, or
  volume;
- project v1 artifacts, raw REN/ERA5-Land data, feature vectors, individual
  predictions or actuals, model files, scalers, or bundles;
- add endpoints, query kinds, public response fields, RAG, embeddings,
  `pgvector`, observability, staging, cloud, or remote exposure;
- fall back from a missing, stale, corrupt, or incompatible required projection
  to an unverified or cached answer;
- treat database contents as proof that the original evidence still exists or
  remains valid.

## Provenance And Citation Contract

Every projected record must retain an internal relationship to:

- `domain`;
- verified loader/result `source_kind`;
- original `schema_version`;
- original opaque `record_id`;
- original or loader-derived SHA-256;
- relevant `effective_at` date, cutoff, generation, or period;
- `observed_at_utc` when mutable external state was observed;
- projection generation, projector contract version, and source Git commit.

The public `EvidenceCitation` contract remains unchanged:

- `evidence_id`;
- `domain`;
- `source_kind`;
- `schema_version`;
- `record_id`;
- `sha256`;
- `effective_at`;
- `observed_at_utc`.

PostgreSQL, its tables, rows, generation, or projector are never exposed as
`source_kind` and never replace the original evidence citation. Each returned
fact and factual summary clause retains the citations produced from the
original verified evidence.

For `latest` or active-alert queries, the reader must verify the authoritative
state before and after selection. A concurrent authoritative change is a
`conflict`. A mismatch between an otherwise valid authoritative file and its
derived database row makes the projection `unavailable`; it does not make the
database an equal authority.

## Failure Semantics

The future configuration is closed to two modes:

- `disabled`: the default; preserve the current filesystem behavior and make
  no PostgreSQL connection;
- `required`: the five projected questions require a compatible, current,
  ready projection and never silently fall back.

There is no `prefer` mode.

| Condition in `required` mode | Operational result |
| --- | --- |
| Projection schema or head is missing or incompatible | `unavailable` |
| Projection is stale or disagrees with valid authoritative evidence | `unavailable` |
| Authoritative store is valid but has no matching evidence | `empty` |
| A valid exact identifier has no matching record | `not_found` |
| Required authoritative file is absent | `unavailable` |
| Authoritative schema, checksum, identity, lineage, or alert chain is corrupt | `corrupt` |
| Authoritative sources disagree or change during one query | `conflict` |
| Cooperative query deadline expires | `timeout` |

An unavailable projection affects only the five projected question kinds. The
three direct deployment/summary questions retain their existing dependencies
and failure behavior. Failure details remain sanitized.

Currentness is evaluated per query and selector, not by wall-clock age:

- report-backed `latest` selectors compare the freshly verified report-state
  identity and digest with the generation and then verify the selected report;
- exact report selectors load that report directly and require an identical
  projected identity and digest;
- reporting-run selectors always call `load_reporting_attempt`; a source record
  absent from or different in PostgreSQL makes the projection stale;
- alert selectors call the full verified alert-history and active-alert loaders
  and compare their ordered-history digest, count, terminal identities, and
  active-state digest with the generation before applying database pagination;
- performance and drift additionally verify the exact report calibration and
  report-scoped model-era association.

A projected lookup that returns no row must still perform the corresponding
authoritative exact or bounded loader check needed to distinguish a true
`not_found` or `empty` state from a stale projection. That check is for
verification only and must not become a fallback answer path.

## Authentication And Authorization

The local implementation target is PostgreSQL 16 in a dedicated service,
database `wind_forecast_operational`, schema `operational_projection`, private
volume, and private network. It must not share the Airflow metadata database.

The required role model is:

| Role | Contract |
| --- | --- |
| `wf_projection_owner` | `NOLOGIN` schema owner. |
| `wf_projection_migrator` | Applies reviewed DDL and records migrations. |
| `wf_projection_writer` | Inserts immutable generations and atomically publishes the projection head; no DDL. |
| `wf_projection_reader` | `SELECT` only, with read-only transactions. |

Privileges are revoked from `PUBLIC`. Migrator, writer, and reader use distinct
DSNs supplied only through environment variables or an approved local
credential store. DSNs and credentials must not be logged, persisted, printed,
or returned in failures. `search_path`, `statement_timeout`, `lock_timeout`, and
application name are explicit.

The exact future configuration keys are:

- `WIND_FORECAST_OPERATIONAL_PROJECTION_MODE`, exactly `disabled` or
  `required`, defaulting to `disabled`;
- `WIND_FORECAST_OPERATIONAL_ENVIRONMENT_ID`, exactly `local` in this contract;
- `WIND_FORECAST_OPERATIONAL_PROJECTION_READER_DSN` for required-mode queries;
- `WIND_FORECAST_OPERATIONAL_PROJECTION_WRITER_DSN` for manual projection;
- `WIND_FORECAST_OPERATIONAL_PROJECTION_MIGRATOR_DSN` for migration commands.

Disabled mode does not read or validate a DSN, import the PostgreSQL driver, or
open a connection. Required mode validates its mode and environment during
operational-query service construction and requires a non-empty reader DSN;
invalid or missing configuration exposes only a sanitized `unavailable`
result. Manual migration and projection commands validate only their required
DSN before acquiring a database connection and exit with a sanitized
configuration error. A database statement timeout is capped by the remaining
cooperative operational-query deadline.

The normal service does not publish the PostgreSQL port. Ephemeral tests may
bind a random port to `127.0.0.1` only. The existing local HTTP loopback
authorization remains unchanged; database credentials do not create product
identity or production authentication. Remote, multi-user, or production use
requires a new reviewed identity, authorization, secret, TLS, network, and
audit decision.

## Data Minimization, Privacy, And Retention

Only normalized fields needed by the five projected questions may be stored.
The projection must not store:

- raw JSON or Markdown documents;
- absolute or relative filesystem paths;
- hostnames, PIDs, usernames, operator principals, request bodies, correlation
  IDs, prompts, or generated answers;
- raw errors, stack traces, connection strings, environment dumps, tokens,
  passwords, or other secrets;
- raw REN/ERA5-Land observations, feature vectors, individual predictions or
  actuals, model/scaler bytes, or artifact bundles.

All ready generations are retained initially. Garbage collection, automatic
purge, and projection backups are not part of the accepted implementation
plans. Removing a dedicated database, schema, generation, or volume is a
destructive action requiring separate explicit authorization. Source evidence
retention remains governed by its existing contracts.

## Conceptual Relational Contract

Plan 2 must implement the following conceptual tables without changing their
meaning:

| Table | Responsibility |
| --- | --- |
| `schema_migration` | Migration version, name, checksum, and application instant. |
| `projection_generation` | Canonical manifest identity, versions, commit, source-set digest, counts, and ready instant. |
| `projection_head` | One atomically published generation per environment. |
| `evidence_record` | Original domain, source kind, schema, record identity, digest, and temporal context. |
| `generation_evidence` | Exact evidence membership for one generation. |
| `model_era` | Report-scoped model-era association and permitted checksum pins. |
| `monitoring_report` | Report identity, dates, source run, calibration/reference, and aggregate quality. |
| `quality_issue` | Allowlisted quality code and severity per report. |
| `monitoring_window` | 30/90-day state, sample count, coverage, and calendar bounds. |
| `performance_metric` | Metric value/state, severity, accepted limits, and direction. |
| `drift_measurement` | Feature, comparator, detector, value, severity, limits, and direction. |
| `alert_event` | Immutable event, rule, type, severity, date, and predecessor. |
| `active_alert_snapshot` | Active rule-to-event association for one generation. |
| `reporting_attempt` | Sanitized reporting attempt and optional report association. |
| `lineage_edge` | Typed relationships among projected evidence records. |

Required constraints include 64-character lowercase hexadecimal SHA-256
values, `window_days IN (30, 90)`, civil dates as `date`, UTC instants as
`timestamptz`, finite numbers, foreign keys, deterministic uniqueness, and no
filesystem-path columns. Raw `jsonb` payloads are not approved; a later need
for one requires explicit review.

These names and semantics are design decisions only. No current database or
persisted relational schema exists.

### Closed Column Allowlist

Plan 2 must implement only these normalized columns, plus surrogate primary
keys where required by PostgreSQL. Every domain row includes its foreign key to
`evidence_record`; no table may add a raw payload column.

| Table | Allowlisted domain columns |
| --- | --- |
| `schema_migration` | `version`, `name`, `sha256`, `applied_at_utc` |
| `projection_generation` | `generation_id`, `environment_id`, `contract_version`, `schema_version`, `projector_version`, `source_git_commit`, `source_set_sha256`, `evidence_record_count`, `generation_evidence_count`, `model_era_count`, `monitoring_report_count`, `quality_issue_count`, `monitoring_window_count`, `performance_metric_count`, `drift_measurement_count`, `alert_event_count`, `active_alert_snapshot_count`, `reporting_attempt_count`, `lineage_edge_count`, `ready_at_utc` |
| `projection_head` | `environment_id`, `generation_id`, `published_at_utc` |
| `evidence_record` | `domain`, `source_kind`, `schema_version`, `record_id`, `sha256`, `effective_at`, `observed_at_utc` |
| `generation_evidence` | `generation_id`, `evidence_record_id` |
| `model_era` | `model_era_id`, `association_kind`, `deployment_id`, `deployment_generation`, `registered_model_name`, `model_version`, `fit_cutoff`, `activation_cutoff`, `bundle_sha256`, `model_sha256`, `dataset_sha256`, `feature_schema_sha256`, `calibration_sha256`, `ledger_sha256`, `calibration_id`, `reference_id` |
| `monitoring_report` | `report_id`, `reporting_run_id`, `created_at_utc`, `through_date`, `source_run_id`, `source_status`, `calibration_id`, `reference_id`, `policy_sha256`, `quality_status`, `batch_status`, `verdict`, `watermark_date`, `watermark_age_days`, `objective_days`, `late_days`, `objective_missed`, `unresolved_late_date_count`, `date_count`, `ren_complete_count`, `era5_complete_count`, `integration_ready_count`, `feature_ready_count`, `model_era_id` |
| `quality_issue` | `report_id`, `position`, `code`, `severity` |
| `monitoring_window` | `report_id`, `window_days`, `status`, `sample_count`, `coverage_ratio`, `coverage_severity`, `minimum_samples`, `calendar_start`, `calendar_end` |
| `performance_metric` | `report_id`, `window_days`, `metric_name`, `value`, `value_status`, `severity`, `warning_threshold`, `critical_threshold`, `direction`, `unit_or_scale` |
| `drift_measurement` | `report_id`, `window_days`, `position`, `feature`, `comparator`, `detector`, `value`, `severity`, `warning_threshold`, `critical_threshold`, `direction` |
| `alert_event` | `alert_event_id`, `rule_id`, `through_date`, `event_type`, `severity`, `previous_alert_event_id` |
| `active_alert_snapshot` | `generation_id`, `rule_id`, `alert_event_id` |
| `reporting_attempt` | `reporting_run_id`, `attempted_at_utc`, `through_date`, `source_run_id`, `source_status`, `status`, `report_id`, `active_alert_count`, `failure_at_utc`, `failure_type`, `failure_message` |
| `lineage_edge` | `generation_id`, `edge_type`, `source_evidence_record_id`, `target_evidence_record_id`, `position` |

`failure_type` and `failure_message` use only the existing sanitized reporting
attempt values. Text lengths, enum/check constraints, nullability, indexes, and
foreign keys must be derived from the current executable schemas and loader
contracts without widening this column allowlist. If a required fact cannot be
represented by these columns, Plan 2 stops for contract review instead of
adding a column implicitly.

## Generation And Publication Contract

`generation_id` is the SHA-256 of a canonical manifest containing:

- environment ID `local`;
- operational-query contract, relational-schema, and projector versions;
- projector source Git commit;
- an ordered list of every source kind, schema version, record ID, digest, and
  effective time;
- verified report-state and active-alert-state digests;
- calibration/reference and model-era associations.

`projected_at_utc` is excluded from the digest so an identical rerun resolves
to the same generation.

Publication is manual, serialized by an environment-scoped advisory lock, and
all-or-nothing:

1. Load evidence only through authorized verified loaders.
2. Capture the initial mutable report and active-alert states.
3. Normalize all five approved domains and calculate the generation manifest.
4. Reverify mutable authoritative states before any database publication.
5. Insert and validate the generation inside one PostgreSQL transaction.
6. Verify counts, foreign keys, identities, and the source-set digest.
7. Mark the generation ready and update `projection_head` in the same
   transaction.

Any domain failure aborts the transaction. No incomplete generation becomes
the head. A valid empty store may produce a ready empty generation. An
identical rerun is `no_op`.

## Migration And Manual Operation Contract

Plan 2 uses `psycopg` 3 and numbered, checksummed, forward-only SQL migrations.
Each migration takes an advisory lock, runs in a transaction, and records its
checksum. A changed applied migration or unsupported future schema fails
closed. Migrations never run during import, API startup, or a health check.

There are no destructive down migrations. Because the projection is derived,
rollback disables the consumer and rebuilds only the dedicated database after
separate destructive authorization.

The manual CLI is limited to:

- `migration-status` and `migrate` in Plan 2;
- `plan`, `project`, and `verify` in Plan 3.

No `rebuild`, purge, scheduled projection, or request-triggered projection is
approved.

### Local Migration Runbook

The dedicated stack is defined in `operational_projection/docker-compose.yml`.
It uses PostgreSQL 16, the `wind_forecast_operational` database, a private
network, and a dedicated volume. The base stack publishes no host port. The
test overlay may bind PostgreSQL to an ephemeral port on `127.0.0.1` only.

Copy `operational_projection/.env.example` to the ignored local `.env`, replace
all placeholders, and start the dedicated database explicitly. For test use,
select an unused loopback port through
`WIND_FORECAST_OPERATIONAL_PROJECTION_TEST_PORT`, then set the three
role-specific DSNs to that port. The migration commands are:

```text
python scripts/manage_operational_projection.py migration-status
python scripts/manage_operational_projection.py migrate
```

Both commands require only the migrator DSN. Output is sanitized JSON; DSNs,
passwords, raw PostgreSQL errors, and environment dumps are never returned.
Migrations are not applied on import, API startup, health checks, or queries.

### Local Projector Runbook

The projector reads the monitoring store selected by
`WIND_FORECAST_MONITORING_STORE_ROOT`, which retains its existing project-root
default. It accepts only the `local` operational environment and requires a
clean tracked Git checkout so every generation is bound to one committed
projector implementation. It never imports the PostgreSQL driver or opens a
connection merely because its modules are imported.

The manual commands are:

```text
python scripts/manage_operational_projection.py plan
python scripts/manage_operational_projection.py project
python scripts/manage_operational_projection.py verify
```

`plan` and `verify` require only the reader DSN. `project` requires only the
writer DSN, obtains the environment advisory lock, revalidates the mutable
source views, and publishes the ready generation and head in one transaction.
An identical ready head is reported as `no_op`; no operational file, pointer,
lock, report, alert, model, or artifact is written.

`verify` returns `ready`, `missing`, `stale`, `mismatch`, or `incompatible`.
Only `ready` exits successfully. `plan` exits successfully with `planned` or
`no_op`, while `project` exits successfully with `projected` or `no_op`.
Failures and negative verification states use sanitized JSON and a non-zero
exit status.

A reporting attempt is mutable between its initial request and its terminal
result. Because the accepted relational schema retains immutable attempt rows,
the projector fails closed with `source_not_stable` while any verified attempt
is still `in_progress`. It neither omits nor prematurely persists that attempt;
the operator reruns the manual command after the reporting attempt reaches
`succeeded` or `failed`.

## Benchmark Readiness Gate

Plan 4 must use deterministic synthetic evidence only:

- 1,000 reports;
- 10,000 reporting attempts;
- 50,000 alert events;
- 200,000 drift measurements;
- two windows and five performance metrics per report.

It runs 30 warm repetitions for alert date interval plus pagination, exact
alert ID, reporting run by run ID, reporting run by report ID, report plus
30-day performance window, and report plus 90-day drift window. Filesystem
measurements call the verified loaders on every repetition. Selectors sharing
the same authoritative enumeration share one loader call per repetition, but
each selector is charged the complete loader duration plus its own selection
duration.

`GO` requires:

- every query remains below the existing five-second maximum deadline;
- alert date interval plus pagination and reporting run by report ID are each
  at least 20% faster by median than equivalent filesystem-fixture enumeration
  on the same process and machine;
- selective queries use the intended indexes under
  `EXPLAIN (ANALYZE, BUFFERS)`;
- PostgreSQL and filesystem selection return identical ordered identities.

A failed condition records `NO-GO` and blocks Plan 5. The benchmark does not
read or write the governed operational store.

### Plan 4 Result — `NO-GO`

Decision date: `2026-08-01`.

The deterministic harness and its PostgreSQL 16 smoke profile were implemented
and passed with exact identity/order equivalence for all six query cases. The
smoke used four reports, twelve terminal reporting attempts, forty alert
events, eight monitoring windows, forty performance metrics, eight hundred
drift measurements, and three repetitions. It ran only against a clean,
ephemeral loopback PostgreSQL instance and a temporary synthetic filesystem
store. Smoke timings are not readiness evidence and did not apply the full
timing or index-plan gates.

The mandatory full profile retained the approved 1,000/10,000/50,000/200,000
cardinalities and 30 repetitions. Multiple local attempts remained active for
hours without producing a complete result. Removing duplicate loader scans
between selectors and bulk-seeding the already loader-normalized snapshot into
the clean ephemeral database did not make the full run complete in a
reasonable operator review window. No complete set of medians, maxima, or
`EXPLAIN (ANALYZE, BUFFERS)` results exists, so none is admitted or inferred.

The readiness claim is therefore unproven and the fail-closed decision is
`NO-GO`. Thresholds, cardinalities, and repetitions were not reduced. Plan 5
must not start unless a later separately approved decision replaces this
`NO-GO` with a complete benchmark design and a reviewed `GO` result.

## Delivery Plans

Each plan starts from updated `master`, uses a separate branch and draft pull
request, passes independent review, and stops for explicit user approval:

1. This documentation-only contract: accepted by this record.
2. Dedicated PostgreSQL foundation, roles, schema, and migrations: implemented.
3. Manual all-or-nothing artifact projector and verifier: implemented.
4. Deterministic benchmark and `GO`/`NO-GO` decision: implemented with
   `NO-GO`; the full readiness evidence did not complete.
5. Optional `disabled|required` query-layer integration after benchmark `GO`:
   blocked by the Plan 4 `NO-GO` and not implemented.

No plan authorizes the next one implicitly. Observability, Copilot, MCP, RAG,
staging, cloud, and production identity remain later independent decisions.

## Acceptance Criteria

- The database remains a disposable derived projection; files and verified
  loaders remain the sole evidence authority.
- The initial five question kinds, direct-only three question kinds, evidence
  boundary, prohibited actions, and failure semantics are closed.
- Public query, answer, citation, HTTP, OpenAPI, and v1 contracts remain
  unchanged.
- PostgreSQL is never cited and every answer revalidates original evidence.
- Authentication, authorization, secret handling, data minimization, retention,
  migration, generation, concurrency, benchmark, rollback, and stop gates are
  decision-complete for Plans 2 through 5.
- PostgreSQL foundation, roles, schema, migrations, manual projector, and
  benchmark harness are implemented without a consumer; the benchmark decision
  is `NO-GO`, and query integration remains blocked and unimplemented.
- The projector is manual, reconstructible, loader-backed, serialized, and
  all-or-nothing. It introduces no consumer, pipeline, scheduler, artifact, or
  operational-state mutation.

## Risks And Controls

| Risk | Control |
| --- | --- |
| PostgreSQL becomes authoritative | Revalidate original files for every answer; never cite the database. |
| A stale projection answers as current | Required-mode head/schema/source-set verification and no fallback. |
| A Registry snapshot is mistaken for live state | Keep deployment and model-metadata questions on direct live verification. |
| Partial projection becomes visible | One all-or-nothing transaction publishes ready generation and head together. |
| Concurrent projectors diverge | Environment-scoped advisory lock and authoritative pre/postchecks. |
| Relational normalization changes semantics | Loader-backed field mapping and filesystem/PostgreSQL equivalence tests. |
| Secrets or private host data are duplicated | Closed column allowlist and negative persistence/output tests. |
| Airflow metadata is affected | Dedicated instance, database, roles, volume, and lifecycle. |
| Database dependency breaks the current API | Default disabled mode performs no import or connection. |
| PostgreSQL adds complexity without benefit | Mandatory benchmark `GO` before query integration. |

## Rollback

Before merge, close the draft pull request. After acceptance, a changed
contract requires a later reviewed ADR that marks this record `Superseded`;
historical evidence and this decision are not silently rewritten.

Later runtime rollback first sets projection mode to `disabled`, verifies the
unchanged filesystem behavior, reverts only projection code, and retains or
removes only the dedicated database after explicit destructive approval.
Rollback never changes Phase 8/9 evidence, deployment, Registry aliases,
MLflow, V1, Airflow, or scheduler state.

## Stop Conditions

Stop and return for explicit review if any later plan requires:

- a new query kind, source, authority, public field, or persisted source
  schema;
- parsing around a verified loader or weaker validation/citation/failure
  behavior;
- mutation of an operational file, pointer, alert, report, artifact, Registry,
  deployment, or scheduler;
- scheduler integration, remote exposure, production authentication, or a new
  user population;
- projection of v1, raw data, feature vectors, individual predictions/actuals,
  models, scalers, or bundles;
- access to Airflow's PostgreSQL database, roles, migrations, or volume;
- provider calls, downloads, training, retraining, or MLflow mutation;
- implementation files outside the separately approved plan allowlist;
- continuation to query integration after benchmark `NO-GO`.
