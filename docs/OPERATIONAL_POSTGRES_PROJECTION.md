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
| Implementation status | Dedicated foundation, migrations, manual projector, verifier, deterministic benchmark with a superseding `GO`, and optional `disabled|required` query-layer integration implemented; projection consumption remains disabled by default |

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

The exact implemented configuration keys are:

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

### Query-layer integration runbook

Plan 5 implements the closed two-mode configuration without changing any
public query, answer, citation, endpoint, OpenAPI, or HTTP status contract.
`disabled` remains the default. In that mode operational service construction
does not read the reader DSN, import `psycopg`, or contact PostgreSQL. To opt
in, configure the dedicated reader credential and set the mode to exactly
`required`; there is no `prefer` value.

In `required`, `data_quality`, `monitoring_performance`,
`monitoring_drift`, `monitoring_alerts`, and `reporting_run` use PostgreSQL
only for current-generation identity selection, causal ordering, and bounded
pagination. Every selected report, calibration, model era, reporting attempt,
alert history, and active-alert state is then reloaded through the existing
verified loaders. Loader-derived normalized values and original evidence
identities must match the projection before the existing fact and citation
logic runs. PostgreSQL is never cited.

The reader opens read-only transactions, validates the exact bundled migration
names and checksums, ready local head, contract/schema/projector versions,
generation evidence count, and canonical source-set checksum, and joins every
entity through the current head's `generation_evidence`. Its statement and
lock timeouts are capped at the remaining query deadline. Alert selection uses
the loader's deterministic `(through_date, rule_id, causal_depth,
alert_event_id)` order; a missing predecessor, cross-rule edge, cycle, count
mismatch, stale value, incompatible head, or invalid checksum fails closed as
`unavailable` without a filesystem fallback. PostgreSQL deadline cancellation
maps to `timeout`.

`operational_summary`, `active_deployment`, and `active_model_metadata` remain
on their unchanged direct loader and live Registry-verification path. An
invalid or unavailable required projection therefore affects only the five
projected query kinds.

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

### Initial Plan 4 Result — `NO-GO` (superseded)

Decision date: `2026-08-02`.

The deterministic harness and its PostgreSQL 16 smoke profile were implemented
and passed with exact identity/order equivalence for all six query cases. The
smoke used four reports, twelve terminal reporting attempts, forty alert
events, eight monitoring windows, forty performance metrics, eight hundred
drift measurements, and three repetitions. It ran only against a clean,
ephemeral loopback PostgreSQL instance and a temporary synthetic filesystem
store. Smoke timings are not readiness evidence and did not apply the full
timing or index-plan gates.

The mandatory full profile retained the approved 1,000/10,000/50,000/200,000
cardinalities and 30 repetitions. After the loader, publication-observability,
and generation-ID recomputation follow-ups documented below, it completed all
six query measurements in 780.192 seconds. Every identity/order comparison was
exact, every maximum PostgreSQL query time was below five seconds, both speed
gates passed, and five of six intended-index gates passed.

The complete fail-closed decision is `NO-GO` solely because
`alert_interval_pagination` did not use `alert_event_date_idx`. Thresholds,
cardinalities, repetitions, constraints, and guarantees were not reduced.
At that revision, Plan 4 was complete with this reviewed `NO-GO`; Plan 5
remained blocked unless a later separately approved decision replaced it with
a reviewed `GO` result. The superseding result is recorded below.

#### Benchmark runtime diagnosis and bounded execution

Follow-up profiling found no blocked PostgreSQL query or accidental repetition
loop. Each full filesystem repetition authoritatively reads approximately
71,001 JSON records: 50,000 alert events, 10,000 requests, 10,000 terminal
outcomes, 1,000 reports referenced by successful attempts, and the selected
monitoring report. Thirty measured repetitions therefore require approximately
2.13 million verified JSON reads, before the warm-up and projection snapshot
passes. This filesystem work, rather than the 180 PostgreSQL statements, is the
dominant local runtime cost.

The harness now reports sanitized phase and repetition progress on stderr,
groups snapshot rows once for bulk publication, and applies a configurable
one-hour hard runtime bound by default. A supervisor process terminates the
single benchmark worker at that bound, so fixture generation, verified loaders,
snapshot construction, publication, and queries cannot continue silently past
it. The supervisor removes only the uniquely tagged synthetic temporary store
after normal completion or forced termination. Reaching either this bound or
any PostgreSQL statement timeout records `NO-GO`; it can never produce `GO` or
a silent fallback. The full cardinalities, thirty repetitions, loader calls,
identity/order comparisons, deadlines, speed thresholds, and index-plan gates
remain unchanged.

A material reduction beyond these harness improvements would require a separate
approved change to the authoritative loaders, with deterministic parallel-read
or equivalent implementation and full corruption, ordering, identity, and
regression evidence. Benchmark-side caching or parallel repetitions are not
accepted because they would weaken or bias the filesystem comparison.

#### Loader performance follow-up

The separately approved loader optimization keeps the authoritative parsing,
validation, public errors, and returned ordering unchanged. Stores below 32
candidate records remain on the sequential path. At 32 records or more,
`load_alert_history` and `load_reporting_attempts` read independent sorted
paths through an order-preserving thread map capped at eight workers. The cap
limits file-descriptor pressure; the ordered result iteration also preserves
which sorted-path error is observed first. The implementation is read-only and
introduces no cache, configuration, environment variable, or API change.

At that stage, this follow-up did not change the Plan 4 decision. The mandatory
full profile still had to complete with all approved cardinalities,
repetitions, equivalence, deadline, speed, and index-plan gates before a
reviewed decision could replace the existing `NO-GO`.

A deterministic local pilot with 40 reports, 400 reporting attempts, 2,000
alert events, and 8,000 drift measurements returned byte-for-byte equivalent
loader results. On the same generated store, alert-history enumeration changed
from 12.522 seconds sequentially to 0.426 seconds in the bounded parallel path;
reporting-attempt enumeration changed from 5.592 seconds to 0.235 seconds. These
pilot timings demonstrate the loader optimization only. They are not projection
readiness evidence and do not satisfy or replace the mandatory full benchmark.

The post-optimization full profile was executed once on `2026-08-02` with the
one-hour hard bound. Migration completed in 0.158 seconds, deterministic fixture
generation in 50.313 seconds, and the complete verified projection snapshot in
139.719 seconds. The run then exhausted the remaining bound in
`snapshot_publish`; query measurement never started. The supervisor recorded
`NO-GO` with `benchmark_runtime:hard_timeout` and removed its synthetic store.
The dedicated PostgreSQL container, network, and volume were also removed.

This intermediate result proved that verified loader enumeration and snapshot
construction were no longer the blocking phase. It did not provide query
medians, maxima, or index plans, so the readiness decision remained `NO-GO`.
The following work therefore targeted benchmark publication separately without
weakening the transactional projector, database constraints, cardinalities,
repetitions, or query gates.

#### Snapshot publication performance follow-up

The separately approved publication optimization changes only the synthetic
benchmark seeding path. Each allowlisted table now has a fixed ordered column
and exact PostgreSQL-type specification derived from migration `0002`.
Psycopg streams those already normalized values with binary `COPY`; the
fixed-length SHA-256 columns use an explicit COPY-only `bpchar` binary dumper.
No manual binary serialization, staging table, unlogged relation, disabled
constraint or trigger, durability relaxation, schema change, or writer-role
expansion is introduced.

All constrained-table copies, the ready marker, and `projection_head` remain in
one explicit transaction, with the head as the final database write before
commit. A failure from binary adaptation, a PostgreSQL constraint, the injected
pre-commit test hook, a statement timeout, or the cooperative global runtime
check before commit aborts and rolls back the transaction. Commit is the
irreversible boundary: after PostgreSQL confirms it, no rollback is claimed.
A later overall timeout or `ANALYZE` failure still produces `NO-GO`, but leaves
only a complete committed synthetic generation in the disposable database,
which is removed with the ephemeral environment. The supervisor and one-hour
full-profile bound remain unchanged.

Sanitized progress now distinguishes preparation, each table COPY, head
publication, commit, and each `ANALYZE`. The successful result adds only the
corresponding millisecond timings; it exposes no DSN, SQL text, path, or source
payload. This follow-up does not alter the full cardinalities, thirty
repetitions, filesystem enumeration, query deadline, speed gates, index gates,
or identity/order equivalence requirements.

The real smoke profile at source commit `06799a661c3a64355270ab3e2ddec24cca09cae4`
completed in 1.348 seconds on `2026-08-02`. Snapshot publication took 0.302
seconds, the ready generation and head were queryable, and all six identity and
ordering comparisons were exact. As required, its timing and index gates were
disabled, so this is functional evidence rather than a readiness decision.

The post-publication-optimization full profile was then executed exactly once
at the same source commit, with the original cardinalities, thirty repetitions,
and 3,600-second supervisor. Migration completed in 0.199 seconds, fixture
generation in 82.063 seconds, and snapshot construction in 147.835 seconds.
Within publication, preparation completed in 0.127 seconds, binary evidence
COPY in 0.726 seconds, and generation insertion in 2.140 seconds. The subsequent
binary `generation_evidence` COPY of 61,004 associations did not complete within
the remaining bound. No projected-table COPY, ready/head publication, commit,
`ANALYZE`, or query measurement followed. The supervisor recorded `NO-GO` with
`benchmark_runtime:hard_timeout`.

Because the timeout occurred before commit, PostgreSQL rolled back the complete
transaction. A read-only verification found zero rows in `projection_head`,
`projection_generation`, `evidence_record`, `generation_evidence`,
`monitoring_report`, and `drift_measurement`. This result identified
`copy_generation_evidence` as the then-remaining publication bottleneck but did
not provide six-query medians, maxima, speedups, or index plans. Plan 4 was
therefore not complete, its readiness decision remained `NO-GO`, and Plan 5
remained blocked. The next attempt required the separately reviewed plan below,
focused on this substep without weakening the preserved constraints or
transaction.

#### Generation-evidence root-cause follow-up

An opt-in PostgreSQL 16 probe compared the existing binary `COPY`, Psycopg text
`COPY`, and an `INSERT ... SELECT` association through the real writer role,
schema, PK, and both FKs. Each trial seeded valid synthetic evidence in one
transaction, verified the exact association identities and cardinality, then
deliberately rolled back and confirmed an empty database. All three methods
completed at 1,000 and 10,000 associations and advanced to three independent
61,004-association trials with a 45-second supervisor and 30-second statement
timeout.

| Method | Trial 1 (ms) | Trial 2 (ms) | Trial 3 (ms) |
|---|---:|---:|---:|
| Binary `COPY` | 879.107 | 846.876 | 882.014 |
| Text `COPY` | 810.468 | 937.702 | 946.482 |
| `INSERT ... SELECT` | 847.299 | 956.123 | 1,059.340 |

The values above cover only association opening/configuration, row transfer,
and server finalization. Every trial wrote exactly 61,004 rows with the same
identity digest and rolled back cleanly. Because binary `COPY` passed every
trial and neither alternative was faster than its best trial in all three
measurements, the fail-closed selection rule retains binary `COPY`.

Code inspection then identified the scale-dependent defect outside Psycopg:
`ProjectionSnapshot.generation_id` is a computed property that canonically
serializes and hashes the full manifest. The association generator referenced
that property once per evidence row, recomputing the complete large-manifest
digest 61,004 times. Publication now computes the generation ID once during
preparation and reuses the immutable value for the manifest row, every
generation-evidence association, ready marker, and head. A regression test
requires exactly one property access. No schema, constraint, privilege,
transaction, durability, cardinality, repetition, or query gate changes.

#### Initial complete full-profile decision

The full profile ran exactly once at source commit
`57071911f5f4c3c3dd06ade5e9e284fb04a307e3` on a clean ephemeral PostgreSQL
16 instance. Migration took 0.134 seconds, fixture generation 47.518 seconds,
snapshot construction 137.933 seconds, snapshot publication 8.451 seconds,
and the complete query measurement 567.568 seconds. Total runtime was 780.192
seconds. Within publication, preparation took 1.646 seconds,
`copy_generation_evidence` 0.654 seconds, all 200,000 drift rows 2.933 seconds,
commit 0.025 seconds, and every `ANALYZE` completed.

| Query | Filesystem median (ms) | PostgreSQL median (ms) | PostgreSQL max (ms) | Speedup | Equivalent | Intended index |
|---|---:|---:|---:|---:|:---:|:---:|
| `alert_interval_pagination` | 12,186.7892 | 2.8616 | 58.5708 | 0.999765 | yes | **no** |
| `exact_alert_id` | 12,177.6591 | 0.71575 | 1.6546 | 0.999941 | yes | yes |
| `reporting_run_by_run_id` | 5,563.47955 | 0.6418 | 1.3373 | 0.999885 | yes | yes |
| `reporting_run_by_report_id` | 5,563.4245 | 0.6276 | 1.2 | 0.999887 | yes | yes |
| `performance_report_window` | 4.72285 | 0.65085 | 1.3514 | 0.862191 | yes | yes |
| `drift_report_window` | 4.7555 | 0.92485 | 11.609 | 0.805520 | yes | yes |

`alert_interval_pagination` used
`alert_event_evidence_record_id_key` and `projection_head_pkey`, but not its
required `alert_event_date_idx`; the harness therefore recorded the single
failure `alert_interval_pagination:index`. This is a complete `NO-GO`, not a
timeout or missing-evidence result. At that revision Plan 4 was concluded and
Plan 5 remained blocked and unimplemented.

#### Superseding alert-pagination full-profile decision — `GO`

The isolated alert-pagination follow-up changed only the benchmark SQL. A
`MATERIALIZED` CTE selects the alert identity, evidence identity, date, and rule
for the requested date interval. The outer query still validates membership in
the current `projection_head` generation before applying the unchanged
`through_date`, `rule_id`, and `alert_event_id` order and the unchanged limit
and offset. The expected index remains `alert_event_date_idx`; no migration,
schema, constraint, transaction, role, planner setting, cardinality,
repetition, threshold, API, or projector behavior changed.

A PostgreSQL 16 probe with 50,000 synthetic alerts first confirmed a bitmap
index scan on `alert_event_date_idx`. The committed implementation then passed
the real smoke profile with exact identity/order equivalence for all six query
cases. All 30 benchmark contract and PostgreSQL integration tests passed,
including fail-closed pre-commit rollback, real-constraint rejection, and
binary publication coverage.

The mandatory full profile ran exactly once on `2026-08-02` at source commit
`a6d34cb6fdd8e1ef706d1695e981a897d395cbb0`, on a clean ephemeral PostgreSQL
16 instance with the original cardinalities, 30 repetitions, and 3,600-second
supervisor. Migration took 0.037 seconds, fixture generation 91.694 seconds,
snapshot construction 277.749 seconds, snapshot publication 13.477 seconds,
and query measurement 1,053.255 seconds. The exact phase measurements were
37.107 ms, 91,693.733 ms, 277,749.259 ms, 13,476.666 ms, and 1,053,254.591 ms,
respectively; exact total runtime was 1,456,530.665 ms.
Publication completed every constrained binary `COPY`, published the head as
the final database write, committed in exactly 40.243 ms, and completed every
`ANALYZE`.

| Query | Filesystem median (ms) | PostgreSQL median (ms) | PostgreSQL max (ms) | Speedup | Equivalent | Intended index |
|---|---:|---:|---:|---:|:---:|:---:|
| `alert_interval_pagination` | 15,800.7812 | 3.43485 | 5.987 | 0.999783 | yes | yes |
| `exact_alert_id` | 15,788.1386 | 1.0576 | 1.8248 | 0.999933 | yes | yes |
| `reporting_run_by_run_id` | 7,111.33025 | 0.8419 | 1.84 | 0.999882 | yes | yes |
| `reporting_run_by_report_id` | 7,111.3412 | 0.80605 | 1.759 | 0.999887 | yes | yes |
| `performance_report_window` | 7.93765 | 0.83815 | 1.9221 | 0.894408 | yes | yes |
| `drift_report_window` | 7.97865 | 1.14865 | 2.1955 | 0.856035 | yes | yes |

All six identity/order comparisons were exact, all six PostgreSQL maxima were
below five seconds, both required speed gates exceeded 20%, and all six
intended-index gates passed. `alert_interval_pagination` used
`alert_event_date_idx` and `generation_evidence_evidence_idx`. The fail-closed
decision is therefore `GO`, replacing the historical `NO-GO` above. Plan 4 is
concluded. This result only makes Plan 5 eligible for a separate plan and
explicit authorization; at that benchmark decision, Plan 5 had not been
started.

## Delivery Plans

Each plan starts from updated `master`, uses a separate branch and draft pull
request, passes independent review, and stops for explicit user approval:

1. This documentation-only contract: accepted by this record.
2. Dedicated PostgreSQL foundation, roles, schema, and migrations: implemented.
3. Manual all-or-nothing artifact projector and verifier: implemented.
4. Deterministic benchmark and `GO`/`NO-GO` decision: implemented and concluded
   with a superseding `GO`; all six intended indexes and all other gates passed.
5. Optional `disabled|required` query-layer integration after benchmark `GO`:
   implemented with default-disabled, fail-closed consumption and authoritative
   loader revalidation.

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
- PostgreSQL foundation, roles, schema, migrations, manual projector, benchmark
  harness, and optional query consumer are implemented; the benchmark decision
  is `GO`, and the consumer remains disabled by default.
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
