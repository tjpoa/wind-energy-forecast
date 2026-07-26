# Phase 9 — Historical Batch Monitoring Contract

## Decision Record

| Field | Value |
| --- | --- |
| Decision | Phase 9 Stage 0 temporal and operational contract |
| Status | `Accepted` |
| Decision date | `2026-07-20` |
| Contract version | `historical_batch_monitoring_v1` |
| Operating mode | Historical batch monitoring with delayed REN and ERA5-Land data |
| Current implementation status | Stage 3 read-only API projection and retrospective dashboard implemented over the immutable Stage 2 evidence |

## Purpose And Scope

This record defines exactly what the first monitoring workflow will estimate,
when it will run, when its inputs and actual value are considered available,
and how delayed or revised records will be handled.

The accepted mode is a delayed historical batch over the existing REN and
ERA5-Land v2 contracts. Its output is a retrospective estimate, or hindcast,
for a completed local calendar day. It is not an ex-ante forecast and must not
be presented as real-time, same-day, D+1, or multi-day forecasting.

Stage 0 was a documentation-only decision and Stage 1 implemented the approved
prediction, actual, and metric evidence ledger. Stage 2 adds offline quality,
drift, performance, immutable reporting, and local alert evidence. A separately
approved Stage 3 projects that evidence through a read-only API and dashboard.
Orchestration, provider calls, external alert delivery, notebooks, retraining,
and model promotion remain outside this phase.

## Decision Summary

| Contract item | Accepted decision |
| --- | --- |
| Target date | Completed civil calendar day `D` in `Europe/Lisbon` |
| Target-day offset | `target_day_offset=0` |
| Forecast horizon | `forecast_horizon=null`; no ex-ante lead time exists |
| Weather role | ERA5-Land historical reanalysis for the completed day `D` |
| Actual-value role | REN historical wind-production observations for `D` |
| Schedule | Daily, including weekends, at `12:00 Europe/Lisbon` |
| Nominal objective | First permitted attempt for `D` at `D+5 12:00 Europe/Lisbon`; never issue earlier |
| Source-freshness deadline | Mark source delay at `D+7 12:00 Europe/Lisbon` if `D` is not complete in both sources |
| Estimate deadline | After the estimate layer is activated, mark estimate delay at `D+7 12:00 Europe/Lisbon` only when `D` was otherwise eligible |
| D+1 or multi-day forecast | `NO-GO` until a forecast-weather source and contract are approved |
| Historical gaps | Explicit per-date states; earlier gaps do not block later dates |
| Persistence | Append-only snapshots and revisioned results |

The D+5 objective and D+7 deadline are internal engineering policies. They are
not supplier service-level agreements. The ECMWF product guide describes
ERA5-Land as updated daily at approximately five days behind real time and
warns that delivery times can vary.

## Temporal Contract

### Canonical Day And Daylight Saving Time

`D` is the complete `Europe/Lisbon` civil day from local `00:00` inclusive to
the next local `00:00` exclusive. Source timestamps retain their source
identity, but daily integration follows the accepted local-day contract.

Completeness requires:

- REN: `96` 15-minute physical intervals on ordinary days, `92` on the spring
  DST transition, and `100` on the autumn DST transition;
- ERA5-Land: `24` local-day hours per approved point on ordinary days, `23` on
  the spring DST transition, and `25` on the autumn DST transition;
- all approved ERA5-Land points and all other existing v2 validation rules.

No interval is imputed, interpolated, prorated, forward-filled, backfilled, or
silently dropped to make a day complete.

### Event Timeline

| Event | Contract |
| --- | --- |
| Observation window | The complete local day `D` |
| Actual first observed | First retrieval at which the REN partition for `D` is complete and passes validation |
| Nominal scheduled attempt | `D+5 12:00 Europe/Lisbon`; this is the earliest permitted issuance time |
| Estimate issued | First scheduled or catch-up run where `D` is individually eligible |
| Source-freshness deadline | `D+7 12:00 Europe/Lisbon` when either source for `D` remains incomplete |
| Estimate deadline | `D+7 12:00 Europe/Lisbon` when the estimate layer is active and `D` was otherwise eligible |
| After the deadline | Keep retrying in later daily runs until recovery or an explicit manual exclusion |

Every schedule instant is stored both with its `Europe/Lisbon` offset and as a
UTC instant. `issued_at_utc` is the actual persistence time, not the nominal
schedule. Because the full ERA5-Land day is required, issuance occurs after
`D`; `target_day_offset=0` describes the target date only and must never be
interpreted as a zero-hour forecast lead time.

Even if both sources are complete earlier, the historical estimate for `D`
must not be issued before `D+5 12:00 Europe/Lisbon`. This fixes the as-of policy
for comparable monitoring runs and prevents opportunistic early issuance from
changing the contract between dates.

## Target Contract

The accepted Stage 0 target preserves the current v2 numerical contract.

| Field | Value |
| --- | --- |
| Target contract ID | `ren_wind_production_15min_mw_sum_v1` |
| Column | `Wind_Production` |
| Formula | `sum(wind_production_mw)` across every validated physical interval in `D` |
| Reported scale | `sum_of_15_minute_MW_observations` |
| Physical unit | Not applicable |
| Missing-data policy | Target unavailable when the REN day is incomplete |

This value is a legacy sum of MW observations. It must not be labelled `MW`,
`MWh`, daily energy, or another physical quantity. A future move to MWh is a
material target change and requires separate approval, a new versioned dataset
and schema, regenerated target-derived features, refitted scalers, retraining,
and re-baselining. Existing v1 or v2 artifacts must never be silently renamed,
rescaled, or reinterpreted as MWh.

## Watermarks, Eligibility, And Catch-Up

The workflow keeps four distinct concepts:

- `ren_source_watermark`: the greatest target date for which a complete,
  validated REN partition exists;
- `era5_source_watermark`: the greatest target date for which the local ERA5-
  Land day is complete for every approved point;
- `common_source_watermark`: the greatest individual target date that is
  complete in both REN and ERA5-Land; it is the maximum of the intersection of
  their per-date complete sets, not the minimum of two maximum dates and not a
  contiguous-history guarantee;
- `evaluation_eligible_dates`: individual dates for which both source days,
  required feature history, and the relevant schemas and artifacts are valid.

A source watermark is a freshness summary, not a claim that every earlier date
is complete. Eligibility and lateness are also evaluated per date. The six
known historical REN gaps and any future unavailable date therefore do not
freeze later processing. At each noon run, the common watermark freshness
objective is at least the local run date minus seven calendar days; an older
common watermark raises `common_watermark_late`. Independently, any particular
date still incomplete at its D+7 deadline raises `source_late`, even if later
dates allowed the common watermark to advance.

Each daily run scans all unresolved or previously late dates up to the source
watermarks. It issues every newly eligible date whose D+5 earliest-issuance
time has passed and that has not already been issued for the same immutable
input and model identities. It never processes only the maximum date.

The run also scans already issued dates for newly observed source checksums.
Every source revision triggers a dependency scan over the source-snapshot IDs
recorded by already issued model-input snapshots:

- a REN revision for source day `S` creates a new actual revision and restated
  metrics for `S`, when an estimate for `S` exists;
- because REN production also feeds lag and rolling features, the same revision
  creates a separately identified `restated` run for every already issued
  target date whose input lineage depends on `S`;
- an ERA5-Land revision likewise creates a `restated` run for every already
  issued target date whose current-day, lag, or rolling weather inputs depend
  on the revised source partition;
- the affected target-date set is derived from the exact versioned input
  dependency map, not from an assumed fixed lookback window;
- each restated run retains the original model, scaler, feature schema, and
  transformation versions, replaces only the revised dependency and its
  causally derived features, and retains the original identities for every
  unaffected dependency;
- when an actual is available for a restated target date, its metrics are also
  written as a new immutable metric revision.

The as-issued estimate always remains unchanged. A restated estimate records
`restates_run_id` and cannot replace or mutate the as-issued record. Current
pointers for as-issued and restated views remain distinct.

A gap remains explicit and moves through these states:

| State | Meaning |
| --- | --- |
| `pending_source` | One or more required source partitions are not complete |
| `source_late` | One or more source partitions were incomplete at the D+7 source deadline; retries continue |
| `blocked_prerequisite` | Estimate-layer artifacts or approvals are not active; this is not an estimate-lateness event |
| `eligible` | Sources, history, schemas, and approved artifacts satisfy the contract |
| `issued` | The immutable retrospective estimate has been persisted |
| `estimate_late` | The estimate layer is active and an otherwise eligible date was not persisted by D+7; retries continue |
| `excluded` | An operator recorded a durable exclusion with a reason and evidence |

`source_late` and `estimate_late` both roll up to the reporting category
`late`, while retaining their distinct causes. No automatic timeout converts
either state to `excluded`. Estimate-lateness alerts remain disabled until all
estimate-layer activation gates are satisfied.

## Actual Availability And Revision Policy

The actual is available to this system only after the first complete REN
partition has been retrieved and validated. The record stores the retrieval
timestamp, raw and normalized checksums, source date, validation result, and
source identity. It does not infer an earlier supplier publication time.

The provider's provisional or final status for the ingested endpoint remains
`unknown`. REN currently describes its live electricity values as provisional
and updated every 15 minutes, but this does not establish a finality flag for
the historical endpoint used by the repository.

Before any performance join, the estimate is persisted independently. The
model-input snapshot excludes the actual target for `D`, even if the actual is
already available when the retrospective estimate runs. A feature-ready table
that also carries the label must not itself be treated as the model-input
snapshot.

Snapshots, estimates, actual revisions, and metric revisions are append-only:

- a changed source checksum creates a new source revision;
- a revised actual creates new metrics that reference the unchanged estimate;
- a revision records `supersedes_id` when it replaces the current view;
- a `current` pointer may advance to a later revision;
- advancing the pointer never deletes, overwrites, or mutates prior evidence.

## Minimum Persistent Record

Any future implementation must preserve at least:

- `run_id`, `target_date`, `scheduled_at`, `issued_at_utc`, and processing
  state;
- operating-mode and target-contract versions;
- REN and ERA5-Land source versions, retrieval timestamps, paths, and
  checksums;
- `model_input_snapshot_id`, immutable path and checksum, with the exact
  target-free serialized feature values used for inference;
- the per-feature source-dependency map, including every source snapshot ID,
  partition date, and checksum needed for revision-impact scans;
- feature-schema version and exact feature names/order;
- transformation version and source-code commit SHA used to produce the model
  input snapshot;
- scaler and model versions/checksums;
- retrospective estimate value and its declared scale;
- actual revision ID, value, retrieval timestamp, checksum, validation status,
  and provider finality `unknown`;
- `supersedes_id` for same-view revision lineage, `restates_run_id` for
  recalculated estimates, and an explicit exclusion reason when state is
  `excluded`.

The Stage 1 ledger satisfies this append-only evidence contract without
changing the existing dataset builders or their safety behavior.

## Stage 1 Implementation — Prediction Evidence Ledger

`wind_forecast.monitoring` implements `historical_batch_monitoring_v1` as a
local filesystem ledger. It verifies the Phase 8 `current.json` checksum chain,
the accepted v2 reference manifests, exact feature order, model checksum,
`selected_not_promoted` decision, hindcast task, and absence of a scaler before
issuing an estimate.

The public interfaces are:

- `plan_historical_monitoring`, a strictly read-only plan;
- `run_historical_monitoring`, an exclusive-lock, append-only execution;
- `load_prediction_evidence`, which verifies the complete evidence chain;
- `replay_prediction`, which reloads the immutable model/input and checks
  numerical equivalence with `rtol=1e-12` and `atol=1e-9`;
- `load_verified_current_state` in `wind_forecast.incremental`, the supported
  downstream bridge to the Phase 8 state.

The default ledger root is `data/processed/v2/monitoring/`. Immutable records
live under `activations/`, `model_snapshots/`, `input_snapshots/`,
`predictions/`, `actuals/`, `metrics/`, and `runs/`. Only the derived
`state/current.json` pointer advances atomically. IDs are SHA-256 hashes of
canonical JSON with an explicit record type, and existing content-addressed
paths accept only byte-identical retries.

The first run requires an immutable activation date. Dates before activation
are considered only through an explicit, bounded backfill. Eligible dates are
never issued before `D+5 12:00 Europe/Lisbon`; later first issuance is marked
`catch_up`, while backfills and causal recalculations are explicitly marked.
No ex-ante or multi-day mode is accepted.

The persisted input is target-free and retains ordered names/values, feature
partition identity, transformation version/code evidence, and the exact
REN/ERA5-Land revisions used by every calendar, direct, lag, or rolling
feature. A semantic source revision can therefore create a separate restated
view while preserving the original as-issued prediction. A physical rewrite
with the same semantic revision does not recalculate it.

Revision occurrence is retained independently from semantic identity. A
sequence `A -> B -> A` therefore creates a third actual/metric revision and a
new restatement that supersedes the `B` view; it never reuses the first `A`
record or leaves the current pointer on stale `B` evidence.

The prediction is written before the actual is consulted. REN actual revisions
and metric revisions are immutable and linked with `supersedes_id`; errors are
created only when both records exist. A retry can reconcile a prediction left
by a crash before the actual/pointer stage without issuing a duplicate.

Run locally without provider calls:

```powershell
.\venv\Scripts\python.exe .\scripts\run_historical_monitoring.py `
  --through-date YYYY-MM-DD `
  --activation-date YYYY-MM-DD `
  --source-store-root data\processed\v2\incremental_update `
  --model-bundle outputs\training\v2_reference_mlflow `
  --dry-run
```

Remove `--dry-run` only after reviewing the plan. Subsequent runs omit
`--activation-date` or repeat the same value. `--backfill-start` and
`--backfill-end` must always be supplied together and must precede activation.
`--model-bundle` is always explicit; the example names the locally verified
accepted output and the CLI never guesses or promotes a model directory.

## Stage 2 Implementation — Drift And Performance Reports

`wind_forecast.monitoring_reporting` builds a content-addressed reference from
the exact train-plus-validation population used to fit the accepted v2 model
(`2010-01-15` through `2024-12-31`). It verifies dataset, model, feature-order,
transformation, and artifact checksums before calculating reference
predictions. Those predictions describe prediction drift only and are never
written to the Stage 1 prediction ledger or treated as performance evidence.

The tracked policy `config/monitoring_policy_v1.json` fixes 30- and 90-day
civil-calendar windows, sample gates, D+5/D+7 freshness rules, 95th/99th
calibration quantiles, protected MAPE, and three-distinct-date persistence for
statistical alerts. It also carries the hard-quality tolerance and optional
fully-qualified warning/critical threshold overrides, which are resolved into
the immutable calibration rather than read dynamically by reports. Calibration
uses historical pseudo-windows, both global and exact month/day seasonal
references, normalized Wasserstein distance, and the KS statistic. Performance
thresholds use only the sealed v2 test predictions. Wind direction is evaluated
through its sine/cosine pairs so the 0°/360° boundary cannot create a scalar
discontinuity.

Create the immutable reference and calibration without training or network
access:

```powershell
.\venv\Scripts\python.exe .\scripts\calibrate_monitoring.py `
  --model-bundle outputs\training\v2_reference_mlflow
```

Then generate a report for one explicit Phase 8 batch manifest:

```powershell
.\venv\Scripts\python.exe .\scripts\run_monitoring_report.py `
  --source-run-manifest data\processed\v2\incremental_update\runs\<RUN_ID>\manifest.json `
  --monitoring-store-root data\processed\v2\monitoring `
  --calibration-dir data\processed\v2\monitoring\reporting\calibrations\<CALIBRATION_ID> `
  --through-date YYYY-MM-DD `
  --dry-run
```

Reports are immutable JSON plus Markdown. Primary performance always uses the
`as_issued` view; restatements remain diagnostic. MAE, RMSE, signed bias,
protected MAPE, and sample-gated R² are reported. Hard contract violations and
D+7 source lateness open local alerts immediately; drift/performance breaches
require three distinct reporting dates. Same-date reruns do not increment the
counter. Alert delivery is a local append-only event record; external delivery
belongs to a later orchestration decision. Public loaders verify active alerts
and return their causally ordered immutable history; a report date older than
the derived alert state is rejected.

The local acceptance calibration completed against the accepted v2 artifacts
with reference ID
`3a1f8a357136bf89dfa1248906486bf726fbfd7bc7dbe4af2f41f347808794c7`
and calibration ID
`ff56dd507607a95aea81f76ab6ce694f1fd8eb51a97175f834bdb83c16b2fe58`.
It evaluated 691 valid 30-day and 690 valid 90-day feature backtest windows.
All 56 model inputs are covered by 50 alert entities because the ten
wind-direction sine/cosine components are evaluated as five circular pairs and
the raw degree column is represented by the current-direction pair. The
protected-MAPE epsilon resolved to `18573.143` on the unchanged legacy target
scale. The model checksum remained
`9d2bf8ed179b6720c3736de0fa674492909e96bffb87e6bb0ce3868e193e3041`.

## Monitoring Scope And Activation Gates

Monitoring is activated in layers:

1. Source freshness, completeness, validation status, late dates, and explicit
   exclusions may start once the scheduled ingestion and persistence path is
   implemented.
2. Feature drift is active only with the verified, immutable v2 monitoring
   reference and calibrated threshold artifact.
3. Model-performance monitoring is active only with the accepted unpromoted v2
   model, temporal cutoff, sealed-test baseline, and immutable Stage 1 evidence.

The current v1 scalers, models, and metrics are not valid for v2. Stage 2 does
not promote an artifact, alter predictions, or authorize ex-ante forecasting
claims.

## Stage 3 Implementation — Read-Only API And Dashboard

`wind_forecast.monitoring_projection` reads the configured
`WIND_FORECAST_MONITORING_STORE_ROOT` (default
`data/processed/v2/monitoring`) through identity-, checksum-, and alert-chain
verified loaders. It exposes sanitized projections through:

- `GET /api/v1/monitoring/latest`;
- `GET /api/v1/monitoring/history`;
- `GET /api/v1/monitoring/runs/{run_id}`.

No reports or runs is a valid connected empty state. Unknown runs return
`404`, invalid pagination returns `422`, and corrupt evidence returns a
sanitized `503`. Calibration references remain relocatable after a store is
moved between hosts: if the recorded absolute path no longer exists, the
loader verifies `reporting/references/{reference_id}` instead.

The React dashboard opens on a `Monitoring` view and retains
`Historical performance` as a separate view backed by the unchanged
`/api/v1/performance` contract. Monitoring shows D+5/D+7 source freshness,
the verified unpromoted model snapshot, source/report run status, 30/90-day
as-issued performance against sealed-test v2 thresholds, top feature drift,
and local alert/run history. It refreshes only on view entry or explicit user
action and is permanently labelled “retrospective historical batch
monitoring — not real time.”

## Rejected Alternatives

| Alternative | Decision and reason |
| --- | --- |
| Operational D+1 | `NO-GO`; ERA5-Land is reanalysis and no versioned forecast-weather source or future-feature contract exists |
| Multi-day forecast | `NO-GO` for the same reason and because no horizon-specific target contract exists |
| Calling the output a same-day forecast | Rejected; the complete target day and delayed reanalysis precede issuance |
| Relabelling the legacy target as MWh | Rejected; it would silently change the target semantics and artifact contract |
| Contiguous all-history watermark | Rejected; known unavailable dates would freeze later processing |
| Automatic exclusion at D+7 | Rejected; D+7 creates an alert and continued retries, not silent data loss |
| Overwriting revised results | Rejected; auditability requires append-only revisions |

## Risks And Controls

| Risk | Control |
| --- | --- |
| ERA5-Land arrives later than the internal objective | D+7 alert, continued catch-up, and measured availability evidence before changing the policy |
| REN observations are revised | Immutable checksummed snapshots and revisioned metrics |
| Legacy target is mistaken for energy | Explicit non-physical scale and prohibition on MW/MWh labels |
| DST changes the number of observations in the legacy target | Treat the 92/96/100-sample effect as part of the frozen scale and account for it when approving drift thresholds and performance baselines |
| Target leakage in a post-factum batch | Persist target-free model inputs and estimate before the actual join |
| Historical gaps stall processing | Per-date eligibility and explicit gap states |
| DST or timezone errors change daily membership | `Europe/Lisbon` day boundaries and the accepted 92/96/100 and 23/24/25 rules |
| Ledger is mistaken for complete monitoring | Project status and this record distinguish evidence persistence from drift, reporting, alerts, and orchestration |

Rollback of this documentation decision means marking this contract
`Superseded` in a later reviewed decision. It does not mean editing historical
data or silently returning to an ambiguous operating mode.

## Acceptance Checklist

- [x] Historical batch mode is selected and distinguished from forecasting.
- [x] Target date, timezone, DST behavior, and absence of forecast lead time are explicit.
- [x] Target formula, reported scale, physical-unit limitation, and missing-data policy are explicit.
- [x] Schedule, nominal objective, late deadline, and retry behavior are explicit.
- [x] Source watermarks, common watermark, and per-date eligibility have distinct definitions.
- [x] Source-freshness and estimate-lateness alerts have distinct activation rules.
- [x] Actual availability, provider finality, revisions, and current-view behavior are explicit.
- [x] Minimum immutable record identity, target-free inputs, transformation lineage, and leakage control are explicit.
- [x] Monitoring layers and their activation gates are explicit.
- [x] D+1 and multi-day operation remain `NO-GO` without forecast weather.
- [x] Stage 0 itself approved no code, data, model, scaler, pipeline, notebook, or generated artifact change.
- [x] Stage 1 persists target-free predictions before actuals and metrics.
- [x] Activation/backfill, D+5, idempotency, revision, restatement, corruption, and replay behavior have synthetic offline tests.
- [x] Feature, prediction, and target drift use calibrated 30/90-day global and seasonal comparisons.
- [x] Performance reports MAE, RMSE, bias, sample-gated R², and protected MAPE from immutable as-issued evidence.
- [x] Phase 8 quality sidecars cover succeeded, no-op, and failed batch attempts.
- [x] Reports and local alert transitions are append-only; statistical alerts require persistence.
- [x] Controlled tests cover no-drift/drift, circular direction, quality failures, metrics, persistence, dry-run, and corruption boundaries.
- [x] A read-only API/dashboard exposes verified monitoring evidence without changing `/api/v1/performance`.
- [x] Loading, empty, error, delayed, partial-history, refresh, cancellation, pagination, and run-detail paths have offline tests.
- [x] Registry serving, orchestration, retraining, provider calls, real-time claims, and external alert delivery remain out of scope.

## External Primary References

- [ECMWF ERA5-Land hourly time-series Product User Guide](https://confluence.ecmwf.int/spaces/CKB/pages/536218894/ERA5-Land%2Bhourly%2Btime-series%2Bdata%2Bon%2Bsingle%2Blevels%2Bfrom%2B1950%2Bto%2Bpresent%2BProduct%2BUser%2BGuide%2BPUG) — accessed `2026-07-20`.
- [REN Data Hub](https://datahub.ren.pt/pt/) — accessed `2026-07-20`.

## Stop Gate

Phase 9 and its separately approved read-only projection stop at local
append-only evidence, sanitized API reads, and retrospective visualization.
Phase 10 orchestration, external notifications, forecast-weather selection,
D+1 forecasting, target migration, dataset regeneration, scaler fitting,
model training, write APIs, and baseline or Registry promotion require
separate approved work.
