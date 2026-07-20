# Phase 9 — Historical Batch Monitoring Contract

## Decision Record

| Field | Value |
| --- | --- |
| Decision | Phase 9 Stage 0 temporal and operational contract |
| Status | `Accepted` |
| Decision date | `2026-07-20` |
| Contract version | `historical_batch_monitoring_v1` |
| Operating mode | Historical batch monitoring with delayed REN and ERA5-Land data |
| Current implementation status | Not started |

## Purpose And Scope

This record defines exactly what the first monitoring workflow will estimate,
when it will run, when its inputs and actual value are considered available,
and how delayed or revised records will be handled.

The accepted mode is a delayed historical batch over the existing REN and
ERA5-Land v2 contracts. Its output is a retrospective estimate, or hindcast,
for a completed local calendar day. It is not an ex-ante forecast and must not
be presented as real-time, same-day, D+1, or multi-day forecasting.

This Stage 0 decision is documentation only. It does not implement monitoring,
orchestration, ingestion schedules, alerting, prediction persistence, drift
statistics, or performance reports. It does not modify data, feature tables,
models, scalers, baselines, APIs, notebooks, or generated artifacts.

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

Current builders and their explicit overwrite options do not satisfy this
append-only operational contract. A later implementation must introduce
versioned run or revision storage without changing the safety of existing
manual scripts.

## Monitoring Scope And Activation Gates

Monitoring is activated in layers:

1. Source freshness, completeness, validation status, late dates, and explicit
   exclusions may start once the scheduled ingestion and persistence path is
   implemented.
2. Feature drift may start only after a v2 reference dataset, reference period,
   exact feature schema, and drift thresholds are separately approved.
3. Model-performance monitoring may start only after a v2 scaler, model,
   temporal training cutoff, evaluation baseline, and immutable estimate path
   are approved and validated.

The current v1 scalers, models, and metrics are not valid for v2. Stage 0 does
not promote any existing artifact and does not authorize prediction or
performance claims.

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
| Documentation is mistaken for an implemented capability | Project status and this record continue to state that monitoring is not implemented |

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
- [x] No code, data, model, scaler, pipeline, notebook, or generated artifact change is approved here.

## External Primary References

- [ECMWF ERA5-Land hourly time-series Product User Guide](https://confluence.ecmwf.int/spaces/CKB/pages/536218894/ERA5-Land%2Bhourly%2Btime-series%2Bdata%2Bon%2Bsingle%2Blevels%2Bfrom%2B1950%2Bto%2Bpresent%2BProduct%2BUser%2BGuide%2BPUG) — accessed `2026-07-20`.
- [REN Data Hub](https://datahub.ren.pt/pt/) — accessed `2026-07-20`.

## Stop Gate

Phase 9 Stage 0 is accepted as a temporal and operational architecture
contract. Monitoring implementation has not started. Phase 9 implementation,
Phase 10 orchestration, forecast-weather selection, D+1 forecasting, target
migration, dataset regeneration, scaler fitting, model training, and baseline
promotion require separate approved work.
