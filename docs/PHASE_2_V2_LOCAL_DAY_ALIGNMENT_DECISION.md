# Phase 2 V2 Local-Day Alignment Decision

## Purpose And Scope

This document records the Phase 2 REN and ERA5-Land canonical daily alignment decision for the integrated v2 dataset. It is a documentation-only checkpoint for local-day derivation, join semantics, and Step 2A.17 validation requirements.

No dataset build is performed by this decision. Step 2B, feature regeneration, scaler refitting, model training, metric re-baselining, downstream training work, and Phase 3 remain paused.

## Decision Summary

The canonical daily key for the integrated v2 dataset is the civil calendar date in `Europe/Lisbon`.

REN production records are interpreted on timezone-aware `Europe/Lisbon` timestamps. ERA5-Land accepted hourly UTC source data remains unchanged, but the integrated-v2 derivation converts those UTC hours to `Europe/Lisbon` and recomputes daily weather aggregates over local-day boundaries.

Integration joins use the canonical `Europe/Lisbon` local daily key only. REN local dates must not be aligned with existing ERA5 UTC aggregate labels.

## Relationship To Existing Phase 2 Decisions

This is a later and narrower decision than the existing Phase 2 v2 data-contract record. It clarifies the integrated daily derivation while preserving UTC source storage and accepted UTC weather evidence.

The older UTC wording remains valid for ERA5-Land source storage and previously accepted UTC aggregate evidence. This decision does not modify that evidence; it defines how the integrated v2 daily dataset must derive local-day weather aggregates before joining with REN production.

## Considered Alternatives

Rejected alternatives:

- Use UTC dates as the integrated v2 daily key.
- Join REN local dates directly to existing ERA5 UTC aggregate labels.
- Use REN `source_date` as a substitute for timestamp-derived local dates.
- Use silent inner joins, silent drop behavior, interpolation, or forward fill (`ffill`) to handle coverage gaps.
- Modify accepted UTC ERA5-Land daily aggregate evidence instead of deriving separate local-day integrated aggregates.

## Canonical Daily Key

The canonical daily key is the civil calendar date in `Europe/Lisbon`.

For every integrated v2 row, the key represents the complete local day from local midnight to the next local midnight in `Europe/Lisbon`, including daylight-saving transitions.

## REN Timestamp And Source-Date Policy

REN timestamps must be parsed and preserved as timezone-aware `Europe/Lisbon` timestamps.

The REN daily key derives from the timestamp, not from `source_date`. `source_date` is an integrity check only. Every REN record assigned to a local daily key must have a matching `source_date`; otherwise validation must fail.

## REN DST Interval Expectations

REN interval counts are evaluated against physical `Europe/Lisbon` local-day expectations:

- Ordinary local day: `96` 15-minute intervals.
- Spring DST local day: `92` 15-minute intervals.
- Autumn DST local day: `100` 15-minute intervals.

Repeated local wall-clock intervals on autumn DST days are expected physical behavior and must be preserved through timezone-aware timestamp handling.

## Production Aggregation And Unit Warning

The integrated v2 production derivation must preserve the v1 recovered production aggregation formula. This decision does not replace that formula or introduce a new production target definition.

Unit warning: a simple sum of 15-minute `MW` observations is not `MWh`. Energy in `MWh` requires multiplying each 15-minute `MW` observation by `0.25h` before summing.

## ERA5-Land UTC-To-Local Policy

Accepted hourly ERA5-Land UTC source data remains unchanged.

For the integrated v2 dataset, ERA5-Land hourly UTC timestamps must be converted to `Europe/Lisbon` before daily weather aggregation. Daily weather aggregates for integration must then be recomputed over `Europe/Lisbon` local-day boundaries.

Accepted UTC daily aggregate evidence must not be modified or reinterpreted as local-day evidence.

## ERA5-Land DST And Month-Boundary Requirements

ERA5-Land hourly counts are evaluated against `Europe/Lisbon` local-day expectations after UTC-to-local conversion:

- Ordinary local day: `24` hourly observations.
- Spring DST local day: `23` hourly observations.
- Autumn DST local day: `25` hourly observations.

Month-boundary processing must include adjacent UTC hours needed to produce complete `Europe/Lisbon` local days. A local day must not be marked complete merely because the UTC calendar month partition is complete.

## Integration Join And Coverage Policy

Integration joins on the canonical `Europe/Lisbon` local daily key.

The integrated v2 build must not:

- Align REN local dates with existing ERA5 UTC aggregate labels.
- Interpolate missing values.
- Forward fill (`ffill`) missing values.
- Use a silent inner join.
- Silently drop requested local dates.

There must be an explicit coverage record for every requested local calendar date. Missing REN, missing ERA5-Land, invalid, incomplete, and excluded dates must be recorded separately.

The final requested date, `2026-06-27`, must be assessed using complete `Europe/Lisbon` local-day coverage.

## Validation Requirements For Step 2A.17

Step 2A.17 validation must confirm:

- The canonical integrated daily key is the `Europe/Lisbon` civil calendar date.
- REN timestamps are timezone-aware `Europe/Lisbon` values.
- REN timestamp-derived local dates match `source_date` for every assigned record.
- REN local-day interval counts match `96`, `92`, or `100` as appropriate.
- ERA5-Land hourly UTC source data remains unchanged.
- Integrated ERA5-Land daily aggregates are recomputed after UTC-to-`Europe/Lisbon` conversion.
- ERA5-Land local-day hourly counts match `24`, `23`, or `25` as appropriate.
- Month-boundary processing includes adjacent UTC hours needed for complete local days.
- Integration coverage records every requested local date and separates missing REN, missing ERA5-Land, invalid, incomplete, and excluded statuses.
- Joins do not use interpolation, forward fill, silent inner joins, or silent drops.
- The requested end date `2026-06-27` has complete `Europe/Lisbon` local-day coverage before being accepted.

## Implications For Integrated V2 Dataset Build

A future integrated v2 dataset build must derive both production and weather daily rows on the same `Europe/Lisbon` civil calendar key before joining.

This decision may require a local-day weather derivative separate from accepted UTC aggregate outputs. It does not invalidate accepted UTC source storage, accepted UTC source-data checks, or prior UTC aggregate evidence.

V2 modelling remains a separate future step. Existing v1 scalers, models, datasets, and baselines are not claimed valid for the integrated v2 dataset.

## Non-Goals

This decision does not:

- Build the integrated v2 dataset.
- Modify REN raw, normalized, status, manifest, or generated data.
- Modify ERA5-Land raw, hourly, daily-point, daily-aggregate, manifest, or generated data.
- Modify code, scripts, notebooks, dependencies, configuration, existing docs, roadmap, or prior decision records.
- Run network requests, ingestion scripts, notebooks, training, or pipelines.
- Refit scalers, retrain models, re-baseline metrics, start Step 2B, or start Phase 3.

## Acceptance Criteria

- The canonical daily key is documented as the civil calendar date in `Europe/Lisbon`.
- REN timezone-aware timestamp parsing, `source_date` integrity checking, DST interval expectations, and production-unit warning are documented.
- ERA5-Land UTC source preservation, UTC-to-local integrated derivation, local-day hourly expectations, and month-boundary requirements are documented.
- Integration join and coverage rules are documented, including explicit per-date coverage records and separate status categories.
- The relationship to older UTC wording is clarified without modifying accepted UTC evidence.
- The document states that no dataset build is performed and that Step 2B, training work, and Phase 3 remain paused.
