# Phase 2 V2 Data Contract Decision

## Purpose And Scope

This document records the Phase 2 Checkpoint 4 v2 data-contract decision for the wind-energy forecasting project. It selects the v2 production and weather source stack, aggregation rules, versioning policy, validation requirements, and model-compatibility consequences before any v2 ingestion or model work begins.

This is a documentation-only decision record. It does not modify v1 data, create v2 data, execute notebooks, run live CDS or REN network calls, regenerate features, refit scalers, retrain models, re-baseline metrics, start Step 2B, or start Phase 3.

Coverage checklist: production source; ERA5-Land product; spatial strategy; temporal resolution; wind-speed aggregation; wind-direction aggregation; temperature aggregation; timezone; versioned paths; manifest; retrieval; revision; validators; scaler refitting; model retraining; metric re-baselining.

## Evidence Base

Local evidence used:

| Evidence | Use in this decision |
| --- | --- |
| `docs/ML_ENGINEERING_ROADMAP.md` | Phase 2 objective, validation focus, and requirement to preserve existing modelling workflow. |
| `docs/PHASE_2_DATA_REFRESH_ASSESSMENT.md` | v1 coverage, current contracts, source-candidate comparison, versioning proposal, validator reuse, and retraining impact. |
| `docs/PHASE_2_SOURCE_PROBE_FINDINGS.md` | REN endpoint probe, overlap comparison, recent-data availability, IPMA station mapping, and direct-append rejection. |
| `docs/PHASE_2_ERA5_LAND_V1_COMPARISON.md` | ERA5-Land pilot formulas, multi-point comparison, calm threshold, distribution-shift evidence, and model/scaler implications. |

Evidence classification used below:

- Confirmed fact: directly supported by repository evidence or completed local probe/comparison documentation.
- Engineering judgment: selected policy or design choice based on confirmed evidence and project constraints.
- Unresolved assumption: material item not yet verified and therefore required before production ingestion or daily aggregation.

## Baseline Preservation Decision

Decision: preserve v1 as the immutable reproducible baseline.

Classification: engineering judgment supported by confirmed facts.

Rationale:

- The roadmap requires the existing modelling workflow to remain usable.
- `docs/PHASE_2_DATA_REFRESH_ASSESSMENT.md` records the current v1 raw, processed, model, and scaler contracts and states that current scalers and models must not be claimed valid for a materially changed v2 weather source.
- `docs/PHASE_2_SOURCE_PROBE_FINDINGS.md` records a later REN overlap difference on `2025-01-25`, which means v2 production must not be silently appended to the frozen v1 snapshot.

Implication: v2 work must use separate raw, processed, manifest, model, scaler, and metric lineage. V1 raw data, processed files, scalers, trained models, baselines, and existing validators remain v1-only unless an explicit future migration step authorizes otherwise.

## Decision Summary

| Topic | Decision | Classification |
| --- | --- | --- |
| production source | Select REN `ElectricityProductionBreakdownDaily` for v2 production reconstruction. Reject silent append to v1. | Engineering judgment based on confirmed probe evidence. |
| ERA5-Land product | Select CDS `reanalysis-era5-land`. Reject unverified daily-statistics shortcuts for now. | Engineering judgment based on pilot evidence and assessment. |
| spatial strategy | Select point extraction at the 17 mapped IPMA station coordinates. Reject guessing unmatched `1200579`, one arbitrary coordinate, full grid, and capacity weighting for this decision. | Engineering judgment based on confirmed mapping evidence. |
| temporal resolution | Store 15-minute REN raw, hourly ERA5-Land raw, and daily canonical model base/features. | Engineering judgment based on source cadence and model contract. |
| wind-speed aggregation | Compute hourly `sqrt(u10^2 + v10^2)`, daily scalar mean per point, then equal-weight mean across valid points. | Engineering judgment using confirmed pilot formula. |
| wind-direction aggregation | Aggregate from `u10` and `v10` vectors; calculate meteorological wind-from direction; use an explicit calm/null threshold of `0.5 m/s`. | Engineering judgment using confirmed pilot formula and calm evidence. |
| temperature aggregation | Convert Kelvin to Celsius; daily mean per point; equal-weight mean across valid points. | Engineering judgment using confirmed pilot formula. |
| timezone | Use UTC as canonical for v2 weather, storage, and aggregation. REN source timezone semantics remain unresolved and must be verified before production daily aggregation. | Engineering judgment plus unresolved assumption. |
| versioned paths | Use separate v2 paths: `data/raw/v2/`, `data/processed/v2/`, and `data/manifests/historical_v2.json`. Do not overwrite or move v1 files. | Engineering judgment based on repository versioning proposal. |
| manifest | Require a v2 manifest with source, coverage, unit, formula, checksum, validation, attribution, revision, and compatibility metadata. | Engineering judgment based on assessment requirements. |
| retrieval and revision policy | Preserve raw responses, checksums, retrieval timestamps, immutable snapshots or partitions, overlap comparisons, and explicit v2 revisions when values change. Never silently mutate v1. | Engineering judgment based on revision evidence. |
| required validators | Require existing production, daily, and merged validators plus v2-specific REN, ERA5-Land, spatial, aggregation, manifest, and alignment validators. | Engineering judgment based on validator reuse assessment. |
| scaler refitting | Required for v2. V1 scalers remain v1-only. | Confirmed compatibility implication. |
| model retraining | Required for v2. V1 trained models remain v1-only. | Confirmed compatibility implication. |
| metric re-baselining | Required for v2 before comparing or promoting v2 models. | Engineering judgment based on distribution-shift evidence. |

## Production Source

Selected production source: REN `ElectricityProductionBreakdownDaily`.

Classification: engineering judgment based on confirmed facts.

Confirmed facts from `docs/PHASE_2_SOURCE_PROBE_FINDINGS.md`:

- The probed endpoint identifier is `REN ElectricityProductionBreakdownDaily`.
- The tested request pattern used one date per request with `culture=pt-PT` and `date=YYYY-MM-DD`.
- Tested complete dates returned 96 records at inferred 15-minute cadence.
- The wind series is `Eolica` in the source evidence and the response unit is `MW`.
- On `2010-01-01`, the REN overlap comparison aligned 96 timestamps and found 96 exact matches.
- On `2025-01-25`, the comparison aligned 96 timestamps but found nonzero differences, with high correlation and nonzero mean absolute difference.
- The recent date `2026-06-27` returned 96 records and is beyond current v1 production coverage.

Decision details:

- Select REN for v2 production because repository probes show strong target compatibility with the current wind-production target.
- Treat compatibility as strong but not complete. The exact `2010-01-01` match supports target compatibility; the `2025-01-25` differences prove overlapping values can differ.
- Reject silent append to v1.
- Require full v2 production reconstruction or explicitly partitioned v2 revision tracking instead of mutating the v1 production file.

Unresolved assumptions:

- REN timezone semantics for the endpoint must be verified before production daily aggregation.
- Source license, attribution, oldest comparable date, and provisional/final status behavior must be recorded before full ingestion.
- The cause of the `2025-01-25` overlap difference remains unresolved.

## ERA5-Land Product

Selected ERA5-Land product: CDS `reanalysis-era5-land`.

Classification: engineering judgment based on confirmed pilot evidence.

Confirmed facts from `docs/PHASE_2_ERA5_LAND_V1_COMPARISON.md`:

- The pilot used ERA5-Land hourly variables `2m_temperature`, `10m_u_component_of_wind`, and `10m_v_component_of_wind`.
- Pilot validation passed for the existing multi-point pilot outputs.
- ERA5-Land can be transformed into station-day comparison rows aligned with v1 station IDs and dates for the sampled stations.
- Temperature aligned strongly in the pilot, while wind speed and wind direction showed material distribution differences.

Decision details:

- Select CDS `reanalysis-era5-land` as the v2 weather product.
- Use hourly component fields for canonical v2 derivations.
- Reject unverified daily-statistics shortcuts for this decision. Daily-statistics products or pre-aggregated shortcuts may be evaluated later only after equivalence checks against hourly-derived aggregation.

Non-goals:

- This decision does not authorize a full CDS download.
- This decision does not create a production ERA5-Land ingestion implementation.
- This decision does not claim ERA5-Land is distribution-compatible with v1 scalers or models.

## Spatial Strategy

Selected spatial strategy: point extraction at the 17 mapped IPMA station coordinates.

Classification: engineering judgment based on confirmed facts.

Confirmed facts from `docs/PHASE_2_SOURCE_PROBE_FINDINGS.md`:

- Current IPMA metadata matched 17 of the 18 v1 station identifiers exactly.
- There were no ambiguous matches.
- Station identifier `1200579` remains unmatched.
- Coordinates are available for the 17 matched identifiers.

Decision details:

- Use the 17 mapped station coordinates for v2 ERA5-Land point extraction.
- Exclude unmatched `1200579` unless a future evidence-backed mapping resolves it.
- Equal-weight across valid mapped points for canonical daily aggregate weather features.

Rejected alternatives:

- Guessing the unmatched `1200579` coordinate is rejected.
- A single arbitrary coordinate is rejected as a national-weather representation.
- Full-grid extraction is rejected for this decision because point extraction is sufficient for the v2 contract and better aligned with v1 station comparability.
- Installed-capacity weighting is rejected for this decision because reliable wind-farm capacity and geolocation lineage have not been approved for the current checkpoint.

Unresolved assumptions:

- Historical station continuity, moves, openings, closures, and metadata changes remain unresolved.
- The 17-point strategy is selected as the canonical v2 starting contract, not as proof of exact equivalence to the v1 18-station weather matrices.

## Temporal Resolution

Selected temporal resolution:

- REN raw production: 15-minute source records.
- ERA5-Land raw weather: hourly source records.
- Canonical model base/features: daily records.

Classification: engineering judgment based on confirmed facts.

Rationale:

- `docs/PHASE_2_SOURCE_PROBE_FINDINGS.md` records REN daily responses with 96 records on tested complete days, implying 15-minute cadence.
- `docs/PHASE_2_ERA5_LAND_V1_COMPARISON.md` records hourly ERA5-Land pilot rows and daily point and aggregate outputs.
- The current v1 processed training base is daily, and v2 model features must be daily before any scaler or model work.

Unresolved assumption:

- REN source timezone semantics must be resolved before final daily production aggregation rules are used in production.

## Weather Aggregation Rules

### Wind-Speed Aggregation

Selected wind-speed aggregation:

1. Compute hourly point wind speed as `sqrt(u10^2 + v10^2)`.
2. Compute daily scalar mean wind speed per point from hourly speeds.
3. Compute daily aggregate wind speed as an equal-weight mean across valid mapped points.

Classification: engineering judgment using confirmed pilot formula.

### Wind-Direction Aggregation

Selected wind-direction aggregation:

1. Use `u10` and `v10` components as the source of direction.
2. Aggregate direction through vector components, not through scalar degree averaging.
3. Convert resulting components to meteorological wind-from direction using `(180 + degrees(atan2(u10, v10))) % 360`.
4. Apply an explicit calm/null threshold to vector-mean speed before emitting direction.
5. Set the initial calm/null threshold to `0.5 m/s`.

Classification: engineering judgment using confirmed pilot formula and evidence.

Confirmed fact from `docs/PHASE_2_ERA5_LAND_V1_COMPARISON.md`: one pilot direction value was intentionally null because the vector-mean speed was `0.414 m/s`, below the documented `0.5 m/s` calm threshold, even though daily scalar mean speed was nonzero.

Implication: v2 validators and downstream feature generation must allow documented nullability for calm or near-calm direction rather than imputing silently.

### Temperature Aggregation

Selected temperature aggregation:

1. Convert ERA5-Land `2m_temperature` from Kelvin to Celsius.
2. Compute daily mean temperature per point.
3. Compute daily aggregate temperature as an equal-weight mean across valid mapped points.

Classification: engineering judgment using confirmed pilot formula.

## Timezone

Selected timezone policy: UTC canonical for v2 weather, storage, and aggregation.

Classification: engineering judgment with one unresolved assumption.

Decision details:

- Store v2 ERA5-Land weather timestamps in UTC.
- Store canonical v2 aggregation boundaries in UTC unless a future verified production-source rule requires an explicitly documented conversion.
- Record timezone, source timezone semantics, and aggregation boundary rules in the manifest.

Unresolved assumption:

- REN source timezone semantics remain unresolved and must be verified before production daily aggregation. This is a blocking requirement for full production ingestion, because local-day versus UTC-day boundaries can change daily production targets.

## Versioned Paths

Selected versioned paths:

```text
data/raw/v2/
data/processed/v2/
data/manifests/historical_v2.json
```

Classification: engineering judgment based on the versioning proposal in `docs/PHASE_2_DATA_REFRESH_ASSESSMENT.md`.

Rules:

- Do not overwrite, move, or reinterpret current v1 files.
- Do not place v2 raw or processed data in current root-level v1-compatible paths.
- Keep v2 raw snapshots, processed features, manifests, scalers, models, and metrics separately identifiable.
- Any future v1 path migration must be a separate approved task with compatibility paths and rollback behavior.

## Manifest Requirements

Required manifest contents for `data/manifests/historical_v2.json`:

- Dataset version.
- Production and weather sources, providers, URLs, endpoints, and request parameters.
- Retrieval timestamps.
- Coverage start and end by source and by canonical merged dataset.
- Timezone and aggregation-boundary policy.
- Raw and canonical resolutions.
- Units.
- Source variables.
- Coordinates or grid cells used for each mapped station point.
- Wind-speed, wind-direction, temperature, and production aggregation formulas.
- Calm/null threshold for wind direction.
- Raw response checksums and processed-file checksums.
- Row counts and coverage counts.
- Validation results and warnings.
- Software versions and transformation version.
- License and attribution notes.
- Revision notes and overlap-comparison summaries.
- Model, scaler, and metric compatibility flags.

Classification: engineering judgment based on current manifest proposal and v2 compatibility requirements.

## Retrieval And Revision Policy

Selected retrieval and revision policy:

- Preserve raw source responses or raw source files.
- Record checksums and retrieval timestamps for every raw snapshot or partition.
- Use immutable snapshots or immutable date partitions for v2 source captures.
- Compare overlaps against prior v2 snapshots and frozen v1 where applicable.
- Create a new v2 revision when source values change.
- Never silently mutate v1.
- Never silently mutate prior v2 snapshots without revision metadata.

Classification: engineering judgment based on confirmed revision evidence.

Rationale:

- `docs/PHASE_2_SOURCE_PROBE_FINDINGS.md` found exact agreement on `2010-01-01` but nonzero overlap differences on `2025-01-25`.
- The v2 data contract must therefore treat source values as versioned observations, not as an append-only extension of v1.

## Required Validators

Required validators for v2:

- Existing raw production validator where the v2 parser yields comparable timestamp and target fields.
- Existing daily production validator after canonical daily production is produced.
- Existing merged base validator after canonical v2 weather and production are joined.
- V2 REN response validation for endpoint shape, series presence, cadence, unit, row counts, timestamps, missing values, duplicates, and target extraction.
- ERA5-Land hourly completeness validation for expected hours, required variables, nulls, units, bounds, and timestamp monotonicity.
- ERA5-Land physical bounds validation for temperature, wind components, wind speed, and direction.
- Spatial coverage validation for the 17 mapped coordinates, station identifiers, and missing or invalid points.
- Aggregation validation for daily counts, formulas, calm-threshold handling, equal-weight point means, and null propagation.
- Manifest checksum validation for raw and processed files.
- Production-weather alignment validation for canonical daily date coverage, timezone policy, and join completeness.

Classification: engineering judgment based on validator reuse assessment and pilot findings.

Compatibility rule: v2 validators must extend current validation coverage without weakening v1 validators or changing v1 pass/fail behavior.

## Model And Metric Compatibility

Scaler refitting decision: required for v2.

Model retraining decision: required for v2.

Metric re-baselining decision: required for v2.

Classification: confirmed compatibility implication for scaler refitting and model retraining; engineering judgment for metric re-baselining based on confirmed distribution shift.

Confirmed facts from `docs/PHASE_2_ERA5_LAND_V1_COMPARISON.md`:

- ERA5-Land wind speed was lower than v1 by `0.522 m/s` on average in the sampled aggregate comparison.
- Wind direction required explicit circular handling and showed material differences.
- The report concludes that current v1 scalers and trained models must not be claimed valid for ERA5-Land-derived weather.

Decision details:

- V1 scalers remain v1-only.
- V1 trained models remain v1-only.
- V1 baseline metrics remain v1-only.
- V2 processed features require v2 scaler fitting before model training or inference.
- V2 models must be retrained against v2 feature distributions.
- V2 metrics must be re-baselined before comparing performance, promoting a model, or using v2 artifacts operationally.

## Explicit Non-Goals

This decision record does not authorize or perform:

- Live CDS, REN, IPMA, WeatherAPI, DGEG, ENTSO-E, or other network calls.
- Full ingestion.
- Bulk downloads.
- Step 2B work.
- Phase 3 work.
- Notebook execution.
- Training runs.
- Pipeline runs.
- Output regeneration.
- Feature regeneration.
- Scaler refitting.
- Model retraining.
- Metric re-baselining.
- Moving, overwriting, or deleting v1 data, models, scalers, predictions, reports, or notebooks.
- Creating v2 raw data, processed data, manifests, models, or metrics.

## Acceptance Criteria For Future Implementation

A future implementation of this v2 contract must:

- Preserve all v1 artifacts and current v1 paths unless a separate approved migration task changes that.
- Retrieve REN and ERA5-Land into versioned v2 raw paths.
- Record raw checksums, retrieval timestamps, source parameters, coverage, units, and revision notes.
- Apply the selected 17-point spatial strategy without guessing `1200579`.
- Apply UTC canonical weather and storage rules.
- Resolve and document REN timezone semantics before production daily aggregation.
- Use hourly ERA5-Land component-derived weather formulas.
- Apply the `0.5 m/s` calm/null threshold unless a later evidence-backed decision changes it.
- Produce daily canonical v2 base/features separately from v1.
- Run required v2 validators and record results in the manifest.
- Refit scalers, retrain models, and re-baseline metrics before any v2 model claim.

## Risks And Rollback

Risks:

- REN historical revision behavior may produce differences from frozen v1.
- REN timezone semantics may affect daily production aggregation.
- The unmatched station `1200579` may affect comparability with v1.
- ERA5-Land point extraction may introduce stable source and spatial biases.
- Wind-direction nullability may affect feature generation if downstream code assumes complete direction values.
- V2 features will invalidate v1 scaler, model, and metric comparability.

Rollback and containment:

- Keep v1 immutable and usable.
- Keep v2 under separate versioned paths.
- Keep raw snapshots and manifests sufficient to reproduce each v2 revision.
- Do not promote v2 artifacts without v2-specific validation, scaler refitting, model retraining, and metric re-baselining.

## Stop Gate

Checkpoint 4 status: decision record created for the approved documentation-only scope.

Step 2B remains paused. Full ingestion was not started. Feature regeneration was not started. Scaler refitting was not started. Model retraining was not started. Metric re-baselining was not started. Phase 3 was not started.
