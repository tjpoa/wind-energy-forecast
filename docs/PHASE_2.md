# Phase 2 - Data Validation And V2 Data Work

This file consolidates the Phase 2 records that were previously split across multiple documents. The goal is to keep one documentation file per roadmap phase while preserving the original audit, decision, readiness, and acceptance content.

## Contents

- [Phase 2 Data Refresh Assessment](#phase-2-data-refresh-assessment)
- [Phase 2 Source Probe Findings](#phase-2-source-probe-findings)
- [Phase 2 ERA5-Land And V1 Weather Comparison](#phase-2-era5land-and-v1-weather-comparison)
- [Phase 2 V2 Data Contract Decision](#phase-2-v2-data-contract-decision)
- [Phase 2 V2 Local-Day Alignment Decision](#phase-2-v2-localday-alignment-decision)
- [Phase 2 REN Backfill Readiness](#phase-2-ren-backfill-readiness)
- [Phase 2 REN Full Backfill Acceptance Audit](#phase-2-ren-full-backfill-acceptance-audit)
- [Phase 2 ERA5-Land Grid Readiness](#phase-2-era5land-grid-readiness)
- [Phase 2 ERA5-Land Full Backfill Acceptance](#phase-2-era5land-full-backfill-acceptance)
- [Phase 2 ERA5-Land Monthly-Bbox Full Backfill Acceptance](#phase-2-era5land-monthlybbox-full-backfill-acceptance)
- [Phase 2 Integrated V2 Daily Dataset Acceptance](#phase-2-integrated-v2-daily-dataset-acceptance)
- [Phase 2 Feature-Ready V2 Dataset Acceptance](#phase-2-featureready-v2-dataset-acceptance)
- [Phase 2 Feature-Ready V2 Validation Acceptance](#phase-2-featureready-v2-validation-acceptance)

---

## Phase 2 Data Refresh Assessment

Original file before consolidation: `PHASE_2_DATA_REFRESH_ASSESSMENT.md`.

## Purpose And Context

This document records the Phase 2 Step 2A.5 historical data refresh assessment. It evaluates whether the current historical production and weather datasets can be refreshed, extended, acquired reproducibly, and versioned without breaking the v1 baseline.

This is an assessment only. Step 2A remains complete, the current v1 datasets and validators remain unchanged, no v2 source has been selected, no data refresh has occurred, and Step 2B remains paused.

Evidence is separated as follows:

- Confirmed repository evidence: local files, notebooks, validators, processed data, and model artifacts.
- Confirmed official-source documentation: public documentation from REN Data Hub, WeatherAPI, IPMA, DGEG, ENTSO-E, and Copernicus CDS.
- Calculated estimates: request counts and paired-history estimates derived from current file dates.
- Unresolved assumptions: items requiring source-verification pilots.

Reference documentation:

- [REN Data Hub](https://datahub.ren.pt/)
- [REN Data Hub API](https://datahub.ren.pt/pt/api/)
- [WeatherAPI documentation](https://www.weatherapi.com/docs/)
- [IPMA public API](https://api.ipma.pt/)
- [DGEG energy statistics](https://www.dgeg.gov.pt/pt/estatistica/energia/)
- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- [Copernicus ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview)

## Current V1 Coverage And Contracts

Confirmed repository evidence:

| Asset | Coverage / Contract |
| --- | --- |
| `data/raw/ReparticaoProducao.csv` | 537,308 rows, 16 columns, 15-minute mode, `2010-01-01 00:00:00` to `2025-04-28 23:45:00` |
| Production target | Raw `Eolica` / canonical `Wind_Production`; metadata unit is MW |
| `data/raw/IntensidadeMediaVento10m.csv` | 4,017 rows, 18 station columns, `2013-01-01` to `2023-12-31` |
| `data/raw/DirecaoMediaVento10m.csv` | 4,017 rows, 18 station columns, `2013-01-01` to `2023-12-31` |
| `data/raw/TemperaturaMedia.csv` | 4,017 rows, 18 station columns, `2013-01-01` to `2023-12-31` |
| `data/processed/agg_data.csv` | 4,017 rows, 5 columns, `2013-01-01` to `2023-12-31` |
| `data/processed/agg_data_ml.csv` | 4,017 rows, 58 columns, `2013-01-01` to `2023-12-31` |
| Model input feature contract | 56 features from saved X scalers, using the legacy training feature names |

The usable paired v1 supervised-training period is constrained by the weather files, not by the production file. The v1 model and scaler artifacts depend on the current preprocessing choices: daily production aggregation, weather station imputation, station aggregation, wind-direction vector averaging, feature names, feature order, lag windows, rolling windows, schema mappings, and fitted scaler distributions.

Step 2A validators currently pass the v1 files with expected warnings only: duplicate/non-chronological/missing 15-minute raw production timestamps and missing station values in the weather matrices.

## Production-Source Evidence

Repository evidence strongly suggests REN as the production source, but does not prove it definitively. The strongest evidence is the `ReparticaoProducao.csv` filename, the metadata rows, the generation-mix columns, the `Eolica` target, 15-minute MW values, and the `Informacao acedida em` metadata style.

REN publicly presents electricity data and API access, and REN electricity values are publicly presented at 15-minute resolution. However, the exact historical export or API endpoint used to create `ReparticaoProducao.csv` is not yet confirmed.

Open production-source questions:

- Oldest comparable historical date available from the exact source.
- Latest downloadable date.
- License and attribution requirements.
- Whether the target definition exactly matches current `Eolica`.
- Whether data after `2025-04-28` can be appended without changing the target definition.
- Whether pre-`2010-01-01` data exists with comparable definition and granularity.
- Incremental-update method and endpoint stability.

DGEG is an official statistical source, but appears better suited to energy statistics than to reproducing the current 15-minute operational target. ENTSO-E is a plausible official alternative for generation by production type, but token access, market-area mapping, historical coverage, and target equivalence must be verified before use.

## Current Weather-Source Evidence

The weather files are daily station matrices with columns `ANO`, `MES`, `DIA`, and 18 numeric station identifiers. Repository files do not record provider, source URL, retrieval timestamp, license, station coordinates, station names, or station metadata.

IPMA's currently documented public observation API exposes recent station observations, including latest-24-hour observation resources. That API alone does not establish an accessible full historical archive for rebuilding the current 2013-2023 matrices. IPMA station metadata may still be useful for mapping the 18 numeric station IDs, but matching an ID does not by itself prove IPMA was the original data source.

Unresolved weather-source questions:

- Whether the same provider offers data after `2023-12-31`.
- Whether the same 18 station identifiers remain available.
- Whether stations were renamed, moved, opened, or closed.
- Whether equivalent historical files can be downloaded programmatically.
- Whether the current aggregation can be reproduced exactly.
- Whether data before `2013-01-01` is available.

## Candidate Source Comparison

| Option | Likely strengths | Main limitations | Retraining impact |
| --- | --- | --- | --- |
| A - Extend current weather source | Highest compatibility with v1 station matrices and preprocessing if the original source is recovered | Weak reproducibility until provider, station metadata, coverage, license, and programmatic access are confirmed | Low to medium if station contract is unchanged; still requires re-baselining after extension |
| B - Historical WeatherAPI | Strong consistency with current operational WeatherAPI inference path; advertised historical data from `2010-01-01` | Full historical access depends on subscription tier; account quotas and bulk/range capabilities must be checked; 18-location reconstruction may be expensive | Medium to high because source/spatial strategy differs from v1 station matrices |
| C - ERA5-Land | Leading candidate for reproducible historical weather; data from 1950 onward; suitable for point or grid extraction | Not yet final selected source; CDS auth, processing complexity, update delay, and distribution shift require pilot | High because weather source and fitted distributions change materially |

Calculated WeatherAPI estimate: from `2010-01-01` to `2026-06-28` is 6,023 calendar days. A one-request-per-day reconstruction for 18 locations would require about 108,414 requests. If account-specific range or bulk capabilities allow 30-day chunks, the same 18-location period would still require about 3,618 chunks. These are planning estimates, not confirmed account behavior.

For ERA5-Land, hourly `u10`, `v10`, and temperature should be used when necessary to reproduce the intended wind-direction aggregation accurately. Wind speed can be derived from `sqrt(u10^2 + v10^2)`. Wind direction should be derived from the vector components using a meteorological convention and then aggregated using vector or circular methods. Daily statistics products may be evaluated as an optimization, but must not be assumed equivalent before comparison.

## Supervised-History Limitation

Weather-only history before the production target begins does not increase supervised model-training rows. The longest usable supervised history is the intersection of production coverage and weather coverage after all source-specific validation and aggregation.

| Combination | Production-only history | Weather-only history | Estimated paired supervised history |
| --- | --- | --- | --- |
| Current v1 | `2010-01-01` to `2025-04-28` | `2013-01-01` to `2023-12-31` | `2013-01-01` to `2023-12-31` |
| REN plus current weather source | Depends on confirmed REN endpoint | Unknown until weather source is recovered | Unknown; potentially extends v1 if same weather source continues |
| Official production plus WeatherAPI | Potentially from `2010-01-01`, if source confirms | Advertised from `2010-01-01`, subject to subscription | Potentially `2010` to latest common date, subject to cost and source limits |
| Official production plus ERA5-Land | Potentially from `2010-01-01`, if source confirms | 1950 onward, subject to update delay | Potentially production-start to latest common date |

## Spatial Strategy

One arbitrary coordinate should not be used as the national-weather representation except for a low-cost smoke test. It has low representativeness and high train-serving skew risk.

Recommended pilot order:

1. Existing 18 station locations, if coordinates can be recovered, because this maximizes v1 comparability.
2. ERA5-Land point time-series extraction at the recovered station locations or a small representative coordinate set.
3. A regular mainland Portugal grid if station coordinates cannot be recovered or if reproducibility is prioritized over v1 similarity.
4. Installed-capacity-weighted wind-farm regions for the highest portfolio value, once reliable capacity and geolocation data are available.

For the ERA5-Land pilot, point time-series extraction should be preferred over downloading a full Portugal grid. Full-grid extraction can be evaluated later if the point strategy is not representative enough.

## Provisional Recommendation

Provisional recommendation: official Portuguese wind-production data, preferably REN if confirmed, plus ERA5-Land weather.

This is not a final source selection. The final decision depends on:

1. Production endpoint verification.
2. Station-ID and coordinate investigation.
3. Limited ERA5-Land pilot.
4. V1-versus-candidate distribution comparison.

This option is favored because it is likely to be reproducible, automatable, long-running historically, and less dependent on a commercial weather vendor. It also has the highest retraining impact and must be treated as a v2 dataset, not as a silent v1 replacement.

## Dataset-Versioning Proposal

Use a non-destructive structure that preserves v1:

```text
data/
  raw/
    v1/
    v2/
  processed/
    v1/
    v2/
  manifests/
    historical_v1.json
    historical_v2.json
```

A future manifest should record:

- Dataset version.
- Source/provider.
- Source URL or endpoint identifier.
- Retrieval timestamp.
- Coverage start and end.
- Granularity.
- Units.
- Geographic coverage.
- Station or coordinate identifiers.
- Raw-file checksums.
- Row and column counts.
- Preprocessing version.
- Known data-quality warnings.
- License or attribution notes.

Do not move or overwrite the current root-level raw files until a dedicated migration commit defines v1 compatibility paths and rollback behavior.

## Compatibility And Retraining Impact

| Component | Compatibility classification |
| --- | --- |
| Step 2A validation primitives | Reusable unchanged |
| Raw production parsing | Reusable with configuration if timestamp and target columns remain comparable |
| Daily production validation | Reusable unchanged after canonical daily schema is produced |
| Current weather matrix parsing | V1-contract-specific; reusable for Option A only |
| ERA5-Land or WeatherAPI historical parsing | Requires v2 implementation |
| Date alignment | Reusable with configuration |
| Station aggregation | Reusable only if station-matrix contract survives; otherwise v2 implementation |
| Missing-value handling | Requires source-specific v2 policy |
| Circular wind-direction aggregation | Reusable conceptually; implementation must be validated for source units/components |
| Daily aggregation | Reusable conceptually; source timezone and resolution must be validated |
| 58-column processed contract | Structurally reusable if v2 base columns match |
| 56 model input features | Structurally reusable, but not sufficient to preserve model validity |
| Scaler feature names | Reusable for compatibility checks |
| Scaler fitted distributions | Requires refitting after material source/spatial change |
| Trained models | Require retraining after material source/spatial change |
| Baseline metrics | Require re-baselining |
| Inference WeatherAPI compatibility | Must be reassessed if training weather source changes |

Current scalers and models must not be claimed valid for a materially changed v2 weather source.

## Step 2A Validator Reuse

| Validator | Reuse assessment |
| --- | --- |
| `ValidationReport` and generic checks | Source-independent and reusable |
| `validate_raw_production_data` | Reusable through existing parameters if v2 parser yields comparable timestamp/target columns |
| `validate_daily_production_data` | Source-independent and reusable |
| `validate_weather_matrix` | V1-contract-specific to `ANO`/`MES`/`DIA` plus station matrix; likely separate v2 validator for ERA5-Land |
| `validate_weather_alignment` | Reusable for same-shape station matrices; likely extension for grid/coordinate metadata |
| `validate_parsed_weather_api_data` | Reusable for current parsed WeatherAPI schema |
| `validate_merged_base_data` | Source-independent after weather and production are transformed into the canonical merged base schema |

Backward-compatible extensions should be preferred. Step 2A should not be weakened to accommodate unverified v2 data.

## Pilot Specification

Before a complete refresh, run a limited source-verification and ERA5-Land pilot:

- Use one representative overlap year, preferably `2023`.
- Optionally include a small `2024` append check after production endpoint verification.
- Use only required variables.
- Use recovered station coordinates if available; otherwise use a small representative point set.
- Store temporary outputs outside tracked dataset paths.
- Do not overwrite v1 data, processed files, models, scalers, or predictions.

The pilot must evaluate:

- Source accessibility and authentication.
- Schema and field names.
- Units.
- Timezone and date boundaries.
- Temporal coverage.
- Missing values.
- Coordinate or station coverage.
- Aggregation logic.
- Wind-direction calculation.
- Comparison with v1 weather.
- Expected storage and runtime.
- Incremental-update feasibility.

Quantitative comparison metrics:

- Date coverage.
- Missingness.
- Mean and standard deviation.
- Quantiles.
- Min and max.
- Correlation.
- Mean absolute difference.
- Seasonal profiles.
- Feature-distribution shift.

## Recommended Phase 2 Ordering

Pause Step 2B until the data-source pilot and v2 contract decision are complete.

Strict feature/scaler validation remains useful for v1, and implementing it now would not be technically wrong. However, postponing it better follows the current project priority and avoids encoding premature v2 assumptions before the source stack, spatial strategy, and feature contract are decided.

## Future Implementation Sequence

1. Verify production source endpoint, historical coverage, target definition, license, and update method.
2. Investigate the 18 weather station IDs and recover coordinates if possible.
3. Define v1 and pilot manifest schemas.
4. Run a limited production-source pilot.
5. Run a limited ERA5-Land point time-series pilot.
6. Produce a v1-versus-candidate comparison report.
7. Decide the v2 source stack and spatial strategy.
8. Implement versioned ingestion.
9. Implement versioned preprocessing.
10. Extend validators backward-compatibly.
11. Regenerate v2 processed features.
12. Refit scalers.
13. Retrain models.
14. Re-baseline metrics.
15. Document dataset selection, rollback, and operational refresh commands.

## Risks And Rollback

Risks:

- REN may not be the exact original source.
- Production target definitions may drift.
- Weather station IDs may not map cleanly to public metadata.
- WeatherAPI may be cost- or quota-prohibitive for full reconstruction.
- ERA5-Land may introduce distribution shift relative to v1 station data.
- Timezone, daylight-saving, and daily aggregation choices may change targets or features.
- Wind-direction component conversion may be implemented incorrectly if conventions are not validated.
- V2 data may invalidate v1 scalers, models, and metrics.

Rollback strategy:

- Preserve v1 raw, processed, model, and scaler artifacts.
- Record checksums in manifests.
- Keep v2 paths separate.
- Never silently replace v1 files.
- Keep v1 validators and scripts usable during migration.
- Require explicit model/scaler version selection before v2 inference is used.

## Uncertainties Requiring Confirmation

- Exact REN export/API endpoint used for `ReparticaoProducao.csv`.
- REN oldest comparable date, latest downloadable date, license, and incremental-update process.
- Whether the current `Eolica` target definition is stable across append periods.
- Original weather-data provider.
- Station coordinates and continuity for the 18 numeric weather columns.
- Whether IPMA or another source offers full historical access for the required station variables.
- WeatherAPI account-specific historical access, quotas, bulk/range capabilities, and cost.
- ERA5-Land pilot runtime, storage, and distribution similarity to v1.
- Whether daily-statistics products are equivalent enough to replace hourly-derived aggregation.

## Final Status

- Step 2A remains complete.
- The current v1 datasets and validators remain unchanged.
- No v2 source has yet been selected.
- No data refresh has yet occurred.
- Step 2B remains paused.
- Phase 3 was not started.
- The next task is a limited source-verification and ERA5-Land pilot plan.

---

## Phase 2 Source Probe Findings

Original file before consolidation: `PHASE_2_SOURCE_PROBE_FINDINGS.md`.

## 1. Purpose

This document records the Phase 2 Step 2A.7 REN and IPMA source-probe findings before any v2 dataset is created. It preserves the evidence needed to decide whether official production and station-metadata sources can support a future v2 data contract.

This is a findings document only. It does not select a final v2 source, modify v1 data, or start the ERA5-Land pilot.

## 2. Scope And Safety

Only limited official-source probes were run:

- REN: three explicit single-date requests.
- IPMA: current station metadata resources only.
- No bulk history was downloaded.
- No ERA5-Land, Copernicus CDS, notebook, pipeline, training, or inference execution occurred.
- Probe outputs were written only under ignored `data/pilot/`.
- Existing v1 raw data, processed data, models, scalers, notebooks, and baselines were not changed.

The probe tooling was committed in `03760fb1 feat: add REN and IPMA source probes`.

## 3. REN Endpoint Evidence

Confirmed API observations:

| Item | Evidence |
| --- | --- |
| Endpoint identifier | `REN ElectricityProductionBreakdownDaily` |
| Request pattern | One date per request with `culture=pt-PT` and `date=YYYY-MM-DD` |
| Tested dates | `2010-01-01`, `2025-01-25`, `2026-06-27` |
| Daily record count | 96 records on each tested complete day |
| Returned cadence | 15 minutes, inferred from `00:00` to `23:45` |
| Wind series | `Eólica` |
| Unit exposed by response | `MW` |
| Top-level response keys | `xAxis`, `yAxis`, `series` |

Confirmed local comparison results:

| Date | Aligned Timestamps | Exact Matches | MAE | Max Absolute Difference | Correlation | Median REN/Local Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2010-01-01` | 96 | 96 | `0.0` | `0.0` | `1.0` | `1.0` |
| `2025-01-25` | 96 | 0 | `15.21666666666667 MW` | `18.200000000000728 MW` | `0.9999960649429729` | `1.005018136527911` |

The `2026-06-27` probe returned 96 records at 15-minute cadence with the `Eólica` series and `MW` units. No local v1 comparison exists for that date because it is beyond the current v1 production-file coverage.

Interpretation:

- The REN endpoint appears compatible with the v1 production target shape: 15-minute wind production values in MW.
- The exact `2010-01-01` overlap is strong evidence of target compatibility.
- The `2025-01-25` overlap shows that later overlapping values can differ and must not be silently appended to the frozen v1 snapshot.

## 4. Production-Source Conclusion

The exact `2010-01-01` correspondence provides strong evidence that the v1 wind-production target is compatible with the official REN source and contract.

This does not prove that every historical v1 value has been verified. It also does not prove the original CSV download mechanism. The current repository evidence supports REN as the leading official v2 production-source candidate, but the full historical contract still requires more verification before v2 selection.

## 5. Historical-Revision Finding

The `2025-01-25` overlap aligned on all 96 timestamps but differed slightly. The correlation remained very high, but the values were not exact.

Unresolved hypotheses:

- Later REN revisions changed values after the local CSV snapshot.
- The local CSV contained provisional values while the current API returns definitive values, or the reverse.
- The local CSV and API were retrieved at different source snapshots.
- Another upstream update process changed overlapping values.
- Less likely: parsing, encoding, timezone, or transformation differences in the probe.

No single explanation is selected because the available evidence does not identify the cause.

## 6. V2 Production Implication

A future v2 production dataset should provisionally be retrieved consistently from the official REN source if further verification confirms the contract.

The v2 production process should record:

- Endpoint identifier and request parameters.
- Retrieval timestamp.
- Units and temporal granularity.
- Raw response checksums.
- Coverage start and end.
- Whether values are provisional or definitive where discoverable.
- Overlap-comparison results against frozen v1 snapshots.

Direct append to v1 is rejected for now. Overlapping values may change, so future work must choose explicitly between:

- full historical reconstruction from a consistent source snapshot;
- controlled append with revision tracking and clear boundary rules.

The v1 dataset remains immutable.

## 7. Recent-Data Availability

The `2026-06-27` REN probe returned successfully. This proves that official REN data exists beyond the current v1 production end date of `2025-04-28`.

This single successful recent date does not prove complete coverage for every intervening date.

## 8. IPMA Mapping Results

Confirmed mapping results from current IPMA metadata:

| Item | Result |
| --- | --- |
| Total v1 station identifiers | 18 |
| Exact current metadata matches | 17 |
| Ambiguous matches | 0 |
| Unmatched identifiers | 1 |
| Unmatched ID | `1200579` |
| Coordinates available for pilot | Yes, for the 17 matched identifiers |

The matched metadata provides current station names and coordinates suitable for a limited ERA5-Land pilot. Raw metadata dumps are intentionally not copied into this document.

## 9. Station-Mapping Limitations

Present-day IPMA identifier correspondence does not prove IPMA was the original historical-weather source.

Limitations:

- Station coordinates or metadata may have changed historically.
- The unmatched `1200579` identifier must not be guessed.
- The 17 confirmed coordinates can support a v2 pilot, but the missing station must be documented.
- Alternative spatial strategies remain possible, including a documented regular grid or representative coordinate set.

## 10. ERA5-Land Pilot Readiness

The source probes establish enough evidence to begin a limited ERA5-Land technical pilot using:

- one confidently mapped station;
- a one-week period in 2023;
- `2m_temperature`;
- `10m_u_component_of_wind`;
- `10m_v_component_of_wind`.

The ERA5-Land pilot must happen before any full download, v2 source contract selection, feature regeneration, scaler refitting, or model retraining.

## 11. Decision Register

| Decision | Status | Evidence |
| --- | --- | --- |
| Preserve v1 unchanged | Approved | Existing baseline and possible historical revisions |
| Use REN as leading v2 production source | Provisional | Exact `2010-01-01` overlap and recent availability |
| Append directly to v1 | Rejected for now | Overlap differences on `2025-01-25` |
| Use 17 mapped coordinates for pilot | Approved | Exact current IPMA metadata matches |
| Guess mapping for `1200579` | Rejected | No exact IPMA metadata match |
| Select ERA5-Land as final v2 weather source | Not yet decided | Pilot still required |

## 12. Remaining Questions

- Why do the `2025-01-25` overlapping REN values differ?
- Does REN expose provisional/final status for the endpoint or data points?
- What is the earliest continuously available comparable REN date?
- Are all dates after `2025-04-28` available with the same contract?
- What was the historical status of the IPMA station metadata during v1 coverage?
- What is the identity or replacement history of `1200579`?
- Are ERA5-Land distributions sufficiently compatible with the intended v2 model?
- Which spatial strategy and daily aggregation formulas should be adopted?

## 13. Next Step

The next approved activity is `Phase 2 — Step 2A.8: ERA5-Land one-point technical pilot`.

Step 2B remains paused. Phase 3 was not started.

---

## Phase 2 ERA5-Land And V1 Weather Comparison

Original file before consolidation: `PHASE_2_ERA5_LAND_V1_COMPARISON.md`.

## Purpose and scope

This document records Phase 2 Checkpoint 3: a read-only comparison between the limited ERA5-Land multi-point pilot and the current v1 weather matrices.

The scope is diagnostic only:

- Compare the existing ERA5-Land pilot outputs with overlapping v1 station-day weather rows.
- Assess schema compatibility, technical data quality, distribution shift, and model/scaler implications.
- Preserve the current v1 raw data, processed data, models, scalers, scripts, notebooks, and pilot artifacts unchanged.
- Do not select a final v2 weather source, regenerate outputs, execute notebooks, run training, or run live CDS/network requests.

This report uses the existing ignored pilot outputs under `data/pilot/era5_land/`, especially `era5_land_multi_point_2023_winter_summer_*`.

## Evidence inputs and checksums

Read-only evidence files:

| File | Role | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `docs/ML_ENGINEERING_ROADMAP.md` | Phase 2 objective and constraints | 17,662 | `32bd36617ddf78ca2b921b15743dde2a4d60069dd811e79d244578d06eca7f83` |
| `docs/PHASE_2.md` | Data-refresh assessment and ERA5-Land pilot requirements | 17,767 | `d9db760e4875e7f8ee36ca9fdb198719ab26910c04234840838d3d35c61f8f81` |
| `docs/PHASE_2.md` | REN/IPMA source-probe findings and pilot readiness | 7,586 | `7323899a9e27fc9285a7c456db115b8dce0f28c467dc00f3199fc59cc7be58e2` |
| `scripts/pilot_era5_land_one_point.py` | ERA5-Land point extraction formulas and output units | 26,097 | `f51f4e59b02f2c66e2e8b27b0e5c20bf162e9fc947001c0b364ab1bdcf1a3f0a` |
| `scripts/pilot_era5_land_multi_point.py` | Multi-point pilot, v1 comparison, validation, and aggregation logic | 37,102 | `0a44e17e0b35386e7f37a86b31b3f30eb3161964e17ff0266ab7be04d2785f28` |
| `data/raw/IntensidadeMediaVento10m.csv` | v1 wind-speed matrix | 352,665 | `e09e94f07618ddb4d52d0b53f46a69e1d6b64814a142e69ee57161f4199e842c` |
| `data/raw/DirecaoMediaVento10m.csv` | v1 wind-direction matrix | 469,994 | `289aa375b4c6a371ab4ca649b999ee4ac1cb16e29a5efea01ede86bca8ba08bd` |
| `data/raw/TemperaturaMedia.csv` | v1 temperature matrix | 400,139 | `ce2829f328e46a99f942343ff0a5a27f18319b5a1dc5728c5c110692596fcef4` |

Pilot output evidence:

| File | Role | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_hourly.csv` | ERA5-Land hourly pilot rows | 146,328 | `c0e6332c38ade4a0fd19a0fbded8d44efdccec0bcc346058fc6ccfc5489deedf` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_daily_points.csv` | ERA5-Land station-day rows | 10,919 | `9d0c0cc54848f49b9a4302c8f7efbc6b3756ea8f5e405d8ef2c8b42763298741` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_daily_aggregate.csv` | ERA5-Land three-station daily aggregate | 2,213 | `1cfacc2c78b58d9e9aa1a86490f2075e9f65d8a0a46b730feeee1587a430501b` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_season_summary.csv` | Pilot seasonal summary | 1,817 | `c883bc5e2d6d8bc1fb51dd488ff9d69c682b329ac96f4a05b1a14c27e5539f53` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_v1_comparison.csv` | Paired ERA5-Land/v1 station-day comparison | 15,492 | `dd046e7c817e7b47c8d91df42aeb81e9631db43b65326b8209ea03d108b4e39d` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_metadata.json` | Pilot metadata | 21,718 | `78db650dd3673aee4b23ac67aa6540776d6d815ed5ef9d317082f114d5a550ef` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_validation.json` | Pilot validation report | 6,715 | `d04738e78257b7c14fe79e92e0bdb4756c09c0009bfd5d2a3fbd0cae64e746cb` |

Raw NetCDF pilot checksums are recorded in the metadata and validation JSON. The six raw keys are:

| Raw key | SHA-256 |
| --- | --- |
| `winter_1210622` | `c3a5e087ed2d412242e7e529b60ea7b018b55523e04ac9222c2067711a9293b2` |
| `winter_1210683` | `350deefb2289d1de4a9c8da8767a2cebfc7f2d02e6db3e51635e345740bee40c` |
| `winter_1200562` | `d4c8fdfa6049f44a432ad59c7f034fdaa4580ed6dc9febadb7d6640801aae90c` |
| `summer_1210622` | `b688212d50a863b0f257d2ad7a5e53d490c0066103cfda948c29d3b61b2fafff` |
| `summer_1210683` | `8c2bd46e11de4a4e22f20d55f3fed011d05eabf8024320a60ecef88dd9483e5a` |
| `summer_1200562` | `9321e854829dfa62edc8666d6ec32fbedcaf6c7354e119be1a579c5706b9ba39` |

## Pilot coverage and schema compatibility

The v1 weather files are daily station matrices with `ANO`, `MES`, `DIA`, and 18 station columns. Read-only checks found 4,017 rows in each matrix, covering `2013-01-01` through `2023-12-31`.

The pilot comparison covers:

| Item | Value |
| --- | ---: |
| ERA5-Land hourly rows | 1,008 |
| ERA5-Land station-day rows | 42 |
| ERA5-Land aggregate daily rows | 14 |
| Paired v1 comparison rows | 42 |
| Seasons | `winter`, `summer` |
| Date windows | `2023-01-01` to `2023-01-07`; `2023-07-01` to `2023-07-07` |
| Stations | `1200562` Beja; `1210622` Braga Merelim; `1210683` Guarda |
| v1 available rows | 42 of 42 |

The pilot validation JSON reports `passed: true`, `issues: []`, and the expected counts above. The multi-point metadata records `service_status: not_contacted_reuse_raw` for the six raw files, meaning the comparison artifacts were rebuilt from existing raw pilot files without contacting CDS during that validation run.

Missingness in the 42 paired rows:

| Column group | Missing values |
| --- | ---: |
| v1 temperature, wind speed, wind direction | 0 |
| ERA5-Land temperature and wind speed | 0 |
| ERA5-Land vector wind direction | 1 |
| Circular wind-direction differences | 1 |

The single missing ERA5-Land vector direction is `Beja` on `2023-01-06`. The daily scalar mean wind speed is `0.962 m/s`, but the vector-mean speed is `0.414 m/s`, below the pilot calm threshold of `0.5 m/s`; the direction is therefore intentionally null rather than imputed.

Schema compatibility is sufficient for a diagnostic station-day comparison: station IDs and dates align, v1 speed is in `m/s`, v1 temperature is in `deg C`, and both sources express wind direction as degrees from. This is structural compatibility for comparison only, not evidence of source equivalence or model compatibility.

## Calculation methodology

ERA5-Land pilot rows use:

- `2m_temperature`, converted from Kelvin to `deg C`.
- `10m_u_component_of_wind` and `10m_v_component_of_wind`.
- Wind speed as `sqrt(u10^2 + v10^2)`.
- Meteorological wind direction from as `(180 + degrees(atan2(u10, v10))) % 360`.
- A calm-or-near-calm threshold of `0.5 m/s`; vector directions below that threshold are null.

Station-day ERA5-Land values come from hourly UTC aggregation. Daily vector direction uses daily mean `u10` and `v10`.

The v1 comparison reads the three raw v1 matrices in read-only mode, creates `date_utc` from `ANO`, `MES`, and `DIA`, melts the three pilot station columns, and joins on `date_utc` and `station_id`.

For temperature and wind speed:

- Difference is `ERA5-Land - v1`.
- Signed mean difference is the mean of that difference.
- Mean absolute difference is the mean absolute value of that difference.
- Pearson and Spearman correlations are calculated on paired non-null rows.

For wind direction, raw-degree correlation is not used as primary evidence. The primary metric is signed circular difference:

```text
((era5_deg - v1_deg + 180) % 360) - 180
```

Absolute circular difference is the absolute value of that signed circular difference.

For aggregate results, ERA5-Land aggregate rows come from `era5_land_multi_point_2023_winter_summer_daily_aggregate.csv`. V1 aggregate rows are computed from the paired station-day comparison:

- Temperature is the daily mean across the three v1 stations.
- Wind speed is the daily mean across the three v1 stations.
- V1 aggregate direction uses speed-weighted vector components matching the project convention:
  - `u = -speed * sin(direction_rad)`
  - `v = -speed * cos(direction_rad)`
  - mean `u` and mean `v` are converted back with `(180 + degrees(atan2(mean_u, mean_v))) % 360`.

No generated outputs were rewritten for this report.

## Station-level comparison

Overall paired station-day temperature and wind-speed metrics:

| Variable | n | ERA5 missing | v1 missing | ERA5 mean | ERA5 std | ERA5 min | ERA5 q25 | ERA5 q50 | ERA5 q75 | ERA5 max | v1 mean | v1 std | v1 min | v1 q25 | v1 q50 | v1 q75 | v1 max | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temperature, deg C | 42 | 0 | 0 | 16.052 | 8.071 | 3.880 | 8.892 | 17.199 | 22.818 | 29.766 | 15.745 | 7.732 | 4.100 | 8.550 | 16.400 | 22.175 | 27.600 | 0.981 | 0.966 | 0.307 | 1.350 |
| Wind speed, m/s | 42 | 0 | 0 | 2.395 | 0.905 | 0.939 | 1.717 | 2.236 | 2.883 | 4.460 | 2.917 | 1.367 | 0.600 | 1.925 | 2.650 | 3.975 | 5.700 | 0.607 | 0.564 | -0.522 | 1.055 |

Temperature by station:

| Station | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1200562` Beja | 14 | 19.196 | 18.221 | 0.997 | 0.974 | 0.975 | 1.386 |
| `1210622` Braga Merelim | 14 | 14.962 | 16.029 | 0.996 | 0.947 | -1.066 | 1.191 |
| `1210683` Guarda | 14 | 13.998 | 12.986 | 0.988 | 0.950 | 1.013 | 1.473 |

Wind speed by station:

| Station | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1200562` Beja | 14 | 2.530 | 3.443 | 0.957 | 0.982 | -0.913 | 0.913 |
| `1210622` Braga Merelim | 14 | 2.556 | 1.757 | 0.898 | 0.771 | 0.799 | 0.799 |
| `1210683` Guarda | 14 | 2.098 | 3.550 | 0.872 | 0.869 | -1.452 | 1.452 |

Temperature by season:

| Season | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Summer | 21 | 23.390 | 22.786 | 0.840 | 0.816 | 0.604 | 1.783 |
| Winter | 21 | 8.714 | 8.705 | 0.924 | 0.913 | 0.009 | 0.917 |

Wind speed by season:

| Season | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Summer | 21 | 2.524 | 3.171 | 0.018 | 0.090 | -0.648 | 1.228 |
| Winter | 21 | 2.266 | 2.662 | 0.787 | 0.691 | -0.396 | 0.882 |

The station-level results show high temperature agreement and mixed wind-speed agreement. Wind speed has good within-station correlations, but the sign and size of bias differ by station. Summer wind-speed correlation across pooled station-days is near zero because station-specific offsets and a short seven-day window dominate the pooled sample.

## Aggregate comparison

Overall aggregate daily temperature and wind-speed metrics:

| Variable | n | ERA5 mean | ERA5 std | ERA5 min | ERA5 q25 | ERA5 q50 | ERA5 q75 | ERA5 max | v1 mean | v1 std | v1 min | v1 q25 | v1 q50 | v1 q75 | v1 max | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temperature, deg C | 14 | 16.052 | 7.861 | 6.633 | 7.937 | 17.026 | 22.381 | 26.094 | 15.745 | 7.562 | 6.533 | 8.117 | 15.933 | 21.500 | 26.100 | 0.997 | 0.969 | 0.307 | 0.617 |
| Wind speed, m/s | 14 | 2.395 | 0.825 | 1.274 | 1.732 | 2.372 | 2.654 | 4.262 | 2.917 | 1.039 | 1.533 | 2.042 | 2.983 | 3.492 | 4.733 | 0.952 | 0.955 | -0.522 | 0.538 |

Aggregate temperature by season:

| Season | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Summer | 7 | 23.390 | 22.786 | 0.999 | 1.000 | 0.604 | 0.606 |
| Winter | 7 | 8.714 | 8.705 | 0.943 | 0.750 | 0.009 | 0.629 |

Aggregate wind speed by season:

| Season | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Summer | 7 | 2.524 | 3.171 | 0.636 | 0.857 | -0.648 | 0.680 |
| Winter | 7 | 2.266 | 2.662 | 0.983 | 0.775 | -0.396 | 0.396 |

Aggregation dampens station-level noise. The aggregate wind-speed correlation is high, but ERA5-Land remains lower than v1 by `0.522 m/s` on average across the 14 aggregate days.

## Seasonal profiles

Seasonal station-day profiles:

| Season | Temp ERA5 mean | Temp v1 mean | Temp signed diff | Temp MAD | Speed ERA5 mean | Speed v1 mean | Speed signed diff | Speed MAD | Direction signed circular diff | Direction mean abs circular diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Summer | 23.390 | 22.786 | 0.604 | 1.783 | 2.524 | 3.171 | -0.648 | 1.228 | 18.711 | 21.868 |
| Winter | 8.714 | 8.705 | 0.009 | 0.917 | 2.266 | 2.662 | -0.396 | 0.882 | -19.190 | 26.066 |

Temperature preserves the broad winter/summer profile. Wind speed is lower in ERA5-Land in both sampled seasons. Direction differences are seasonally asymmetric: ERA5-Land is clockwise relative to v1 on average in summer and counter-clockwise in winter, but the sample is too small to treat this as a stable seasonal rule.

## Wind-direction circular-difference analysis

Station-day circular differences:

| Scope | n | Signed mean | Signed std | Signed min | Signed q25 | Signed q50 | Signed q75 | Signed max | Abs mean | Abs q50 | Abs q75 | Abs q95 | Abs max | Percent abs <= 45 | Percent abs <= 90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 41 | 0.223 | 35.426 | -109.301 | -4.811 | 5.330 | 15.626 | 67.511 | 23.916 | 14.503 | 35.523 | 67.511 | 109.301 | 80.488 | 97.561 |
| `1200562` | 13 | 3.059 | 21.407 | -48.705 | 1.407 | 11.624 | 15.527 | 21.588 | 16.415 | 14.503 | 18.592 | 40.796 | 48.705 | 92.308 | 100.000 |
| `1210622` | 14 | 2.739 | 53.756 | -109.301 | -18.401 | 13.871 | 44.744 | 67.511 | 41.576 | 33.769 | 59.285 | 96.091 | 109.301 | 57.143 | 92.857 |
| `1210683` | 14 | -4.927 | 22.484 | -65.188 | -4.540 | 2.280 | 5.259 | 17.428 | 13.221 | 5.187 | 11.881 | 51.415 | 65.188 | 92.857 | 100.000 |
| Summer | 21 | 18.711 | 22.788 | -21.660 | 6.246 | 15.527 | 21.588 | 67.511 | 21.868 | 15.626 | 21.660 | 60.360 | 67.511 | 80.952 | 100.000 |
| Winter | 20 | -19.190 | 36.346 | -109.301 | -43.473 | -0.589 | 5.115 | 19.962 | 26.066 | 10.125 | 43.473 | 89.994 | 109.301 | 80.000 | 95.000 |

Aggregate daily circular differences:

| Scope | n | Signed mean | Signed std | Signed min | Signed q25 | Signed q50 | Signed q75 | Signed max | Abs mean | Abs q50 | Abs q75 | Abs q95 | Abs max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 14 | 3.213 | 26.795 | -64.403 | -1.256 | 9.397 | 19.988 | 36.012 | 19.788 | 15.320 | 22.642 | 50.492 | 64.403 |
| Summer | 7 | 18.373 | 11.829 | -3.146 | 15.320 | 20.789 | 22.158 | 36.012 | 19.272 | 20.789 | 22.158 | 32.146 | 36.012 |
| Winter | 7 | -11.947 | 29.654 | -64.403 | -24.236 | 4.416 | 6.304 | 12.225 | 20.303 | 6.569 | 27.613 | 57.982 | 64.403 |

Direction agreement is usable for diagnostics but not equivalent. The aggregate view reduces the maximum absolute circular difference from `109.301 deg` at station-day level to `64.403 deg`, but aggregate mean absolute circular difference remains about `19.788 deg`.

## Interpretation: schema compatibility, technical data quality, expected source differences, unexplained discrepancies, distribution shift

Schema compatibility:

- The pilot proves the selected ERA5-Land point output can be transformed into a station-day comparison schema aligned with v1 station IDs and dates.
- Required temperature and wind-speed fields are non-null for all 42 paired station-days.
- Direction requires circular handling and explicit calm-threshold nullability.
- The existing v1 station-matrix contract is structurally comparable for the three pilot stations, but ERA5-Land is not a drop-in replacement for the 18-column v1 matrix.

Technical data quality:

- The multi-point validation report passed with no validation issues.
- Hourly, station-day, aggregate, and v1 comparison row counts match the pilot expectations.
- The single direction null is explainable from the documented calm-threshold logic.
- This report used existing pilot outputs only; it did not regenerate data or contact CDS.

Expected source differences:

- ERA5-Land grid cells represent reanalysis model grid points, not necessarily the same physical exposure as v1 station observations.
- Station siting, elevation, local terrain, and model-grid representativeness can cause station-specific temperature and wind-speed offsets.
- ERA5-Land uses hourly UTC component fields; v1 weather files are daily station matrices whose original provider and detailed aggregation method remain unconfirmed.
- Direction comparison is sensitive to calm or variable-wind days, and vector aggregation can differ from any scalar daily direction method used by the original v1 provider.

Unexplained discrepancies:

- Braga Merelim has ERA5-Land wind speed higher than v1 on average (`+0.799 m/s`), while Beja and Guarda are lower (`-0.913 m/s` and `-1.452 m/s`).
- Summer pooled station-day wind-speed correlation is near zero despite stronger within-station correlations.
- Braga Merelim has the largest direction spread, with mean absolute circular difference `41.576 deg` and maximum absolute circular difference `109.301 deg`.
- The pilot does not establish whether these discrepancies come from source definitions, station/grid geography, aggregation differences, short sample windows, or original v1 provider behavior.

Distribution shift:

- Temperature is strongly aligned overall (`Pearson 0.981`, station-day MAD `1.350 deg C`), but station-specific biases are visible.
- Wind speed shows material shift: ERA5-Land is lower than v1 by `0.522 m/s` on average at both station-day and aggregate levels, and station-day MAD is `1.055 m/s`.
- Wind direction differs enough to require explicit circular validation if ERA5-Land becomes a v2 source.
- The observed shifts are large enough that a v2 ERA5-Land weather dataset must be treated as a new data contract, not as a silent extension of v1.

## Model/scaler compatibility implications

The current v1 scalers and trained models must not be claimed valid for ERA5-Land-derived weather.

Compatibility assessment:

- Feature names and high-level concepts can likely be mapped after versioned preprocessing: temperature, wind speed, and wind direction all exist.
- Fitted scaler distributions are not compatible by evidence alone because wind speed and direction distributions shift.
- Current model weights were trained against the v1 weather-source distribution and v1 aggregation choices.
- Replacing v1 weather with ERA5-Land without refitting scalers, retraining models, and re-baselining metrics would create train-serving and historical-distribution risk.

Required future work before any v2 model use:

- Define a versioned ERA5-Land data contract and manifest.
- Extend validators without weakening v1 validation.
- Generate v2 processed features in separate paths.
- Refit scalers on the v2 training distribution.
- Retrain models and compare against v1 baselines.
- Document source selection, rollback, and inference-source compatibility.

## Limitations and deferred questions

Limitations:

- The pilot covers only three mapped stations, not all 17 currently mapped IPMA station IDs or all 18 v1 weather columns.
- The pilot covers only two seven-day windows in 2023.
- No full-year seasonality, weather extremes, long-term drift, or production-target relationship was evaluated.
- The original v1 weather provider, station metadata history, and exact v1 daily aggregation method remain unconfirmed.
- The unmatched v1 station `1200579` remains unresolved.
- This report does not evaluate full-grid ERA5-Land strategies, capacity-weighted wind-farm regions, or WeatherAPI alternatives.
- Direction analysis intentionally uses circular differences; raw-degree correlations are not reported as primary evidence.

Deferred questions:

- Are the station-specific wind-speed biases stable across full years?
- Would all 17 mapped station coordinates improve or worsen the aggregate distribution shift?
- Should a future v2 source use station points, a regular grid, or installed-capacity-weighted regions?
- How should calm and near-calm direction nullability be represented in a future canonical schema?
- What production-source overlap and revision policy will be paired with any v2 weather source?
- How large is the downstream forecast-performance impact after v2 scaler refitting and model retraining?

## Checkpoint status

Checkpoint 3 documentation status: complete for the approved scope.

Safety status:

- Only a documentation comparison was produced.
- No v1 data, processed data, models, scalers, scripts, notebooks, or pilot artifacts were modified.
- No live CDS/network request, notebook execution, training run, pipeline run, or generated-output write was performed.
- No final v2 source was selected.
- Step 2B remains paused and Phase 3 was not started.

Conclusion:

ERA5-Land is technically usable for a versioned v2 weather-source pilot path, but the observed wind-speed and wind-direction shifts mean it is not compatible with the current v1 scalers or trained models as a silent replacement. Any adoption must proceed through versioned ingestion, validation, preprocessing, scaler refitting, model retraining, and metric re-baselining.

---

## Phase 2 V2 Data Contract Decision

Original file before consolidation: `PHASE_2_V2_DATA_CONTRACT_DECISION.md`.

## Purpose And Scope

This document records the Phase 2 Checkpoint 4 v2 data-contract decision for the wind-energy forecasting project. It selects the v2 production and weather source stack, aggregation rules, versioning policy, validation requirements, and model-compatibility consequences before any v2 ingestion or model work begins.

This is a documentation-only decision record. It does not modify v1 data, create v2 data, execute notebooks, run live CDS or REN network calls, regenerate features, refit scalers, retrain models, re-baseline metrics, start Step 2B, or start Phase 3.

Coverage checklist: production source; ERA5-Land product; spatial strategy; temporal resolution; wind-speed aggregation; wind-direction aggregation; temperature aggregation; timezone; versioned paths; manifest; retrieval; revision; validators; scaler refitting; model retraining; metric re-baselining.

## Evidence Base

Local evidence used:

| Evidence | Use in this decision |
| --- | --- |
| `docs/ML_ENGINEERING_ROADMAP.md` | Phase 2 objective, validation focus, and requirement to preserve existing modelling workflow. |
| `docs/PHASE_2.md` | v1 coverage, current contracts, source-candidate comparison, versioning proposal, validator reuse, and retraining impact. |
| `docs/PHASE_2.md` | REN endpoint probe, overlap comparison, recent-data availability, IPMA station mapping, and direct-append rejection. |
| `docs/PHASE_2.md` | ERA5-Land pilot formulas, multi-point comparison, calm threshold, distribution-shift evidence, and model/scaler implications. |

Evidence classification used below:

- Confirmed fact: directly supported by repository evidence or completed local probe/comparison documentation.
- Engineering judgment: selected policy or design choice based on confirmed evidence and project constraints.
- Unresolved assumption: material item not yet verified and therefore required before production ingestion or daily aggregation.

## Baseline Preservation Decision

Decision: preserve v1 as the immutable reproducible baseline.

Classification: engineering judgment supported by confirmed facts.

Rationale:

- The roadmap requires the existing modelling workflow to remain usable.
- `docs/PHASE_2.md` records the current v1 raw, processed, model, and scaler contracts and states that current scalers and models must not be claimed valid for a materially changed v2 weather source.
- `docs/PHASE_2.md` records a later REN overlap difference on `2025-01-25`, which means v2 production must not be silently appended to the frozen v1 snapshot.

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

Confirmed facts from `docs/PHASE_2.md`:

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

Confirmed facts from `docs/PHASE_2.md`:

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

Confirmed facts from `docs/PHASE_2.md`:

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

- `docs/PHASE_2.md` records REN daily responses with 96 records on tested complete days, implying 15-minute cadence.
- `docs/PHASE_2.md` records hourly ERA5-Land pilot rows and daily point and aggregate outputs.
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

Confirmed fact from `docs/PHASE_2.md`: one pilot direction value was intentionally null because the vector-mean speed was `0.414 m/s`, below the documented `0.5 m/s` calm threshold, even though daily scalar mean speed was nonzero.

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

Classification: engineering judgment based on the versioning proposal in `docs/PHASE_2.md`.

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

- `docs/PHASE_2.md` found exact agreement on `2010-01-01` but nonzero overlap differences on `2025-01-25`.
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

Confirmed facts from `docs/PHASE_2.md`:

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

---

## Phase 2 V2 Local-Day Alignment Decision

Original file before consolidation: `PHASE_2_V2_LOCAL_DAY_ALIGNMENT_DECISION.md`.

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

---

## Phase 2 REN Backfill Readiness

Original file before consolidation: `PHASE_2_REN_BACKFILL_READINESS.md`.

## Purpose

This document records Phase 2 Step 2A.10b REN backfill operational readiness after the January 2010 recovery pilot. It is a readiness note only.

## Scope And Non-Goals

Scope:

- Assess whether the current REN v2 raw ingestion implementation is operationally ready for a complete backfill.
- Record the January 2010 bounded live retry, resume/idempotency check, dry-run check, file inventory, and v1 comparison results.
- Preserve the operational decision for v2 raw file layout.

Non-goals:

- No full historical backfill was run.
- No v1 data, processed data, models, scalers, notebooks, or baselines were changed.
- No ERA5-Land, CDS, IPMA, WeatherAPI, Step 2B, feature regeneration, scaler refitting, model training, or metric re-baselining work was started.
- No new data source assumption is introduced beyond the existing REN v2 production-source decision.

Ignored v2 pilot outputs under `data/raw/v2/production` remain uncommitted.

## Implementation And Recovery Defect

The initial sandbox run attempted the January 2010 REN pilot. Dates `2010-01-04` through `2010-01-31` failed with `WinError 10013` network/socket permission errors and left status-only `unavailable` metadata partitions. That failure exposed a same-range resume defect: `--resume` correctly skipped verified partitions, but then refused to retry existing unverified status-only partitions; `--resume --overwrite` was intentionally disallowed.

The narrow correction adds `--retry-unavailable` for use with `--resume`. The retry path recognizes only strict status-only unavailable partitions: `status.json` exists, raw and normalized files do not exist, validation status is exactly `unavailable`, and status metadata has no raw or normalized paths or checksums. Verified partitions are still skipped and are never overwritten by retry recovery.

## Bounded Live Pilot

The bounded live retry used:

```python
run_ingestion(
    start_date="2010-01-01",
    end_date="2010-01-31",
    output_root=Path("data/raw/v2/production"),
    request_delay=1.0,
    resume=True,
    retry_unavailable=True,
    compare_v1_csv=Path("data/raw/ReparticaoProducao.csv"),
)
```

Validated result:

| Item | Result |
| --- | --- |
| Runtime seconds | `91.352694200119` |
| Requests made | `28` |
| Daily result status counts | `{"complete": 31}` |
| Manifest | `data/raw/v2/production/ren/manifests/ren_production_manifest.json` |
| Manifest SHA-256 | `a50b2ae3f50f868c5017e77914bb8860ab01a4add0b4cf7c96bc1e91b3c8d1cd` |

Previously verified smoke partition hashes for `2010-01-01` through `2010-01-03` and the separately verified `2026-06-27` partition were captured before the live retry and were preserved. The recovery request count of 28 matches only the Jan 4-31 status-only unavailable partitions.

## Monthly Recovery Metrics

| Item | Result |
| --- | --- |
| January raw files | `31` |
| January normalized files | `31` |
| January status files | `31` |
| January file count | `93` |
| January size | `1,403,959 bytes` |
| Total REN tree file count | `97` |
| Total REN tree size | `1,483,469 bytes` |
| Manifest size | `34,683 bytes` |
| January status counts | `complete: 31` |
| Warnings | `0` |
| Errors | None |
| Expected rows | `2,976` |
| Actual rows | `2,976` |
| First timestamp | `2010-01-01T00:00:00` |
| Last timestamp | `2010-01-31T23:45:00` |
| Missing expected timestamps | `0` |
| Duplicate timestamps | `0` |
| Unit | `MW` for 31 days |
| Manifest row count | `3,072`, including January plus `2026-06-27` |
| Manifest coverage start | `2010-01-01` |
| Manifest coverage end | `2026-06-27` |
| Unavailable dates | Empty |
| Incomplete dates | Empty |

## Resume And Idempotency

Resume verification used a guarded request function after the recovery was complete.

| Item | Result |
| --- | --- |
| Runtime seconds | `27.808218399994075` |
| Requests made reported | `0` |
| Guarded request calls | `0` |
| Daily result count | `31` |
| Complete results | `31` |
| Skipped existing count | `31` |
| Partition inventory unchanged | `true` |
| Partition file count before / after | `96 / 96` |
| Partition row counts unchanged | `true` |
| Manifest JSON parseable | `true` |
| Manifest hash changed | `true`, because run metadata changed while partition inventory stayed byte-stable |

This confirms that a same-range resume does not call the network and does not rewrite verified raw, normalized, or status partition files. Manifest metadata may change on a resumed run.
After the resume verification, the current manifest SHA-256 was `032c12b26db5499fc199186f002e41724e0ef336fb931b20fa2dc96219603bb4`.

## Dry-Run Verification

Dry-run verification used a guarded request function and performed no network calls or writes.

| Item | Result |
| --- | --- |
| Runtime seconds | `0.0037366000469774008` |
| `dry_run` | `true` |
| Network requests planned | `31` |
| Writes planned | `false` |
| Guarded request calls | `0` |
| Partition plan count | `31` |
| First planned date | `2010-01-01` |
| First planned paths | raw, normalized, and status paths for `date=2010-01-01` |
| Last planned date | `2010-01-31` |
| Last planned paths | raw, normalized, and status paths for `date=2010-01-31` |
| Inventory unchanged | `true` |
| File count before / after | `97 / 97` |

## V1 Comparison

The January 2010 recovered REN partitions were compared with the frozen v1 production CSV.

| Metric | Result |
| --- | --- |
| Aligned rows | `2,976` |
| Exact match count | `2,976` |
| Mean absolute error | `0.0` |
| Maximum absolute difference | `0.0` |
| Pearson correlation | `1.0` |
| Dates with nonzero differences | None |

This confirms exact overlap for the bounded January 2010 pilot. It does not prove full historical equivalence.

## File-Layout Decision

Decision A: keep daily raw, normalized, and status partitions for v2 raw REN ingestion.

Rationale:

- The partition layout isolated the sandbox failure to Jan 4-31 and allowed selective retry without replacing verified partitions.
- Per-day metadata, checksums, and validation status remain inspectable.
- A complete backfill can use the same layout and manifest conventions without a minimal further code change.

Compact derivatives may be produced separately in a later approved step. They should not replace the raw v2 partition layout. After transient unavailable failures, the required recovery path is to rerun with `--resume --retry-unavailable`.

## Operational Readiness

Assessment: ready for a complete REN production backfill only with explicit operator approval.

Readiness conditions:

- Use a conservative request delay.
- Keep outputs under the ignored v2 output path.
- Monitor request counts, status counts, file inventory, row counts, warnings, and manifest parseability.
- Preserve v1 as immutable; do not append to, overwrite, or reinterpret v1 files.
- Treat network/socket failures as recoverable only when they leave strict status-only unavailable partitions.
- Stop and inspect any corrupt, partial, raw-only, normalized-only, checksum-mismatched, incomplete, or schema-invalid partition.

No full backfill should be started automatically from this checkpoint.

## Future Full-Backfill Template

The latest separately verified date for this readiness note is `2026-06-27`.

NOT RUN:

```python
from pathlib import Path

from scripts.ingest_ren_production_v2 import run_ingestion

result = run_ingestion(
    start_date="2010-01-01",
    end_date="2026-06-27",
    output_root=Path("data/raw/v2/production"),
    request_delay=1.0,
    resume=True,
    retry_unavailable=True,
    compare_v1_csv=Path("data/raw/ReparticaoProducao.csv"),
)
```

The command requires explicit operator approval and live REN network access before use.

## Monitoring Checklist

Before running:

- Confirm `git status --short` and ensure only expected uncommitted files are present.
- Confirm output root is `data/raw/v2/production` and remains ignored.
- Confirm the requested end date and any separately verified latest-date evidence.
- Confirm `--resume --retry-unavailable` is used for recovery runs after transient unavailable failures.

During running:

- Track elapsed time, request count, request delay, and any HTTP or socket failures.
- Track daily result status counts and warnings.
- Stop on repeated non-transient failures, corrupt metadata, or validator errors.

After running:

- Count raw, normalized, and status files.
- Validate expected 96 rows per complete day unless a documented incomplete date is accepted.
- Check manifest JSON parseability, row count, coverage start/end, unavailable dates, incomplete dates, and checksums.
- Compare overlapping dates with frozen v1 where applicable.
- Confirm v1 files and existing model artifacts were not modified.

## Risks, Fallbacks, And Rollback

Risks:

- REN historical values may differ from frozen v1 outside the January 2010 pilot.
- REN service availability, throttling, socket permissions, or transient HTTP failures may interrupt a long run.
- Manifest metadata can change on resume even when partition files remain byte-stable.
- REN timezone semantics and source status remain unresolved for downstream daily aggregation and model use.

Fallbacks:

- For strict status-only unavailable partitions, rerun the same range with `--resume --retry-unavailable`.
- For verified partitions, rerun with plain `--resume` to skip without network calls.
- For any other existing unverified or corrupt partition, stop and inspect manually before choosing a repair path.

Rollback:

- Do not mutate v1.
- Because v2 outputs are isolated under the ignored v2 raw path, rollback can be handled by archiving or removing the affected v2 output tree only after explicit approval.
- Do not delete data, manifests, or pilot outputs without operator approval.

## Acceptance Criteria Checklist

- [x] January 2010 recovered to 31 complete daily partitions.
- [x] Jan 4-31 status-only unavailable partitions were retried without overwriting verified Jan 1-3 partitions.
- [x] Previously verified smoke partitions, including `2026-06-27`, were preserved.
- [x] Same-range resume made zero guarded network calls and skipped all verified partitions.
- [x] Dry run planned 31 partitions without network calls or writes.
- [x] January row count was `2,976` with no missing or duplicate timestamps.
- [x] January v1 overlap comparison was exact across all aligned rows.
- [x] Manifest was parseable and recorded coverage through `2026-06-27`.
- [x] V1 data, models, scalers, notebooks, and baselines were not changed.
- [x] Full backfill remains not run and requires explicit operator approval.

Step 2B and downstream training remain paused.

---

## Phase 2 REN Full Backfill Acceptance Audit

Original file before consolidation: `PHASE_2_REN_FULL_BACKFILL_ACCEPTANCE.md`.

## Audit Scope

This document records the Phase 2 Step 2A.11 local acceptance audit for the REN v2 raw production full backfill. It is an audit of the locally generated raw REN v2 backfill artifacts and manifest evidence only.

The audit covers coverage, status metadata, row counts, timestamp/DST behavior, value bounds, partition integrity, checksums, manifest consistency, frozen v1 overlap, and resume-safety implications.

No network calls were made for this audit. No notebooks, pipelines, data regeneration, model training, scaler fitting, or downstream aggregation were run. Ignored v2 backfill data was not altered by this documentation step.

## Verdict

Qualified acceptance: the raw REN v2 full backfill is locally auditable and checksum-consistent.

This verdict is qualified by known caveats:

- Six dates are unavailable in the local backfill.
- One date, `2010-03-28`, is marked incomplete because legacy status metadata expected 96 rows, while the physical Europe/Lisbon spring-DST expectation is 92 intervals and matches the CSV.
- The manifest requested-ranges union covers fewer dates than status coverage.
- REN timezone semantics, source license, and provisional/final source-status behavior remain unresolved.
- Frozen v1 overlap contains late-period revisions and therefore does not support replacing v1 without a separate approved migration, validation, and re-baselining step.

This audit does not approve downstream daily aggregation, v2 processed features, scaler validity, model validity, metric comparison, or replacement of frozen v1 artifacts.

## Repository And Data Locations

Repository context:

- Roadmap phase: Phase 2, data validation and sanity checks.
- Step: Phase 2 Step 2A.11, REN full backfill acceptance audit.
- Audit document: `docs/PHASE_2.md`.
- Commit message for this audit: `docs: accept REN full backfill audit`.

Data context:

- Raw REN v2 backfill artifacts remain under the ignored v2 production data area.
- Frozen v1 production data remains the immutable comparison baseline.
- No v1 raw data, processed data, models, scalers, notebooks, reports, or baselines were modified or approved for mutation.

## Evidence Summary

| Item | Evidence |
| --- | ---: |
| Coverage start | `2010-01-01` |
| Coverage end | `2026-06-27` |
| Calendar dates covered | `6,022` |
| Total files | `18,055` |
| Total storage | `279,440,105 bytes` |
| JSON files | `12,039` |
| CSV files | `6,016` |
| Status rows, complete | `6,015` |
| Status rows, incomplete | `1` |
| Status rows, unavailable | `6` |
| Status rows, invalid | `0` |
| Manifest row sum | `577,532` |
| Status row sum | `577,532` |
| Manifest SHA-256 | `97cdd28e609a7a4573c4dda4e5fddd55fd6be60f00742c7d03b14897ddd68cd0` |

## Coverage And Status

The local status coverage spans `2010-01-01` through `2026-06-27`, covering `6,022` calendar dates.

Status distribution:

- Complete: `6,015` dates.
- Incomplete: `1` date.
- Unavailable: `6` dates.
- Invalid: `0` dates.

Unavailable dates:

- `2014-05-03`
- `2016-02-03`
- `2016-02-04`
- `2021-10-03`
- `2023-08-30`
- `2025-08-02`

Incomplete date:

- `2010-03-28`

For `2010-03-28`, the CSV contains `92` rows, which matches the physical Europe/Lisbon spring-DST expectation. The incomplete status is therefore attributed to older status metadata that expected `96` rows for every date, not to physically missing intervals in the CSV.

The current DST-aware REN normalized-day validator accepts all `6,016` normalized CSV partitions as complete; the single incomplete status above is a persisted metadata caveat, not a current validation failure.

## DST And Timestamp Policy

The row evidence is consistent with physical Europe/Lisbon DST behavior:

| Interval class | Dates | Rows |
| --- | ---: | ---: |
| Ordinary days | `5,983` | `574,368` |
| Spring DST days | `17` | `1,564` |
| Fall DST days | `16` | `1,600` |

Timestamp checks:

- Timestamp identity duplicates: `0`.
- Physical missing intervals: `0`.
- Expected fall local-wall-clock duplicates: `16` dates / `128` rows.

Policy implication: fall-DST repeated local wall-clock intervals are expected physical behavior and must be preserved by timestamp handling. Spring-DST days may contain `92` physical intervals rather than `96`; legacy metadata expecting `96` rows requires interpretation rather than silent correction.

REN timezone semantics remain unresolved for downstream daily aggregation. This audit accepts only the local raw backfill integrity evidence and does not approve canonical daily aggregation boundaries.

## Value Validation

All `577,532` rows have finite, non-negative values in unit `MW`.

Observed production-value bounds:

- Minimum: `0.0 MW`.
- Maximum: `5094.6 MW`.

No value evidence in this audit indicates negative production, non-finite values, or mixed units.

## Partition And Checksum Integrity

The backfill tree contains `18,055` files totaling `279,440,105 bytes`:

- `12,039` JSON files.
- `6,016` CSV files.

Checksum and partition checks:

- Manifest records `18,054` file checksums.
- Missing files: `0`.
- Checksum mismatches: `0`.
- Raw/normalized partition mismatches: `0`.

The local partition evidence supports the qualified acceptance verdict for raw v2 storage integrity.

## Manifest Assessment

The manifest is parseable and deterministic serialization matches the file.

Manifest SHA-256:

```text
97cdd28e609a7a4573c4dda4e5fddd55fd6be60f00742c7d03b14897ddd68cd0
```

Manifest row evidence:

- Manifest row sum: `577,532`.
- Status row sum: `577,532`.
- Manifest checksum count: `18,054`.

Manifest caveat: the requested-ranges union covers `5,665` dates, while status coverage has `6,022` dates. Therefore, `357` status-covered dates are not represented in requested ranges.

The manifest is accepted as parseable and checksum-consistent for this local audit. It is not described as perfect because the requested-range gap remains a caveat to resolve or document before any stronger operational claim.

## Frozen V1 Overlap

Frozen v1 overlap through `2025-04-28` was compared to the REN v2 backfill.

| Metric | Result |
| --- | ---: |
| V1 rows | `537,308` |
| REN-aligned rows | `536,828` |
| Exact matches | `515,533` |
| Nonzero differences | `21,295` rows |
| Dates with nonzero differences | `258` |
| First nonzero-difference date | `2024-06-01` |
| Last nonzero-difference date | `2025-04-28` |
| MAE | `0.2445271111 MW` |
| Maximum absolute difference | `130.0 MW` |
| Correlation | `0.9999981644` |

REN aligns fewer rows than v1 because five unavailable dates fall inside v1 coverage.

Interpretation: the overlap is extremely close overall, but late-period differences confirm that v2 must not be silently substituted for frozen v1. The differences are consistent with unresolved source revision behavior or source-status differences, and this audit does not identify their cause.

## Resume Safety

Prior readiness evidence established the intended v2 raw layout and recovery behavior:

- Daily raw, normalized, and status partitions isolate failures by date.
- Verified partitions can be skipped with resume behavior.
- Strict status-only unavailable partitions can be retried through the approved retry path.
- Verified raw and normalized partitions must not be overwritten during recovery.

For this full-backfill audit, the checksum and partition evidence shows no missing files, no checksum mismatches, and no raw/normalized partition mismatches. That supports local resume-safety and inspectability of the partitioned raw v2 layout.

Any future retry or repair must remain explicit, bounded, and non-destructive. Corrupt, partial, raw-only, normalized-only, checksum-mismatched, or otherwise ambiguous partitions require manual inspection before choosing a repair path.

## Acceptance Checklist

- [x] Coverage spans `2010-01-01` through `2026-06-27`.
- [x] Status coverage includes `6,022` calendar dates.
- [x] Status counts are recorded: `6,015` complete, `1` incomplete, `6` unavailable, `0` invalid.
- [x] Unavailable dates are explicitly listed.
- [x] The single incomplete date is explained by legacy 96-row status metadata versus physical spring-DST expectation.
- [x] Manifest and status row sums match at `577,532`.
- [x] Ordinary, spring-DST, and fall-DST interval classes are recorded.
- [x] Timestamp identity duplicates are `0`.
- [x] Physical missing intervals are `0`.
- [x] Expected fall local-wall-clock duplicates are documented.
- [x] Values are finite, non-negative, and in `MW` for all rows.
- [x] File inventory and storage totals are recorded.
- [x] Manifest checksum evidence reports no missing files, checksum mismatches, or raw/normalized partition mismatches.
- [x] Manifest parseability, deterministic serialization, and SHA-256 are recorded.
- [x] Manifest requested-range gap is documented as a caveat.
- [x] Frozen v1 overlap is summarized without approving v1 replacement.
- [x] No network calls were made for this audit.
- [x] No downstream aggregation, scaler fitting, model training, notebook execution, or pipeline regeneration is approved by this document.

## Risks And Caveats

Known caveats:

- Six unavailable dates remain in the raw v2 backfill.
- The `2010-03-28` status is incomplete because of legacy expected-row metadata, even though the CSV matches physical spring-DST behavior.
- Manifest requested ranges do not cover all status-covered dates.
- REN timezone semantics remain unresolved.
- REN license and attribution requirements remain unresolved.
- REN provisional/final source-status behavior remains unresolved.
- Late v1 overlap revisions exist from `2024-06-01` through `2025-04-28`.
- The cause of the nonzero v1 overlap differences is not resolved by this audit.

Operational risks:

- Treating local raw integrity as daily aggregation approval could introduce timezone-boundary errors.
- Treating v2 overlap closeness as v1 replacement approval could invalidate existing baselines.
- Treating current v1 scalers or trained models as v2-compatible would violate the Phase 2 data-contract decision.

## Non-Goals

This audit does not:

- Run network calls.
- Download or regenerate data.
- Modify ignored v2 backfill data.
- Modify frozen v1 data.
- Execute notebooks.
- Run pipelines.
- Create processed v2 features.
- Approve downstream daily aggregation.
- Refit scalers.
- Retrain models.
- Re-baseline metrics.
- Approve v2 model or scaler validity.
- Approve replacing v1 with v2.
- Start Step 2B or Phase 3.

## Rollback And Containment

This documentation step creates only this audit file. Rollback for this step is to remove `docs/PHASE_2.md` before any future staging or commit, if the audit document is rejected.

Raw v2 backfill artifacts remain isolated under ignored v2 storage and were not modified by this documentation step. Any rollback, deletion, archiving, or repair of v2 data artifacts requires separate explicit approval.

Frozen v1 data, processed files, models, scalers, notebooks, reports, and baselines remain unchanged and must continue to be treated as the current reproducible baseline.

## Final Status

Phase 2 Step 2A.11 status: qualified local acceptance audit documented.

The raw REN v2 full backfill is locally auditable and checksum-consistent, subject to the caveats recorded above. This is not approval for downstream aggregation, v1 replacement, v2 model validity, scaler validity, or metric promotion.

Step 2B was not started. Phase 3 was not started.

---

## Phase 2 ERA5-Land Grid Readiness

Original file before consolidation: `PHASE_2_ERA5_LAND_GRID_READINESS.md`.

## Scope

This note records the Phase 2 Step 2A.13 grid-readiness policy and bounded July 2023 readiness pilot for ERA5-Land v2 weather ingestion.

It does not perform historical backfill, start Step 2B, regenerate features, refit scalers, retrain models, execute notebooks, or start Phase 3.

## Root Cause Evidence

The ignored Step 2A.12 status at `data/raw/v2/weather/era5_land/metadata/station_id=1200551/period=2023-07-01_2023-07-31/status.json` records a completed CDS retrieval for station `1200551` and period `2023-07-01` through `2023-07-31`, but the requested nearest grid cell returned all-null required variables.

Recorded request evidence:

| Field | Value |
| --- | --- |
| station_id | `1200551` |
| station coordinate | `41.648875`, `-8.804606` |
| requested ERA5-Land area | `[41.6, -8.8, 41.6, -8.8]` |
| period | `2023-07-01` through `2023-07-31` |
| validation status | `invalid` |
| null evidence | 744 null values for each required weather value derived from `2m_temperature`, `10m_u_component_of_wind`, and `10m_v_component_of_wind` |

Interpretation: the v2 point-extraction contract needs an explicit, deterministic policy for coastal or otherwise invalid nearest ERA5-Land grid cells. The station coordinate remains the station evidence; the selected ERA5 grid coordinate is separate operational metadata.

## Evaluated Alternatives

| Alternative | Decision | Rationale |
| --- | --- | --- |
| Keep single-cell nearest grid only | Preserved only as default/radius-0 compatibility | It reproduces Step 2A.12 behavior but can block valid coastal stations. |
| Search an unbounded neighbourhood | Rejected | It expands spatial ambiguity and request count beyond the approved readiness step. |
| Download full regional grids | Rejected for Step 2A.13 | The v2 contract selected point extraction; full-grid extraction is a separate future decision. |
| Silently impute or clean all-null cells | Rejected | Validation must remain separate from cleaning and must not hide source-grid failures. |
| Deterministic nearest-valid 3x3 search | Selected | It is bounded, auditable, preserves station coordinates, and records the selected grid coordinate and distance. |

## Final Policy

Step 2A.13 adds optional grid policy `nearest-valid` with `--grid-search-radius 1`.

Rules:

- Candidate set is bounded to one ERA5-Land grid step around the nearest rounded grid cell, for at most 9 candidates.
- Candidate ordering is deterministic: haversine distance ascending, then absolute latitude delta, absolute longitude delta, grid latitude, and grid longitude.
- One bounded neighbourhood NetCDF is retrieved per station/chunk, then each candidate cell is evaluated from that shared file against the existing hourly and daily validation rules.
- Required variables must not be all-null or partially invalid.
- The first valid candidate is selected.
- The station latitude and longitude remain the requested station coordinates in normalized outputs.
- The selected ERA5 grid latitude, grid longitude, selected-candidate rank, and station-to-grid distance are recorded in status metadata and manifest metadata.
- Default single-cell/radius-0 behavior and paths are preserved for compatibility.
- Radius-1 nearest-valid outputs use policy-specific paths under `grid_policy=nearest_valid_r1` so prior Step 2A.12 evidence is not overwritten.
- Candidate evidence points to the shared raw neighbourhood NetCDF rather than separate per-candidate downloads.

Readiness statuses:

| Selection outcome | Status |
| --- | --- |
| nearest grid candidate validates | `READY` |
| non-nearest candidate validates | `READY_WITH_WARNING` |
| no candidate validates | `BLOCKED` |

## Maximum Search Radius And Distance

The approved search radius is one ERA5-Land grid step. ERA5-Land grid spacing for this ingestion helper is `0.1` degree, so the candidate neighbourhood is a 3x3 grid around the nearest rounded station point.

The exact station-to-grid distance is calculated per selected candidate using haversine distance and recorded as `station_to_grid_distance_km`. A fixed kilometre maximum is not hardcoded because longitude spacing varies with latitude; the operational bound is the one-step 3x3 candidate set.

## Bounded Readiness Pilot

The bounded readiness pilot used:

- Dataset: `reanalysis-era5-land`
- Period: `2023-07-01` through `2023-07-31`
- Variables: `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`
- Station set: all 17 approved exact-match IPMA station coordinates
- Grid policy: `nearest-valid`
- Search radius: `1`
- Output path: `data/raw/v2/weather/era5_land/grid_policy=nearest_valid_r1/`

The diagnostic run for station `1200551` made one bounded 3x3 neighbourhood request with area `[41.7, -8.9, 41.5, -8.7]`. The nearest candidate `41.6, -8.8` remained all-null, while candidate rank `1` at `41.7, -8.8` validated successfully.

The full-station readiness run used `--resume`, reused the verified `1200551` diagnostic partition, and made 16 additional station/month CDS requests. The aggregate output contains 31 daily rows with `point_count=17`, `expected_point_count=17`, and `missing_point_count=0` for every day.

Readiness summary:

| Status | Count |
| --- | ---: |
| `READY` | 15 |
| `READY_WITH_WARNING` | 2 |
| `BLOCKED` | 0 |

| station_id | period | grid_policy | search_radius | readiness_status | selected_grid | station_to_grid_distance_km | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1200545` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `41.2, -8.7` | `4.039` | nearest candidate valid |
| `1200548` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `40.2, -8.5` | `6.232` | nearest candidate valid |
| `1200551` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY_WITH_WARNING` | `41.7, -8.8` | `5.698` | nearest candidate all-null; selected rank `1` |
| `1200554` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `37.0, -8.0` | `3.099` | nearest candidate valid |
| `1200558` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `38.5, -7.9` | `3.982` | nearest candidate valid |
| `1200560` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `40.7, -7.9` | `1.695` | nearest candidate valid |
| `1200562` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `38.0, -7.9` | `3.988` | nearest candidate valid |
| `1200567` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `41.3, -7.7` | `3.206` | nearest candidate valid |
| `1200570` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `39.8, -7.5` | `4.673` | nearest candidate valid |
| `1200571` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `39.3, -7.4` | `1.791` | nearest candidate valid |
| `1200575` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `41.8, -6.7` | `3.577` | nearest candidate valid |
| `1210622` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `41.6, -8.5` | `5.560` | nearest candidate valid |
| `1210683` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `40.5, -7.3` | `4.656` | nearest candidate valid |
| `1210702` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `40.6, -8.7` | `5.200` | nearest candidate valid |
| `1210718` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `39.8, -8.8` | `2.662` | nearest candidate valid |
| `1210734` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `39.2, -8.7` | `3.162` | nearest candidate valid |
| `1210770` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY_WITH_WARNING` | `38.6, -8.9` | `5.787` | nearest candidate all-null; selected rank `1` |

## Manifest And Status Evidence

For each partition, status JSON includes:

- `grid_selection.grid_policy`
- `grid_selection.grid_search_radius`
- `grid_selection.readiness_status`
- `grid_selection.selected_candidate_rank`
- `grid_selection.selected_grid_coordinate`
- `grid_selection.station_to_grid_distance_km`
- `grid_selection.candidate_evidence`

Manifest metadata includes the same policy contract and partition-level grid-selection evidence.

## Unresolved Risks

- Neighbour selection may introduce a small spatial shift for affected stations. The selected grid coordinate and distance must be reviewed before full historical ingestion.
- The July 2023 readiness pilot proves operational readiness for the sampled calendar month, but it does not prove that every historical month will have identical candidate validity.
- ERA5-Land remains a v2 weather data contract and is not compatible with v1 scalers or trained models without v2 refitting, retraining, and re-baselining.
- CDS service behavior, licensing, accepted terms, and operational availability remain external dependencies for live ingestion.

## Historical Backfill Decision

The approved 17-station ERA5-Land grid policy is operationally ready for a future historical ingestion command because all 17 approved stations validated for the bounded July 2023 readiness pilot and no station remains `BLOCKED`.

Historical backfill was not started by this checkpoint. A future backfill must use the documented `nearest-valid` radius-1 policy, preserve v1 data, write only to approved v2 paths, and keep manifest/status evidence for every station/month partition.

No full historical backfill, Step 2B work, feature regeneration, scaler refitting, model training, notebook execution, or Phase 3 work was started by Step 2A.13.

---

## Phase 2 ERA5-Land Full Backfill Acceptance

Original file before consolidation: `PHASE_2_ERA5_LAND_FULL_BACKFILL_ACCEPTANCE.md`.

## Scope And Non-Goals

This document records the Phase 2 Step 2A.14 local audit and acceptance decision for ERA5-Land v2 historical weather outputs under:

```text
data/raw/v2/weather/era5_land/grid_policy=nearest_valid_r1/
```

The audit is based only on actual local ignored files present in that policy directory. Dry-run request planning, planned request ranges, or intended backfill commands are not used as evidence that a chunk exists or is missing.

This documentation-only step did not run network calls, repair files, overwrite files, backfill missing periods, execute notebooks, regenerate features, modify generated data, start Step 2B, train models, refit scalers, or start Phase 3.

## Verdict

Decision: `FAIL`.

The actual local ERA5-Land v2 weather outputs are internally complete for the two period partitions present locally, but they do not constitute a full historical backfill for `2010-01-01` through `2026-06-27`.

ERA5-Land and REN v2 integration must not begin from this weather backfill because the full historical weather period is absent.

## Expected Versus Actual Coverage

Requested full coverage: `2010-01-01` through `2026-06-27`, chunked by calendar month with the final chunk ending on `2026-06-27`.

| Item | Expected | Actual | Result |
| --- | ---: | ---: | --- |
| Monthly period chunks | `198` | `2` | `196` missing |
| Station-period partitions for 17 approved stations | `3,366` | `34` | `3,332` missing |
| Hourly rows for station-period outputs | `2,456,976` | `23,664` | incomplete full coverage |
| Daily point rows for station-period outputs | `102,374` | `986` | incomplete full coverage |
| Daily aggregate rows | `6,022` | `58` | incomplete full coverage |

Actual local periods present:

| Period | Calendar days | Station count | Station-periods | Hourly rows | Daily point rows | Aggregate rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2023-07-01_2023-07-31` | `31` | `17` | `17` | `12,648` | `527` | `31` |
| `2026-06-01_2026-06-27` | `27` | `17` | `17` | `11,016` | `459` | `27` |

The actual local date coverage is therefore:

- `2023-07-01T00:00:00Z` through `2023-07-31T23:00:00Z` for 17 stations.
- `2026-06-01T00:00:00Z` through `2026-06-27T23:00:00Z` for 17 stations.

Missing periods are exactly every expected monthly period except `2023-07-01_2023-07-31` and `2026-06-01_2026-06-27`. In compact form, the missing coverage is:

- `2010-01-01_2010-01-31` through `2023-06-01_2023-06-30`.
- `2023-08-01_2023-08-31` through `2026-05-01_2026-05-31`.

For station-periods, each of the 17 approved stations is missing each of those 196 periods, for `3,332` missing station-periods. No incomplete station-period was found among the 34 actual local partitions.

## Station Evidence

The actual outputs represent all 17 approved mapped station IDs:

```text
1200545, 1200548, 1200551, 1200554, 1200558, 1200560, 1200562, 1200567, 1200570, 1200571, 1200575, 1210622, 1210683, 1210702, 1210718, 1210734, 1210770
```

Status counts across the 34 actual station-period partitions:

| Readiness status | Count |
| --- | ---: |
| `READY` | `30` |
| `READY_WITH_WARNING` | `4` |
| `BLOCKED` | `0` |

All 34 status files report validation status `complete`.

## Selected Grid Coordinates

The selected grid coordinates and station-to-grid distances are stable across both actual periods.

| station_id | selected grid latitude | selected grid longitude | distance km | readiness |
| --- | ---: | ---: | ---: | --- |
| `1200545` | `41.2` | `-8.7` | `4.039268` | `READY` |
| `1200548` | `40.2` | `-8.5` | `6.231658` | `READY` |
| `1200551` | `41.7` | `-8.8` | `5.697706` | `READY_WITH_WARNING` |
| `1200554` | `37.0` | `-8.0` | `3.098502` | `READY` |
| `1200558` | `38.5` | `-7.9` | `3.981716` | `READY` |
| `1200560` | `40.7` | `-7.9` | `1.694505` | `READY` |
| `1200562` | `38.0` | `-7.9` | `3.988029` | `READY` |
| `1200567` | `41.3` | `-7.7` | `3.206311` | `READY` |
| `1200570` | `39.8` | `-7.5` | `4.672728` | `READY` |
| `1200571` | `39.3` | `-7.4` | `1.790695` | `READY` |
| `1200575` | `41.8` | `-6.7` | `3.577181` | `READY` |
| `1210622` | `41.6` | `-8.5` | `5.560361` | `READY` |
| `1210683` | `40.5` | `-7.3` | `4.655948` | `READY` |
| `1210702` | `40.6` | `-8.7` | `5.200200` | `READY` |
| `1210718` | `39.8` | `-8.8` | `2.661790` | `READY` |
| `1210734` | `39.2` | `-8.7` | `3.161513` | `READY` |
| `1210770` | `38.6` | `-8.9` | `5.786876` | `READY_WITH_WARNING` |

The four `READY_WITH_WARNING` station-periods are:

| station_id | period | warning |
| --- | --- | --- |
| `1200551` | `2023-07-01_2023-07-31` | Nearest requested grid cell was invalid; selected nearest valid neighbour rank `1` at `41.7, -8.8`. |
| `1200551` | `2026-06-01_2026-06-27` | Nearest requested grid cell was invalid; selected nearest valid neighbour rank `1` at `41.7, -8.8`. |
| `1210770` | `2023-07-01_2023-07-31` | Nearest requested grid cell was invalid; selected nearest valid neighbour rank `1` at `38.6, -8.9`. |
| `1210770` | `2026-06-01_2026-06-27` | Nearest requested grid cell was invalid; selected nearest valid neighbour rank `1` at `38.6, -8.9`. |

This warning handling is accepted as local partition evidence for those actual periods only. It does not compensate for the absent historical periods.

## Row, Timestamp, Null, And Range Evidence

For the 34 actual station-periods, status timestamp coverage matches requested coverage:

| Period | Expected start | Expected end | Actual start | Actual end | Stations |
| --- | --- | --- | --- | --- | ---: |
| `2023-07-01_2023-07-31` | `2023-07-01T00:00:00Z` | `2023-07-31T23:00:00Z` | `2023-07-01T00:00:00Z` | `2023-07-31T23:00:00Z` | `17` |
| `2026-06-01_2026-06-27` | `2026-06-01T00:00:00Z` | `2026-06-27T23:00:00Z` | `2026-06-01T00:00:00Z` | `2026-06-27T23:00:00Z` | `17` |

Timestamp checks inside actual partitions:

| Check | Finding |
| --- | ---: |
| Hourly duplicate timestamps | `0` |
| Hourly missing timestamps | `0` |
| Hourly unexpected timestamps | `0` |
| Daily point duplicate dates | `0` |
| Daily aggregate duplicate dates | `0` |

Null and non-finite findings across actual CSV outputs:

| Output | Finding |
| --- | --- |
| Hourly | `355` null `wind_direction_deg_from` values; no nulls in temperature, wind components, wind speed, coordinates, station IDs, or timestamps. |
| Daily points | `16` null `vector_mean_wind_direction_deg_from` values; no nulls in temperature means/min/max, wind-speed metrics, vector components, counts, coordinates, station IDs, or dates. |
| Daily aggregate | `1` null `vector_mean_wind_direction_deg_from` value; no nulls in point counts, temperature means, wind-speed means, vector components, or dates. |
| Non-finite values | `0` found in scanned numeric weather columns. |

The wind-direction nulls align with the documented calm/near-calm direction policy and are not evidence of missing rows.

Observed ranges across actual outputs had no range violations under the documented weather sanity bounds:

| Field group | Observed min | Observed max |
| --- | ---: | ---: |
| Hourly `temperature_2m_k` | `281.04678` | `312.1629` |
| Hourly `temperature_2m_c` | `7.8967896` | `39.01291` |
| Hourly `u10_m_s` | `-9.299988` | `7.988373` |
| Hourly `v10_m_s` | `-9.165871` | `6.2831697` |
| Hourly `wind_speed_m_s` | `0.01982022` | `9.571904` |
| Hourly `wind_direction_deg_from` | `0.020324707` | `359.99652` |
| Daily point `temperature_2m_c_mean` | `13.728826` | `30.491388` |
| Daily point `wind_speed_m_s_mean` | `0.82374126` | `7.843986` |
| Daily point `vector_mean_wind_direction_deg_from` | `2.65445241141947` | `357.53920910263287` |
| Aggregate `temperature_2m_c_mean` | `16.784411588235297` | `26.549537882352944` |
| Aggregate `wind_speed_m_s_mean` | `1.9895143588235291` | `4.650475676470587` |
| Aggregate `vector_mean_wind_direction_deg_from` | `35.51056594012775` | `358.7976597753362` |

Recorded units include UTC timestamps and dates, degrees for station/grid coordinates, `K` and degree Celsius for temperature, `m s-1` for wind components and speed, `degree_from` for wind direction, and boolean calm/near-calm flags.

## Aggregate Coverage

Both actual aggregate period outputs have complete point coverage:

| point_count | expected_point_count | missing_point_count | Aggregate rows |
| ---: | ---: | ---: | ---: |
| `17` | `17` | `0` | `58` |

This confirms complete aggregate station coverage for the two actual periods only. It does not establish aggregate coverage for the missing 196 monthly periods.

## File Inventory And Storage

The policy directory contains `141` files totaling `9,303,253` bytes.

| Top-level area | File count |
| --- | ---: |
| `raw` | `34` |
| `hourly` | `34` |
| `daily_points` | `34` |
| `metadata` | `34` |
| `daily_aggregate` | `2` |
| `comparisons` | `2` |
| `manifests` | `1` |

| Extension | File count |
| --- | ---: |
| `.nc` | `34` |
| `.csv` | `72` |
| `.json` | `35` |

CSV data rows:

| CSV output | Rows |
| --- | ---: |
| Hourly station outputs | `23,664` |
| Daily point outputs | `986` |
| Daily aggregate outputs | `58` |
| Prior pilot comparison outputs | `21` |
| Total CSV data rows | `24,729` |

The weather-output row total excluding prior-pilot comparison files is `24,708`.

## Checksum And Manifest Assessment

Status-file checksum evidence:

- `102` status-referenced checksums were checked: one raw NetCDF, one hourly CSV, and one daily-points CSV for each of 34 station-periods.
- Missing status-referenced files: `0`.
- Status checksum mismatches: `0`.

Manifest evidence:

- Manifest path: `data/raw/v2/weather/era5_land/grid_policy=nearest_valid_r1/manifests/era5_land_weather_manifest.json`.
- Manifest SHA-256: `c832181cf6e6577081569f5bc3e1e961ff96cb3e3042ea5b24867839b5bc9fca`.
- Manifest `coverage_start`: `2026-06-01`.
- Manifest `coverage_end`: `2026-06-27`.
- Manifest `row_count`: `27`.
- Manifest station count: `17`.
- Manifest checksum entries checked: `70`.
- Manifest checksum missing files: `0`.
- Manifest checksum mismatches: `0`.

Assessment: checksums are accurate for the files referenced by status metadata and by the manifest. The manifest is not a full-backfill manifest: it records the latest actual period coverage only and explicitly does not claim full historical coverage. It also does not represent both actual local periods as one complete historical dataset. Therefore, manifest checksum consistency does not change the full-backfill decision from `FAIL`.

## Duplicate Partition Assessment

No duplicate partition keys were found for:

- Raw NetCDF station-period files.
- Hourly station-period CSV files.
- Daily point station-period CSV files.
- Status station-period JSON files.
- Daily aggregate period CSV files.
- Prior pilot comparison period CSV files.

No duplicate timestamps or dates were found inside the actual hourly, daily point, or daily aggregate CSV outputs.

## Data And Artifact Safety

This audit did not modify generated data or source artifacts. It did not modify v1 data, REN files, notebooks, models, scalers, manifests, dependencies, configuration, source code, or existing documentation.

The only intended tracked-file change for this documentation step is this file:

```text
docs/PHASE_2.md
```

## Acceptance Checklist

- [x] Actual local evidence was used instead of dry-run request planning.
- [x] Expected month count is recorded as `198`.
- [x] Actual month count is recorded as `2`.
- [x] Expected station-period count is recorded as `3,366`.
- [x] Actual station-period count is recorded as `34`.
- [x] Missing periods and station-periods are recorded.
- [x] Present periods and station IDs are recorded.
- [x] Status, raw, hourly, daily point, and aggregate coverage are recorded.
- [x] Requested versus actual date coverage is recorded.
- [x] Expected versus actual hourly, daily point, and aggregate rows are recorded.
- [x] Duplicate and missing timestamp findings are recorded.
- [x] Null, non-finite, unit, and range findings are recorded.
- [x] Selected grid coordinates and distances are recorded.
- [x] `READY_WITH_WARNING` handling for `1200551` and `1210770` is recorded.
- [x] Aggregate `point_count=17`, `expected_point_count=17`, and `missing_point_count=0` evidence is recorded for actual periods.
- [x] Checksum and manifest accuracy are assessed.
- [x] File counts, row counts, and storage size are recorded.
- [x] Duplicate partition assessment is recorded.
- [x] Acceptance decision is explicit: `FAIL`.
- [x] ERA5-Land and REN v2 integration decision is explicit: must not begin from this weather backfill.
- [x] No v1, REN, model, scaler, notebook, source-code, dependency, configuration, or generated-output changes are approved or made by this audit.

## Stop Gate

Phase 2 Step 2A.14 status: full-backfill acceptance record created; acceptance decision is `FAIL`.

The two actual ERA5-Land period partitions may remain useful as local evidence for grid-policy behavior and partition-level validation, but they do not satisfy the requested full historical backfill. Step 2B, ERA5-Land/REN v2 integration, feature regeneration, scaler refitting, model training, metric re-baselining, notebook execution, and Phase 3 remain paused.

---

## Phase 2 ERA5-Land Monthly-Bbox Full Backfill Acceptance

Original file before consolidation: `PHASE_2_ERA5_LAND_MONTHLY_BBOX_FULL_BACKFILL_ACCEPTANCE.md`.

## Scope / Safety

This document records the Phase 2 Step 2A.16 documentation-only acceptance checkpoint for the ERA5-Land v2 monthly-bbox full backfill outputs under:

```text
data/raw/v2/weather/era5_land/grid_policy=nearest_valid_r1/request_mode=monthly_bbox/
```

The evidence in this record comes from verified read-only local audit results for the monthly-bbox output tree. This documentation task made no network calls, ran no ingestion scripts, executed no notebooks, started no training, and did not modify, repair, regenerate, or delete generated ERA5-Land data.

The previous historical `FAIL` record remains unchanged:

```text
docs/PHASE_2.md
```

Non-goals for this checkpoint:

- Do not start Step 2B.
- Do not start REN/ERA5 integration work.
- Do not refit scalers, retrain models, regenerate features, or re-baseline metrics.
- Do not start Phase 3.
- Do not treat generated-data acceptance as scientific or model acceptance.

## Decision

Decision: `PASS WITH WARNINGS`.

GO/NO-GO decision for generated data: `GO` for REN + ERA5-Land v2 integration based on monthly-bbox generated data acceptance.

Warning: the manifest is latest-run-scoped rather than full-backfill-scoped. Integration must not rely solely on the latest-run manifest as proof of full historical coverage. Full-coverage evidence must come from the monthly-bbox file inventory, station-period status coverage, row counts, timestamp checks, and checksum checks recorded here.

Scientific and model acceptance remain separate. This checkpoint does not claim that ERA5-Land v2 weather is distribution-compatible with current v1 scalers or trained models.

## Expected vs Actual Coverage

| Item | Expected | Actual monthly-bbox evidence | Result |
| --- | ---: | ---: | --- |
| Monthly periods | `198` | `198` unique raw periods | Pass |
| Station-periods | `3,366` | `3,366` unique status station-periods | Pass |
| Hourly station-periods | `3,366` | `3,366` unique hourly station-periods | Pass |
| Daily-point station-periods | `3,366` | `3,366` unique daily-point station-periods | Pass |
| Aggregate periods | `198` | `198` unique aggregate periods | Pass |
| Days / aggregate rows | `6,022` | `6,022` aggregate rows | Pass |
| Hourly rows | `2,456,976` | `2,456,976` rows | Pass |
| Daily-point rows | `102,374` | `102,374` rows | Pass |
| Coverage start | `2010-01-01` | `2010-01-01` | Pass |
| Coverage end | `2026-06-27` | `2026-06-27` | Pass |

File counts supporting the monthly-bbox coverage:

| File type | Count |
| --- | ---: |
| Raw NetCDF files | `198` |
| Status JSON files | `3,366` |
| Hourly CSV files | `3,366` |
| Daily-point CSV files | `3,366` |
| Daily aggregate CSV files | `198` |
| Comparison CSV files | `198` |
| Manifest JSON files | `1` |

## Final Partial Period

| Field | Evidence |
| --- | --- |
| Period | `2026-06-01_2026-06-27` |
| Calendar days | `27` |
| Hours per station | `648` |
| Status files | `17` |
| Raw NetCDF | Present |
| Daily aggregate | Present |

The final partial period is accepted as complete for the requested end date of `2026-06-27`.

## Station Readiness Summary

| Readiness or validation status | Count |
| --- | ---: |
| `READY` | `2,970` |
| `READY_WITH_WARNING` | `396` |
| `BLOCKED` | `0` |
| validation status `complete` | `3,366` |

`1200551` and `1210770` account for all `READY_WITH_WARNING` station-periods: each has `198` warning station-periods. All other 15 stations are `READY` for all `198` periods.

## Selected Grid Table

| station_id | selected_grid | distance_km | readiness evidence | selected_rank evidence | warning_count |
| --- | --- | ---: | --- | --- | ---: |
| `1200545` | `41.2,-8.7` | `4.039268` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1200548` | `40.2,-8.5` | `6.231658` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1200551` | `41.7,-8.8` | `5.697706` | `READY_WITH_WARNING` in all `198` periods | rank `1` in all `198` periods | `198` |
| `1200554` | `37.0,-8.0` | `3.098502` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1200558` | `38.5,-7.9` | `3.981716` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1200560` | `40.7,-7.9` | `1.694505` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1200562` | `38.0,-7.9` | `3.988029` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1200567` | `41.3,-7.7` | `3.206311` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1200570` | `39.8,-7.5` | `4.672728` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1200571` | `39.3,-7.4` | `1.790695` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1200575` | `41.8,-6.7` | `3.577181` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1210622` | `41.6,-8.5` | `5.560361` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1210683` | `40.5,-7.3` | `4.655948` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1210702` | `40.6,-8.7` | `5.2002` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1210718` | `39.8,-8.8` | `2.66179` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1210734` | `39.2,-8.7` | `3.161513` | `READY` in all `198` periods | rank `0` in all `198` periods | `0` |
| `1210770` | `38.6,-8.9` | `5.786876` | `READY_WITH_WARNING` in all `198` periods | rank `1` in all `198` periods | `198` |

The rank `1` warnings for `1200551` and `1210770` are accepted operationally under the nearest-valid-grid policy. They are not `BLOCKED` statuses.

## Content Validation

Timestamp findings:

| Check | Finding |
| --- | ---: |
| Hourly duplicate timestamps | `0` |
| Hourly missing timestamps | `0` |
| Hourly unexpected timestamps | `0` |
| Daily-point duplicate dates | `0` |
| Daily-point missing dates | `0` |
| Aggregate duplicate dates | `0` |
| Aggregate missing dates | `0` |

Content and physical-bound findings:

| Check | Finding |
| --- | ---: |
| Negative wind-speed values | `0` |
| Direction range violations | `0` |
| Calm direction policy violations | `0` |
| Aggregate point-count violations | `0` |
| Non-finite numeric values | `0` |
| Required-null counts excluding documented direction nulls | `0` |

Documented direction nulls:

| Output | Direction field | Null count | Assessment |
| --- | --- | ---: | --- |
| Hourly | `wind_direction_deg_from` | `42,906` | Aligned with documented calm/near-calm direction policy |
| Daily point | `vector_mean_wind_direction_deg_from` | `2,997` | Aligned with documented calm/near-calm direction policy |
| Daily aggregate | `vector_mean_wind_direction_deg_from` | `134` | Aligned with documented calm/near-calm direction policy |

Observed range summary:

| Output | Field | Minimum | Maximum |
| --- | --- | ---: | ---: |
| Hourly | `temperature_2m_k` | `267.71875` | `317.24658` |
| Hourly | `temperature_2m_c` | `-5.431244` | `44.09659` |
| Hourly | `u10_m_s` | `-13.671036` | `17.339203` |
| Hourly | `v10_m_s` | `-12.723648` | `16.012146` |
| Hourly | `wind_speed_m_s` | `0.001712319` | `18.385456` |
| Hourly | `wind_direction_deg_from` | `0.0` | `359.99976` |
| Daily point | `temperature_2m_c_mean` | `-2.1394145` | `36.337994` |
| Daily point | `wind_speed_m_s_mean` | `0.27518642` | `14.908436` |
| Daily point | `vector_mean_wind_speed_m_s` | `0.0035094626889771` | `13.29230234402292` |
| Daily point | `vector_mean_wind_direction_deg_from` | `0.0032883451810334` | `359.99415770877306` |
| Daily aggregate | `temperature_2m_c_mean` | `2.583810874705882` | `31.257067117647058` |
| Daily aggregate | `wind_speed_m_s_mean` | `1.1090769929411766` | `8.467178552941176` |
| Daily aggregate | `vector_mean_wind_speed_m_s` | `0.0476270831345727` | `7.289820996409188` |
| Daily aggregate | `vector_mean_wind_direction_deg_from` | `0.0599566541876015` | `359.98024668067444` |

## Integrity / Manifests

Integrity findings:

| Check | Finding |
| --- | ---: |
| Status checksum mismatches | `0` |
| Status missing referenced files | `0` |
| Manifest checksum mismatches | `0` |
| Manifest missing files | `0` |
| Zero-byte files | `0` |

Manifest evidence:

| Field | Value |
| --- | --- |
| Path | `data/raw/v2/weather/era5_land/grid_policy=nearest_valid_r1/request_mode=monthly_bbox/manifests/era5_land_weather_manifest.json` |
| Request mode | `monthly-bbox` |
| Checksum entries | `54` |
| Station IDs | `17` |
| Known warnings | `5` |
| `coverage_start` | `2026-06-01` |
| `coverage_end` | `2026-06-27` |
| `row_count` | `27` |
| `raw_file_paths_count` | `1` |
| `monthly_bbox_partition_count` | `17` |

Assessment: manifest checksums pass for referenced files, but the manifest describes the latest run, not the full 198-period backfill. It is accepted as latest-run metadata only and must not be used by itself as full-coverage proof.

## File Inventory / Storage

Total raw NetCDF size: `1,051,439,778` bytes.

Total generated dataset size: `1,570,819,742` bytes.

Storage by top-level area:

| Area | Size bytes |
| --- | ---: |
| `raw` | `1,051,439,778` |
| `hourly` | `424,622,550` |
| `metadata` | `63,507,630` |
| `daily_points` | `29,838,380` |
| `daily_aggregate` | `1,041,094` |
| `comparisons` | `77,332` |
| `manifests` | `292,978` |

Monthly-bbox file inventory:

| Output area | Count |
| --- | ---: |
| Raw NetCDF | `198` |
| Status JSON | `3,366` |
| Hourly CSV | `3,366` |
| Daily-point CSV | `3,366` |
| Daily aggregate CSV | `198` |
| Comparison CSV | `198` |
| Manifest JSON | `1` |

## Legacy Station-Month Separation

Legacy station-month outputs are reported separately and are excluded from all monthly-bbox counts above.

| Legacy output area | Count |
| --- | ---: |
| Raw | `42` |
| Metadata | `42` |
| Hourly | `42` |
| Daily points | `42` |
| Daily aggregate | `2` |

Legacy periods observed:

```text
2010-01-01_2010-01-31
2023-07-01_2023-07-31
2026-06-01_2026-06-27
```

These legacy outputs do not change the monthly-bbox acceptance decision and must not be mixed into monthly-bbox full-backfill counts.

## Comparison With Previous FAIL Audit

The previous acceptance record remains valid historical evidence and is not modified by this checkpoint:

```text
docs/PHASE_2.md
```

That earlier record concluded `FAIL` because the local evidence then represented only partial period coverage. The monthly-bbox output tree now has verified local evidence for all `198` expected periods, all `3,366` station-periods, all expected row counts, and passing integrity checks.

This new record does not rewrite the prior audit. It records acceptance for the separate monthly-bbox request-mode output tree.

## Generated-Data vs Scientific/Model Acceptance

Generated-data acceptance: `PASS WITH WARNINGS`.

Integration readiness: `GO` for REN + ERA5-Land v2 integration based on generated monthly-bbox data acceptance, with the manifest-scoping warning above.

Scientific/model acceptance remains open:

- This checkpoint does not assess forecast skill.
- This checkpoint does not validate ERA5-Land as scientifically equivalent to the current v1 weather source.
- This checkpoint does not claim current v1 scalers, trained models, datasets, or baselines remain valid for ERA5-Land-derived v2 features.
- Any v2 modelling workflow still requires explicit integration, feature generation, scaler refitting, model retraining, metric re-baselining, and scientific/model review when those steps are approved.

## Acceptance Checklist

- [x] The document is scoped to Phase 2 Step 2A.16 monthly-bbox full-backfill acceptance.
- [x] The previous `FAIL` document is preserved unchanged as historical evidence.
- [x] No network calls were made by this documentation task.
- [x] No generated ERA5-Land data was modified, repaired, regenerated, or deleted by this documentation task.
- [x] Expected monthly periods are recorded as `198`.
- [x] Expected station-periods are recorded as `3,366`.
- [x] Expected aggregate rows are recorded as `6,022`.
- [x] Expected hourly rows are recorded as `2,456,976`.
- [x] Expected daily-point rows are recorded as `102,374`.
- [x] Actual monthly-bbox counts match expected counts.
- [x] Final partial period `2026-06-01_2026-06-27` is recorded as complete for `27` days and `648` hours per station.
- [x] Readiness status counts are recorded.
- [x] Selected grid coordinates, distances, ranks, and warnings are recorded for all 17 stations.
- [x] Timestamp, content, null, non-finite, and range validation findings are recorded.
- [x] Status and manifest checksum findings are recorded.
- [x] Latest-run manifest scoping is called out as a warning.
- [x] File inventory and storage totals are recorded.
- [x] Legacy station-month outputs are separated and excluded from monthly-bbox counts.
- [x] Decision is explicit: `PASS WITH WARNINGS`.
- [x] GO/NO-GO decision is explicit: `GO` for REN + ERA5-Land v2 integration based on generated data acceptance.
- [x] Scientific/model acceptance is kept separate.

## Stop Gate

Phase 2 Step 2A.16 status: monthly-bbox full-backfill acceptance record created with decision `PASS WITH WARNINGS`.

The approved documentation-only scope stops here. Step 2B, REN/ERA5 integration, feature regeneration, scaler refitting, model training, metric re-baselining, notebook execution, and Phase 3 were not started.

---

## Phase 2 Integrated V2 Daily Dataset Acceptance

Original file before consolidation: `PHASE_2_INTEGRATED_V2_DAILY_DATASET_ACCEPTANCE.md`.

## Scope And Inputs

This document records the Phase 2 Step 2A.17 local build and validation of the integrated REN + ERA5-Land v2 daily dataset.

The build uses the local-day contract from `docs/PHASE_2.md`: the canonical daily key is the `Europe/Lisbon` civil calendar date. No network calls, notebook execution, scaler fitting, model training, Step 2B work, or Phase 3 work were performed.

Input roots:

| Input | Path |
| --- | --- |
| REN v2 production | `data/raw/v2/production/` |
| ERA5-Land monthly-bbox weather | `data/raw/v2/weather/era5_land/grid_policy=nearest_valid_r1/request_mode=monthly_bbox/` |
| Station mapping | `data/pilot/ipma/ipma_station_mapping.csv` |
| Frozen v1 production comparison | `data/raw/ReparticaoProducao.csv` |

Output root, ignored by Git:

```text
data/processed/v2/daily_merged/integrated_ren_era5_land_v2/
```

## Local-Day And DST Policy

- REN timestamps are interpreted as `Europe/Lisbon` local physical instants.
- REN `source_date` is an integrity check against the timestamp-derived local date.
- Expected REN intervals are `96` on ordinary days, `92` on spring DST days, and `100` on autumn DST days.
- ERA5-Land hourly source timestamps remain UTC in source files.
- Integrated ERA5-Land weather is recomputed after converting UTC timestamps to `Europe/Lisbon`.
- Expected ERA5-Land hourly counts are `24` on ordinary days, `23` on spring DST days, and `25` on autumn DST days.
- Month-boundary local days use adjacent UTC hours from neighboring monthly partitions where required.

## Production Target And Unit

The production target column is `Wind_Production`.

The aggregation preserves the recovered frozen v1 behavior:

```text
sum(wind_production_mw)
```

Unit warning: this is a sum of 15-minute `MW` observations, not `MWh`. Energy in `MWh` would require multiplying each 15-minute `MW` value by `0.25h` before summing.

## Generated Outputs

| Output | Rows | SHA-256 |
| --- | ---: | --- |
| `ren_daily_production_local.csv` | `6,016` | `f49120e5384f17059bd295657b205363f31bd598c94f4edcc5c71c898470be31` |
| `era5_land_daily_points_local.csv` | `102,374` | `5e586708bf277bb7175edb1e1fe407ee936c9890e1fcc7f6ef20cca92ae3803a` |
| `era5_land_daily_aggregate_local.csv` | `6,022` | `f40a807f560ed0f9af3ad1313007d311d34c4be0486f63011af89e307c192870` |
| `daily_merged.csv` | `6,016` | `45a3d5fec21e8a81dffef0f2e14e1ad3ca8c46c5efeeb5bb4a4d3ae44ae13f15` |
| `coverage.csv` | `6,022` | `cbe48df1b100e576a25b4003fec3e650f8e5a1f17d35e26e146c06a6cde4e73c` |
| `validation.json` | n/a | `cd3bfc9abd3f743cf8537ced108700843f8fc7b8456a2f65761046d7f251245d` |
| `manifest.json` | n/a | `904364d662c2427f8e43dae4d543519bf163d6d9190d7f06aa1eafcb504025ef` |

`daily_merged.csv` has `27` columns: `date_local`, REN production lineage fields, and approved ERA5-Land aggregate weather fields.

## Coverage And Exclusions

| Status | Count |
| --- | ---: |
| `integration-ready` | `6,016` |
| `excluded-downstream-ren-unavailable` | `6` |

REN status counts:

| Status | Count |
| --- | ---: |
| `complete` | `6,016` |
| `unavailable` | `6` |

ERA5-Land status counts:

| Status | Count |
| --- | ---: |
| `complete` | `6,022` |

Explicitly excluded REN-unavailable dates:

- `2014-05-03`
- `2016-02-03`
- `2016-02-04`
- `2021-10-03`
- `2023-08-30`
- `2025-08-02`

No interpolation, forward fill, silent inner join, or silent date drop was used. Every requested local calendar date has one coverage row.

The final requested date, `2026-06-27`, is integration-ready with `96` REN intervals, `17` ERA5-Land points, and `408` local-day hourly weather observations.

## Validation Results

Validation verdict: `PASS WITH WARNINGS`.

Checks passed:

- Synthetic validation covered duplicate temporal keys, REN `source_date` mismatch, ordinary and DST interval counts, ERA5 UTC-to-`Europe/Lisbon` conversion, month-boundary local-day windows, missing hourly data, incomplete station coverage, unexpected units, non-finite values, deterministic ordering, checksum stability, and source-input non-mutation.
- Coverage table records every requested local date.
- REN interval counts match the expected `96/92/100` policy.
- ERA5-Land hourly counts match the expected `24/23/25` policy.
- Final date `2026-06-27` has complete local-day weather coverage.
- Merged rows match explicit integration-ready coverage.
- No interpolation or forward fill is used.
- Generated outputs are ignored by Git.
- A second explicit `--overwrite` rerun produced identical output checksums.

Warnings:

- Six REN-unavailable dates are excluded downstream.
- V2 REN daily production differs from frozen v1 on `257` overlapping local days.

No validation failures were reported.

## Frozen V1 Production Comparison

| Metric | Value |
| --- | ---: |
| Overlap day count | `5,592` |
| Exact-match day count | `5,335` |
| Differing day count | `257` |
| First overlap date | `2010-01-01` |
| Last overlap date | `2025-04-28` |
| Mean difference, v2 minus v1 | `18.494670958512298` |
| Mean absolute difference | `21.56112303290948` |
| Maximum absolute difference | `3891.600000000035` |

Sample differing dates:

```text
2024-06-01
2024-06-03
2024-06-13
2024-06-15
2024-06-17
2024-06-18
2024-06-19
2024-06-20
2024-06-28
2024-08-01
```

This comparison is production-only. ERA5-Land weather is not expected to match the former v1 weather provider.

## Commands Run

```powershell
.\venv\Scripts\python.exe -c "import wind_forecast.integration; print('integration import ok')"
.\venv\Scripts\python.exe -m compileall -q src\wind_forecast scripts\build_integrated_v2_dataset.py
.\venv\Scripts\python.exe -c "from wind_forecast.integration import run_synthetic_alignment_checks; import json; print(json.dumps(run_synthetic_alignment_checks(), indent=2, sort_keys=True))"
git check-ignore -v data\processed\v2\daily_merged\integrated_ren_era5_land_v2\daily_merged.csv
.\venv\Scripts\python.exe .\scripts\build_integrated_v2_dataset.py --start-date 2010-01-01 --end-date 2026-06-27 --ren-root data\raw\v2\production --era5-root data\raw\v2\weather\era5_land\grid_policy=nearest_valid_r1\request_mode=monthly_bbox --station-mapping data\pilot\ipma\ipma_station_mapping.csv --output-root data\processed\v2\daily_merged\integrated_ren_era5_land_v2
.\venv\Scripts\python.exe .\scripts\build_integrated_v2_dataset.py --start-date 2010-01-01 --end-date 2026-06-27 --ren-root data\raw\v2\production --era5-root data\raw\v2\weather\era5_land\grid_policy=nearest_valid_r1\request_mode=monthly_bbox --station-mapping data\pilot\ipma\ipma_station_mapping.csv --output-root data\processed\v2\daily_merged\integrated_ren_era5_land_v2 --overwrite
```

The non-overwrite rerun was also checked and correctly failed with `FileExistsError`, requiring explicit `--overwrite` for regeneration.

## Decision

Decision: `PASS WITH WARNINGS`.

GO/NO-GO for the next approved work:

- Feature regeneration: `GO`, with the warnings above and only after an explicit future checkpoint starts it.
- Step 2B: `GO`, with the warnings above and only after explicit approval to start Step 2B.

This checkpoint does not claim model, scaler, or metric validity for v2. Scaler refitting, model retraining, and metric re-baselining remain future work.

## Stop Gate

Phase 2 Step 2A.17 status: integrated v2 daily dataset built locally and validated.

Step 2B was not started. Feature regeneration was not started. Notebook execution, scaler fitting, model training, metric re-baselining, and Phase 3 were not started.

---

## Phase 2 Feature-Ready V2 Dataset Acceptance

Original file before consolidation: `PHASE_2_FEATURE_READY_V2_DATASET_ACCEPTANCE.md`.

## Scope And Inputs

This document records the Phase 2 Step 2A.18 local build and validation of the
feature-ready REN + ERA5-Land v2 daily dataset.

Source of truth:

- `docs/PHASE_2.md`
- `docs/PHASE_2.md`
- `src/wind_forecast/features.py`
- local v1 feature table: `data/processed/agg_data_ml.csv`
- accepted Step 2A.17 integrated source root:
  `data/processed/v2/daily_merged/integrated_ren_era5_land_v2/`

No network calls, notebook execution, scaler fitting, model training, train/test
splitting, Step 2B validator changes, or Phase 3 work were performed.

Output root, ignored by Git:

```text
data/processed/v2/ml_features/feature_ready_ren_era5_land_v2/
```

## Recovered Contract

The output feature table is `feature_ready_daily.csv`.

The output columns exactly match the local v1 feature table
`data/processed/agg_data_ml.csv`: `58` columns in the same order.

V2 base-column mapping:

| V1 feature base column | V2 integrated source column |
| --- | --- |
| `Date` | `date_local` |
| `Wind_Production` | `Wind_Production` |
| `Average_Wind_Speed` | `wind_speed_m_s_mean` |
| `Average_Temperature` | `temperature_2m_c_mean` |
| `Average_Wind_Direction` | `vector_mean_wind_direction_deg_from` |

Feature formulas come from
`wind_forecast.features.apply_feature_engineering`.

`handle_final_nans` is not called for v2. No backfill, forward fill,
interpolation, zero-fill, or hidden data cleaning is used.

Feature-ready row eligibility requires:

- current integrated row is integration-ready;
- current target, wind speed, temperature, and direction are finite;
- prior production lag dates `1`, `2`, `3`, `7`, and `14` are finite;
- prior speed and temperature lag dates `1`, `2`, `3`, and `7` are finite;
- prior 14 local calendar days are finite for target, speed, and temperature so
  rolling windows do not skip gaps;
- prior direction lag dates `1`, `2`, `3`, and `7` are finite.

The direction-history rule is intentionally tied to the actual v1 direction lag
features produced by `apply_feature_engineering`: `1`, `2`, `3`, and `7`. This
matches the recovered expected row counts.

## Generated Outputs

| Output | Rows | SHA-256 |
| --- | ---: | --- |
| `feature_ready_daily.csv` | `5,322` | `d0d073748c5d963cba30212e6b0ab666ec2000197b8f61a5c439b4aaf786b2a6` |
| `feature_schema.json` | n/a | `bcb862238f893d599e3ac3d35663d2a30be2eb0a53f27bc7ffea279fdd9df130` |
| `feature_coverage.csv` | `6,022` | `b6e18d172fa956109e62351b91700b7a94fbde5f1f517901dfcc167c7ad4cf67` |
| `v1_structure_comparison.json` | n/a | `7469e9d9a47b392381be289b5977c665b37ec8a3c22cf4842d6b5ca4e17935a5` |
| `validation.json` | n/a | `f6d32f6b808256b9eb0aa0dbfd1771e2c885210efabd724c886167d174f38115` |
| `manifest.json` | n/a | `6634e14b74becd6f775f5fdf352b5a31eb9f93ddd07a1a70c493d65168c9ea72` |

Source checksums recorded in `manifest.json` include:

| Source | SHA-256 |
| --- | --- |
| `data/processed/agg_data_ml.csv` | `888d9629f89ea18d3f704a0e5298b041a4aedb9b56cc3255b4dcd749798fafc5` |
| `daily_merged.csv` | `45a3d5fec21e8a81dffef0f2e14e1ad3ca8c46c5efeeb5bb4a4d3ae44ae13f15` |
| `coverage.csv` | `cbe48df1b100e576a25b4003fec3e650f8e5a1f17d35e26e146c06a6cde4e73c` |
| `validation.json` | `cd3bfc9abd3f743cf8537ced108700843f8fc7b8456a2f65761046d7f251245d` |
| `manifest.json` | `904364d662c2427f8e43dae4d543519bf163d6d9190d7f06aa1eafcb504025ef` |

## Coverage And Exclusions

Coverage lineage records all `6,022` requested local calendar dates, including
the six REN-unavailable dates inherited from Step 2A.17.

| Status | Count |
| --- | ---: |
| `feature-ready` | `5,322` |
| `excluded-ren-unavailable` | `6` |
| `excluded-current-weather-or-target-null` | `134` |
| `excluded-warmup-insufficient-14-day-history` | `13` |
| `excluded-gap-in-prior-14-day-history` | `67` |
| `excluded-direction-null-in-prior-7-day-history` | `480` |

Row-count summary:

| Metric | Count |
| --- | ---: |
| Coverage rows | `6,022` |
| Integrated-ready rows | `6,016` |
| Current complete base rows | `5,882` |
| Feature-ready rows | `5,322` |

Feature-ready date range:

- first feature-ready date: `2010-01-15`
- last feature-ready date: `2026-06-27`

## V1 Structure Comparison

| Check | Result |
| --- | --- |
| V1 rows | `4,017` |
| V2 feature-ready rows | `5,322` |
| V1 column count | `58` |
| V2 column count | `58` |
| Exact column order match | `true` |
| Missing v1 columns from v2 | none |
| Extra v2 columns | none |
| Numeric feature columns match | `true` |
| V2 numeric sample finite | `true` |

## Validation Results

Validation verdict: `PASS WITH WARNINGS`.

Checks passed:

- accepted Step 2A.17 integrated validation passed;
- feature coverage records the full local calendar;
- feature-ready row count matches the coverage status count;
- v2 feature columns and order exactly match v1;
- no `handle_final_nans` fill policy is used;
- feature-ready output has no NaNs;
- feature-ready numeric values are finite;
- no unexpected integrated-not-ready rows were found beyond REN-unavailable
  dates.

Warnings:

- Step 2A.17 integrated dataset verdict was `PASS WITH WARNINGS`.
- Six REN-unavailable dates are explicitly excluded downstream:
  `2014-05-03`, `2016-02-03`, `2016-02-04`, `2021-10-03`,
  `2023-08-30`, and `2025-08-02`.
- V2 REN daily production differs from frozen v1 on `257` overlapping local
  days.

## Commands Run

```powershell
.\venv\Scripts\python.exe -c "import wind_forecast.v2_features; print('v2 feature import ok')"
.\venv\Scripts\python.exe -m compileall -q src\wind_forecast scripts\build_feature_ready_v2_dataset.py
git check-ignore -v data\processed\v2\ml_features\feature_ready_ren_era5_land_v2\feature_ready_daily.csv
.\venv\Scripts\python.exe -c "from wind_forecast.v2_features import run_synthetic_feature_checks; import json; print(json.dumps(run_synthetic_feature_checks(), indent=2, sort_keys=True))"
.\venv\Scripts\python.exe .\scripts\build_feature_ready_v2_dataset.py --input-root data\processed\v2\daily_merged\integrated_ren_era5_land_v2 --v1-feature-table data\processed\agg_data_ml.csv --output-root data\processed\v2\ml_features\feature_ready_ren_era5_land_v2
.\venv\Scripts\python.exe .\scripts\build_feature_ready_v2_dataset.py --input-root data\processed\v2\daily_merged\integrated_ren_era5_land_v2 --v1-feature-table data\processed\agg_data_ml.csv --output-root data\processed\v2\ml_features\feature_ready_ren_era5_land_v2
.\venv\Scripts\python.exe .\scripts\build_feature_ready_v2_dataset.py --input-root data\processed\v2\daily_merged\integrated_ren_era5_land_v2 --v1-feature-table data\processed\agg_data_ml.csv --output-root data\processed\v2\ml_features\feature_ready_ren_era5_land_v2 --overwrite
```

The non-overwrite rerun correctly failed with `FileExistsError`. The explicit
`--overwrite` rerun produced identical checksums for all six generated outputs.

## Decision

Decision: `PASS WITH WARNINGS`.

GO/NO-GO for the next approved work:

- Step 2B validation: `GO`, with the warnings above and only after explicit
  approval to start Step 2B.
- Later scaler refitting, model retraining, and metric re-baselining: `GO` as
  future work after validation confirms the v2 feature contract, but this
  checkpoint does not perform or claim those results.

Existing v1 scalers, models, and metrics are not claimed valid for v2.

## Stop Gate

Phase 2 Step 2A.18 status: feature-ready v2 daily dataset built locally and
validated.

Step 2B was not started. Notebook execution, scaler fitting, model training,
metric re-baselining, and Phase 3 were not started.

---

## Phase 2 Feature-Ready V2 Validation Acceptance

Original file before consolidation: `PHASE_2_FEATURE_READY_V2_VALIDATION_ACCEPTANCE.md`.

## Scope

Phase 2 Step 2B.1 adds formal validation for the accepted feature-ready REN +
ERA5-Land v2 dataset.

Validated dataset root:

```text
data/processed/v2/ml_features/feature_ready_ren_era5_land_v2/
```

This step does not rebuild v2 data, modify generated CSV or JSON outputs, fit
scalers, train models, execute notebooks, call network services, or start Phase
3.

## Validator

Reusable validation lives in:

```text
src/wind_forecast/validation/feature_ready.py
```

The validator checks:

- the six required feature-ready files and absence of unexpected files;
- manifest, schema, validation, lineage, row-count, path, and checksum metadata;
- exact 58-column v1 feature order;
- duplicate, missing, extra, and reordered feature columns;
- unique sorted daily `Date` values;
- coverage lineage for dates missing from the feature table;
- the six REN-unavailable dates are excluded from feature rows and present in
  coverage;
- finite numeric values and absence of NaN or infinity;
- non-negative target and wind-speed features;
- calendar, wind-direction, and cyclical feature domains;
- feature-ready coverage consistency;
- date-based lag and rolling recomputation from accepted integrated inputs;
- deterministic report serialization.

The validator records that scaler fitting is not performed in this step and that
existing v1 scalers are not claimed valid for v2.

## CLI

Run the validator with:

```powershell
.\venv\Scripts\python.exe .\scripts\validate_feature_ready_v2_dataset.py --feature-root data\processed\v2\ml_features\feature_ready_ren_era5_land_v2 --integrated-root data\processed\v2\daily_merged\integrated_ren_era5_land_v2 --v1-feature-table data\processed\agg_data_ml.csv
```

The command prints deterministic JSON to stdout. It exits with `1` only when
validation errors are present. `PASS` and `PASS WITH WARNINGS` both exit with
`0`.

Optional report writing is explicit:

```powershell
.\venv\Scripts\python.exe .\scripts\validate_feature_ready_v2_dataset.py --feature-root data\processed\v2\ml_features\feature_ready_ren_era5_land_v2 --integrated-root data\processed\v2\daily_merged\integrated_ren_era5_land_v2 --v1-feature-table data\processed\agg_data_ml.csv --report-output $env:TEMP\feature_ready_v2_validation_report.json
```

## Acceptance Status

Acceptance decision: `PASS WITH WARNINGS`.

The warnings are inherited from the accepted feature-ready and integrated v2
dataset lineage:

- the integrated Step 2A.17 source verdict is `PASS WITH WARNINGS`;
- six REN-unavailable dates remain explicitly excluded;
- v2 REN daily production differs from frozen v1 on known overlapping days.

These warnings do not block formal validation, but they remain required context
for later scaler fitting, model retraining, and metric re-baselining.

GO/NO-GO decision:

- GO for creating and fitting a new v2 scaler in a later explicitly approved
  checkpoint.
- NO-GO for reusing existing v1 scalers as valid for v2.
- NO-GO for model training, model inference, or metric re-baselining in this
  checkpoint.

## Stop Gate

Phase 2 Step 2B.1 is limited to formal validation. It does not start scaler
fitting, model training, feature regeneration, Step 2B follow-up validators, or
Phase 3.
