# Phase 2 Data Refresh Assessment

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
