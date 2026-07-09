# Phase 2 ERA5-Land Monthly-Bbox Full Backfill Acceptance

## Scope / Safety

This document records the Phase 2 Step 2A.16 documentation-only acceptance checkpoint for the ERA5-Land v2 monthly-bbox full backfill outputs under:

```text
data/raw/v2/weather/era5_land/grid_policy=nearest_valid_r1/request_mode=monthly_bbox/
```

The evidence in this record comes from verified read-only local audit results for the monthly-bbox output tree. This documentation task made no network calls, ran no ingestion scripts, executed no notebooks, started no training, and did not modify, repair, regenerate, or delete generated ERA5-Land data.

The previous historical `FAIL` record remains unchanged:

```text
docs/PHASE_2_ERA5_LAND_FULL_BACKFILL_ACCEPTANCE.md
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
docs/PHASE_2_ERA5_LAND_FULL_BACKFILL_ACCEPTANCE.md
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
