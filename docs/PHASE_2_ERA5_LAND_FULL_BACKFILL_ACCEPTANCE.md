# Phase 2 ERA5-Land Full Backfill Acceptance

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
docs/PHASE_2_ERA5_LAND_FULL_BACKFILL_ACCEPTANCE.md
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
