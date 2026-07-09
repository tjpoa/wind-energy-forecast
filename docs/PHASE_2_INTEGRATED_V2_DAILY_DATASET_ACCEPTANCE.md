# Phase 2 Integrated V2 Daily Dataset Acceptance

## Scope And Inputs

This document records the Phase 2 Step 2A.17 local build and validation of the integrated REN + ERA5-Land v2 daily dataset.

The build uses the local-day contract from `docs/PHASE_2_V2_LOCAL_DAY_ALIGNMENT_DECISION.md`: the canonical daily key is the `Europe/Lisbon` civil calendar date. No network calls, notebook execution, scaler fitting, model training, Step 2B work, or Phase 3 work were performed.

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
