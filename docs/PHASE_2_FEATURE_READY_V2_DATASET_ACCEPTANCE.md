# Phase 2 Feature-Ready V2 Dataset Acceptance

## Scope And Inputs

This document records the Phase 2 Step 2A.18 local build and validation of the
feature-ready REN + ERA5-Land v2 daily dataset.

Source of truth:

- `docs/PHASE_2_INTEGRATED_V2_DAILY_DATASET_ACCEPTANCE.md`
- `docs/PHASE_2_V2_LOCAL_DAY_ALIGNMENT_DECISION.md`
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
