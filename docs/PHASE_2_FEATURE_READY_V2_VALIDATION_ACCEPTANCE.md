# Phase 2 Feature-Ready V2 Validation Acceptance

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
