# Phase 2 — V2 scaler fit acceptance

## Scope

This checkpoint fits a new scaler bundle for the accepted REN + ERA5-Land v2
feature-ready dataset. It does not replace or modify any v1 data, models,
scalers, predictions, notebooks, or serving paths.

The bundle is prepared for a future v2 ANN training/inference path. The current
v2 reference model is tree-based and remains unscaled and separate from v1
serving.

## Input and fit contract

| Field | Value |
| --- | --- |
| Input | `data/processed/v2/ml_features/feature_ready_ren_era5_land_v2/feature_ready_daily.csv` |
| Input SHA-256 | `d0d073748c5d963cba30212e6b0ab666ec2000197b8f61a5c439b4aaf786b2a6` |
| Dataset version | `v2` |
| Transformation version | `feature_ready_ren_era5_land_v2_2A.18` |
| Total rows | `5,322` |
| Features | `56` |
| Fit scope | train + validation, excluding the sealed test period |
| Fit dates | `2010-01-15` through `2024-12-31` |
| Fit rows | `4,872` |
| Feature scaler | `MinMaxScaler` for X, original and log target paths |
| Target transforms | original identity and `log1p` |

The fit window matches the v2 reference split's train and validation periods;
the test period beginning `2025-01-01` is not used to fit any scaler.

## Generated bundle

Output directory:

```text
models/v2/scalers/feature_ready_ren_era5_land_v2/
```

| File | SHA-256 |
| --- | --- |
| `scaler_X_original_ann.joblib` | `aba2ec41bc5c1b65dc305dba032b6be37d3029b2162ef37ebacc1f2c4ccc3133` |
| `scaler_X_log_ann.joblib` | `aba2ec41bc5c1b65dc305dba032b6be37d3029b2162ef37ebacc1f2c4ccc3133` |
| `scaler_y_original_ann.joblib` | `f18ceb86d13d2d2be67e4f9302d31f9c1804e2defe7abda61fe78d29c97c8d6d` |
| `scaler_y_log_ann.joblib` | `daf7586b568705cdee59b8a1b5d479d7bf975dd786a644b3b1637c2b4ad78e15` |
| `scaler_manifest.json` | `da670ada2310385fb0915459651f58d4b317e6c8ad0a2979aacd53a3a4f71829` |

The manifest records the input digest, fit window, feature order and schema
hash, target transformations, individual artifact hashes, and the explicit
`v1_artifacts_untouched` marker.

## Reproduction

```powershell
.\venv\Scripts\python.exe .\scripts\fit_v2_scalers.py
```

The CLI revalidates the accepted feature-ready dataset and refuses to replace
an existing output directory. A new fit therefore requires a new, explicitly
versioned output location.

## Validation and stop gate

Checks passed:

- accepted v2 feature-ready validation ran before fitting;
- all four joblib artifacts load as `MinMaxScaler` instances with 56 X
  features and one target feature;
- the fitted maxima and target log transform use only the declared fit window;
- the manifest hashes match the generated files;
- the four tracked v1 scaler SHA-256 values remain unchanged;
- local scaler and v2 training tests pass, and Ruff reports no findings.

No model was trained, retrained, promoted, served, or re-baselined in this
checkpoint. Existing v1 scalers and models remain v1-only. The next approved
work is a separate v2 model-training and metric re-baselining checkpoint.
