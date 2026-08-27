# Baseline Training CLI

This document describes the lightweight baseline training command extracted
from the notebook workflow. It is intended to make the project easier to review
and reproduce without opening `notebooks/Modeling.ipynb`.

The CLI does not replace the notebook workflow, does not run Optuna, does not
train ANN/Keras models, and does not overwrite the existing saved artifacts
under `models/`.

## Inputs

The default input is the local feature-ready v1 table:

```text
data/processed/agg_data_ml.csv
```

This file is generated locally and intentionally ignored by Git. A fresh clone
may need to regenerate it through the existing data-preparation workflow before
training can run.

## Command

Run a deterministic ExtraTrees baseline:

```powershell
.\venv\Scripts\python.exe .\scripts\train_baseline.py `
  --input data\processed\agg_data_ml.csv `
  --output-dir outputs\training\baseline `
  --overwrite
```

Optional arguments:

```powershell
--model extra_trees
--model random_forest
--test-fraction 0.2
--seed 42
--n-estimators 100
--mlflow
--tracking-mode local
--tracking-mode off
```

## Outputs

The CLI writes known output files under the selected output directory:

```text
model.joblib
metrics.json
predictions.csv
run_summary.json
```

Project-local training outputs must be placed under `outputs/`. The
`outputs/training/` tree is ignored by Git so local training artifacts are not
committed accidentally.

The output contract now also includes `dataset_manifest.json`,
`model_manifest.json`, `environment.json`, `validation_sample.csv`, and
`actual_vs_predicted.png`. These make a tracked run independently inspectable.

## Evaluation Contract

- Rows are sorted by `Date`.
- The final holdout split is chronological.
- The default test fraction is `0.2`, matching the notebook's final 80/20
  split intent.
- Metrics are calculated on the original target scale:
  - `R2`
  - `MAE`
  - `RMSE`
  - `MAPE (%)`

## What This Proves

- The historical feature table can be trained and evaluated from a script.
- The final holdout split is reproducible and test-covered.
- Metrics are captured as JSON artifacts.
- A baseline model artifact can be produced without touching the existing
  production-facing `models/` directory.

## What This Does Not Prove

- It does not reproduce the tuned notebook ANN artifacts.
- It does not validate v2 model or scaler compatibility.
- It does not make Registry aliases part of FastAPI serving.
- It does not replace future fuller training, tuning, or monitoring work.

## Baseline Model Card

| Field | Description |
| --- | --- |
| Model name | Lightweight historical baseline training CLI. |
| Implementation | `wind_forecast.training` and `scripts/train_baseline.py`. |
| Default estimator | `ExtraTreesRegressor` with deterministic `random_state`. |
| Optional estimator | `RandomForestRegressor`. |
| Task | Supervised regression for daily Portuguese wind-energy production. |
| Target | `Wind_Production`, evaluated on the original target scale. |
| Intended use | Portfolio-grade reproducible baseline training and holdout evaluation. |
| Non-goals | Production model promotion, tuned ANN reproduction, live forecasting, registry operations, and v2 model validation. |

### Inputs

The baseline expects a feature-ready CSV containing:

- `Date`.
- `Wind_Production`.
- Numeric feature columns compatible with the project schema.

Known legacy column names are normalized through `wind_forecast.schemas` before
training. Rows are sorted by `Date` before the split.

### Training And Evaluation

The CLI uses a chronological final holdout split. By default, the first 80% of
rows are used for training and the last 20% for evaluation. This preserves the
time-series intent from the notebook workflow and avoids random leakage across
time.

The reported metrics are:

- `R2`.
- `MAE`.
- `RMSE`.
- `MAPE (%)`.

The CLI writes metrics, predictions, model artifact, and run summary under
`outputs/training/`. These outputs are intentionally ignored by Git.

### Appropriate Use

This baseline is appropriate for:

- Demonstrating reproducible model training from a script.
- Showing deterministic data loading, schema normalization, chronological
  splitting, metric capture, and artifact writing.
- Comparing future training refactors against a small, clear baseline.

It is not appropriate for:

- Claiming the tuned notebook ANN has been reproduced.
- Claiming current v1 models or scalers are valid for v2 data.
- Production deployment or automated model promotion.
- Monitoring live model quality.

### Main Risks And Limitations

- The full tuned workflow remains in `notebooks/Modeling.ipynb`.
- The v1 feature table is generated locally and ignored by Git.
- Current saved model and scaler artifacts depend on the recovered v1 feature
  contract.
- Material source or feature-distribution changes require retraining,
  re-baselining, and scaler validation.
- Registry state is local SQLite state. It is auditable but not a remote shared
  service and is not itself placed in release bundles.

## Phase 4B MLflow Lifecycle

The baseline training command uses `http://127.0.0.1:5000` by default. Start
MLflow with a SQLite backend and proxied local artifacts:

```powershell
.\venv\Scripts\python.exe -m mlflow server `
  --backend-store-uri sqlite:///var/mlflow/mlflow.db `
  --artifacts-destination ./var/mlflow/artifacts `
  --host 127.0.0.1 `
  --port 5000
```

Every tracked run records the Git commit/dirty flag, dependency versions,
dataset and feature hashes, temporal split, four metrics, evaluation outputs,
dataset lineage, and a sklearn model with signature and input example. A run
created from a dirty Git tree remains inspectable but cannot be registered as a
candidate.

Candidate registration is deliberately separate from training:

```powershell
.\venv\Scripts\python.exe .\scripts\register_candidate.py --run-id <RUN_ID>
```

It requires a finished clean run, finite metrics, the approved v1/original
target contract, all manifests, and reload-equivalent predictions. It then
creates a model version and moves only `candidate`.

`champion` promotion never happens automatically. The operator supplies the
expected candidate version, expected current champion (or `none`), and an
approval note. The receipt records the previous champion and supports an
optimistic rollback. No metric threshold is invented because the legacy ANN
models do not have a comparable reproducible Registry run.

## Phase 4B Artifact Bundles

GitHub Releases are the selected first distribution mechanism. The builder
creates a deterministic ZIP and SHA-256 sidecar containing the v1 processed
training table, validated candidate MLflow package, baseline outputs, manifests,
environment, plot, and validation sample. Raw v1 files already tracked in Git,
legacy Keras/scaler serving files, MLflow local state, and all v2 artifacts are
excluded.

The release catalog deliberately marks `artifacts-v1.0.0` as blocked and leaves
its pinned bundle SHA-256 empty until
source, licence, attribution, and redistribution permission are confirmed.
Builder tooling may be exercised locally, but a public-release or
cross-machine reproducibility claim is not allowed before that gate and a clean
clone round-trip pass.

This checkpoint was additionally exercised against a real local MLflow 3.14
server with a SQLite backend: a clean v1 baseline run completed, candidate
validation and registration succeeded, two candidate bundles had matching
SHA-256 values, bundle verification and deterministic retraining passed, and
manual promotion followed by rollback succeeded. All resulting Registry and
artifact state remained local. It does not claim that a public release or
clean-clone/cross-machine round-trip has run; both remain blocked until the
catalog has approved source, licence, attribution, redistribution permission,
and a pinned bundle SHA-256.

## Baseline Data Card

| Field | Description |
| --- | --- |
| Primary v1 target | Historical wind-production data under `data/raw/`. |
| Training table | `data/processed/agg_data_ml.csv`, generated locally and ignored by Git. |
| Current schema | English project columns such as `Date`, `Wind_Production`, `Average_Wind_Speed`, `Average_Temperature`, and `Average_Wind_Direction`. |
| Feature families | Calendar features, cyclic encodings, lags, rolling means, rolling standard deviations, and weather-derived inputs. |
| v2 status | REN + ERA5-Land work is isolated under v2 paths and does not replace v1. |

### Data Contract

The baseline data contract is intentionally conservative:

- Raw v1 data is treated as immutable.
- Processed v1 CSV files remain local generated artifacts.
- V2 datasets must not be silently appended to or substituted for v1.
- Any material data-source change invalidates claims about current model,
  scaler, and metric compatibility until explicitly revalidated.

### Known Data Limitations

- The exact provenance of every historical v1 source remains a documented
  engineering concern.
- A fresh clone may need local processed artifacts regenerated before full model
  training or serving can run.
- Recent API-period evaluation is not the same as live future forecasting.
- V2 REN + ERA5-Land evidence supports future data-refresh work but does not
  certify v1 artifact compatibility.

### Presentation Guidance

When presenting the project, describe this as a reproducible Data/ML
Engineering baseline around a historical forecasting workflow. Avoid describing
it as a deployed production forecasting system or as a completed model lifecycle
platform.

## Stage 2 — First v2 Reference Model

The dedicated v2 workflow is isolated from `train_baseline.py`, all v1 models,
scalers, and serving paths. It uses the accepted
`feature_ready_ren_era5_land_v2` table with SHA-256
`d0d073748c5d963cba30212e6b0ab666ec2000197b8f61a5c439b4aaf786b2a6`.
The contract is a historical daily **hindcast**: contemporaneous ERA5-Land
weather is allowed for the target date, so the result is not an operational
day-ahead forecast.

Run the fully validated and locally tracked workflow with:

```powershell
$env:PYTHONUTF8="1"
.\venv\Scripts\python.exe .\scripts\train_v2_reference.py `
  --output-dir outputs\training\v2_reference
```

This uses the local SQLite-backed MLflow server documented above at
`http://127.0.0.1:5000`; start that server before running the command.

The fixed chronological contract is train `2010-01-15`–`2022-12-31`
(4,209 rows), validation `2023-01-01`–`2024-12-31` (663 rows), and sealed test
`2025-01-01`–`2026-06-27` (450 rows). ExtraTrees and RandomForest use 100 trees,
seed 42, and `n_jobs=-1`; validation MAE selects the candidate, with ExtraTrees
as the exact-tie winner. The selected estimator is refitted on train plus
validation and evaluated once against `Wind_Production_Lag1` persistence.

The local acceptance run selected RandomForest. Its test MAE was
`25,541.036826667`, compared with persistence MAE `69,577.153111111`, for MAE
skill `0.632910579`; RMSE was `33,448.655794793`, R² `0.886607984`, signed bias
`-7,684.841942222`, and MAPE `22.137047390%`. The strict MAE gate passed and the
result is `selected_not_promoted`. No scaler is required or created. Two
independent tracking-off executions produced byte-identical model, prediction,
metric, plot, manifest, environment, audit, sample, and summary artifacts.

Before training, the CLI runs the accepted feature-ready validator, including
checksum verification and date-based recomputation of every lag and rolling
feature from the integrated base. It records split isolation, feature order,
dataset/model/split hashes, full estimator parameters, environment, Git state,
reload evidence, metrics, predictions, and the reference decision.

MLflow logging creates one run in `wind-energy-forecast-v2-reference`, logs
dataset lineage and the sklearn model with signature, reloads that logged model,
and tags the result as not promoted. It never calls the Registry. The real local
MLflow 3.14 run `ddb478bde0404fbf8e48f496326c3d41` finished successfully against
the SQLite-backed server. Its logged model URI was
`models:/m-4eef011bd42c4f9e9f4d8fedaa19361f`; five reloaded predictions matched the
saved sample with maximum absolute difference `1.7462298274040222e-10`. The
`PYTHONUTF8` setting prevents MLflow's run-link emoji from failing on a Windows
CP1252 console. Use `--tracking-mode off` only for an explicitly untracked local
reproduction.

This stage does not replace `/predict`, modify v1 artifacts, register a model,
or promote a model automatically.

The separately versioned v2 scaler fit is documented in
[`PHASE_2_V2_SCALER_ACCEPTANCE.md`](PHASE_2_V2_SCALER_ACCEPTANCE.md). It is
prepared for a future ANN path and is not consumed by this tree-based
reference model.
