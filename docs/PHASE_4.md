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

This checkpoint validates the implementation with synthetic/fake MLflow tests.
It does not claim that a real local server/SQLite Registry smoke, public
release, or clean-clone round-trip has already run.

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
