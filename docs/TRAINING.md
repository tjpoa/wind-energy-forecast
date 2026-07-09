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
- It does not implement a model registry or promotion workflow.
- It does not replace future fuller training, tuning, or monitoring work.
