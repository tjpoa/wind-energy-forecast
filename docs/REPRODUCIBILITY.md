# Reproducibility And Model Lifecycle

This guide covers the local MLflow Registry and immutable GitHub Release bundle
workflow introduced in Phase 4B. It does not change the FastAPI serving paths,
publish v2 data, or make the local Registry a production service.

For the no-credentials full-stack demonstration, use the tracked
[`demo/v1`](../demo/v1/) bundle and run
`scripts/smoke_demo_stack.py`. That bundle is clearly labelled deterministic
synthetic evidence and is separate from the real-data release process below.

## 1. Start MLflow

```powershell
.\venv\Scripts\python.exe -m mlflow server `
  --backend-store-uri sqlite:///var/mlflow/mlflow.db `
  --artifacts-destination ./var/mlflow/artifacts `
  --host 127.0.0.1 `
  --port 5000
```

The SQLite database and artifact directory are local and ignored. Do not commit
or bundle them.

## 2. Train And Register A Candidate

Run from a clean Git commit with the approved v1 feature table:

```powershell
.\venv\Scripts\python.exe .\scripts\train_baseline.py --overwrite
.\venv\Scripts\python.exe .\scripts\register_candidate.py --run-id <RUN_ID>
```

The second command validates lineage, required artifacts, finite metrics,
signature-compatible input, and prediction equivalence before moving
`candidate`.

## 3. Promote Or Roll Back

```powershell
.\venv\Scripts\python.exe .\scripts\promote_model.py promote `
  --expected-candidate-version <VERSION> `
  --expected-champion-version none `
  --approval-note "manual review against the approved v1 contract"
```

For an existing champion, replace `none` with its expected version. Rollback is
allowed only while champion still points to the version in the receipt:

```powershell
.\venv\Scripts\python.exe .\scripts\promote_model.py rollback `
  --receipt outputs\registry\promotion-v<VERSION>.json
```

## 4. Build A Local Bundle

```powershell
.\venv\Scripts\python.exe .\scripts\build_reproduction_bundle.py `
  --release artifacts-v1.0.0 `
  --model-version <CANDIDATE_VERSION>
```

This produces a deterministic ZIP and SHA-256 file under `dist/`. Record that
digest in the tracked `bundle_sha256` catalog field, review the complete bundle,
and commit the catalog before creating the release tag. The command prints
`PUBLICATION BLOCKED` while the matching entry lacks explicit redistribution
approval. Publishing remains a separate manual, authorized action; released
assets are never silently replaced, and corrections use a new version such as
`artifacts-v1.0.1`.

## 5. Restore On Another Computer

After a release has been explicitly approved and published:

```powershell
git clone https://github.com/tjpoa/wind-energy-forecast.git
cd wind-energy-forecast
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -e .
.\venv\Scripts\python.exe .\scripts\fetch_reproduction_bundle.py `
  --release artifacts-v1.0.0 `
  --materialize
.\venv\Scripts\python.exe .\scripts\verify_reproduction.py `
  --release artifacts-v1.0.0 `
  --retrain
```

The fetcher requires the SHA-256 pinned in the tracked catalog, verifies the
release checksum and every declared manifest member, rejects duplicate,
undeclared, or unsafe members, and uses atomic temporary destinations. It
rejects
unsafe ZIP paths and existing destination data, and only then materializes the
training table. The reproduction command repeats the recorded deterministic
training configuration and compares metrics and predictions with
`rtol=1e-12`, `atol=1e-9`.

Do not claim cross-machine reproducibility until this sequence passes from a
clean clone. The current catalog intentionally blocks public data distribution
pending provenance and licence confirmation.
