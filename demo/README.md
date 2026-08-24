# Reproducible dashboard demo

The tracked [`v1/`](v1/) directory is the default Docker Compose input. It is
a small, clearly labelled deterministic synthetic evidence bundle, not a
redistributed REN/CDS dataset or production model artifact.

Rebuild it with:

```powershell
.\venv\Scripts\python.exe .\scripts\build_demo_bundle.py --overwrite
```

The bundle manifest records its fixed seed, evidence type, claims boundary,
and SHA-256 for every member. It includes performance observations, a
retrospective monitoring report, deployment attribution, and a succeeded
synthetic pipeline-run receipt. No credentials, network requests, ignored
files, MLflow server, or external release asset are required.
