# Deterministic synthetic demo bundle

This is `demo-v1`, a clearly labelled synthetic evidence set for the
local dashboard. It is not REN data, CDS/ERA5-Land data, a trained
model release, or a production deployment. Values and identities are
generated deterministically by `scripts/build_demo_bundle.py` with
seed 2026. No credentials, network calls, MLflow state, or ignored
files are required.

The bundle contains a tiny historical-performance artifact, one
verified retrospective monitoring report, deployment attribution, and
a succeeded synthetic pipeline-run receipt.
