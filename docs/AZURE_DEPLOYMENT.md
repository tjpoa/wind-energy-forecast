# Azure Container Apps deployment

This is the cloud runbook for the synthetic, retrospective portfolio demo. It
uses a public React/Nginx Container App, an internal FastAPI Container App, and
a daily scheduled Container Apps Job that verifies the immutable `demo/v1`
bundle. The API is not publicly addressable; Nginx exposes only the dashboard's
read-only health, performance, and monitoring paths.

## Runtime contract

- Region: West Europe.
- Frontend: one warm 0.25-vCPU/0.5-GiB replica.
- API: internal-only HTTP ingress (the public frontend is the sole caller),
  zero to two 0.5-vCPU/1-GiB replicas.
- Evidence: `demo/v1` is copied into the API image and deployed by digest.
- Job: one 0.25-vCPU/0.5-GiB replica at `06:00 UTC` daily, with one retry and
  a five-minute timeout. It performs checksum and projection validation only;
  it never updates the dashboard evidence.
- Storage: no mutable Azure Files or PostgreSQL state in v1.
- Secrets: no runtime credentials are needed. Future provider credentials or
  database DSNs must use Key Vault references and managed identities.
- Airflow remains local and documented; Databricks and PySpark remain deferred.

## Legacy Bicep deployment sequence

The manual, approval-protected `Deploy Azure portfolio demo (legacy Bicep)` workflow is the
legacy Bicep path:

1. Validates both Bicep templates.
2. Optionally bootstraps the foundation (resource group, Basic ACR, Log
   Analytics, Container Apps environment, and managed identity) when the
   `bootstrap_foundation` input is enabled.
3. Builds and pushes the API and cloud frontend images, unless a registered
   `release_run_id` is supplied for rollback.
4. Runs Bicep `what-if`, deploys the apps, Job, and €10 monthly budget, and
   records the resulting revision names and digests.
5. Smoke-tests the public URL, SPA routes, proxy allow-list, internal API
   behavior, and a manually started successful validation Job.

The one-time GitHub OIDC identity bootstrap and required environment variables
are documented in [`infra/azure/README.md`](../infra/azure/README.md). Azure
resource provisioning is intentionally not performed by ordinary CI pushes.

## Terraform promotion and rollback

The supported replacement path is the protected Terraform workflow. New
foundation changes use [`foundation-production.yml`](../.github/workflows/foundation-production.yml);
deployments use separate `bootstrap`, `foundation`, and `production` roots;
releases use [`release-production.yml`](../.github/workflows/release-production.yml);
rollbacks use [`rollback-production.yml`](../.github/workflows/rollback-production.yml)
with a registered `release_run_id`, which downloads the exact prior release
manifest instead of accepting arbitrary image digests. The production root reads the
foundation outputs from `foundation.tfstate`. Both workflows plan, require
the `production` approval, apply immutable image references, and run the
public and validation-Job smoke checks. Each protected apply is followed by a
Terraform plan with detailed exit codes; any post-deployment drift fails the
workflow.

All Azure mutation workflows share one concurrency group. A central preflight
requires `PRODUCTION_RELEASE_ENABLED=true` and the mode-specific OIDC/state
configuration before Azure login or image publication. Terraform plans reject
deletes and replacements both before and after the protected approval. Release
and rollback receipts record the source/release run, image digests, active
revisions, approval gate, smoke tests, and drift result.

The Bicep path cannot be retired until the inventory, Terraform parity,
promotion, and rollback gates in
[`AZURE_TERRAFORM_MIGRATION.md`](AZURE_TERRAFORM_MIGRATION.md) are approved.

## Rollback and cleanup

Record the `release_run_id` from each successful release receipt. Roll back by
running the protected rollback workflow with that run ID; the deployment
creates a new revision from the registered prior manifest and repeats the
smoke checks. Do not overwrite the evidence bundle or claim a production
rollback.

When the portfolio demo is no longer needed, delete the dedicated resource
group after confirming that no unrelated resources are in it. Budget alerts
are advisory and do not stop consumption automatically.
