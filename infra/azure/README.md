# Azure deployment

This directory contains the synthetic, retrospective portfolio deployment.
It does not deploy live ingestion, provider credentials, PostgreSQL, Airflow,
Databricks, Registry-based serving, or automatic retraining.

## Sources of truth

- Legacy recovery path: `foundation.bicep`, `workload.bicep`, and
  `.github/workflows/deploy-azure-demo.yml`.
- Protected path: `terraform/` and
  `.github/workflows/foundation-production.yml`,
  `release-production.yml`, and `rollback-production.yml`.
- Static safety policy: `scripts/azure_workflow_policy.py` and its tests.

The Terraform path is configuration-only until bootstrap, state migration,
resource parity, promotion, and rollback evidence exists. Bicep retirement is
not authorized by repository code alone.

## External prerequisites

An operator must create the protected GitHub `production` environment, Azure
OIDC federated credentials, remote Terraform state, and the role assignments
declared by the Terraform roots. Store identifiers in protected GitHub
variables/secrets, never in the repository. No client secret or registry admin
password is expected.

The Terraform workflows run read-only readiness probes before publishing an
image, initializing a backend, or creating a rollback manifest. The probes
check the environment reviewer/branch policy, the exact OIDC subject, the
Entra-authenticated state container and upstream state key, and the least
privilege RBAC assignments. Planner and deployer identities also require a
control-plane `Reader` assignment on the state storage account so the probe can
inspect RBAC without acquiring a Terraform lock. A failed probe publishes a
sanitized immutable `azure_external_readiness.v1` receipt and blocks the
workflow.

The release gate requires `PRODUCTION_RELEASE_ENABLED=true`, successful CI on
protected `master`, and the protected maintainer confirmation. Workflows use
one `azure-production-mutation` concurrency group and immutable image digests.

## Static checks

These commands do not contact Azure:

```powershell
terraform fmt -check -recursive infra/azure/terraform
terraform -chdir=infra/azure/terraform/bootstrap init -backend=false -input=false
terraform -chdir=infra/azure/terraform/bootstrap validate
terraform -chdir=infra/azure/terraform/foundation init -backend=false -input=false
terraform -chdir=infra/azure/terraform/foundation validate
terraform -chdir=infra/azure/terraform/production init -backend=false -input=false
terraform -chdir=infra/azure/terraform/production validate
```

The legacy templates can be syntax-checked with:

```powershell
az bicep build --file infra/azure/foundation.bicep
az bicep build --file infra/azure/workload.bicep
```

## Promotion and rollback

The protected release workflow builds and smoke-tests each image once, records
source SHA and immutable digests, rejects destructive Terraform changes,
requires confirmation, applies the exact reviewed plan, verifies active image
references, and performs a post-deployment drift check.

Rollback is a new protected deployment selected only from a prior registered
release manifest. It never accepts arbitrary image digests and repeats the same
smoke and drift checks. Preserve both release and rollback receipts.

Do not run Azure CLI, Terraform apply/import, or a live workflow without an
approved external environment and an operator-reviewed readiness gate.
