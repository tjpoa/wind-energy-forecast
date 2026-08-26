# Terraform Azure deployment foundation

This directory is the Terraform source for the future Azure Container Apps
deployment path. It is intentionally introduced alongside the existing Bicep
templates. The current Bicep workflow remains the active deployment path until
the Terraform resources have been imported or provisioned, compared, and
accepted in a later increment.

## Layout

- `bootstrap/` describes the remote state storage and the user-assigned
  identities used by GitHub Actions. Its first application requires a
  subscription operator because it creates the trust boundary used by later
  OIDC workflows.
- `production/` describes the existing cost-bounded Container Apps portfolio
  demo: ACR, Log Analytics, Container Apps, the validation Job, managed
  identity permissions, and the monthly budget.

Both roots use the AzureRM provider and contain no credentials or environment
secrets. The production backend contains only non-secret placeholders required
by Terraform validation; protected workflows override all backend coordinates
through `-backend-config` values supplied at initialization time.

## GitHub controls configured in Increment 2

The repository controls are now configured as follows:

- `master` requires a pull request, one approval, resolved conversations,
  linear history, and the `CI gate` check. Administrator bypass and force
  pushes are disabled.
- `production` requires a manual review, accepts deployments only from
  protected branches, and disallows administrator bypass.
- The current repository owner is the environment reviewer and self-review is
  permitted because this is a single-owner repository. Add an independent
  reviewer before enabling `prevent_self_review`.

The Azure identifiers and OIDC client IDs are not stored in the repository.
The later bootstrap/application step must populate the protected environment
with the required Azure identifiers and grant the generated identities the
permissions represented by the Terraform roots.

## Promotion workflow added in Increment 3

[`release-production.yml`](../../../.github/workflows/release-production.yml)
is a promotion workflow, not a second independent build path. It runs only
after the `CI` workflow succeeds for a push to the protected `master` branch,
checks out `github.event.workflow_run.head_sha`, and then:

1. builds and smoke-tests the API and frontend images once;
2. publishes both images to ACR and records their immutable registry digests;
3. stores a small release manifest containing the source SHA and both digests;
4. runs a read-only Terraform plan and publishes only address/action metadata;
5. waits for the protected `production` environment approval;
6. re-plans after approval, rejects destructive changes, applies that exact
   plan, and runs the public dashboard, proxy, and validation-Job smoke tests.

The binary Terraform plan is deliberately not uploaded because it can contain
sensitive state. The post-approval job applies the exact plan it generated
immediately before `terraform apply`; the preceding plan job provides a
non-sensitive review summary. The workflow uses GitHub-to-Azure OIDC for the
publisher, planner, and protected deployer identities. No client secret,
registry admin password, or long-lived cloud credential is expected.

Before enabling the workflow, configure these GitHub Actions repository or
environment values from the outputs of the approved bootstrap:

- Variables: `AZURE_ACR_NAME`, `TFSTATE_RESOURCE_GROUP_NAME`,
  `TFSTATE_STORAGE_ACCOUNT_NAME`, and optionally `TFSTATE_CONTAINER_NAME`.
  The workflow defaults the workload resource group, location, Container Apps
  environment, runtime identity, and budget amount to the Terraform defaults.
- Variables required by the budget contract:
  `AZURE_BUDGET_START_DATE` and `AZURE_BUDGET_END_DATE`.
- Secrets: `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
  `AZURE_PUBLISHER_CLIENT_ID`, `AZURE_PLANNER_CLIENT_ID`,
  `AZURE_PUBLISHER_PRINCIPAL_ID`, `AZURE_PLANNER_PRINCIPAL_ID`,
  `AZURE_DEPLOYER_PRINCIPAL_ID`, and `AZURE_BUDGET_ALERT_EMAIL`.
- Protected `production` environment secret:
  `AZURE_DEPLOYER_CLIENT_ID`.

The client IDs and principal IDs are identifiers, not client credentials; they
are listed as secrets to keep the workflow configuration out of logs and to
make the trust boundary explicit. Do not create placeholder values. The
workflow cannot run successfully until the Terraform bootstrap has created the
remote state, federated credentials, role assignments, and the production
portfolio has been imported or provisioned.

## Rollback workflow added in Increment 4

[`rollback-production.yml`](../../../.github/workflows/rollback-production.yml)
is manually dispatched from the protected `master` branch with the previous
API and frontend image references. Both references must use the same registry,
the expected repositories, and a full `@sha256:<64 hexadecimal characters>`
digest. The workflow records the request manifest, creates a non-sensitive
Terraform plan summary, waits for the protected `production` approval, rejects
destructive changes, applies the exact post-approval plan, and repeats the
public dashboard, proxy, and validation-Job smoke tests.

The rollback workflow never rebuilds or pushes an image. A rollback is a new
Container Apps revision selected by the prior immutable digests. Keep the
workflow summary and manifest as the rollback receipt; do not overwrite the
prior release evidence.

## Static validation

From the repository root, the CI checks both roots without contacting Azure:

```text
terraform fmt -check -recursive infra/azure/terraform
terraform -chdir=infra/azure/terraform/bootstrap init -backend=false -input=false
terraform -chdir=infra/azure/terraform/bootstrap validate
terraform -chdir=infra/azure/terraform/production init -backend=false -input=false
terraform -chdir=infra/azure/terraform/production validate
```

The commands above install provider plugins in the CI runner's temporary
workspace. No `.terraform` directory, state file, plan, or variable file is
committed.

## Deliberate stop boundary

This workspace has not run `terraform plan`, `terraform apply`, `terraform
import`, Azure CLI, or a network-backed provider operation. The four increments
add the release and rollback workflows, but they are not yet operational in
this workspace. Terraform parity, a successful promotion, and a successful
rollback remain mandatory gates before the legacy Bicep workflow can be
retired. The complete gate is documented in
[`AZURE_TERRAFORM_MIGRATION.md`](../../../docs/AZURE_TERRAFORM_MIGRATION.md).
