# Terraform Azure deployment foundation

This directory is the Terraform source for the protected Azure Container Apps
deployment path. It was introduced alongside the existing Bicep templates;
the Bicep workflow remains a legacy recovery path until the Terraform
resources have been imported or provisioned, compared, and accepted against
the target subscription.

## Layout

- `bootstrap/` describes the remote state storage and the user-assigned
  identities used by GitHub Actions, creates the workload resource group, and
  grants the state and workload permissions used by later roots. Its first
  application requires a subscription operator because it creates the trust
  boundary used by later OIDC workflows.
- `foundation/` describes the ACR, Log Analytics workspace, Container Apps
  Environment, runtime identity, and the `AcrPush`/`AcrPull` role assignments.
  It consumes the workload resource group and publisher identity outputs from
  `bootstrap.tfstate`.
- `production/` describes only the API and frontend Container Apps, the
  validation Job, and the monthly budget. It consumes `foundation` outputs via
  the `foundation.tfstate` remote state.

All three roots use the AzureRM provider and contain no credentials or
environment secrets. Their state keys are separate: `bootstrap.tfstate`,
`foundation.tfstate`, and `production.tfstate`. The backends contain only
non-secret placeholders required by Terraform validation; protected workflows
override the backend coordinates through `-backend-config` values supplied at
initialization time.

The first bootstrap run is intentionally local and must be initialized with
`-backend=false`. After the state storage has been created, the operator must
migrate and verify that state in the `bootstrap.tfstate` Azure Blob before
using the `foundation` or `production` roots. This task does not perform that
migration or create Azure resources.

## GitHub controls configured for protected Azure workflows

The repository controls are now configured as follows:

- `master` requires a pull request, resolved conversations, linear history,
  and the `CI gate` check. It requires one independent approval, approval of
  the latest reviewable push, and dismissal of stale reviews. Administrator
  bypass, force pushes, and branch deletion are disabled.
- `production` requires a manual review, accepts deployments only from
  protected branches, disallows administrator bypass, and prevents the person
  who started a deployment from approving it. Keep an independent reviewer in
  the environment reviewer list.

The independent-reviewer configuration is an external GitHub control and must
be rechecked as part of the final evidence matrix; it cannot be represented by
repository code.

The Azure identifiers and OIDC client IDs are not stored in the repository.
The later bootstrap/application step must populate the protected environment
with the required Azure identifiers and grant the generated identities the
permissions represented by the Terraform roots.

## Protected Azure workflow controls

The release, foundation, legacy recovery, and rollback workflows share the
`azure-production-mutation` concurrency group. Only one Azure mutation can run
at a time. Each workflow calls the central preflight action before Azure login
or publication. The preflight requires the explicit
`PRODUCTION_RELEASE_ENABLED=true` repository variable and validates only the
mode-specific configuration without printing values.

The CI workflow rejects tracked Terraform state, plans, non-example `.tfvars`,
password/client-secret paths, and credential-file extensions. The policy is
implemented in testable Python with JSON fixtures.

## Promotion workflow

[`release-production.yml`](../../../.github/workflows/release-production.yml)
is a promotion workflow, not a second independent build path. It runs only
after the `CI` workflow succeeds for a push to the protected `master` branch,
checks out `github.event.workflow_run.head_sha`, and then:

1. builds and smoke-tests the API and frontend images once;
2. labels both images with `org.opencontainers.image.revision` equal to the
   source SHA, publishes them to ACR, and records immutable registry digests;
3. stores a schema-validated release manifest containing the source SHA, CI
   run, release run, image digests, and OCI labels;
4. runs a Terraform plan and rejects deletes or replacements before approval;
5. waits for the protected `production` environment approval;
6. re-plans after approval, rejects deletes or replacements again, applies the
   exact plan, verifies active Container Apps image references against the
   manifest, runs smoke tests, and fails closed if the post-deployment plan
   reports drift;
7. uploads a receipt containing commit/run IDs, digests, revisions, approval
   gate status, smoke results, and the post-deployment drift result.

The binary Terraform plan is deliberately not uploaded because it can contain
sensitive state. The post-approval job applies the exact plan it generated
immediately before `terraform apply`; the preceding plan job provides a
non-sensitive review summary. The workflow uses GitHub-to-Azure OIDC for the
publisher, planner, and protected deployer identities. No client secret,
registry admin password, or long-lived cloud credential is expected.

Before enabling the workflows, configure these GitHub Actions repository or
environment values from the outputs of the approved bootstrap and foundation:

- Variables: `PRODUCTION_RELEASE_ENABLED`, `AZURE_ACR_NAME`,
  `AZURE_RESOURCE_GROUP`,
  `TFSTATE_RESOURCE_GROUP_NAME`, `TFSTATE_STORAGE_ACCOUNT_NAME`, and
  optionally `TFSTATE_CONTAINER_NAME`, `AZURE_ENVIRONMENT_NAME`, and
  `AZURE_RUNTIME_IDENTITY_NAME`. The production Terraform root reads the
  workload resource group, Container Apps Environment, runtime identity, and
  registry server from `foundation.tfstate`; the release and rollback CLI
  smoke/publish steps continue to use `AZURE_RESOURCE_GROUP` and
  `AZURE_ACR_NAME`.
- Variables required by the budget contract:
  `AZURE_BUDGET_START_DATE` and `AZURE_BUDGET_END_DATE`.
- Secrets used by the production workflows: `AZURE_TENANT_ID`,
  `AZURE_SUBSCRIPTION_ID`, `AZURE_PUBLISHER_CLIENT_ID`,
  `AZURE_PLANNER_CLIENT_ID`, and `AZURE_BUDGET_ALERT_EMAIL`.
- Protected `production` environment secret:
  `AZURE_DEPLOYER_CLIENT_ID`.

The bootstrap principal IDs are consumed by `foundation`; they are not runtime
secrets and do not need to be passed to the production roots.
The client IDs are identifiers, not client credentials; they are kept out of
logs to make the trust boundary explicit. Do not create placeholder values.
The production workflow cannot run successfully until the Terraform bootstrap
has created the remote state, federated credentials, role assignments, the
foundation state, and the production portfolio has been imported or
provisioned.

## Foundation and rollback workflows

[`foundation-production.yml`](../../../.github/workflows/foundation-production.yml)
plans and applies only the Terraform `foundation` root. It uses the same
concurrency group and protected `production` approval. Both the pre-approval
and post-approval plans reject deletes and replacements.
The protected identity used for its apply must have the reviewed Azure
permission to create the ACR role assignments in the workload resource group;
the workflow does not infer or elevate that permission.

[`rollback-production.yml`](../../../.github/workflows/rollback-production.yml)
is manually dispatched from the protected `master` branch with a successful
`release_run_id`. It downloads the registered release manifest from that run;
operators cannot supply arbitrary image digests. The workflow records a
rollback manifest, creates a non-sensitive Terraform plan summary, rejects
deletes or replacements before and after approval, applies the exact plan,
verifies active image references, repeats the public dashboard, proxy, and
validation-Job smoke tests, and performs a fail-closed post-rollback drift
check. It uploads a receipt linked to the original release run.

The rollback workflow never rebuilds or pushes an image. A rollback is a new
Container Apps revision selected by the registered prior release manifest.
Keep both the original release evidence and the immutable rollback receipt.

## Static validation

From the repository root, the CI checks all three roots without contacting Azure:

```text
terraform fmt -check -recursive infra/azure/terraform
terraform -chdir=infra/azure/terraform/bootstrap init -backend=false -input=false
terraform -chdir=infra/azure/terraform/bootstrap validate
terraform -chdir=infra/azure/terraform/foundation init -backend=false -input=false
terraform -chdir=infra/azure/terraform/foundation validate
terraform -chdir=infra/azure/terraform/production init -backend=false -input=false
terraform -chdir=infra/azure/terraform/production validate
```

The commands above install provider plugins in the CI runner's temporary
workspace. No `.terraform` directory, state file, plan, or variable file is
committed.

## Deliberate stop boundary

This workspace has not run `terraform plan`, `terraform apply`, `terraform
import`, Azure CLI, or a network-backed provider operation. The roots and
workflow wiring remain configuration-only in this workspace. Terraform parity,
a successful promotion, and a successful rollback remain mandatory gates
before the legacy Bicep workflow can be retired. The complete gate is
documented in
[`AZURE_TERRAFORM_MIGRATION.md`](../../../docs/AZURE_TERRAFORM_MIGRATION.md).
