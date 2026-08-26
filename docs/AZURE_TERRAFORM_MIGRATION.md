# Azure Terraform migration and Bicep retirement gate

Status: **blocked pending Azure evidence** (reviewed 2026-08-25).

The repository now contains separate Terraform `bootstrap`, `foundation`, and
`production` roots, a promotion workflow, and a protected digest-based
rollback workflow. The original Bicep workflow is kept as a legacy recovery
path until the migration has been proven against the actual subscription.
Removing it before that evidence exists would make the deployment path harder
to recover and would not prove parity.

## Required evidence before retirement

An operator with access to the target subscription must attach the following
to a pull request that has passed branch protection and the self-review
checklist, or to a decision record. The evidence must cover the
same resource group and registry names used by the protected workflows.

This repository uses the single-maintainer governance profile documented in
[`GITHUB_GOVERNANCE.md`](GITHUB_GOVERNANCE.md). Unless a real external
reviewer participates, independent human review is not applicable; deployment
approval evidence must be labelled as maintainer confirmation.

1. A read-only inventory of the Bicep-created resource IDs, names, regions,
   SKUs, identities, role assignments, Container Apps settings, Job schedule,
   budget, and Log Analytics configuration.
2. A Terraform import or provisioning record showing that the production state
   contains the intended resources without an unapproved replacement.
3. A successful protected production promotion using the release workflow,
   with the source SHA, API/frontend digests, Terraform plan summary, public
   smoke checks, and validation-Job success.
4. A successful protected rollback using
   [`rollback-production.yml`](../.github/workflows/rollback-production.yml),
   using the prior image digests and producing the same smoke evidence.
5. A final read-only comparison confirming that the rollback-capable Terraform
   state and the deployed Container Apps contract still match the inventory.
6. A successful post-deployment and post-rollback Terraform plan with detailed
   exit code `0`, recorded in the protected workflow summaries.

The comparison must explicitly record any intentional difference. A clean
Terraform plan alone is not sufficient evidence when the state has not first
been imported or compared with the live subscription.

## Retirement action

Only after all evidence above is validated by CI and explicitly accepted by
the maintainer should a separate,
small pull request:

- disable and then remove the legacy Bicep deployment workflow;
- remove `foundation.bicep` and `workload.bicep` only if no recovery or audit
  requirement still depends on them;
- update the Azure runbook and project status to identify Terraform as the
  only supported deployment path; and
- preserve the release and rollback manifests, digest references, plan
  summaries, and approval history as immutable deployment evidence.

This retirement action must not delete Azure resources. Resource deletion is a
separate, explicitly approved operation with an independently checked target.

## Current gate result

- GitHub `master` protection and `production` maintainer-confirmation controls:
  configured for the single-maintainer profile.
- Terraform roots and OIDC release/rollback workflows: present locally.
- Azure subscription inventory and Terraform state comparison: unavailable;
  Azure CLI and an authenticated session are not present in this workspace.
- Successful Terraform promotion and rollback: not executed.
- Bicep retirement: intentionally not performed.
