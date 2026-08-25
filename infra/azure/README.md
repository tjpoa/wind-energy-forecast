# Azure Container Apps demo deployment

This directory deploys the synthetic, retrospective portfolio demo in West
Europe. It does not deploy live ingestion, provider credentials, PostgreSQL,
Airflow, Databricks, MLflow Registry serving, or automatic retraining.

The `foundation.bicep` deployment creates the resource group, Basic Azure
Container Registry, Log Analytics workspace, Container Apps environment, and
managed identity. The `workload.bicep` deployment creates the internal
FastAPI app, public React/Nginx app, scheduled read-only validation Job, and
the monthly budget.

## One-time Azure and GitHub setup

1. Create a Microsoft Entra application or user-assigned identity for GitHub
   Actions and a federated credential limited to this repository's `master`
   branch and the protected `azure-demo` environment.
2. Grant the deployment identity permission to deploy the foundation and create
   the two ACR role assignments for the first bootstrap. Run the workflow once
   with `bootstrap_foundation=true`; then reduce permissions to the resource
   group plus the registry `AcrPush` role for normal deployments.
3. Add `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
   `AZURE_PRINCIPAL_OBJECT_ID`, and `AZURE_BUDGET_ALERT_EMAIL` to the protected
   GitHub environment. No client secret or registry admin password is used.

The workflow is manual and approval-protected. The `bootstrap_foundation` input
is disabled by default after the one-time foundation deployment. It builds images tagged with
the commit SHA, resolves their registry digests, and deploys only those
immutable digests. The frontend image is built with `VITE_API_BASE_URL=/backend`
and `nginx.azure.conf`; the API image contains the tracked `demo/v1` bundle.

## Local preflight

From an authenticated Azure CLI session, validate the templates before using
the protected workflow:

```powershell
az bicep build --file infra/azure/foundation.bicep
az bicep build --file infra/azure/workload.bicep
```

The workflow performs the subscription deployment, image build/push, resource
group deployment, public smoke checks, and a manual validation-job execution.
The generated `wind-forecast-web` Container Apps HTTPS hostname is the v1
public URL. The API has internal ingress and is only reachable through the
allow-listed same-origin Nginx proxy paths.

## Rollback

Record the API and frontend image digests and revision names emitted by each
successful deployment. A rollback is a new protected deployment using the
previous digests, followed by the same frontend, API-proxy, and validation-job
smoke checks. Do not mutate or overwrite the evidence bundle.
