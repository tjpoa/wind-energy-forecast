locals {
  common_tags = {
    project     = "wind-energy-forecast"
    environment = "portfolio-demo"
    managed_by  = "terraform"
  }
}

data "azurerm_resource_group" "workload" {
  name = data.terraform_remote_state.bootstrap.outputs.workload_resource_group_name
}

resource "azurerm_container_registry" "this" {
  name                          = var.acr_name
  resource_group_name           = data.azurerm_resource_group.workload.name
  location                      = data.azurerm_resource_group.workload.location
  sku                           = "Basic"
  admin_enabled                 = false
  public_network_access_enabled = true
  tags                          = local.common_tags
}

resource "azurerm_log_analytics_workspace" "this" {
  name                = "wind-forecast-demo-logs"
  location            = data.azurerm_resource_group.workload.location
  resource_group_name = data.azurerm_resource_group.workload.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.common_tags
}

resource "azurerm_container_app_environment" "this" {
  name                       = var.environment_name
  location                   = data.azurerm_resource_group.workload.location
  resource_group_name        = data.azurerm_resource_group.workload.name
  logs_destination           = "log-analytics"
  log_analytics_workspace_id = azurerm_log_analytics_workspace.this.id
  tags                       = local.common_tags
}

resource "azurerm_user_assigned_identity" "runtime" {
  name                = var.runtime_identity_name
  resource_group_name = data.azurerm_resource_group.workload.name
  location            = data.azurerm_resource_group.workload.location
  tags                = local.common_tags
}

data "azurerm_role_definition" "acr_pull" {
  name = "AcrPull"
}

data "azurerm_role_definition" "acr_push" {
  name = "AcrPush"
}

resource "azurerm_role_assignment" "runtime_acr_pull" {
  scope              = azurerm_container_registry.this.id
  role_definition_id = data.azurerm_role_definition.acr_pull.role_definition_id
  principal_id       = azurerm_user_assigned_identity.runtime.principal_id
  principal_type     = "ServicePrincipal"
}

resource "azurerm_role_assignment" "github_acr_push" {
  scope              = azurerm_container_registry.this.id
  role_definition_id = data.azurerm_role_definition.acr_push.role_definition_id
  principal_id       = data.terraform_remote_state.bootstrap.outputs.publisher_principal_id
  principal_type     = "ServicePrincipal"
}
