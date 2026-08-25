locals {
  common_tags = {
    project     = "wind-energy-forecast"
    environment = "platform"
    managed_by  = "terraform"
  }

  github_oidc_issuer   = "https://token.actions.githubusercontent.com"
  github_oidc_audience = ["api://AzureADTokenExchange"]
  branch_subject       = "repo:${var.github_repository}:ref:refs/heads/${var.github_branch}"
  environment_subject  = "repo:${var.github_repository}:environment:${var.github_environment}"
}

resource "azurerm_resource_group" "state" {
  name     = var.state_resource_group_name
  location = var.location
  tags     = local.common_tags
}

resource "azurerm_storage_account" "state" {
  name                            = var.state_storage_account_name
  resource_group_name             = azurerm_resource_group.state.name
  location                        = azurerm_resource_group.state.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  min_tls_version                 = "TLS1_2"
  https_traffic_only_enabled      = true
  public_network_access_enabled   = true
  shared_access_key_enabled       = false
  allow_nested_items_to_be_public = false

  blob_properties {
    versioning_enabled = true
  }

  tags = local.common_tags
}

resource "azurerm_storage_container" "state" {
  name                  = var.state_container_name
  storage_account_id    = azurerm_storage_account.state.id
  container_access_type = "private"
}

resource "azurerm_user_assigned_identity" "publisher" {
  name                = var.publisher_identity_name
  resource_group_name = azurerm_resource_group.state.name
  location            = azurerm_resource_group.state.location
  tags                = local.common_tags
}

resource "azurerm_user_assigned_identity" "planner" {
  name                = var.planner_identity_name
  resource_group_name = azurerm_resource_group.state.name
  location            = azurerm_resource_group.state.location
  tags                = local.common_tags
}

resource "azurerm_user_assigned_identity" "deployer" {
  name                = var.deployer_identity_name
  resource_group_name = azurerm_resource_group.state.name
  location            = azurerm_resource_group.state.location
  tags                = local.common_tags
}

resource "azurerm_federated_identity_credential" "publisher" {
  name                      = "github-master-publisher"
  user_assigned_identity_id = azurerm_user_assigned_identity.publisher.id
  audience                  = local.github_oidc_audience
  issuer                    = local.github_oidc_issuer
  subject                   = local.branch_subject
}

resource "azurerm_federated_identity_credential" "planner" {
  name                      = "github-master-planner"
  user_assigned_identity_id = azurerm_user_assigned_identity.planner.id
  audience                  = local.github_oidc_audience
  issuer                    = local.github_oidc_issuer
  subject                   = local.branch_subject
}

resource "azurerm_federated_identity_credential" "deployer" {
  name                      = "github-production-deployer"
  user_assigned_identity_id = azurerm_user_assigned_identity.deployer.id
  audience                  = local.github_oidc_audience
  issuer                    = local.github_oidc_issuer
  subject                   = local.environment_subject
}

data "azurerm_role_definition" "storage_blob_data_reader" {
  name = "Storage Blob Data Reader"
}

data "azurerm_role_definition" "storage_blob_data_contributor" {
  name = "Storage Blob Data Contributor"
}

resource "azurerm_role_assignment" "planner_state_reader" {
  scope              = azurerm_storage_account.state.id
  role_definition_id = data.azurerm_role_definition.storage_blob_data_reader.role_definition_id
  principal_id       = azurerm_user_assigned_identity.planner.principal_id
  principal_type     = "ServicePrincipal"
}

resource "azurerm_role_assignment" "deployer_state_contributor" {
  scope              = azurerm_storage_account.state.id
  role_definition_id = data.azurerm_role_definition.storage_blob_data_contributor.role_definition_id
  principal_id       = azurerm_user_assigned_identity.deployer.principal_id
  principal_type     = "ServicePrincipal"
}
