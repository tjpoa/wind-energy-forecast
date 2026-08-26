data "terraform_remote_state" "bootstrap" {
  backend = "azurerm"

  config = {
    resource_group_name  = var.tfstate_resource_group_name
    storage_account_name = var.tfstate_storage_account_name
    container_name       = var.tfstate_container_name
    key                  = "bootstrap.tfstate"
    subscription_id      = var.subscription_id
    tenant_id            = var.tenant_id
    use_oidc             = true
    use_azuread_auth     = true
  }
}
