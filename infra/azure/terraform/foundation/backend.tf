terraform {
  backend "azurerm" {
    # The coordinates are supplied by the operator or a protected workflow.
    resource_group_name  = "replace-with-bootstrap-output"
    storage_account_name = "tfstateplaceholder"
    container_name       = "tfstate"
    key                  = "foundation.tfstate"
    use_oidc             = true
    use_azuread_auth     = true
  }
}
