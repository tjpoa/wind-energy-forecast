terraform {
  backend "azurerm" {
    # The first bootstrap run uses -backend=false locally. The operator then
    # migrates this state to the storage account created by this root.
    resource_group_name  = "replace-with-bootstrap-output"
    storage_account_name = "tfstateplaceholder"
    container_name       = "tfstate"
    key                  = "bootstrap.tfstate"
    use_oidc             = true
    use_azuread_auth     = true
  }
}
