terraform {
  backend "azurerm" {
    # These non-secret placeholders make the backend schema self-validating.
    # The protected workflows override them with bootstrap outputs at init time.
    resource_group_name  = "replace-with-bootstrap-output"
    storage_account_name = "tfstateplaceholder"
    container_name       = "tfstate"
    key                  = "production.tfstate"
    use_oidc             = true
    use_azuread_auth     = true
  }
}
