terraform {
  backend "azurerm" {
    # Storage account, container, and key are injected by the protected
    # workflow. OIDC and Azure AD data-plane auth avoid access keys and SAS.
    use_oidc         = true
    use_azuread_auth = true
  }
}
