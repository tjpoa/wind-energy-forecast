output "resource_group_name" {
  description = "Resource group containing the portfolio demo."
  value       = azurerm_resource_group.this.name
}

output "registry_login_server" {
  description = "ACR login server used by the image publisher."
  value       = azurerm_container_registry.this.login_server
}

output "frontend_fqdn" {
  description = "Public frontend hostname."
  value       = azurerm_container_app.frontend.ingress[0].fqdn
}

output "api_revision_name" {
  description = "Latest API revision created by Container Apps."
  value       = azurerm_container_app.api.latest_revision_name
}

output "frontend_revision_name" {
  description = "Latest frontend revision created by Container Apps."
  value       = azurerm_container_app.frontend.latest_revision_name
}

output "validation_job_name" {
  description = "Scheduled validation Job name."
  value       = azurerm_container_app_job.validation.name
}
