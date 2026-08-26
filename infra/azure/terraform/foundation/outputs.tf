output "resource_group_id" {
  description = "Workload resource group ID created by the bootstrap root."
  value       = data.azurerm_resource_group.workload.id
}

output "resource_group_name" {
  description = "Workload resource group name."
  value       = data.azurerm_resource_group.workload.name
}

output "resource_group_location" {
  description = "Workload resource group Azure region."
  value       = data.azurerm_resource_group.workload.location
}

output "acr_name" {
  description = "Azure Container Registry name."
  value       = azurerm_container_registry.this.name
}

output "registry_login_server" {
  description = "ACR login server used by Container Apps and image publication."
  value       = azurerm_container_registry.this.login_server
}

output "container_app_environment_id" {
  description = "Container Apps Environment ID used by workload resources."
  value       = azurerm_container_app_environment.this.id
}

output "runtime_identity_id" {
  description = "Runtime user-assigned identity ID used by Container Apps."
  value       = azurerm_user_assigned_identity.runtime.id
}

output "runtime_identity_principal_id" {
  description = "Runtime user-assigned identity principal ID."
  value       = azurerm_user_assigned_identity.runtime.principal_id
}
