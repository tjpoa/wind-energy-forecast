output "state_resource_group_name" {
  description = "Resource group containing remote Terraform state."
  value       = azurerm_resource_group.state.name
}

output "state_storage_account_name" {
  description = "Storage account containing remote Terraform state."
  value       = azurerm_storage_account.state.name
}

output "state_container_name" {
  description = "Blob container containing remote Terraform state."
  value       = azurerm_storage_container.state.name
}

output "publisher_client_id" {
  description = "Client ID for the branch-scoped image publisher identity."
  value       = azurerm_user_assigned_identity.publisher.client_id
}

output "publisher_principal_id" {
  description = "Object ID for the branch-scoped image publisher identity."
  value       = azurerm_user_assigned_identity.publisher.principal_id
}

output "planner_client_id" {
  description = "Client ID for the branch-scoped Terraform planner identity."
  value       = azurerm_user_assigned_identity.planner.client_id
}

output "planner_principal_id" {
  description = "Object ID for the branch-scoped Terraform planner identity."
  value       = azurerm_user_assigned_identity.planner.principal_id
}

output "deployer_client_id" {
  description = "Client ID for the environment-scoped production deployer identity."
  value       = azurerm_user_assigned_identity.deployer.client_id
}

output "deployer_principal_id" {
  description = "Object ID for the environment-scoped production deployer identity."
  value       = azurerm_user_assigned_identity.deployer.principal_id
}
