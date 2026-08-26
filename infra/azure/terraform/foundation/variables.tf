variable "subscription_id" {
  description = "Azure subscription ID supplied by the operator or workflow."
  type        = string
  sensitive   = true
}

variable "tenant_id" {
  description = "Microsoft Entra tenant ID supplied by the operator or workflow."
  type        = string
  sensitive   = true
}

variable "tfstate_resource_group_name" {
  description = "Resource group containing the remote Terraform state storage."
  type        = string
}

variable "tfstate_storage_account_name" {
  description = "Storage account containing the remote Terraform state."
  type        = string
}

variable "tfstate_container_name" {
  description = "Blob container containing the remote Terraform state."
  type        = string
  default     = "tfstate"
}

variable "acr_name" {
  description = "Globally unique Azure Container Registry name."
  type        = string
}

variable "environment_name" {
  description = "Azure Container Apps environment name."
  type        = string
  default     = "wind-forecast-demo-env"
}

variable "runtime_identity_name" {
  description = "User-assigned identity used by Container Apps to pull images."
  type        = string
  default     = "wind-forecast-demo-runtime"
}
