variable "location" {
  description = "Azure region for the Terraform control-plane resources."
  type        = string
  default     = "westeurope"
}

variable "state_resource_group_name" {
  description = "Resource group containing the remote Terraform state storage."
  type        = string
  default     = "wind-energy-forecast-tfstate"
}

variable "state_storage_account_name" {
  description = "Globally unique, lowercase Azure Storage account name."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]{3,24}$", var.state_storage_account_name))
    error_message = "The state storage account name must contain 3-24 lowercase letters or digits."
  }
}

variable "state_container_name" {
  description = "Blob container holding Terraform state."
  type        = string
  default     = "tfstate"
}

variable "github_repository" {
  description = "GitHub owner/repository used in federated identity subjects."
  type        = string
  default     = "tjpoa/wind-energy-forecast"
}

variable "github_branch" {
  description = "Protected branch allowed to publish and plan."
  type        = string
  default     = "master"
}

variable "github_environment" {
  description = "Protected GitHub environment allowed to deploy."
  type        = string
  default     = "production"
}

variable "publisher_identity_name" {
  description = "User-assigned identity used only to publish images to ACR."
  type        = string
  default     = "wind-forecast-github-publisher"
}

variable "planner_identity_name" {
  description = "User-assigned identity used for read-only Terraform plans."
  type        = string
  default     = "wind-forecast-github-planner"
}

variable "deployer_identity_name" {
  description = "User-assigned identity used by the protected production deploy job."
  type        = string
  default     = "wind-forecast-github-deployer"
}
