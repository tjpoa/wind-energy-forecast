variable "subscription_id" {
  description = "Azure subscription ID supplied by the workflow environment."
  type        = string
  sensitive   = true
}

variable "tenant_id" {
  description = "Microsoft Entra tenant ID supplied by the workflow environment."
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

variable "api_image" {
  description = "Immutable API image reference, including a digest."
  type        = string

  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.api_image))
    error_message = "api_image must be an image reference ending in a 64-character sha256 digest."
  }
}

variable "frontend_image" {
  description = "Immutable frontend image reference, including a digest."
  type        = string

  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.frontend_image))
    error_message = "frontend_image must be an image reference ending in a 64-character sha256 digest."
  }
}

variable "budget_amount" {
  description = "Monthly budget amount in the subscription billing currency."
  type        = number
  default     = 10

  validation {
    condition     = var.budget_amount > 0
    error_message = "budget_amount must be greater than zero."
  }
}

variable "budget_alert_email" {
  description = "Email address receiving budget notifications."
  type        = string
  sensitive   = true
}

variable "budget_start_date" {
  description = "First day of the budget period in ISO 8601 format."
  type        = string
}

variable "budget_end_date" {
  description = "End of the budget period in ISO 8601 format."
  type        = string
}
