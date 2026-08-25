locals {
  common_tags = {
    project     = "wind-energy-forecast"
    environment = "portfolio-demo"
    managed_by  = "terraform"
  }
}

resource "azurerm_resource_group" "this" {
  name     = var.resource_group_name
  location = var.location
  tags     = local.common_tags
}

resource "azurerm_container_registry" "this" {
  name                          = var.acr_name
  resource_group_name           = azurerm_resource_group.this.name
  location                      = azurerm_resource_group.this.location
  sku                           = "Basic"
  admin_enabled                 = false
  public_network_access_enabled = true
  tags                          = local.common_tags
}

resource "azurerm_log_analytics_workspace" "this" {
  name                = "wind-forecast-demo-logs"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.common_tags
}

resource "azurerm_container_app_environment" "this" {
  name                       = var.environment_name
  location                   = azurerm_resource_group.this.location
  resource_group_name        = azurerm_resource_group.this.name
  logs_destination           = "log-analytics"
  log_analytics_workspace_id = azurerm_log_analytics_workspace.this.id
  tags                       = local.common_tags
}

resource "azurerm_user_assigned_identity" "runtime" {
  name                = var.runtime_identity_name
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  tags                = local.common_tags
}

data "azurerm_role_definition" "acr_pull" {
  name = "AcrPull"
}

data "azurerm_role_definition" "acr_push" {
  name = "AcrPush"
}

data "azurerm_role_definition" "reader" {
  name = "Reader"
}

resource "azurerm_role_assignment" "runtime_acr_pull" {
  scope              = azurerm_container_registry.this.id
  role_definition_id = data.azurerm_role_definition.acr_pull.role_definition_id
  principal_id       = azurerm_user_assigned_identity.runtime.principal_id
  principal_type     = "ServicePrincipal"
}

resource "azurerm_role_assignment" "github_acr_push" {
  scope              = azurerm_container_registry.this.id
  role_definition_id = data.azurerm_role_definition.acr_push.role_definition_id
  principal_id       = var.publisher_principal_id
  principal_type     = "ServicePrincipal"
}

resource "azurerm_role_assignment" "github_resource_group_reader" {
  scope              = azurerm_resource_group.this.id
  role_definition_id = data.azurerm_role_definition.reader.role_definition_id
  principal_id       = var.planner_principal_id
  principal_type     = "ServicePrincipal"
}

resource "azurerm_role_assignment" "github_resource_group_contributor" {
  scope              = azurerm_resource_group.this.id
  role_definition_id = data.azurerm_role_definition.contributor.role_definition_id
  principal_id       = var.deployer_principal_id
  principal_type     = "ServicePrincipal"
}

data "azurerm_role_definition" "contributor" {
  name = "Contributor"
}

locals {
  registry_server = azurerm_container_registry.this.login_server
  registry = {
    server   = local.registry_server
    identity = azurerm_user_assigned_identity.runtime.id
  }
}

resource "azurerm_container_app" "api" {
  name                         = "wind-forecast-api"
  container_app_environment_id = azurerm_container_app_environment.this.id
  resource_group_name          = azurerm_resource_group.this.name
  revision_mode                = "Single"
  tags                         = merge(local.common_tags, { component = "api" })

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.runtime.id]
  }

  registry {
    server   = local.registry.server
    identity = local.registry.identity
  }

  ingress {
    external_enabled           = false
    target_port                = 8000
    transport                  = "http"
    allow_insecure_connections = true

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  template {
    min_replicas = 0
    max_replicas = 2

    http_scale_rule {
      name                = "http-requests"
      concurrent_requests = 10
    }

    container {
      name   = "api"
      image  = var.api_image
      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "WIND_FORECAST_PERFORMANCE_ARTIFACT_DIR"
        value = "/app/demo/v1/performance"
      }

      env {
        name  = "WIND_FORECAST_MONITORING_STORE_ROOT"
        value = "/app/demo/v1/monitoring"
      }

      env {
        name  = "WIND_FORECAST_CORS_ALLOW_ORIGINS"
        value = "http://wind-forecast-web"
      }

      liveness_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/health"
        initial_delay           = 10
        interval_seconds        = 10
        timeout                 = 5
        failure_count_threshold = 3
      }
    }
  }
}

resource "azurerm_container_app" "frontend" {
  name                         = "wind-forecast-web"
  container_app_environment_id = azurerm_container_app_environment.this.id
  resource_group_name          = azurerm_resource_group.this.name
  revision_mode                = "Single"
  tags                         = merge(local.common_tags, { component = "frontend" })

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.runtime.id]
  }

  registry {
    server   = local.registry.server
    identity = local.registry.identity
  }

  ingress {
    external_enabled           = true
    target_port                = 80
    transport                  = "http"
    allow_insecure_connections = false

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  template {
    min_replicas = 1
    max_replicas = 1

    container {
      name   = "web"
      image  = var.frontend_image
      cpu    = 0.25
      memory = "0.5Gi"

      liveness_probe {
        transport               = "HTTP"
        port                    = 80
        path                    = "/"
        initial_delay           = 5
        interval_seconds        = 10
        timeout                 = 5
        failure_count_threshold = 3
      }
    }
  }
}

resource "azurerm_container_app_job" "validation" {
  name                         = "wind-forecast-validation"
  location                     = azurerm_resource_group.this.location
  resource_group_name          = azurerm_resource_group.this.name
  container_app_environment_id = azurerm_container_app_environment.this.id
  replica_timeout_in_seconds   = 300
  replica_retry_limit          = 1
  tags                         = merge(local.common_tags, { component = "validation-job" })

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.runtime.id]
  }

  registry {
    server   = local.registry.server
    identity = local.registry.identity
  }

  schedule_trigger_config {
    cron_expression          = "0 6 * * *"
    parallelism              = 1
    replica_completion_count = 1
  }

  template {
    container {
      name    = "validator"
      image   = var.api_image
      command = ["python", "-m", "wind_forecast.demo_validation", "--bundle-root", "/app/demo/v1"]
      cpu     = 0.25
      memory  = "0.5Gi"
    }
  }
}

resource "azurerm_consumption_budget_resource_group" "this" {
  name              = "wind-forecast-demo-budget"
  resource_group_id = azurerm_resource_group.this.id
  amount            = var.budget_amount
  time_grain        = "Monthly"

  time_period {
    start_date = var.budget_start_date
    end_date   = var.budget_end_date
  }

  notification {
    enabled        = true
    threshold      = 50
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = [var.budget_alert_email]
  }

  notification {
    enabled        = true
    threshold      = 80
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = [var.budget_alert_email]
  }

  notification {
    enabled        = true
    threshold      = 100
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = [var.budget_alert_email]
  }
}
