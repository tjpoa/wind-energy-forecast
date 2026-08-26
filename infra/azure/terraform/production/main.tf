locals {
  common_tags = {
    project     = "wind-energy-forecast"
    environment = "portfolio-demo"
    managed_by  = "terraform"
  }
  foundation = data.terraform_remote_state.foundation.outputs
}

locals {
  registry_server = local.foundation.registry_login_server
  registry = {
    server   = local.registry_server
    identity = local.foundation.runtime_identity_id
  }
}

resource "azurerm_container_app" "api" {
  name                         = "wind-forecast-api"
  container_app_environment_id = local.foundation.container_app_environment_id
  resource_group_name          = local.foundation.resource_group_name
  revision_mode                = "Single"
  tags                         = merge(local.common_tags, { component = "api" })

  identity {
    type         = "UserAssigned"
    identity_ids = [local.foundation.runtime_identity_id]
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
  container_app_environment_id = local.foundation.container_app_environment_id
  resource_group_name          = local.foundation.resource_group_name
  revision_mode                = "Single"
  tags                         = merge(local.common_tags, { component = "frontend" })

  identity {
    type         = "UserAssigned"
    identity_ids = [local.foundation.runtime_identity_id]
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
  location                     = local.foundation.resource_group_location
  resource_group_name          = local.foundation.resource_group_name
  container_app_environment_id = local.foundation.container_app_environment_id
  replica_timeout_in_seconds   = 300
  replica_retry_limit          = 1
  tags                         = merge(local.common_tags, { component = "validation-job" })

  identity {
    type         = "UserAssigned"
    identity_ids = [local.foundation.runtime_identity_id]
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
  resource_group_id = local.foundation.resource_group_id
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
