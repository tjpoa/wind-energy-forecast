targetScope = 'resourceGroup'

@description('Azure region for the demo.')
param location string = 'westeurope'

@description('Existing Azure Container Registry name.')
param acrName string

@description('Existing Container Apps environment name.')
param environmentName string = 'wind-forecast-demo-env'

@description('Existing runtime managed identity name.')
param runtimeIdentityName string = 'wind-forecast-demo-runtime'

@description('Immutable API image reference, including a digest.')
param apiImage string

@description('Immutable frontend image reference, including a digest.')
param frontendImage string

@description('Monthly budget amount in the subscription billing currency.')
param budgetAmount int = 10

@description('Email address to receive budget notifications.')
param budgetAlertEmail string

@description('First day of the budget period in ISO 8601 format.')
param budgetStartDate string

@description('End of the budget period in ISO 8601 format.')
param budgetEndDate string

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

resource environment 'Microsoft.App/managedEnvironments@2023-05-01' existing = {
  name: environmentName
}

resource runtimeIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: runtimeIdentityName
}

var registryCredentials = [
  {
    server: registry.properties.loginServer
    identity: runtimeIdentity.id
  }
]

resource api 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'wind-forecast-api'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${runtimeIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: false
        targetPort: 8000
        transport: 'http'
        // Nginx reaches this internal-only app through Container Apps service
        // discovery using http://wind-forecast-api. No public endpoint exists.
        allowInsecure: true
      }
      registries: registryCredentials
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          env: [
            {
              name: 'WIND_FORECAST_PERFORMANCE_ARTIFACT_DIR'
              value: '/app/demo/v1/performance'
            }
            {
              name: 'WIND_FORECAST_MONITORING_STORE_ROOT'
              value: '/app/demo/v1/monitoring'
            }
            {
              name: 'WIND_FORECAST_CORS_ALLOW_ORIGINS'
              value: 'http://wind-forecast-web'
            }
          ]
          resources: {
            cpu: 0.5
            memory: '1Gi'
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 2
        rules: [
          {
            name: 'http-requests'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
  tags: {
    project: 'wind-energy-forecast'
    environment: 'portfolio-demo'
    component: 'api'
  }
}

resource web 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'wind-forecast-web'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${runtimeIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 80
        transport: 'http'
        allowInsecure: false
      }
      registries: registryCredentials
    }
    template: {
      containers: [
        {
          name: 'web'
          image: frontendImage
          resources: {
            cpu: 0.25
            memory: '0.5Gi'
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/'
                port: 80
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
  tags: {
    project: 'wind-energy-forecast'
    environment: 'portfolio-demo'
    component: 'frontend'
  }
}

resource validationJob 'Microsoft.App/jobs@2023-05-01' = {
  name: 'wind-forecast-validation'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${runtimeIdentity.id}': {}
    }
  }
  properties: {
    environmentId: environment.id
    configuration: {
      triggerType: 'Schedule'
      replicaTimeout: 300
      replicaRetryLimit: 1
      scheduleTriggerConfig: {
        cronExpression: '0 6 * * *'
        parallelism: 1
        replicaCompletionCount: 1
      }
      registries: registryCredentials
    }
    template: {
      containers: [
        {
          name: 'validator'
          image: apiImage
          command: [
            'python'
            '-m'
            'wind_forecast.demo_validation'
            '--bundle-root'
            '/app/demo/v1'
          ]
          resources: {
            cpu: 0.25
            memory: '0.5Gi'
          }
        }
      ]
    }
  }
  tags: {
    project: 'wind-energy-forecast'
    environment: 'portfolio-demo'
    component: 'validation-job'
  }
}

resource budget 'Microsoft.Consumption/budgets@2023-11-01' = {
  name: 'wind-forecast-demo-budget'
  properties: {
    category: 'Cost'
    amount: budgetAmount
    timeGrain: 'Monthly'
    timePeriod: {
      startDate: budgetStartDate
      endDate: budgetEndDate
    }
    notifications: {
      actual50: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 50
        contactEmails: [budgetAlertEmail]
      }
      actual80: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 80
        contactEmails: [budgetAlertEmail]
      }
      actual100: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 100
        contactEmails: [budgetAlertEmail]
      }
    }
  }
}

output frontendFqdn string = web.properties.configuration.ingress.fqdn
output apiFqdn string = api.properties.configuration.ingress.fqdn
output validationJobName string = validationJob.name
output apiRevisionName string = api.properties.latestRevisionName
output frontendRevisionName string = web.properties.latestRevisionName
