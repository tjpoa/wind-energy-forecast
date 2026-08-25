targetScope = 'subscription'

@description('Resource group for the portfolio demo.')
param resourceGroupName string

@description('Azure region for the demo.')
param location string = 'westeurope'

@description('Globally unique Azure Container Registry name.')
param acrName string

@description('Container Apps environment name.')
param environmentName string = 'wind-forecast-demo-env'

@description('User-assigned identity used by Container Apps to pull images.')
param runtimeIdentityName string = 'wind-forecast-demo-runtime'

@description('Object ID of the protected GitHub deployment identity.')
param githubPrincipalObjectId string

resource resourceGroup 'Microsoft.Resources/resourceGroups@2022-09-01' = {
  name: resourceGroupName
  location: location
  tags: {
    project: 'wind-energy-forecast'
    environment: 'portfolio-demo'
    managedBy: 'bicep'
  }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  scope: resourceGroup
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
  tags: {
    project: 'wind-energy-forecast'
    environment: 'portfolio-demo'
  }
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  scope: resourceGroup
  name: 'wind-forecast-demo-logs'
  location: location
  properties: {
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
    sku: {
      name: 'PerGB2018'
    }
  }
  tags: {
    project: 'wind-energy-forecast'
    environment: 'portfolio-demo'
  }
}

resource environment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  scope: resourceGroup
  name: environmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: listKeys(logAnalytics.id, '2022-10-01').primarySharedKey
      }
    }
  }
  tags: {
    project: 'wind-energy-forecast'
    environment: 'portfolio-demo'
  }
}

resource runtimeIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  scope: resourceGroup
  name: runtimeIdentityName
  location: location
  tags: {
    project: 'wind-energy-forecast'
    environment: 'portfolio-demo'
  }
}

resource runtimeAcrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: registry
  name: guid(registry.id, runtimeIdentity.id, 'acr-pull')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7f951dda-4ed3-4680-a7ca-43fe172d538d'
    )
    principalId: runtimeIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource githubAcrPush 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: registry
  name: guid(registry.id, githubPrincipalObjectId, 'acr-push')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '8311e382-0749-4cb8-b61a-304f252e45ec'
    )
    principalId: githubPrincipalObjectId
    principalType: 'ServicePrincipal'
  }
}

output resourceGroupName string = resourceGroup.name
output registryName string = registry.name
output registryLoginServer string = registry.properties.loginServer
output environmentName string = environment.name
output runtimeIdentityResourceId string = runtimeIdentity.id
