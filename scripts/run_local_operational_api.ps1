[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExecutable,
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,
    [Parameter(Mandatory = $true)]
    [string]$DeploymentRoot,
    [Parameter(Mandatory = $true)]
    [string]$MonitoringStoreRoot,
    [ValidateRange(1, 600)]
    [int]$MlflowHealthTimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path -LiteralPath $RepositoryRoot).Path

function Resolve-RepositoryPath([string]$Value) {
    $candidate = $Value
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $repository $candidate
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

$python = Resolve-RepositoryPath $PythonExecutable
$deployment = Resolve-RepositoryPath $DeploymentRoot
$monitoring = Resolve-RepositoryPath $MonitoringStoreRoot
$apiModule = Join-Path $repository "src\wind_forecast\api.py"
if (-not (Test-Path -LiteralPath $apiModule -PathType Leaf)) {
    throw "Operational API module was not found: $apiModule"
}

$mlflowHealthUri = "http://127.0.0.1:5000/health"
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$mlflowReady = $false
while ($stopwatch.Elapsed.TotalSeconds -lt $MlflowHealthTimeoutSeconds) {
    $remaining = $MlflowHealthTimeoutSeconds - $stopwatch.Elapsed.TotalSeconds
    if ($remaining -lt 1) {
        break
    }
    $requestTimeout = [Math]::Min(5, [Math]::Floor($remaining))
    try {
        $response = Invoke-WebRequest `
            -UseBasicParsing `
            -Uri $mlflowHealthUri `
            -TimeoutSec $requestTimeout
        if ($response.StatusCode -eq 200) {
            $mlflowReady = $true
            break
        }
    }
    catch {
        # The bounded loop retries until the local service becomes healthy.
    }
    $remainingAfterRequest = (
        $MlflowHealthTimeoutSeconds - $stopwatch.Elapsed.TotalSeconds
    )
    if ($remainingAfterRequest -le 0) {
        break
    }
    $sleepMilliseconds = [Math]::Min(
        2000,
        [Math]::Floor($remainingAfterRequest * 1000)
    )
    if ($sleepMilliseconds -gt 0) {
        Start-Sleep -Milliseconds $sleepMilliseconds
    }
}
if (-not $mlflowReady) {
    throw "Local MLflow did not become healthy within the configured timeout."
}

# These assignments affect only this runner and the Uvicorn child process.
$env:WIND_FORECAST_DEPLOYMENT_ROOT = $deployment
$env:WIND_FORECAST_MONITORING_STORE_ROOT = $monitoring
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:WIND_FORECAST_OPERATIONAL_PROJECTION_MODE = "disabled"

$logDirectory = Join-Path $repository "var\local_services"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
$logFile = Join-Path $logDirectory "operational-api-$stamp-$PID.log"

Push-Location $repository
$exitCode = 1
try {
    & $python -m uvicorn wind_forecast.api:app `
        --host "127.0.0.1" `
        --port "8000" *>> $logFile
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $exitCode
