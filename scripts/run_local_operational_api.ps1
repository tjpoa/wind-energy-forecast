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
    [Parameter(Mandatory = $true)]
    [string]$ModelBundle,
    [Parameter(Mandatory = $true)]
    [string]$CalibrationDirectory,
    [ValidateRange(1, 600)]
    [int]$MlflowHealthTimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
$runnerName = "operational-api"
$runId = $runnerName + "-" + [guid]::NewGuid().ToString("N")
$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$logDirectory = Join-Path $RepositoryRoot "var\local_services"
$eventsFile = Join-Path $logDirectory "$runnerName-$stamp-$PID.events.jsonl"
$outputFile = Join-Path $logDirectory "$runnerName-$stamp-$PID.output.log"

function Write-RunnerEvent {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string]$Status,
        [Nullable[int]]$ChildExitCode = $null,
        [string]$ExceptionType = $null,
        [string]$ExceptionMessage = $null
    )
    $event = [ordered]@{
        schema_version = "wind_forecast.runner_event.v1"
        timestamp_utc = [DateTime]::UtcNow.ToString("o")
        runner = $runnerName
        run_id = $runId
        stage = $Stage
        status = $Status
        runner_pid = [int]$PID
        child_exit_code = $ChildExitCode
        exception_type = $ExceptionType
        exception_message = $ExceptionMessage
    }
    $json = $event | ConvertTo-Json -Compress
    [System.IO.File]::AppendAllText(
        $eventsFile,
        $json + [Environment]::NewLine,
        $utf8NoBom
    )
}

try {
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    [System.IO.File]::WriteAllText($eventsFile, "", $utf8NoBom)
    [System.IO.File]::WriteAllText($outputFile, "", $utf8NoBom)
    Write-RunnerEvent -Stage "observability" -Status "initialized"
}
catch {
    Write-Error "Runner observability initialization failed."
    exit 1
}

$exitCode = 1
$childExitCode = $null
$locationPushed = $false
$stage = "setup"
try {
    Write-RunnerEvent -Stage $stage -Status "started"
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
    $model = Resolve-RepositoryPath $ModelBundle
    $calibration = Resolve-RepositoryPath $CalibrationDirectory
    $apiModule = Join-Path $repository "src\wind_forecast\api.py"
    if (-not (Test-Path -LiteralPath $apiModule -PathType Leaf)) {
        throw "Operational API module was not found."
    }
    Write-RunnerEvent -Stage $stage -Status "succeeded"

    $stage = "health_wait"
    Write-RunnerEvent -Stage $stage -Status "started"
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
        throw "Local MLflow health wait timed out."
    }
    Write-RunnerEvent -Stage $stage -Status "succeeded"

    # These assignments affect only this runner and the Uvicorn child process.
    $env:WIND_FORECAST_DEPLOYMENT_ROOT = $deployment
    $env:WIND_FORECAST_MONITORING_STORE_ROOT = $monitoring
    $env:WIND_FORECAST_OPERATIONAL_MODEL_BUNDLE = $model
    $env:WIND_FORECAST_OPERATIONAL_CALIBRATION_DIR = $calibration
    $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    $env:WIND_FORECAST_OPERATIONAL_PROJECTION_MODE = "disabled"

    Push-Location $repository
    $locationPushed = $true
    $stage = "child"
    Write-RunnerEvent -Stage $stage -Status "started"
    $nativeErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 promotes native stderr to an ErrorRecord. Keep
        # setup fail-closed, but do not terminate a healthy long-running service
        # merely because it writes normal diagnostics to stderr.
        $ErrorActionPreference = "Continue"
        & $python -m uvicorn wind_forecast.api:app `
            --host "127.0.0.1" `
            --port "8000" *>> $outputFile
        $childExitCode = $LASTEXITCODE
        $exitCode = $childExitCode
    }
    finally {
        $ErrorActionPreference = $nativeErrorActionPreference
    }
    $childStatus = if ($exitCode -eq 0) { "succeeded" } else { "failed" }
    Write-RunnerEvent `
        -Stage $stage `
        -Status $childStatus `
        -ChildExitCode $childExitCode
}
catch {
    $exceptionType = $_.Exception.GetType().FullName
    Write-RunnerEvent `
        -Stage $stage `
        -Status "failed" `
        -ExceptionType $exceptionType `
        -ExceptionMessage "Runner failed during $stage."
    $exitCode = 1
}
finally {
    if ($locationPushed) {
        Pop-Location
    }
}

$runnerStatus = if ($exitCode -eq 0) { "succeeded" } else { "failed" }
Write-RunnerEvent `
    -Stage "runner_exit" `
    -Status $runnerStatus `
    -ChildExitCode $childExitCode
exit $exitCode
