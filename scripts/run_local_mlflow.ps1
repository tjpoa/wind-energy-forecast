[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExecutable,
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
$runnerName = "mlflow"
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
    $backendDatabase = Join-Path $repository "var\mlflow\mlflow.db"
    $artifactsDirectory = Join-Path $repository "var\mlflow\artifacts"
    foreach ($requiredPath in @($backendDatabase, $artifactsDirectory)) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Required local MLflow state was not found."
        }
    }
    Write-RunnerEvent -Stage $stage -Status "succeeded"

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
        & $python -m mlflow server `
            --backend-store-uri "sqlite:///var/mlflow/mlflow.db" `
            --artifacts-destination "./var/mlflow/artifacts" `
            --host "127.0.0.1" `
            --port "5000" *>> $outputFile
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
