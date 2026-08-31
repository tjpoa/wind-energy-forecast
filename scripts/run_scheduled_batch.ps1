[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExecutable,
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,
    [Parameter(Mandatory = $true)]
    [string]$ModelBundle,
    [Parameter(Mandatory = $true)]
    [string]$CalibrationDirectory,
    [Parameter(Mandatory = $true)]
    [string]$DeploymentRoot,
    [Parameter(Mandatory = $true)]
    [string]$SchedulerStateRoot,
    [Parameter(Mandatory = $true)]
    [string]$EnvironmentId,
    [string]$ActivationDate,
    [string]$EnvFile,
    [Parameter(Mandatory = $true)]
    [string]$ReadinessPath
)

$ErrorActionPreference = "Stop"
$runnerName = "scheduled-batch"
$runId = "windows-daily-" + [guid]::NewGuid().ToString("N")
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

function Invoke-LoggedNativeCommand {
    param(
        [Parameter(Mandatory = $true)][string[]]$CommandArguments,
        [Parameter(Mandatory = $true)][bool]$EmitStdout
    )
    $stdoutLines = New-Object System.Collections.Generic.List[string]
    $nativeErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $python @CommandArguments 2>&1 | ForEach-Object {
            $record = $_
            $line = [string]$record
            [System.IO.File]::AppendAllText(
                $outputFile,
                $line + [Environment]::NewLine,
                $utf8NoBom
            )
            if ($record -is [System.Management.Automation.ErrorRecord]) {
                [Console]::Error.WriteLine($line)
            }
            else {
                $stdoutLines.Add($line) | Out-Null
                if ($EmitStdout) {
                    [Console]::Out.WriteLine($line)
                }
            }
        }
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $nativeErrorActionPreference
    }
    return [pscustomobject]@{
        ExitCode = $nativeExitCode
        Stdout = $stdoutLines.ToArray()
    }
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

$repository = $null
$python = $null
$schedulerScript = $null
$leaseId = $null
$exitCode = 1
$childExitCode = $null
$locationPushed = $false
$stage = "setup"
try {
    Write-RunnerEvent -Stage $stage -Status "started"
    $repository = (Resolve-Path -LiteralPath $RepositoryRoot).Path
    $python = (Resolve-Path -LiteralPath $PythonExecutable).Path
    $batchScript = Join-Path $repository "scripts\run_batch_pipeline.py"
    $schedulerScript = Join-Path $repository "scripts\manage_scheduler_owner.py"

    foreach ($requiredScript in @($batchScript, $schedulerScript)) {
        if (-not (Test-Path -LiteralPath $requiredScript -PathType Leaf)) {
            throw "Scheduled CLI wrapper was not found."
        }
    }

    $readinessScript = Join-Path $repository "scripts\verify_local_automation_readiness.py"
    if (-not (Test-Path -LiteralPath $readinessScript -PathType Leaf)) {
        throw "Automation readiness verifier was not found."
    }
    $readiness = if ([System.IO.Path]::IsPathRooted($ReadinessPath)) {
        $ReadinessPath
    } else {
        Join-Path $repository $ReadinessPath
    }
    if (-not (Test-Path -LiteralPath $readiness -PathType Leaf)) {
        throw "Automation readiness receipt was not found."
    }
    $readinessOutput = @(
        & $python $readinessScript `
            --path $readiness `
            --environment-id $EnvironmentId `
            --workflow historical_daily_batch 2>&1
    )
    $readinessExitCode = $LASTEXITCODE
    if ($readinessExitCode -ne 0) {
        foreach ($record in $readinessOutput) {
            [Console]::Error.WriteLine([string]$record)
        }
        throw "Automation readiness does not permit historical_daily_batch."
    }

    $arguments = @(
        $batchScript,
        "run",
        "--model-bundle", $ModelBundle,
        "--calibration-dir", $CalibrationDirectory,
        "--deployment-root", $DeploymentRoot
    )
    if ($ActivationDate) {
        $arguments += @("--activation-date", $ActivationDate)
    }
    if ($EnvFile) {
        $arguments += @("--env-file", $EnvFile)
    }
    Write-RunnerEvent -Stage $stage -Status "succeeded"

    Push-Location $repository
    $locationPushed = $true
    $stage = "lease_acquire"
    Write-RunnerEvent -Stage $stage -Status "started"
    $acquireArguments = @(
        $schedulerScript,
        "acquire",
        "--scheduler-root", $SchedulerStateRoot,
        "--environment-id", $EnvironmentId,
        "--scheduler", "windows_task_scheduler",
        "--workflow", "historical_daily_batch",
        "--run-id", $runId
    )
    $nativeErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $leaseRecords = @(& $python @acquireArguments 2>&1)
        $acquireExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $nativeErrorActionPreference
    }
    $leaseStdout = @(
        $leaseRecords |
            Where-Object {
                -not ($_ -is [System.Management.Automation.ErrorRecord])
            } |
            ForEach-Object { [string]$_ }
    )
    if ($acquireExitCode -ne 0) {
        foreach ($record in $leaseRecords) {
            if ($record -is [System.Management.Automation.ErrorRecord]) {
                [Console]::Error.WriteLine([string]$record)
            }
        }
        throw "Scheduler execution lease acquisition failed."
    }
    $leaseJson = $leaseStdout -join [Environment]::NewLine
    $lease = $leaseJson | ConvertFrom-Json
    $leaseId = $lease.lease_id
    if (-not $leaseId) {
        throw "Scheduler execution lease response was invalid."
    }
    Write-RunnerEvent -Stage $stage -Status "succeeded"

    $stage = "child"
    Write-RunnerEvent -Stage $stage -Status "started"
    $childResult = Invoke-LoggedNativeCommand `
        -CommandArguments $arguments `
        -EmitStdout $true
    $childExitCode = $childResult.ExitCode
    $exitCode = $childExitCode
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
    if ($leaseId) {
        $stage = "lease_release"
        try {
            $releaseArguments = @(
                $schedulerScript,
                "release",
                "--scheduler-root", $SchedulerStateRoot,
                "--environment-id", $EnvironmentId,
                "--lease-id", $leaseId
            )
            $nativeErrorActionPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = "Continue"
                $releaseRecords = @(& $python @releaseArguments 2>&1)
                $releaseExitCode = $LASTEXITCODE
            }
            finally {
                $ErrorActionPreference = $nativeErrorActionPreference
            }
            foreach ($record in $releaseRecords) {
                if ($record -is [System.Management.Automation.ErrorRecord]) {
                    [Console]::Error.WriteLine([string]$record)
                }
            }
            if ($releaseExitCode -ne 0) {
                throw "Scheduler execution lease release failed."
            }
            Write-RunnerEvent -Stage $stage -Status "succeeded"
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
    }
    if ($locationPushed) {
        Pop-Location
    }
}

$runnerStatus = if ($exitCode -eq 0) { "succeeded" } else { "failed" }
try {
    Write-RunnerEvent `
        -Stage "runner_exit" `
        -Status $runnerStatus `
        -ChildExitCode $childExitCode
}
catch {
    Write-Error "Runner observability finalization failed."
    exit 1
}
exit $exitCode
