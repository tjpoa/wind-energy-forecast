[CmdletBinding(SupportsShouldProcess = $true)]
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
    [string]$TaskName = "WindForecastHistoricalBatch"
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$runner = Join-Path $repository "scripts\run_scheduled_batch.ps1"
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Scheduled batch runner was not found: $runner"
}

function Resolve-RepositoryPath([string]$Value) {
    $candidate = $Value
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $repository $candidate
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

$python = Resolve-RepositoryPath $PythonExecutable
$model = Resolve-RepositoryPath $ModelBundle
$calibration = Resolve-RepositoryPath $CalibrationDirectory
$deployment = Resolve-RepositoryPath $DeploymentRoot
$schedulerState = Resolve-RepositoryPath $SchedulerStateRoot
$environmentFile = if ($EnvFile) { Resolve-RepositoryPath $EnvFile } else { $null }
$schedulerManager = Join-Path $repository "scripts\manage_scheduler_owner.py"
& $python $schedulerManager verify `
    --scheduler-root $schedulerState `
    --environment-id $EnvironmentId `
    --scheduler windows_task_scheduler | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Scheduler ownership is not configured for Windows Task Scheduler."
}

$timezone = [System.TimeZoneInfo]::Local.Id
if ($timezone -ne "GMT Standard Time") {
    throw "The local timezone must be GMT Standard Time (Europe/Lisbon contract); found $timezone."
}

function Quote-TaskArgument([string]$Value) {
    if ($Value.Contains('"')) {
        throw "Task arguments must not contain quote characters."
    }
    return '"' + $Value + '"'
}

$actionArguments = @(
    "-NoProfile",
    "-NonInteractive",
    "-ExecutionPolicy", "Bypass",
    "-File", (Quote-TaskArgument $runner),
    "-PythonExecutable", (Quote-TaskArgument $python),
    "-RepositoryRoot", (Quote-TaskArgument $repository),
    "-ModelBundle", (Quote-TaskArgument $model),
    "-CalibrationDirectory", (Quote-TaskArgument $calibration),
    "-DeploymentRoot", (Quote-TaskArgument $deployment),
    "-SchedulerStateRoot", (Quote-TaskArgument $schedulerState),
    "-EnvironmentId", (Quote-TaskArgument $EnvironmentId)
)
if ($ActivationDate) {
    $actionArguments += @("-ActivationDate", (Quote-TaskArgument $ActivationDate))
}
if ($environmentFile) {
    $actionArguments += @("-EnvFile", (Quote-TaskArgument $environmentFile))
}

$actionArgumentText = $actionArguments -join " "
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
if ($WhatIfPreference) {
    [pscustomobject]@{
        TaskName = $TaskName
        Executable = "powershell.exe"
        Arguments = $actionArgumentText
        UserId = $currentUser
        RunLevel = "Limited"
        Schedule = "Daily at 12:00 local time"
        ExecutionTimeLimit = "06:00:00"
        RestartCount = 2
        RestartInterval = "00:30:00"
        MultipleInstances = "IgnoreNew"
        LogonType = "S4U"
        SchedulerStateRoot = $schedulerState
        EnvironmentId = $EnvironmentId
        StartsTask = $false
    }
    return
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $actionArgumentText
$trigger = New-ScheduledTaskTrigger -Daily -At "12:00"
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType S4U `
    -RunLevel Limited
$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Owner-guarded daily D+5 historical wind-forecast batch in a non-interactive S4U session at 12:00 local time."

if ($PSCmdlet.ShouldProcess($TaskName, "Register or replace scheduled task")) {
    Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
    Get-ScheduledTask -TaskName $TaskName
}
