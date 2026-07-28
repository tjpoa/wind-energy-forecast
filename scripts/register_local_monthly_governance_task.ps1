[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExecutable,
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,
    [Parameter(Mandatory = $true)]
    [string]$MonitoringStoreRoot,
    [Parameter(Mandatory = $true)]
    [string]$DeploymentRoot,
    [Parameter(Mandatory = $true)]
    [string]$SchedulerStateRoot,
    [Parameter(Mandatory = $true)]
    [string]$EnvironmentId,
    [string]$TaskName = "WindForecastMonthlyGovernance"
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$runner = Join-Path $repository "scripts\run_scheduled_monthly_governance.ps1"
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Scheduled monthly runner was not found: $runner"
}
if ([System.TimeZoneInfo]::Local.Id -ne "GMT Standard Time") {
    throw "The local timezone must be GMT Standard Time (Europe/Lisbon contract)."
}

function Resolve-RepositoryPath([string]$Value) {
    $candidate = $Value
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $repository $candidate
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}
function Quote-TaskArgument([string]$Value) {
    if ($Value.Contains('"')) {
        throw "Task arguments must not contain quote characters."
    }
    return '"' + $Value + '"'
}

$python = Resolve-RepositoryPath $PythonExecutable
$monitoring = Resolve-RepositoryPath $MonitoringStoreRoot
$deployment = Resolve-RepositoryPath $DeploymentRoot
$schedulerState = Resolve-RepositoryPath $SchedulerStateRoot
$schedulerManager = Join-Path $repository "scripts\manage_scheduler_owner.py"
& $python $schedulerManager verify `
    --scheduler-root $schedulerState `
    --environment-id $EnvironmentId `
    --scheduler windows_task_scheduler | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Scheduler ownership is not configured for Windows Task Scheduler."
}
$arguments = @(
    "-NoProfile",
    "-NonInteractive",
    "-ExecutionPolicy", "Bypass",
    "-File", (Quote-TaskArgument $runner),
    "-PythonExecutable", (Quote-TaskArgument $python),
    "-RepositoryRoot", (Quote-TaskArgument $repository),
    "-MonitoringStoreRoot", (Quote-TaskArgument $monitoring),
    "-DeploymentRoot", (Quote-TaskArgument $deployment),
    "-SchedulerStateRoot", (Quote-TaskArgument $schedulerState),
    "-EnvironmentId", (Quote-TaskArgument $EnvironmentId)
)
$argumentText = $arguments -join " "
if ($WhatIfPreference) {
    [pscustomobject]@{
        TaskName = $TaskName
        Executable = "powershell.exe"
        Arguments = $argumentText
        Schedule = "Monthly on day 8 at 13:00 local time"
        MultipleInstances = "IgnoreNew"
        StartWhenAvailable = $true
        SchedulerStateRoot = $schedulerState
        EnvironmentId = $EnvironmentId
    }
    return
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argumentText
$startBoundary = (Get-Date -Hour 13 -Minute 0 -Second 0).ToString("s")
$trigger = New-CimInstance `
    -ClassName "MSFT_TaskMonthlyTrigger" `
    -Namespace "Root/Microsoft/Windows/TaskScheduler" `
    -ClientOnly `
    -Property @{
        Enabled = $true
        DaysOfMonth = [uint32]128
        MonthsOfYear = [uint16]4095
        StartBoundary = $startBoundary
    }
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Owner-guarded monthly retraining and stability recommendations only."
if ($PSCmdlet.ShouldProcess($TaskName, "Register or replace scheduled task")) {
    Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
    Get-ScheduledTask -TaskName $TaskName
}
