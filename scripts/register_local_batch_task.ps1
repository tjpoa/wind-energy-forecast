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
$environmentFile = if ($EnvFile) { Resolve-RepositoryPath $EnvFile } else { $null }

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
    "-DeploymentRoot", (Quote-TaskArgument $deployment)
)
if ($ActivationDate) {
    $actionArguments += @("-ActivationDate", (Quote-TaskArgument $ActivationDate))
}
if ($environmentFile) {
    $actionArguments += @("-EnvFile", (Quote-TaskArgument $environmentFile))
}

$actionArgumentText = $actionArguments -join " "
if ($WhatIfPreference) {
    [pscustomobject]@{
        TaskName = $TaskName
        Executable = "powershell.exe"
        Arguments = $actionArgumentText
        Schedule = "Daily at 12:00 local time"
        ExecutionTimeLimit = "06:00:00"
        RestartCount = 2
        RestartInterval = "00:30:00"
        MultipleInstances = "IgnoreNew"
        LogonType = "Interactive"
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
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Daily D+5 historical wind-forecast batch at 12:00 local time."

if ($PSCmdlet.ShouldProcess($TaskName, "Register or replace scheduled task")) {
    Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
    Get-ScheduledTask -TaskName $TaskName
}
