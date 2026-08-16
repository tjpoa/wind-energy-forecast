[CmdletBinding(SupportsShouldProcess = $true)]
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
    [int]$MlflowHealthTimeoutSeconds = 120,
    [string]$TaskName = "WindForecastOperationalApi"
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$runner = Join-Path $repository "scripts\run_local_operational_api.ps1"
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Local operational API runner was not found: $runner"
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
$deployment = Resolve-RepositoryPath $DeploymentRoot
$monitoring = Resolve-RepositoryPath $MonitoringStoreRoot
$model = Resolve-RepositoryPath $ModelBundle
$calibration = Resolve-RepositoryPath $CalibrationDirectory
$powershell = Join-Path $PSHOME "powershell.exe"
if (-not (Test-Path -LiteralPath $powershell -PathType Leaf)) {
    throw "Windows PowerShell executable was not found: $powershell"
}
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$actionArguments = @(
    "-NoProfile",
    "-NonInteractive",
    "-ExecutionPolicy", "Bypass",
    "-File", (Quote-TaskArgument $runner),
    "-PythonExecutable", (Quote-TaskArgument $python),
    "-RepositoryRoot", (Quote-TaskArgument $repository),
    "-DeploymentRoot", (Quote-TaskArgument $deployment),
    "-MonitoringStoreRoot", (Quote-TaskArgument $monitoring),
    "-ModelBundle", (Quote-TaskArgument $model),
    "-CalibrationDirectory", (Quote-TaskArgument $calibration),
    "-MlflowHealthTimeoutSeconds", $MlflowHealthTimeoutSeconds.ToString()
)
$actionArgumentText = $actionArguments -join " "

if ($WhatIfPreference) {
    [pscustomobject]@{
        TaskName = $TaskName
        Executable = $powershell
        Arguments = $actionArgumentText
        Trigger = "At logon for $currentUser"
        RunLevel = "Limited"
        LogonType = "S4U"
        ExecutionTimeLimit = "PT0S"
        RestartCount = 3
        RestartInterval = "00:01:00"
        MultipleInstances = "IgnoreNew"
        StartWhenAvailable = $true
        StartsTask = $false
    }
    return
}

$action = New-ScheduledTaskAction `
    -Execute $powershell `
    -Argument $actionArgumentText `
    -WorkingDirectory $repository
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
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
    -Description "Loopback-only read-only API in a non-interactive S4U session."

if ($PSCmdlet.ShouldProcess($TaskName, "Register or replace scheduled task")) {
    Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
    Get-ScheduledTask -TaskName $TaskName
}
