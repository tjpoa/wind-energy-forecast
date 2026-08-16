[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExecutable,
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,
    [string]$TaskName = "WindForecastMlflow"
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$runner = Join-Path $repository "scripts\run_local_mlflow.ps1"
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Local MLflow runner was not found: $runner"
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
    "-RepositoryRoot", (Quote-TaskArgument $repository)
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
    -Description "Loopback-only local MLflow service in a non-interactive S4U session."

if ($PSCmdlet.ShouldProcess($TaskName, "Register or replace scheduled task")) {
    Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
    Get-ScheduledTask -TaskName $TaskName
}
