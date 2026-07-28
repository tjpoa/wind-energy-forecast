[CmdletBinding()]
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
    [string]$PolicyPath = "config\retraining_policy_v1.json",
    [string]$MonitoringPolicyPath = "config\monitoring_policy_v1.json",
    [string]$OutputRoot = "data\processed\v2\retraining\monthly_recommendations",
    [string]$EvaluationOutputRoot = "data\processed\v2\retraining\evaluations"
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$python = (Resolve-Path -LiteralPath $PythonExecutable).Path
$governanceScript = Join-Path $repository "scripts\run_monthly_governance.py"
$schedulerScript = Join-Path $repository "scripts\manage_scheduler_owner.py"
foreach ($requiredScript in @($governanceScript, $schedulerScript)) {
    if (-not (Test-Path -LiteralPath $requiredScript -PathType Leaf)) {
        throw "Scheduled CLI wrapper was not found: $requiredScript"
    }
}
if ([System.TimeZoneInfo]::Local.Id -ne "GMT Standard Time") {
    throw "The local timezone must be GMT Standard Time (Europe/Lisbon contract)."
}

$evaluationPeriod = [DateTimeOffset]::Now.ToString("yyyy-MM")
$runId = "windows-monthly-" + [guid]::NewGuid().ToString("N")
$leaseId = $null
$exitCode = 1
Push-Location $repository
try {
    $leaseJson = & $python $schedulerScript acquire `
        --scheduler-root $SchedulerStateRoot `
        --environment-id $EnvironmentId `
        --scheduler windows_task_scheduler `
        --workflow monthly_governance `
        --run-id $runId
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to acquire the scheduler execution lease."
    }
    $lease = $leaseJson | ConvertFrom-Json
    $leaseId = $lease.lease_id
    & $python $governanceScript `
        --policy-path $PolicyPath `
        --monitoring-policy-path $MonitoringPolicyPath `
        --monitoring-store-root $MonitoringStoreRoot `
        --deployment-root $DeploymentRoot `
        --output-root $OutputRoot `
        --evaluation-output-root $EvaluationOutputRoot `
        --evaluation-period $evaluationPeriod
    $exitCode = $LASTEXITCODE
}
finally {
    if ($leaseId) {
        & $python $schedulerScript release `
            --scheduler-root $SchedulerStateRoot `
            --environment-id $EnvironmentId `
            --lease-id $leaseId
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to release the scheduler execution lease."
        }
    }
    Pop-Location
}
exit $exitCode
