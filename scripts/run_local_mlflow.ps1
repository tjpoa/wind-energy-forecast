[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExecutable,
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot
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
$backendDatabase = Join-Path $repository "var\mlflow\mlflow.db"
$artifactsDirectory = Join-Path $repository "var\mlflow\artifacts"
foreach ($requiredPath in @($backendDatabase, $artifactsDirectory)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required local MLflow state was not found: $requiredPath"
    }
}

$logDirectory = Join-Path $repository "var\local_services"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
$logFile = Join-Path $logDirectory "mlflow-$stamp-$PID.log"

Push-Location $repository
$exitCode = 1
try {
    & $python -m mlflow server `
        --backend-store-uri "sqlite:///var/mlflow/mlflow.db" `
        --artifacts-destination "./var/mlflow/artifacts" `
        --host "127.0.0.1" `
        --port "5000" *>> $logFile
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $exitCode
