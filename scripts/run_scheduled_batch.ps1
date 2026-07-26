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
    [string]$ActivationDate,
    [string]$EnvFile
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$python = (Resolve-Path -LiteralPath $PythonExecutable).Path
$batchScript = Join-Path $repository "scripts\run_batch_pipeline.py"

if (-not (Test-Path -LiteralPath $batchScript -PathType Leaf)) {
    throw "Batch CLI wrapper was not found: $batchScript"
}

$arguments = @(
    $batchScript,
    "run",
    "--model-bundle", $ModelBundle,
    "--calibration-dir", $CalibrationDirectory
)
if ($ActivationDate) {
    $arguments += @("--activation-date", $ActivationDate)
}
if ($EnvFile) {
    $arguments += @("--env-file", $EnvFile)
}

Push-Location $repository
try {
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
