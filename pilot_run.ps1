<#
.SYNOPSIS
    Run one bounded AutoFE-ShiftBench pilot in Conda P12.

.DESCRIPTION
    Installs/verifies requirements, acquires one selected dataset with the
    100,000-row cap, and runs one seed/fold/condition with two pipelines and
    one estimator. This is a smoke/pilot run, not the complete benchmark.
#>

[CmdletBinding()]
param(
    [string]$CondaEnv = "",
    [string]$PythonExe = "python",
    [string]$Dataset = "adult",
    [string]$Pipelines = "Raw,AutoFE_Baseline",
    [string]$Models = "logistic_regression",
    [int]$MaxWorkers = 1,
    [int]$MaxAttempts = 1,
    [int]$WorkerMaxTasks = 16,
    [string]$RunId = "pilot_default",
    [string]$RunRoot = "",
    [switch]$Resume,
    [switch]$SkipDownload,
    [switch]$DryRun,
    [switch]$PlanOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -LiteralPath $projectRoot
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path "reports/runs/pilot" $RunId
}
if ([string]::IsNullOrWhiteSpace($RunId) -or $RunId -in @('.', '..') -or $RunId -match '[\\/]') {
    throw "RunId must be a single stable directory name, for example pilot_001."
}

function Resolve-CondaCommand {
    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -eq $command) { throw "Conda was not found on PATH." }
    if ($command.Path) { return $command.Path }
    return $command.Name
}

function Assert-ExitCode([string]$stage) {
    if ($LASTEXITCODE -ne 0) { throw "$stage failed with exit code $LASTEXITCODE." }
}

$conda = $null
if (-not [string]::IsNullOrWhiteSpace($CondaEnv)) { $conda = Resolve-CondaCommand }
function Invoke-Python([string[]]$Arguments) {
    if ($null -ne $conda) {
        & $conda @("run", "--no-capture-output", "-n", $CondaEnv, "python") @Arguments
    } else {
        & $PythonExe @Arguments
    }
}
$requirementsPath = Join-Path $projectRoot "requirements.txt"
$rawDir = Join-Path $projectRoot "data/raw"
$runDir = Join-Path $projectRoot $RunRoot
$logDir = Join-Path $runDir "logs"
$workerLogDir = Join-Path $logDir "workers"
$metadataDir = Join-Path $runDir "metadata"
$csvPath = Join-Path $rawDir "$Dataset.csv"
$metaPath = Join-Path $rawDir "${Dataset}_meta.json"

Write-Host "AutoFE-ShiftBench bounded pilot" -ForegroundColor Cyan
Write-Host ("Environment: {0}" -f ($(if($CondaEnv){"Conda $CondaEnv"}else{"Python $PythonExe"})))
Write-Host "Dataset: $Dataset (100,000-row cap)"
Write-Host "Pipelines: $Pipelines"
Write-Host "Models: $Models"
Write-Host "Run ID: $RunId"

if ($PlanOnly) {
    Write-Host "Plan only: requirements, acquisition and benchmark execution were not run." -ForegroundColor Yellow
    exit 0
}

Write-Host "Installing/verifying requirements..." -ForegroundColor Yellow
Invoke-Python @("-m", "pip", "install", "--disable-pip-version-check", "-r", $requirementsPath)
Assert-ExitCode "Requirement installation"

New-Item -ItemType Directory -Force -Path $rawDir,$runDir,$logDir,$workerLogDir,$metadataDir | Out-Null
$env:AUTOFE_WORKER_LOG_DIR = $workerLogDir
$launcherMetadata = [ordered]@{
    run_id = $RunId
    run_kind = "pilot"
    run_root = (Resolve-Path -LiteralPath $runDir).Path
    resume_requested = [bool]$Resume
    conda_environment = if($CondaEnv){$CondaEnv}else{$null}
    python_executable = $PythonExe
    dataset = $Dataset
    row_cap = 100000
    invocation_utc = (Get-Date).ToUniversalTime().ToString("o")
}
$metadataName = if (Test-Path -LiteralPath (Join-Path $metadataDir "launcher_config.json")) {
    "launcher_attempt_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss_fff")
} else { "launcher_config.json" }
$launcherMetadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $metadataDir $metadataName) -Encoding UTF8
if (-not $SkipDownload -and (-not (Test-Path $csvPath) -or -not (Test-Path $metaPath))) {
    Write-Host "Acquiring pilot dataset..." -ForegroundColor Yellow
    Invoke-Python @("-B", "-m", "src.data_loader", "--datasets", $Dataset, "--output-dir", "data/raw", "--max-rows", "100000")
    Assert-ExitCode "Pilot dataset acquisition"
}
if (-not (Test-Path $csvPath) -or -not (Test-Path $metaPath)) {
    throw "Dataset and metadata are required: $csvPath and $metaPath"
}

$args = @(
    "-B", "-m", "src.pipeline_runner",
    "--datasets", $Dataset,
    "--max-datasets", "1",
    "--max-seeds", "1",
    "--max-folds", "1",
    "--max-conditions", "1",
    "--pipelines", $Pipelines,
    "--models", $Models,
    "--max-workers", "$MaxWorkers",
    "--max-attempts", "$MaxAttempts",
    "--worker-max-tasks", "$WorkerMaxTasks",
    "--cache-policy", "rolling",
    "--cache-max-gib", "10",
    "--min-free-gib", "20",
    "--dense-budget-gib", "2",
    "--run-id", $RunId,
    "--manifest-db", (Join-Path $RunRoot "pilot.db"),
    "--manifest-path", (Join-Path $RunRoot "pilot_manifest.jsonl"),
    "--results-path", (Join-Path $RunRoot "pilot_results.jsonl"),
    "--log-file", (Join-Path $RunRoot "logs/pilot.log"),
    "--worker-log-dir", (Join-Path $RunRoot "logs/workers")
)
if ($DryRun) { $args += "--dry-run-manifest" }

Write-Host "Starting bounded pilot..." -ForegroundColor Green
Invoke-Python $args
Assert-ExitCode "Pilot benchmark"
Write-Host "Pilot completed successfully." -ForegroundColor Green
