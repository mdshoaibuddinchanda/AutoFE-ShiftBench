<#
.SYNOPSIS
    Acquire the bounded benchmark data and run the complete AutoFE-ShiftBench grid.

.DESCRIPTION
    This is the single Windows launcher for the project. It uses Conda P12,
    acquires only the declared datasets with the 100,000-row cap, validates the
    pinned eligibility bundle, and then resumes or creates one durable run.

    Defaults are conservative and hardware-aware: native numerical libraries
    use one thread, CPU workers scale with logical CPUs and RAM, dense memory is
    bounded, and rolling cache cleanup preserves the free-disk reserve. Use the
    parameters below to override limits on a larger or smaller machine.

    GPU models are included when an NVIDIA GPU is detected. On a machine without
    NVIDIA support, pass -CpuOnly explicitly; that run is not equivalent to the
    full ten-estimator benchmark because XGBoost and CatBoost are omitted.
#>

[CmdletBinding()]
param(
    [string]$CondaEnv = "",
    [string]$PythonExe = "python",
    [int]$MaxWorkers = 0,
    [int]$WorkerMaxTasks = 64,
    [int]$MaxAttempts = 3,
    [double]$DenseBudgetGiB = 0,
    [double]$CacheMaxGiB = 0,
    [double]$MinFreeGiB = 0,
    [string]$RunId = "full_default",
    [string]$RunRoot = "",
    [switch]$Resume,
    [switch]$SkipDownload,
    [switch]$CpuOnly,
    [switch]$DryRun,
    [switch]$PlanOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -LiteralPath $projectRoot

if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path "reports/runs/full" $RunId
}
if ([string]::IsNullOrWhiteSpace($RunId) -or $RunId -in @('.', '..') -or $RunId -match '[\\/]') {
    throw "RunId must be a single stable directory name, for example experiment_001."
}

$datasets = @(
    "haberman",
    "sonar",
    "ionosphere",
    "heart-disease",
    "breast-cancer-wisconsin",
    "blood-transfusion-service-center",
    "diabetes",
    "titanic",
    "credit-g",
    "wine-quality-red",
    "kr-vs-kp",
    "mushroom",
    "spambase",
    "jm1",
    "PhishingWebsites",
    "default-of-credit-card-clients",
    "magic-telescope",
    "dry-bean-dataset",
    "adult",
    "bank-marketing",
    "electricity",
    "aps_failure",
    "covertype",
    "airlines",
    "kddcup99"
)

function Resolve-CondaCommand {
    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Conda was not found on PATH. Initialize Conda in this PowerShell session, then rerun this launcher."
    }
    if ($command.Path) { return $command.Path }
    return $command.Name
}

function Get-DriveFreeGiB {
    $driveName = (Get-Location).Drive.Name
    $drive = Get-PSDrive -Name $driveName
    if ($null -eq $drive.Free) { throw "Cannot determine free space on drive $driveName`:." }
    return [math]::Floor([double]$drive.Free / 1GB)
}

function Get-HardwareDefaults {
    $logical = [math]::Max(1, [Environment]::ProcessorCount)
    $computer = Get-CimInstance -ClassName Win32_ComputerSystem
    $ramGiB = [math]::Round([double]$computer.TotalPhysicalMemory / 1GB, 1)
    $freeGiB = Get-DriveFreeGiB

    # Leave two logical CPUs for Windows/browser work and reserve roughly 4 GiB
    # per active worker. The cap avoids turning a large workstation into a
    # memory-bandwidth bottleneck.
    $ramWorkers = [int][math]::Floor($ramGiB / 4)
    $cpuWorkers = [math]::Max(1, $logical - 2)
    $workers = [int][math]::Max(1, [math]::Min(12, [math]::Min($cpuWorkers, $ramWorkers)))

    $dense = [int][math]::Max(2, [math]::Min(24, [math]::Floor($ramGiB * 0.45)))
    $minFree = [int][math]::Max(10, [math]::Min(50, [math]::Floor($freeGiB * 0.15)))
    $cache = [int][math]::Max(20, [math]::Min(100, [math]::Floor($freeGiB - $minFree - 20)))

    return [pscustomobject]@{
        LogicalProcessors = $logical
        RamGiB = $ramGiB
        FreeGiB = $freeGiB
        Workers = $workers
        DenseBudgetGiB = $dense
        CacheMaxGiB = $cache
        MinFreeGiB = $minFree
    }
}

function Get-NvidiaInfo {
    $command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $command) { return $null }
    try {
        $output = & $command.Source --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $output) { return (($output -join "; ").Trim()) }
    } catch { }
    return $null
}

function Assert-ExitCode([string]$stage) {
    if ($LASTEXITCODE -ne 0) {
        throw "$stage failed with exit code $LASTEXITCODE."
    }
}

function Install-Requirements {
    $requirementsPath = Join-Path $projectRoot "requirements.txt"
    if (-not (Test-Path -LiteralPath $requirementsPath)) {
        throw "requirements.txt was not found: $requirementsPath"
    }
    Write-Host "Installing/verifying Python requirements..." -ForegroundColor Yellow
    $installArgs = @(
        "-m", "pip", "install", "--disable-pip-version-check",
        "-r", $requirementsPath
    )
    Invoke-Python $installArgs
    Assert-ExitCode "Python dependency installation"
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

if (-not $PlanOnly) {
    Install-Requirements
}
$hardware = Get-HardwareDefaults
$gpuInfo = Get-NvidiaInfo

if ($MaxWorkers -le 0) { $MaxWorkers = $hardware.Workers }
if ($DenseBudgetGiB -le 0) { $DenseBudgetGiB = $hardware.DenseBudgetGiB }
if ($CacheMaxGiB -le 0) { $CacheMaxGiB = $hardware.CacheMaxGiB }
if ($MinFreeGiB -le 0) { $MinFreeGiB = $hardware.MinFreeGiB }
if ($MaxWorkers -lt 1 -or $WorkerMaxTasks -lt 1 -or $MaxAttempts -lt 1) {
    throw "Worker, worker-task, and retry limits must be positive."
}
if ($DenseBudgetGiB -le 0 -or $CacheMaxGiB -le 0 -or $MinFreeGiB -lt 0) {
    throw "Memory and disk limits are invalid."
}

$rawDir = Join-Path $projectRoot "data/raw"
$runDir = Join-Path $projectRoot $RunRoot
$logDir = Join-Path $runDir "logs"
$workerLogDir = Join-Path $logDir "workers"
$metadataDir = Join-Path $runDir "metadata"
$eligibilityPath = Join-Path $projectRoot "reports/benchmark_runs/preflight_eligibility.json"
$eligibilitySha256 = "34f87b818d9c3705b691baa0ed088178e7bef6af4ad032a9e555267e513adbcb"
$localEligibilityPath = Join-Path $projectRoot "reports/benchmark_runs/laptop2/preflight_eligibility.local.json"
if (Test-Path -LiteralPath $localEligibilityPath) {
    $eligibilityPath = $localEligibilityPath
    $eligibilitySha256 = "9dc453f42d51a6c574c1518a235aa98687b098948058c3f9603187e9e1dd146b"
}
if (-not (Test-Path -LiteralPath $eligibilityPath) -and -not $PlanOnly) {
    throw "Pinned eligibility bundle not found: $eligibilityPath"
}

if (-not $PlanOnly) {
    New-Item -ItemType Directory -Force -Path $rawDir,$runDir,$logDir,$workerLogDir,$metadataDir | Out-Null
    $env:AUTOFE_WORKER_LOG_DIR = $workerLogDir
    $launcherMetadata = [ordered]@{
        run_id = $RunId
        run_kind = "full"
        run_root = (Resolve-Path -LiteralPath $runDir).Path
        resume_requested = [bool]$Resume
        conda_environment = if($CondaEnv){$CondaEnv}else{$null}
        python_executable = $PythonExe
        datasets = $datasets
        row_cap = 100000
        protocol_eligibility_path = $eligibilityPath
        invocation_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
    $metadataName = if (Test-Path -LiteralPath (Join-Path $metadataDir "launcher_config.json")) {
        "launcher_attempt_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss_fff")
    } else { "launcher_config.json" }
    $launcherMetadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $metadataDir $metadataName) -Encoding UTF8
}

$missing = [System.Collections.Generic.List[string]]::new()
$incomplete = [System.Collections.Generic.List[string]]::new()
foreach ($dataset in $datasets) {
    $csv = Join-Path $rawDir "$dataset.csv"
    $meta = Join-Path $rawDir "${dataset}_meta.json"
    if ((Test-Path -LiteralPath $csv) -and (Test-Path -LiteralPath $meta)) {
        continue
    }
    if ((Test-Path -LiteralPath $csv) -or (Test-Path -LiteralPath $meta)) {
        $incomplete.Add($dataset)
    } else {
        $missing.Add($dataset)
    }
}
if ($incomplete.Count -gt 0) {
    throw "Incomplete raw dataset pairs found; refusing to overwrite them: $($incomplete -join ', ')"
}
$nativeEnv = @{
    OPENBLAS_NUM_THREADS = "1"
    OMP_NUM_THREADS = "1"
    MKL_NUM_THREADS = "1"
    LOKY_MAX_CPU_COUNT = "1"
    PYTHONUNBUFFERED = "1"
}
foreach ($entry in $nativeEnv.GetEnumerator()) { Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value }

$modelArgs = @()
if ($CpuOnly) {
    $modelArgs = @("--models", "logistic_regression,random_forest,extra_trees,linear_svm,knn,gaussian_nb,mlp,lightgbm")
}
Write-Host "AutoFE-ShiftBench full-project launcher" -ForegroundColor Cyan
Write-Host ("Environment: {0}" -f ($(if($CondaEnv){"Conda $CondaEnv"}else{"Python $PythonExe"})))
Write-Host ("CPU/RAM/disk: {0} logical CPUs, {1} GiB RAM, {2} GiB free" -f $hardware.LogicalProcessors,$hardware.RamGiB,$hardware.FreeGiB)
Write-Host ("Limits: {0} workers, {1} GiB dense budget, {2} GiB cache admission, {3} GiB free reserve" -f $MaxWorkers,$DenseBudgetGiB,$CacheMaxGiB,$MinFreeGiB)
Write-Host ("GPU: {0}" -f ($(if($gpuInfo){$gpuInfo}else{"not detected; CPU-only mode"})))
Write-Host ("Datasets: {0} (100,000-row hard cap)" -f $datasets.Count)

if ($PlanOnly) {
    if ($missing.Count -gt 0) { Write-Host ("Would download: {0}" -f ($missing -join ", ")) }
    if (-not $gpuInfo) { Write-Host "No NVIDIA GPU detected; a full run would require -CpuOnly or a supported GPU." -ForegroundColor Yellow }
    Write-Host "Plan only: no download or benchmark command was executed." -ForegroundColor Yellow
    exit 0
}

if ($missing.Count -gt 0 -and $SkipDownload) {
    throw "Raw datasets are missing ($($missing -join ', ')); rerun without -SkipDownload."
}
if (-not $gpuInfo -and -not $CpuOnly) {
    throw "No usable NVIDIA GPU was detected. The full benchmark requires XGBoost/CatBoost GPU tasks; rerun with -CpuOnly only if a reduced CPU-only run is scientifically acceptable."
}

if ($missing.Count -gt 0) {
    Write-Host ("Downloading {0} missing dataset(s) with the 100,000-row cap..." -f $missing.Count) -ForegroundColor Yellow
    $downloadArgs = @("-B", "-m", "src.data_loader", "--datasets", ($missing -join ","), "--output-dir", "data/raw", "--max-rows", "100000")
    Invoke-Python $downloadArgs
    Assert-ExitCode "Dataset acquisition"
}

$runnerArgs = @(
    "-B", "-m", "src.pipeline_runner",
    "--eligibility", $eligibilityPath,
    "--eligibility-sha256", $eligibilitySha256,
    "--datasets"
)
$runnerArgs += $datasets
$runnerArgs += @(
    "--enable-fsva-diagnostics",
    "--fsva-max-rows", "128",
    "--max-workers", "$MaxWorkers",
    "--dense-budget-gib", "$DenseBudgetGiB",
    "--max-attempts", "$MaxAttempts",
    "--worker-max-tasks", "$WorkerMaxTasks",
    "--cache-policy", "rolling",
    "--cache-max-gib", "$CacheMaxGiB",
    "--min-free-gib", "$MinFreeGiB",
    "--stale-after-seconds", "3600",
    "--run-id", $RunId,
    "--manifest-db", (Join-Path $RunRoot "benchmark.db"),
    "--manifest-path", (Join-Path $RunRoot "manifest.jsonl"),
    "--results-path", (Join-Path $RunRoot "results.jsonl"),
    "--log-file", (Join-Path $RunRoot "logs/benchmark.log"),
    "--worker-log-dir", (Join-Path $RunRoot "logs/workers")
)
$runnerArgs += $modelArgs
if ($DryRun) { $runnerArgs += "--dry-run-manifest" }

Write-Host ("Starting durable run '{0}'..." -f $RunId) -ForegroundColor Green
Invoke-Python $runnerArgs
Assert-ExitCode "Benchmark run"
Write-Host "Benchmark launcher completed successfully." -ForegroundColor Green
