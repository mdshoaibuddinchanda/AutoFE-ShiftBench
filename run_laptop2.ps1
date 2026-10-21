# AutoFE-ShiftBench: Laptop 2 Runner (Reverse Dataset Order)
# Hardware profile: 32GB RAM, 4GB VRAM, i5 11th Gen

$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:LOKY_MAX_CPU_COUNT = "1"

# Datasets in reverse order (from dataset 25 down to 13)
$datasets = @(
    "kddcup99",
    "airlines",
    "covertype",
    "aps_failure",
    "electricity",
    "bank-marketing",
    "adult",
    "dry-bean-dataset",
    "magic-telescope",
    "default-of-credit-card-clients",
    "PhishingWebsites",
    "jm1",
    "spambase"
)

conda run -n py12 python -m src.pipeline_runner `
    --eligibility reports/gemini_benchmark/preflight_eligibility.json `
    --eligibility-sha256 34f87b818d9c3705b691baa0ed088178e7bef6af4ad032a9e555267e513adbcb `
    --datasets $datasets `
    --enable-fsva-diagnostics `
    --max-workers 8 `
    --dense-budget-gib 20 `
    --worker-max-tasks 64 `
    --cache-policy rolling `
    --cache-max-gib 60 `
    --min-free-gib 15 `
    --stale-after-seconds 1.0 `
    --run-id laptop2_run `
    --manifest-db reports/gemini_benchmark/laptop2/benchmark.db `
    --manifest-path reports/gemini_benchmark/laptop2/manifest.jsonl `
    --results-path reports/gemini_benchmark/laptop2/results.jsonl `
    --log-file reports/gemini_benchmark/laptop2/benchmark.log
