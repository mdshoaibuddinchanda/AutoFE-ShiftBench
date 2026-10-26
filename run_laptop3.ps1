# AutoFE-ShiftBench: Laptop 3 Runner (Middle Partition)
# Datasets: electricity, bank-marketing, adult, dry-bean-dataset

$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:LOKY_MAX_CPU_COUNT = "1"

$datasets = @(
    "electricity",
    "bank-marketing",
    "adult",
    "dry-bean-dataset"
)

conda run -n py12 python -m src.pipeline_runner `
    --eligibility reports/gemini_benchmark/preflight_eligibility.json `
    --eligibility-sha256 34f87b818d9c3705b691baa0ed088178e7bef6af4ad032a9e555267e513adbcb `
    --datasets $datasets `
    --enable-fsva-diagnostics `
    --max-workers 8 `
    --dense-budget-gib 16 `
    --worker-max-tasks 64 `
    --cache-policy rolling `
    --cache-max-gib 50 `
    --min-free-gib 15 `
    --stale-after-seconds 1.0 `
    --run-id laptop3_run `
    --manifest-db reports/gemini_benchmark/laptop3/benchmark.db `
    --manifest-path reports/gemini_benchmark/laptop3/manifest.jsonl `
    --results-path reports/gemini_benchmark/laptop3/results.jsonl `
    --log-file reports/gemini_benchmark/laptop3/benchmark.log
