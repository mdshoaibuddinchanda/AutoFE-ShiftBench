@echo off
echo ============================================================
echo   AutoFE-ShiftBench — One-Command Setup and Run
echo ============================================================
echo.

REM Step 1: Install dependencies
echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo.

REM Step 2: Download datasets
echo [2/3] Downloading datasets from OpenML...
python -c "from src.data_loader import download_datasets_from_list; download_datasets_from_list()"
if errorlevel 1 (
    echo ERROR: Failed to download datasets.
    pause
    exit /b 1
)
echo.

REM Step 3: Run the benchmark
echo [3/3] Starting benchmark with full parallelization...
python -m src.pipeline_runner
echo.
echo ============================================================
echo   Benchmark complete! Results in reports/tables/results_stream.jsonl
echo ============================================================
pause
