@echo off
setlocal
cd /d "%~dp0"

set "P12_PYTHON=D:\Conda\p12\python.exe"
if not exist "%P12_PYTHON%" (
    echo ERROR: Existing p12 Python was not found at %P12_PYTHON%.
    exit /b 1
)

echo ============================================================
echo   AutoFE-ShiftBench bounded smoke in existing p12
echo   Full 25-dataset campaign: DO NOT LAUNCH until gates pass
echo ============================================================

"%P12_PYTHON%" main.py --max-datasets 1 --max-seeds 1 --max-folds 1 --max-conditions 2 --pipelines Raw AutoFE_Baseline --models logistic_regression --resource-policy adaptive --adaptive-worker-cap 1 --gpu-policy auto
if errorlevel 1 exit /b 1

echo Bounded smoke complete. Inspect corrected_runs by generated run ID.
exit /b 0
