@echo off
setlocal
REM Use the supported existing environment and forward all selected run limits.
conda run --no-capture-output -n P12 python -B -m src.pipeline_runner %*
exit /b %errorlevel%
