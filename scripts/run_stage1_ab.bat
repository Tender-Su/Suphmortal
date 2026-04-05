@echo off
REM Stage 1 oracle-dropout profile A/B: fixed full-aux recipe + dropout schedule sweep

echo Starting Stage 1 A/B runner...
cd /d "%~dp0..\mortal"

python run_stage1_ab.py %*
if errorlevel 1 (
    echo ERROR: Stage 1 A/B runner failed.
    exit /b 1
)

echo Stage 1 A/B runner complete!
