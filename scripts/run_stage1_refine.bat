@echo off
REM Stage 1: Oracle Dropout Supervised Refinement

echo Starting Stage 1 oracle-dropout supervised refinement...
cd /d "%~dp0..\mortal"

python train_stage1_refine.py %*
if errorlevel 1 (
    echo ERROR: Stage 1 oracle-dropout supervised refinement failed.
    exit /b 1
)

echo Stage 1 oracle-dropout supervised refinement complete!
