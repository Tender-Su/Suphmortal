@echo off
REM Stage 0: Train GRP (Game Result Predictor)
REM Run this first before any other training stage

echo Starting GRP training (Stage 0)...
cd /d "%~dp0..\mortal"

python train_grp.py
if errorlevel 1 (
    echo ERROR: GRP training failed.
    exit /b 1
)

echo GRP training complete!
echo Check checkpoints/grp.pth for the trained model.
