@echo off
REM Stage 0.5: Formal supervised pretraining / protocol replay
REM Default protocol arm is selected inside run_stage05_formal.py; extra args are forwarded.

echo Starting formal supervised pretraining (Stage 0.5)...
cd /d "%~dp0..\mortal"

python run_stage05_formal.py %*
if errorlevel 1 (
    echo ERROR: Formal supervised pretraining failed.
    exit /b 1
)

echo Formal supervised pretraining complete!
echo Primary-arm runs refresh the canonical Stage 0.5 handoff checkpoint.
echo Check logs/stage05_ab/ for outputs.
