@echo off
REM Supervised phase: formal supervised training / protocol replay
REM Default protocol arm is selected inside run_sl_formal.py; extra args are forwarded.

echo Starting formal supervised training...
cd /d "%~dp0..\mortal"

python run_sl_formal.py %*
if errorlevel 1 (
    echo ERROR: Formal supervised training failed.
    exit /b 1
)

echo Formal supervised training complete!
echo Primary-arm runs refresh the canonical supervised checkpoint.
echo Check logs/sl_ab/ for outputs.
