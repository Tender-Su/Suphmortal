@echo off
REM Stage 2: Online PPO Self-Play Training
REM Requires: a Stage 1 handoff checkpoint at [control].state_file
REM Current default handoff is refreshed by Stage 1 oracle-dropout refinement.
REM Set online = true in config.toml before running

echo Starting Online PPO training (Stage 2)...
cd /d "%~dp0..\mortal"

set "STATE_FILE="
for /f "usebackq delims=" %%I in (`python -c "from config import config; print(config['control']['state_file'])"`) do set "STATE_FILE=%%I"
if not defined STATE_FILE (
    echo ERROR: Failed to resolve [control].state_file from config.toml
    exit /b 1
)

if not exist "%STATE_FILE%" (
    echo ERROR: Stage 1 handoff checkpoint not found at %STATE_FILE%
    echo Please run the current Stage 1 mainline first.
    exit /b 1
)

python train_online.py
if errorlevel 1 (
    echo ERROR: Online PPO training failed.
    exit /b 1
)

echo Online PPO training complete!
