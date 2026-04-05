@echo off
REM Reinforcement learning phase: online PPO self-play training
REM Resumes from [control].state_file when present; otherwise bootstraps from [online].init_state_file
REM Set online = true in config.toml before running

echo Starting online PPO training...
cd /d "%~dp0..\mortal"

set "STATE_FILE="
set "INIT_STATE_FILE="
for /f "usebackq delims=" %%I in (`python -c "from config import config; print(config['control']['state_file'])"`) do set "STATE_FILE=%%I"
for /f "usebackq delims=" %%I in (`python -c "from config import config; import train_online; print(train_online.resolve_online_init_state_file(config))"`) do set "INIT_STATE_FILE=%%I"
if not defined STATE_FILE (
    echo ERROR: Failed to resolve [control].state_file from config.toml
    exit /b 1
)

if exist "%STATE_FILE%" goto RUN_ONLINE
if defined INIT_STATE_FILE if exist "%INIT_STATE_FILE%" goto RUN_ONLINE

if defined INIT_STATE_FILE (
    echo ERROR: Neither RL resume checkpoint nor supervised init checkpoint exists.
    echo Missing [control].state_file: %STATE_FILE%
    echo Missing [online].init_state_file: %INIT_STATE_FILE%
) else (
    echo ERROR: RL resume checkpoint not found at %STATE_FILE%
    echo [online].init_state_file is empty; please point it to the canonical supervised checkpoint.
)
    exit /b 1

:RUN_ONLINE

python train_online.py
if errorlevel 1 (
    echo ERROR: Online PPO training failed.
    exit /b 1
)

echo Online PPO training complete!
