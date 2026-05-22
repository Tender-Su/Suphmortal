@echo off
REM Pretrain Oracle critic on non-GRP score-rank returns without actor imitation.

echo Starting Oracle critic pretraining...
cd /d "%~dp0.."

set "PYTHON_EXE=C:\ProgramData\anaconda3\envs\mortal\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

"%PYTHON_EXE%" -m mortal.online.pretrain_oracle_critic %*
if errorlevel 1 (
    echo ERROR: Oracle critic pretraining failed.
    exit /b 1
)

echo Oracle critic pretraining complete!
