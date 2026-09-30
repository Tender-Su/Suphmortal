@echo off
setlocal

if "%~1"=="" (
    echo Usage: %~nx0 ^<config_path^> [oracle_arm] [artifact_suffix]
    exit /b 1
)

set "CFG=%~f1"
set "ARM=%~2"
set "SUFFIX=%~3"

cd /d "%~dp0.."

set "COMMON_ARGS=--config \"%CFG%\""
if not "%ARM%"=="" set "COMMON_ARGS=%COMMON_ARGS% --arm %ARM%"
if not "%SUFFIX%"=="" set "COMMON_ARGS=%COMMON_ARGS% --artifact-suffix %SUFFIX%"

start "MahjongAI Online Server" cmd /k python -m mortal.online.online_role_runner server %COMMON_ARGS%
start "MahjongAI Online Trainer" cmd /k python -m mortal.online.online_role_runner trainer %COMMON_ARGS%
start "MahjongAI Online Worker" cmd /k python -m mortal.online.online_role_runner client %COMMON_ARGS%
