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

set "ARGS=client --config \"%CFG%\""
if not "%ARM%"=="" set "ARGS=%ARGS% --arm %ARM%"
if not "%SUFFIX%"=="" set "ARGS=%ARGS% --artifact-suffix %SUFFIX%"

python -m mortal.online.online_role_runner %ARGS%
