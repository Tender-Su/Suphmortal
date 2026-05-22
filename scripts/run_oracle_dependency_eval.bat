@echo off
REM Evaluate one checkpoint under true / zero / shuffled Oracle input modes.

cd /d "%~dp0.."

set "STATE_FILE_ARG="
if not "%~1"=="" set "STATE_FILE_ARG=--state-file %~1"

python -m mortal.eval.oracle_dependency_eval %STATE_FILE_ARG% %2 %3 %4 %5 %6 %7 %8 %9
if errorlevel 1 (
    echo ERROR: Oracle dependency evaluation failed.
    exit /b 1
)
