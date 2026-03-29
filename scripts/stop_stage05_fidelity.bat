@echo off
setlocal
set RUN_NAME=%~1
if "%RUN_NAME%"=="" set RUN_NAME=s05_fidelity_main
"C:\ProgramData\anaconda3\envs\mortal\python.exe" "%~dp0stop_stage05_fidelity.py" --run-name "%RUN_NAME%"
endlocal
