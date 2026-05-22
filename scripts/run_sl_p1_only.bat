@echo off
setlocal
cd /d "%~dp0.."
"C:\ProgramData\anaconda3\envs\mortal\python.exe" -m mortal.supervised.run_sl_p1_only %*
