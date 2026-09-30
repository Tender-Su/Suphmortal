@echo off
setlocal

set "PY=C:\ProgramData\anaconda3\envs\mortal\python.exe"
"%PY%" -m integrations.majsoul.login %*
