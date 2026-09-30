@echo off
setlocal

cd /d "%~dp0.."
python -m mortal.research.run_online_fidelity %*
