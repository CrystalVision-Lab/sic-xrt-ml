@echo off
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m sic_xrt_ml.workstation.server stop --repo "%CD%"
if errorlevel 1 pause
