@echo off
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m sic_xrt_ml.workstation.doctor --require-cuda
if errorlevel 1 goto end
".venv\Scripts\python.exe" -m sic_xrt_ml.workstation.inventory --repo "%CD%"
:end
pause
