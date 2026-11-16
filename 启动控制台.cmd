@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_workbench.ps1" -Port 8880 -DataRoot "%~dp0data\sessions-liquid" %*
