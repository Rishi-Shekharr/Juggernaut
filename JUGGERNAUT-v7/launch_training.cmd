@echo off
title JUGGERNAUT V7 Symmetric Gait Training v4
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
if defined JUGGERNAUT_PYTHON goto custom_python
where python >nul 2>nul
if not errorlevel 1 goto path_python
where py >nul 2>nul
if not errorlevel 1 goto py_launcher
echo Activate the training Python environment, or set JUGGERNAUT_PYTHON to its python.exe.
pause
exit /b 1
:custom_python
"%JUGGERNAUT_PYTHON%" -u "%~dp0training_console.py" %*
exit /b %ERRORLEVEL%
:path_python
python -u "%~dp0training_console.py" %*
exit /b %ERRORLEVEL%
:py_launcher
py -3 -u "%~dp0training_console.py" %*
exit /b %ERRORLEVEL%
