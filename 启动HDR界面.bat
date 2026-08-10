@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON=D:\Users\13\anaconda3\envs\pyqt\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" hdr_gui.py
pause
