@echo off
REM Check if parameter is provided
if "%~1"=="" (
    echo Usage: deploy_and_run.bat ^<command_or_path^>
    echo Example: deploy_and_run.bat "python train.py"
    echo Example: deploy_and_run.bat "source ~/triton-env-stable"
    pause
    exit /b 1
)

REM Call PowerShell script with the parameter
powershell -ExecutionPolicy Bypass -File "%~dp0deploy_and_run.ps1" -Command "%~1"