@echo off
setlocal enabledelayedexpansion

echo ========================================
echo GitHub Upload and WSL Environment Setup
echo ========================================

REM Check if parameter is provided
if "%~1"=="" (
    echo Usage: deploy_and_run.bat ^<command_or_path^>
    echo Example: deploy_and_run.bat "python train.py"
    echo Example: deploy_and_run.bat "./scripts/run_model.sh"
    pause
    exit /b 1
)

set "COMMAND_TO_RUN=%~1"

echo.
echo Step 1: Adding files to git...
git add .
if errorlevel 1 (
    echo Error: Failed to add files to git
    pause
    exit /b 1
)

echo.
echo Step 2: Committing changes...
git commit -m "Auto-deploy: Update AI-learning project files"
if errorlevel 1 (
    echo Warning: No changes to commit or commit failed
)

echo.
echo Step 3: Pushing to GitHub...
git push origin master
if errorlevel 1 (
    echo Error: Failed to push to GitHub
    pause
    exit /b 1
)

echo.
echo Step 4: Opening WSL Ubuntu 22.04 and setting up environment...
echo Command to be executed: %COMMAND_TO_RUN%
echo.

REM Create a temporary script for WSL execution
echo #!/bin/bash > wsl_commands.sh
echo echo "========================================" >> wsl_commands.sh
echo echo "Setting up Triton environment..." >> wsl_commands.sh
echo echo "========================================" >> wsl_commands.sh
echo source ~/miniconda3/etc/profile.d/conda.sh >> wsl_commands.sh
echo conda activate triton-env-stable >> wsl_commands.sh
echo if [ $? -ne 0 ]; then >> wsl_commands.sh
echo     echo "Error: Failed to activate triton-env-stable environment" >> wsl_commands.sh
echo     echo "Available environments:" >> wsl_commands.sh
echo     conda env list >> wsl_commands.sh
echo     exit 1 >> wsl_commands.sh
echo fi >> wsl_commands.sh
echo echo "Environment activated successfully" >> wsl_commands.sh
echo echo "Navigating to AI learning directory..." >> wsl_commands.sh
echo cd ~/Coding/AI-learning >> wsl_commands.sh
echo if [ $? -ne 0 ]; then >> wsl_commands.sh
echo     echo "Error: Failed to navigate to ~/Coding/AI-learning" >> wsl_commands.sh
echo     echo "Current directory: $(pwd)" >> wsl_commands.sh
echo     echo "Available directories in ~/Coding:" >> wsl_commands.sh
echo     ls -la ~/Coding/ 2^>/dev/null ^|^| echo "~/Coding directory not found" >> wsl_commands.sh
echo     exit 1 >> wsl_commands.sh
echo fi >> wsl_commands.sh
echo echo "Current directory: $(pwd)" >> wsl_commands.sh
echo echo "Executing command: %COMMAND_TO_RUN%" >> wsl_commands.sh
echo %COMMAND_TO_RUN% >> wsl_commands.sh
echo echo "Command execution completed" >> wsl_commands.sh
echo exec bash >> wsl_commands.sh

REM Execute the script in WSL
wsl -d Ubuntu-22.04 bash ./wsl_commands.sh

REM Clean up temporary script
del wsl_commands.sh 2>nul

echo.
echo Script execution completed!
pause