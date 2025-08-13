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
echo #!/bin/bash > wsl_setup.sh
echo echo "========================================" >> wsl_setup.sh
echo echo "Setting up Triton environment..." >> wsl_setup.sh
echo echo "========================================" >> wsl_setup.sh
echo. >> wsl_setup.sh
echo # Find conda installation >> wsl_setup.sh
echo if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then >> wsl_setup.sh
echo     source ~/miniconda3/etc/profile.d/conda.sh >> wsl_setup.sh
echo elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then >> wsl_setup.sh
echo     source ~/anaconda3/etc/profile.d/conda.sh >> wsl_setup.sh
echo elif command -v conda ^^^>/dev/null 2^^^>^^^&1; then >> wsl_setup.sh
echo     echo "Conda already in PATH" >> wsl_setup.sh
echo else >> wsl_setup.sh
echo     echo "Error: Conda not found. Please check your conda installation." >> wsl_setup.sh
echo     echo "Trying alternative activation methods..." >> wsl_setup.sh
echo     if [ -f ~/.bashrc ]; then >> wsl_setup.sh
echo         source ~/.bashrc >> wsl_setup.sh
echo     fi >> wsl_setup.sh
echo fi >> wsl_setup.sh
echo. >> wsl_setup.sh
echo # Try to activate environment >> wsl_setup.sh
echo if command -v conda ^^^>/dev/null 2^^^>^^^&1; then >> wsl_setup.sh
echo     conda activate triton-env-stable >> wsl_setup.sh
echo     if [ $? -ne 0 ]; then >> wsl_setup.sh
echo         echo "Error: Failed to activate triton-env-stable environment" >> wsl_setup.sh
echo         echo "Available environments:" >> wsl_setup.sh
echo         conda env list >> wsl_setup.sh
echo         exit 1 >> wsl_setup.sh
echo     fi >> wsl_setup.sh
echo     echo "Environment activated successfully" >> wsl_setup.sh
echo else >> wsl_setup.sh
echo     echo "Warning: Conda not available, continuing without virtual environment" >> wsl_setup.sh
echo fi >> wsl_setup.sh
echo. >> wsl_setup.sh
echo echo "Navigating to AI learning directory..." >> wsl_setup.sh
echo cd ~/Coding/AI-learning >> wsl_setup.sh
echo if [ $? -ne 0 ]; then >> wsl_setup.sh
echo     echo "Error: Failed to navigate to ~/Coding/AI-learning" >> wsl_setup.sh
echo     echo "Current directory: $$(pwd)" >> wsl_setup.sh
echo     echo "Trying alternative paths..." >> wsl_setup.sh
echo     if [ -d ~/Coding ]; then >> wsl_setup.sh
echo         echo "Available directories in ~/Coding:" >> wsl_setup.sh
echo         ls -la ~/Coding/ >> wsl_setup.sh
echo     else >> wsl_setup.sh
echo         echo "~/Coding directory not found" >> wsl_setup.sh
echo         echo "Available directories in home:" >> wsl_setup.sh
echo         ls -la ~/ >> wsl_setup.sh
echo     fi >> wsl_setup.sh
echo     cd ~/AI-learning 2^^^>/dev/null ^^^|^^^| cd ~/coding/AI-learning 2^^^>/dev/null ^^^|^^^| echo "Could not find AI-learning directory" >> wsl_setup.sh
echo fi >> wsl_setup.sh
echo. >> wsl_setup.sh
echo echo "Current directory: $$(pwd)" >> wsl_setup.sh
echo echo "Executing command: %COMMAND_TO_RUN%" >> wsl_setup.sh
echo %COMMAND_TO_RUN% >> wsl_setup.sh
echo echo "Command execution completed" >> wsl_setup.sh
echo exec bash >> wsl_setup.sh

REM Convert to Unix line endings and execute in WSL
wsl -d Ubuntu-22.04 bash -c "dos2unix wsl_setup.sh 2>/dev/null; chmod +x wsl_setup.sh; ./wsl_setup.sh"

REM Clean up
del wsl_setup.sh 2>nul


echo.
echo Script execution completed!
pause