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

REM Create a temporary script for WSL execution (with proper Unix line endings)
(
echo #!/bin/bash
echo echo "========================================"
echo echo "Setting up Triton environment..."
echo echo "========================================"
echo # Find conda installation
echo if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
echo     source ~/miniconda3/etc/profile.d/conda.sh
echo elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
echo     source ~/anaconda3/etc/profile.d/conda.sh
echo elif command -v conda ^>/dev/null 2^>^&1; then
echo     echo "Conda already in PATH"
echo else
echo     echo "Error: Conda not found. Please check your conda installation."
echo     echo "Trying alternative activation methods..."
echo     if [ -f ~/.bashrc ]; then
echo         source ~/.bashrc
echo     fi
echo fi
echo.
echo # Try to activate environment
echo if command -v conda ^>/dev/null 2^>^&1; then
echo     conda activate triton-env-stable
echo     if [ $? -ne 0 ]; then
echo         echo "Error: Failed to activate triton-env-stable environment"
echo         echo "Available environments:"
echo         conda env list
echo         exit 1
echo     fi
echo     echo "Environment activated successfully"
echo else
echo     echo "Warning: Conda not available, continuing without virtual environment"
echo fi
echo.
echo echo "Navigating to AI learning directory..."
echo cd ~/Coding/AI-learning
echo if [ $? -ne 0 ]; then
echo     echo "Error: Failed to navigate to ~/Coding/AI-learning"
echo     echo "Current directory: $(pwd^)"
echo     echo "Trying alternative paths..."
echo     if [ -d ~/Coding ]; then
echo         echo "Available directories in ~/Coding:"
echo         ls -la ~/Coding/
echo     else
echo         echo "~/Coding directory not found"
echo         echo "Available directories in home:"
echo         ls -la ~/
echo     fi
echo     # Try alternative paths
echo     cd ~/AI-learning 2^>/dev/null ^|^| cd ~/coding/AI-learning 2^>/dev/null ^|^| echo "Could not find AI-learning directory"
echo fi
echo.
echo echo "Current directory: $(pwd^)"
echo echo "Executing command: %COMMAND_TO_RUN%"
echo %COMMAND_TO_RUN%
echo echo "Command execution completed"
echo exec bash
) > wsl_setup.sh

REM Convert to Unix line endings and execute in WSL
wsl -d Ubuntu-22.04 bash -c "dos2unix wsl_setup.sh 2>/dev/null; chmod +x wsl_setup.sh; ./wsl_setup.sh"

REM Clean up
del wsl_setup.sh 2>nul


echo.
echo Script execution completed!
pause