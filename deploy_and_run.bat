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

REM Execute commands directly in WSL to avoid line ending issues
wsl -d Ubuntu-22.04 bash -c "echo '========================================'
echo 'Setting up Triton environment...'
echo '========================================'
# Find conda installation
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    echo 'Conda already in PATH'
else
    echo 'Error: Conda not found. Please check your conda installation.'
    echo 'Trying alternative activation methods...'
    if [ -f ~/.bashrc ]; then
        source ~/.bashrc
    fi
fi

# Try to activate environment
if command -v conda >/dev/null 2>&1; then
    conda activate triton-env-stable
    if [ \$? -ne 0 ]; then
        echo 'Error: Failed to activate triton-env-stable environment'
        echo 'Available environments:'
        conda env list
        exit 1
    fi
    echo 'Environment activated successfully'
else
    echo 'Warning: Conda not available, continuing without virtual environment'
fi

echo 'Navigating to AI learning directory...'
cd ~/Coding/AI-learning
if [ \$? -ne 0 ]; then
    echo 'Error: Failed to navigate to ~/Coding/AI-learning'
    echo 'Current directory:' \$(pwd)
    echo 'Trying alternative paths...'
    if [ -d ~/Coding ]; then
        echo 'Available directories in ~/Coding:'
        ls -la ~/Coding/
    else
        echo '~/Coding directory not found'
        echo 'Available directories in home:'
        ls -la ~/
    fi
    # Try alternative paths
    cd ~/AI-learning 2>/dev/null || cd ~/coding/AI-learning 2>/dev/null || echo 'Could not find AI-learning directory'
fi

echo 'Current directory:' \$(pwd)
echo 'Executing command: %COMMAND_TO_RUN%'
%COMMAND_TO_RUN%
echo 'Command execution completed'
exec bash"


echo.
echo Script execution completed!
pause