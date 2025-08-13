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

REM Execute commands directly in WSL using here-document approach
wsl -d Ubuntu-22.04 bash -c "
echo '========================================'
echo 'Setting up Triton environment...'
echo '========================================'

# Find conda installation
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh  
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    echo 'Conda already in PATH'
else
    echo 'Conda not found. Trying to source common locations...'
    [ -f ~/.bashrc ] && source ~/.bashrc
    [ -f ~/.profile ] && source ~/.profile
fi

# Try to activate environment
if command -v conda >/dev/null 2>&1; then
    echo 'Activating triton-env-stable environment...'
    conda activate triton-env-stable
    if [ \$? -eq 0 ]; then
        echo 'Environment activated successfully'
    else
        echo 'Failed to activate triton-env-stable. Available environments:'
        conda env list
    fi
else
    echo 'Warning: Conda not available, continuing without virtual environment'
fi

echo 'Navigating to AI learning directory...'
if cd ~/Coding/AI-learning 2>/dev/null; then
    echo \"Successfully navigated to: \$(pwd)\"
elif cd ~/AI-learning 2>/dev/null; then
    echo \"Found alternative path: \$(pwd)\"
elif cd ~/coding/AI-learning 2>/dev/null; then
    echo \"Found alternative path: \$(pwd)\"
else
    echo 'Could not find AI-learning directory. Available options:'
    [ -d ~/Coding ] && echo 'In ~/Coding:' && ls -la ~/Coding/ | head -10
    echo 'In home directory:' && ls -la ~/ | grep -E '^d.*[Cc]oding|[Aa][Ii]' | head -5
    echo \"Staying in: \$(pwd)\"
fi

echo \"Current directory: \$(pwd)\"
echo \"Executing command: %COMMAND_TO_RUN%\"
%COMMAND_TO_RUN%
echo 'Command execution completed'

exec bash
"


echo.
echo Script execution completed!
pause