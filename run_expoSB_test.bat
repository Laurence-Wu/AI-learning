@echo off
echo ========================================
echo Running expoSB Implementation Test
echo ========================================
echo.
echo This script will:
echo 1. Upload current changes to GitHub
echo 2. Open WSL Ubuntu 22.04
echo 3. Activate triton-env-stable environment
echo 4. Navigate to AI learning directory
echo 5. Run expoSB_implementation/test_expoSB
echo.

REM Use the deploy_and_run.bat script to set up environment and run the test
call "%~dp0deploy_and_run.bat" "source ~/triton-env-stable/bin/activate && python expoSB_implementation/test_expoSB.py"