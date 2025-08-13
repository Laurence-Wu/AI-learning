param(
    [Parameter(Mandatory=$true)]
    [string]$Command
)

Write-Host "========================================"
Write-Host "GitHub Upload and WSL Environment Setup"
Write-Host "========================================"

# Step 1: Git operations
Write-Host "`nStep 1: Adding files to git..."
git add .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to add files to git" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host "`nStep 2: Committing changes..."
git commit -m "Auto-deploy: Update AI-learning project files"

Write-Host "`nStep 3: Pushing to GitHub..."
git push origin master
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to push to GitHub" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host "`nStep 4: Opening WSL Ubuntu 22.04 and setting up environment..."
Write-Host "Command to be executed: $Command"
Write-Host ""

# Create the bash script content
$bashScript = @'
echo "========================================"
echo "Setting up Triton environment..."
echo "========================================"

# Find conda installation
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    echo "Sourced miniconda3 conda"
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    echo "Sourced anaconda3 conda"
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    echo "Sourced /opt/conda conda"
elif command -v conda >/dev/null 2>&1; then
    echo "Conda already in PATH"
else
    echo "Conda not found. Trying to source common locations..."
    [ -f ~/.bashrc ] && source ~/.bashrc
    [ -f ~/.profile ] && source ~/.profile
fi

# Try to activate environment
if command -v conda >/dev/null 2>&1; then
    echo "Activating triton-env-stable environment..."
    conda activate triton-env-stable
    if [ $? -eq 0 ]; then
        echo "Environment activated successfully"
        conda info --envs | grep "^\*"
    else
        echo "Failed to activate triton-env-stable. Available environments:"
        conda env list
    fi
else
    echo "Warning: Conda not available, continuing without virtual environment"
fi

echo "Navigating to AI learning directory..."
if cd ~/Coding/AI-learning 2>/dev/null; then
    echo "Successfully navigated to: $(pwd)"
elif cd ~/AI-learning 2>/dev/null; then
    echo "Found alternative path: $(pwd)"
elif cd ~/coding/AI-learning 2>/dev/null; then
    echo "Found alternative path: $(pwd)"
else
    echo "Could not find AI-learning directory. Available options:"
    if [ -d ~/Coding ]; then
        echo "In ~/Coding:"
        ls -la ~/Coding/ | head -10
    fi
    echo "In home directory:"
    ls -la ~/ | grep -E "^d.*[Cc]oding" | head -5
    echo "Staying in: $(pwd)"
fi

echo "Current directory: $(pwd)"
echo "Executing command: COMMAND_PLACEHOLDER"
COMMAND_PLACEHOLDER
echo "Command execution completed"

exec bash
'@

# Replace the placeholder with the actual command
$bashScript = $bashScript -replace "COMMAND_PLACEHOLDER", $Command

# Execute in WSL
wsl -d Ubuntu-22.04 bash -c $bashScript

Write-Host "`nScript execution completed!"
Read-Host "Press Enter to continue"