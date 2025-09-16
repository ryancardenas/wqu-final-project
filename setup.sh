#!/usr/bin/env bash

set -e  # Exit on error

ENV_NAME="wqufp"
ENV_FILE="environment.yml"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest"

# Detect platform
OS="$(uname)"
if [[ "$OS" == "Linux" ]]; then
    MINICONDA_URL+="-Linux-x86_64.sh"
elif [[ "$OS" == "Darwin" ]]; then
    MINICONDA_URL+="-MacOSX-x86_64.sh"
elif [[ "$OS" =~ MINGW|MSYS|CYGWIN ]]; then
    MINICONDA_URL+="-Windows-x86_64.exe"
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Check for conda
if command -v conda &> /dev/null; then
    echo "✅ Conda found: $(conda --version)"
else
    echo "🔧 Conda not found. Installing Miniconda..."
    curl -LO "$MINICONDA_URL"
    
    if [[ "$OS" == "Windows_NT" || "$OS" =~ MINGW|MSYS|CYGWIN ]]; then
        echo "⚠️ Please run the Miniconda installer manually: Miniconda3-latest-Windows-x86_64.exe"
        echo "After installation, restart your terminal and re-run this script."
        exit 1
    else
        bash "$(basename "$MINICONDA_URL")" -b -p "$HOME/miniconda"
        export PATH="$HOME/miniconda/bin:$PATH"
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        conda init
    fi
fi

if conda info --envs | grep -q "^wqufp"; then
    echo "✅ Environment 'wqufp' already exists."
else
    # Create environment
    echo "📦 Creating Conda environment: $ENV_NAME"
    conda env create -f "$ENV_FILE"
fi

# Activate environment
echo "🚀 Activating environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install pre-commit hooks
echo "🔍 Installing pre-commit hooks"
pre-commit install

echo "✅ Setup complete. Environment '$ENV_NAME' is ready."