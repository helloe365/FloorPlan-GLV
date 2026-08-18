#!/usr/bin/env bash
#
# ===============================================================
#  FloorPlan-GLV Environment Setup Script
#  System: Ubuntu, GPU: RTX 5090 (32GB), CUDA Driver: 13.0
#  Proxy: 127.0.0.1:9999
#  Mirror: Tsinghua (conda pre-configured, pip via -i flag)
# ===============================================================
set -euo pipefail

ENV_NAME="floorplan-glv"
PROJECT_DIR="/root/FloorPlan-GLV"
PROXY="http://127.0.0.1:9999"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# ===============================================================
#  Step 1: Check conda availability
# ===============================================================
echo ">>> [1/5] Checking conda ..."
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install miniconda/anaconda first."
    exit 1
fi
echo "  conda path: $(which conda)"

# ===============================================================
#  Step 2: Remove old environment (if exists) and create a new one
# ===============================================================
echo ">>> [2/5] Creating conda environment: ${ENV_NAME} ..."
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
conda create -n "${ENV_NAME}" python=3.12 -y

echo "  conda environment ${ENV_NAME} created successfully"

# ===============================================================
#  Step 3: Install PyTorch (CUDA 13.0, Blackwell architecture)
# ===============================================================
echo ">>> [3/5] Installing PyTorch 2.12.0 + torchvision 0.27.0 (CUDA 13.0) ..."
# Set proxy
export http_proxy="${PROXY}"
export https_proxy="${PROXY}"
export HTTP_PROXY="${PROXY}"
export HTTPS_PROXY="${PROXY}"

conda install -n "${ENV_NAME}" \
    pytorch=2.12.0=gpu_cuda130_py312hfe95348_300 \
    torchvision=0.27.0=cuda130py312hfb2e9fc_100 \
    -c pytorch -y

echo "  PyTorch installed successfully"

# ===============================================================
#  Step 4: Install project and dev dependencies
# ===============================================================
echo ">>> [4/5] Installing project dependencies (pip) ..."

conda run -n "${ENV_NAME}" pip install -e "${PROJECT_DIR}[dev]" \
    --proxy "${PROXY}" \
    -i "${PIP_MIRROR}" \
    --trusted-host pypi.tuna.tsinghua.edu.cn

echo "  Project dependencies installed successfully"

# ===============================================================
#  Step 5: Verify environment
# ===============================================================
echo ">>> [5/5] Verifying environment ..."
echo ""

# Verify PyTorch + CUDA
echo "  --- PyTorch / CUDA check ---"
conda run -n "${ENV_NAME}" python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU name:        {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version:    {torch.version.cuda}')
    print(f'  GPU count:       {torch.cuda.device_count()}')
else:
    print('  !! WARNING: CUDA is not available, please check PyTorch installation !!')
"

echo ""

# Verify torchvision
echo "  --- torchvision check ---"
conda run -n "${ENV_NAME}" python -c "
import torchvision
print(f'  torchvision version: {torchvision.__version__}')
"

echo ""

# Verify project imports
echo "  --- Project module check ---"
conda run -n "${ENV_NAME}" python -c "
import floorplan_glv
print(f'  floorplan_glv imported successfully')
from floorplan_glv.cli.app import app
print(f'  CLI app loaded successfully')
"

echo ""

# Verify CLI entry point
echo "  --- CLI entry point check ---"
conda run -n "${ENV_NAME}" floorplan-glv --help

echo ""
echo "==============================================================="
echo "  Environment setup complete!"
echo "  Activate with: conda activate ${ENV_NAME}"
echo "==============================================================="
