#!/usr/bin/env bash
#
# ===============================================================
#  FloorPlan-GLV 环境配置脚本
#  系统: Ubuntu, GPU: RTX 5090 (32GB), CUDA Driver: 13.0
#  代理: 127.0.0.1:9999
#  镜像: 清华 (conda 已预配置, pip 通过 -i 指定)
# ===============================================================
set -euo pipefail

ENV_NAME="floorplan-glv"
PROJECT_DIR="/root/FloorPlan-GLV"
PROXY="http://127.0.0.1:9999"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# ===============================================================
#  第 1 步: 检查 conda 是否可用
# ===============================================================
echo ">>> [1/5] 检查 conda ..."
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda, 请先安装 miniconda/anaconda"
    exit 1
fi
echo "  conda 路径: $(which conda)"

# ===============================================================
#  第 2 步: 删除旧环境 (如果存在) 并创建新环境
# ===============================================================
echo ">>> [2/5] 创建 conda 环境: ${ENV_NAME} ..."
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
conda create -n "${ENV_NAME}" python=3.12 -y

echo "  conda 环境 ${ENV_NAME} 创建完成"

# ===============================================================
#  第 3 步: 安装 PyTorch (CUDA 13.0, Blackwell 架构)
# ===============================================================
echo ">>> [3/5] 安装 PyTorch 2.12.0 + torchvision 0.27.0 (CUDA 13.0) ..."
# 设置代理
export http_proxy="${PROXY}"
export https_proxy="${PROXY}"
export HTTP_PROXY="${PROXY}"
export HTTPS_PROXY="${PROXY}"

conda install -n "${ENV_NAME}" \
    pytorch=2.12.0=gpu_cuda130_py312hfe95348_300 \
    torchvision=0.27.0=cuda130py312hfb2e9fc_100 \
    -c pytorch -y

echo "  PyTorch 安装完成"

# ===============================================================
#  第 4 步: 安装项目及开发依赖
# ===============================================================
echo ">>> [4/5] 安装项目依赖 (pip) ..."

conda run -n "${ENV_NAME}" pip install -e "${PROJECT_DIR}[dev]" \
    --proxy "${PROXY}" \
    -i "${PIP_MIRROR}" \
    --trusted-host pypi.tuna.tsinghua.edu.cn

echo "  项目依赖安装完成"

# ===============================================================
#  第 5 步: 验证环境
# ===============================================================
echo ">>> [5/5] 验证环境 ..."
echo ""

# 验证 PyTorch + CUDA
echo "  --- PyTorch / CUDA 检查 ---"
conda run -n "${ENV_NAME}" python -c "
import torch
print(f'  PyTorch 版本: {torch.__version__}')
print(f'  CUDA 可用:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU 名称:    {torch.cuda.get_device_name(0)}')
    print(f'  CUDA 版本:   {torch.version.cuda}')
    print(f'  GPU 数量:    {torch.cuda.device_count()}')
else:
    print('  !! 警告: CUDA 不可用, 请检查 PyTorch 安装 !!')
"

echo ""

# 验证 torchvision
echo "  --- torchvision 检查 ---"
conda run -n "${ENV_NAME}" python -c "
import torchvision
print(f'  torchvision 版本: {torchvision.__version__}')
"

echo ""

# 验证项目导入
echo "  --- 项目模块检查 ---"
conda run -n "${ENV_NAME}" python -c "
import floorplan_glv
print(f'  floorplan_glv 导入成功')
from floorplan_glv.cli.app import app
print(f'  CLI app 加载成功')
"

echo ""

# 验证 CLI
echo "  --- CLI 入口检查 ---"
conda run -n "${ENV_NAME}" floorplan-glv --help

echo ""
echo "==============================================================="
echo "  环境配置完成!"
echo "  激活命令: conda activate ${ENV_NAME}"
echo "==============================================================="
