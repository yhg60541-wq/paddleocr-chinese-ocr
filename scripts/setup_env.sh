#!/bin/bash
# ============================================================
# 训练环境初始化（Ubuntu / CUDA 11.8 示例）
# 用法: bash scripts/setup_env.sh
# ============================================================
set -e

echo "=== GPU ==="
nvidia-smi || echo "no GPU found"

echo "=== Python ==="
python --version

# PaddlePaddle GPU 版（按实例 CUDA 版本选择）
pip install paddlepaddle-gpu==2.6.1.post117 \
  -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# PaddleOCR 2.7 源码（训练脚本与配置均基于该版本）
cd "$(dirname "$0")/.."
if [ ! -d "PaddleOCR" ]; then
  git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git
fi

pip install -r PaddleOCR/requirements.txt
pip install opencv-python-headless shapely pyclipper

python -c "import paddle; print(paddle.__version__, paddle.is_compiled_with_cuda())"
echo "环境准备完成"
