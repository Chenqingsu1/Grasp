#!/bin/bash

echo "=== Isaac Gym GPU Pipeline 修复脚本启动 ==="

# 确保 libcuda.so.1 存在
if [ ! -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    echo "[错误] 未找到 libcuda.so.1"
    exit 1
fi

echo "[步骤1] 清理 torch 扩展缓存"
rm -rf ~/.cache/torch_extensions

echo "[步骤2] 设置环境变量"
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/home/zhangboran/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/include
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
export CUDA_LAUNCH_BLOCKING=1
export GYM_DEBUG=1

echo "[步骤3] 重建 gymtorch 扩展模块"
python -c "import torch; import isaacgym; from isaacgym import gymtorch; print('gymtorch loaded')"

echo "=== 修复完成，请重新运行 oneobj.sh 或训练脚本 ==="
