#!/bin/bash
set -euo pipefail

# ================= 1. 核心路径定义 =================
PROJECT_ROOT="/hpc2hdd/home/kluo573/vlaadapter/VLA-Adapter"

# ================= 2. 接收命令行参数 =================
TASK_SUITE="${1:-libero_spatial}"
if [ -z "${1:-}" ]; then
    echo "⚠️  未指定任务，默认运行: libero_spatial"
fi

# ================= 3. 自动匹配模型权重 =================
case "$TASK_SUITE" in
    "libero_spatial") MODEL_DIR="LIBERO-Spatial-Pro-8bit" ;;
    "libero_object")  MODEL_DIR="LIBERO-Object-Pro-8bit" ;;
    "libero_goal")    MODEL_DIR="LIBERO-Goal-Pro-8bit" ;;
    "libero_10")      MODEL_DIR="LIBERO-Long-Pro-8bit" ;;
    *) echo "❌ 错误：未知的任务名: $TASK_SUITE"; exit 1 ;;
esac
CHECKPOINT="${PROJECT_ROOT}/outputs/${MODEL_DIR}"

# ================= 4. 运行前准备 =================
if [ ! -d "$PROJECT_ROOT" ]; then echo "❌ 错误：找不到项目根目录"; exit 1; fi
cd "$PROJECT_ROOT"

# ================= 5. 环境配置 =================
# 【关键】强制单卡运行，解决 device map 冲突
export CUDA_VISIBLE_DEVICES=0

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
# 防止未定义时报错
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CONDA_PREFIX:-}/lib/libOSMesa.so"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/LIBERO"

# ================= 6. 运行评测 =================
echo "========================================"
echo "Job started at $(date)"
echo "Mode: OSMesa (CPU Rendering)"
echo "GPU: Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Task Suite: $TASK_SUITE"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"

python -u "${PROJECT_ROOT}/experiments/robot/libero/run_libero_eval.py" \
    --model_family "openvla" \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --use_pro_version True \
    --load_in_8bit True \
    --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name "$TASK_SUITE"

echo "Job finished at $(date)"

