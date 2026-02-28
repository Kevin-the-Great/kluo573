#!/usr/bin/env bash
set -euo pipefail

USER_HOME="/hpc2hdd/home/kluo573"

# 你明确点名的 4 个路径
TARGETS=(
  "/hpc2hdd/home/kluo573/VLAadapter/_trash"
  "/hpc2hdd/home/kluo573/VLAadapter/VLA-Adapter"
  "/hpc2hdd/home/kluo573/VLAadapter/gpu_live_monitor.py"
  "/hpc2hdd/home/kluo573/VLAadapter/run_monitor.py"
)

# 额外：把整个 VLAadapter 根目录也清掉（因为你说权重/代码/所有统统删掉）
# 如果你希望只删上面 4 个，而保留 VLAadapter 根目录，请注释掉下一行
TARGET_ROOT="/hpc2hdd/home/kluo573/VLAadapter"

# conda env（按你之前的路径推断）
CONDA_ENV_NAME="vla-adapter"
CONDA_ENV_PATH="/hpc2hdd/home/kluo573/.conda/envs/${CONDA_ENV_NAME}"

echo "============================================================"
echo "[1/5] 将要删除的路径（请仔细核对）："
for p in "${TARGETS[@]}"; do
  echo "  - $p"
done
echo "  - $TARGET_ROOT"
echo
echo "[2/5] 将要删除的 conda 环境："
echo "  - env name: ${CONDA_ENV_NAME}"
echo "  - env path: ${CONDA_ENV_PATH}"
echo "============================================================"
echo

# 轻量检查：确保路径都在你自己的家目录下
for p in "${TARGETS[@]}" "$TARGET_ROOT" "$CONDA_ENV_PATH"; do
  if [[ "$p" != "$USER_HOME"* ]]; then
    echo "ERROR: 发现不在用户目录下的路径：$p"
    echo "为安全起见退出。"
    exit 1
  fi
done

echo "为避免误删，请输入 DELETE（全大写）确认："
read -r CONFIRM
if [[ "$CONFIRM" != "DELETE" ]]; then
  echo "未确认，退出。"
  exit 0
fi

echo
echo "[3/5] 尝试停止可能占用的 python 进程（仅杀你用户的）..."
# 只杀明显相关的进程名，避免误伤
pkill -u "$(id -u)" -f "run_libero_eval.py" || true
pkill -u "$(id -u)" -f "gpu_live_monitor.py" || true
pkill -u "$(id -u)" -f "run_monitor.py" || true

echo "[4/5] 删除 conda 环境（如果 conda 命令存在则优先用 conda，否则直接 rm）..."
if command -v conda >/dev/null 2>&1; then
  # 防止 conda 交互卡住
  conda env remove -n "${CONDA_ENV_NAME}" -y || true
fi
rm -rf "${CONDA_ENV_PATH}"

echo "[5/5] 删除目录/文件..."
for p in "${TARGETS[@]}"; do
  rm -rf "$p"
done
rm -rf "$TARGET_ROOT"

echo
echo "✅ 清理完成。你现在可以重新开作业/镜像从零开始。"
