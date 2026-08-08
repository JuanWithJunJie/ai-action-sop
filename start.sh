#!/usr/bin/env bash
# AI-SOP Vision GUI 一键启动脚本
# 用法：./start.sh
set -euo pipefail

# 切换到脚本所在目录（兼容从任意位置调用）
cd "$(dirname "$0")"

VENV=".venv"
PYBIN="$VENV/bin/python"

# === 1. 检查 venv 是否存在，不存在则提示并退出 ===
if [ ! -x "$PYBIN" ]; then
    echo "❌ 未找到 venv：$VENV"
    echo "   请先初始化环境："
    echo "   python3.11 -m venv $VENV"
    echo "   . $VENV/bin/activate"
    echo "   pip install 'numpy<2' 'opencv-python==4.10.0.84' \\"
    echo "       'ultralytics==8.3.63' 'torch==2.2.2' 'torchvision==0.17.2' \\"
    echo "       'pandas==2.3.3' 'Pillow==10.2.0' 'mediapipe==0.10.9' 'PyQt5==5.15.10'"
    exit 1
fi

# === 2. 检查模型文件是否存在 ===
YOLO_PT="runs_detect/yolov8s_mirror_v14_no_earlystop/weights/best.pt"
LSTM_PT="lstm_runs_fine/best_lstm_fine.pt"
LSTM_CFG="lstm_runs_fine/config.json"
MISSING=()
[ -f "$YOLO_PT" ] || MISSING+=("$YOLO_PT")
[ -f "$LSTM_PT" ] || MISSING+=("$LSTM_PT")
[ -f "$LSTM_CFG" ] || MISSING+=("$LSTM_CFG")

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "⚠️  缺少模型文件："
    for f in "${MISSING[@]}"; do
        echo "   - $f"
    done
    echo "   GUI 仍会启动，但推理会失败。"
    echo
fi

# === 3. 启动 GUI ===
echo "🚀 启动 AI-SOP GUI ..."
exec "$PYBIN" ai_sop_gui.py
