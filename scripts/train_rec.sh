#!/bin/bash
# ============================================================
# 识别头训练
# 用法:
#   bash scripts/train_rec.sh                     # 训练最终版 rec_final_v4
#   bash scripts/train_rec.sh configs/recognition/rec_ablation_baseline.yml
# ============================================================
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:-$ROOT/configs/recognition/rec_final_v4.yml}"
PADDLEOCR="$ROOT/PaddleOCR"
LOG="$ROOT/train_rec_$(basename "$CFG" .yml).log"

cd "$PADDLEOCR"
nohup env PYTHONPATH="$PADDLEOCR" python tools/train.py -c "$CFG" > "$LOG" 2>&1 &
echo "训练已启动，日志: $LOG"
echo "查看进度: tail -f $LOG"
