#!/bin/bash
# ============================================================
# 检测头训练 / 评估
# 用法:
#   bash scripts/train_det.sh            # 训练最终版 det_final_v3
#   bash scripts/train_det.sh eval       # 仅评估
# 数据目录需与 config 中的 data_dir / label_file_list 保持一致
# ============================================================
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$ROOT/configs/detection/det_final_v3.yml"
PADDLEOCR="$ROOT/PaddleOCR"

cd "$PADDLEOCR"

if [ "$1" = "eval" ]; then
  # 评估前把输出目录里 config.yml 的 resume 路径改成本实验，
  # 否则会加载到 baseline 的权重（常见坑）
  sed -i 's#exp_baseline_v2#det_final_v3#g' output/det_final_v3/config.yml
  PYTHONPATH="$PADDLEOCR" python tools/eval.py -c "$CFG"
else
  nohup env PYTHONPATH="$PADDLEOCR" python tools/train.py -c "$CFG" \
    > "$ROOT/train_det.log" 2>&1 &
  echo "训练已启动，日志: $ROOT/train_det.log"
  echo "查看进度: tail -f $ROOT/train_det.log"
fi
