# -*- coding: utf-8 -*-
"""
===============================================================================
  Chinese OCR · Flask 推理服务
===============================================================================
  加载自训练的检测头（DBNet v3）与识别头（CRNN），对外提供 HTTP 接口。

  请求链路：客户端 → POST /ocr → 检测头找文字框 → 识别头逐框识别 → 返回结果

  启动：
      python inference/app.py

  依赖：
      pip install -r requirements.txt
      并把 PaddleOCR（release/2.7）源码克隆到项目根目录下的 PaddleOCR/ 目录

  接口：
      GET  /health       健康检查
      GET  /model_info   模型信息与验证集指标
      POST /ocr          端到端识别（multipart 字段 image，或 JSON 的 base64 image）
===============================================================================
"""

# ---------------------------------------------------------------------------
# 1. 路径配置（必须在导入 ppocr 之前完成）
# ---------------------------------------------------------------------------
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))      # inference/
PROJECT = os.path.dirname(BASE)                        # 项目根目录
POCR_DIR = os.path.join(PROJECT, "PaddleOCR")          # PaddleOCR 2.7 源码
MODEL_D = os.path.join(PROJECT, "weights")             # 自训练权重目录
CFG_D = os.path.join(PROJECT, "configs")               # 训练时使用的配置文件

sys.path.insert(0, POCR_DIR)

# ---------------------------------------------------------------------------
# 2. 导入依赖
# ---------------------------------------------------------------------------
import base64
import time

import cv2
import numpy as np
import paddle
import yaml
from flask import Flask, jsonify, request

from ppocr.modeling.architectures import build_model   # 按 config 搭建网络结构
from ppocr.postprocess import build_post_process       # 后处理工厂
from ppocr.utils.save_load import load_model           # 载入 .pdparams 权重

# 中文字典：6623 个常用汉字 + 英文 + 数字
# 识别头 CTC 输出维度为 6625（6624 个字符 + 1 个 blank 占位符）
DICT = os.path.join(POCR_DIR, "ppocr", "utils", "ppocr_keys_v1.txt")

DET_WT = os.path.join(MODEL_D, "det_v3_best.pdparams")
REC_WT = os.path.join(MODEL_D, "rec_exp_v4_best.pdparams")

DET_CFG = os.path.join(CFG_D, "detection", "det_final_v3.yml")
REC_CFG = os.path.join(CFG_D, "recognition", "rec_final_v4.yml")

# ---------------------------------------------------------------------------
# 3. Paddle 初始化
# ---------------------------------------------------------------------------
USE_GPU = paddle.is_compiled_with_cuda()
paddle.device.set_device("gpu" if USE_GPU else "cpu")
paddle.disable_static()

app = Flask(__name__)


# ---------------------------------------------------------------------------
# 4. 配置裁剪：训练配置 → 推理配置
# ---------------------------------------------------------------------------
def make_cfg(cfg_path, model_path, is_rec=False):
    """训练配置里含 Optimizer/Loss/Train/Eval 等推理不需要的字段，需要剔除。"""
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    cfg.setdefault("Global", {})
    cfg["Global"]["pretrained_model"] = model_path
    cfg["Global"]["load_static_weights"] = False
    cfg["Global"].pop("checkpoints", None)
    for k in ["Optimizer", "Loss", "Metric", "Train", "Eval"]:
        cfg.pop(k, None)
    cfg["PreProcess"] = {}

    if is_rec:
        # 识别头必须显式补 out_channels：6624 个字符 + 1 个 blank
        cfg["Architecture"]["Head"]["out_channels"] = 6625
    return cfg


# ---------------------------------------------------------------------------
# 5. 模型加载（启动时执行一次）
# ---------------------------------------------------------------------------
def load_all():
    print("loading self-trained models ...")
    det_cfg = make_cfg(DET_CFG, DET_WT)
    det_model = build_model(det_cfg["Architecture"])
    load_model(det_cfg, det_model)          # 注意：config 在前，model 在后
    det_model.eval()

    rec_cfg = make_cfg(REC_CFG, REC_WT, is_rec=True)
    rec_model = build_model(rec_cfg["Architecture"])
    load_model(rec_cfg, rec_model)
    rec_model.eval()

    det_post = build_post_process({
        "name": "DBPostProcess",
        "thresh": 0.3,            # 概率图二值化阈值
        "box_thresh": 0.5,        # 框置信度阈值
        "max_candidates": 1000,   # 最多保留框数
        "unclip_ratio": 1.6,      # 框外扩比例
    })
    rec_post = build_post_process({
        "name": "CTCLabelDecode",
        "character_dict_path": DICT,
        "use_space_char": True,
    })
    return det_model, rec_model, det_post, rec_post


DET_MODEL, REC_MODEL, DET_POST, REC_POST = load_all()


# ---------------------------------------------------------------------------
# 6. 预处理
# ---------------------------------------------------------------------------
def det_prep(img):
    """检测头：缩放到 32 的倍数 + ImageNet 归一化，并返回还原坐标所需的 shape_list。"""
    h, w = img.shape[:2]
    new_h = max(32, (h // 32) * 32)
    new_w = max(32, (w // 32) * 32)
    img_r = cv2.resize(img, (new_w, new_h))
    img_r = (img_r.astype("float32") / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return (img_r.transpose(2, 0, 1)[np.newaxis].astype("float32"),
            [h, w, new_h / h, new_w / w])


def rec_prep(crop):
    """识别头：高固定 32px，宽等比例缩放且不超过 320px（与训练 image_shape 一致）。"""
    h, w = crop.shape[:2]
    new_w = min(320, max(1, int(w * 32.0 / h)))
    crop = cv2.resize(crop, (new_w, 32))
    crop = (crop.astype("float32") / 255 - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]
    return crop.transpose(2, 0, 1)[np.newaxis].astype("float32")


def load_font(size=24):
    """OpenCV 的 putText 不支持中文，标注中文需借助 PIL 字体。"""
    from PIL import ImageFont
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT = load_font()


# ---------------------------------------------------------------------------
# 7. HTTP 接口
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "gpu": USE_GPU})


@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "det_model": "DBNet_v3 (MobileNetV3-Large x0.5 + RSEFPN)",
        "det_weights": os.path.basename(DET_WT),
        "rec_model": "CRNN (ResNet34 + RNN SequenceEncoder hidden=256 + CTC)",
        "rec_weights": os.path.basename(REC_WT),
        "dict_chars": 6624,
        "det_f1_val": 0.656,
        "rec_val_acc": 0.7738,
        "device": "GPU" if USE_GPU else "CPU",
    })


@app.route("/ocr", methods=["POST"])
def ocr_api():
    t0 = time.time()
    try:
        # 1) 接收图片：multipart 或 base64
        if request.content_type and "multipart" in request.content_type:
            if "image" not in request.files:
                return jsonify({"error": "no image field"}), 400
            img_bytes = request.files["image"].read()
        else:
            data = request.get_json(force=True)
            img_b64 = data.get("image", "")
            if not img_b64:
                return jsonify({"error": "missing image"}), 400
            img_bytes = base64.b64decode(img_b64)

        # 2) 解码
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "image decode fail"}), 400

        h_orig, w_orig = img.shape[:2]

        # 3) 检测
        with paddle.no_grad():
            prep_tensor, shape_list = det_prep(img)
            dt = DET_MODEL(paddle.to_tensor(prep_tensor))
        result = DET_POST(dt, [shape_list])
        polys = result[0]["points"] if result else []

        detections = []
        annotated = img.copy()

        from PIL import Image as PILImage, ImageDraw

        # 4) 逐框裁剪 + 识别
        for poly in polys:
            poly = np.array(poly, dtype=np.int32)
            x1, y1 = poly.min(axis=0)
            x2, y2 = poly.max(axis=0)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                continue

            text, conf = "", 0.0
            try:
                with paddle.no_grad():
                    rc_out = REC_MODEL(paddle.to_tensor(rec_prep(crop)))
                # 注意：不要传 lengths 参数，否则会报 unhashable ndarray
                rp = REC_POST(rc_out, [])
                if rp and rp[0]:
                    tup = rp[0][0]
                    text = tup[0] if tup else ""
                    conf = float(tup[1]) if len(tup) > 1 else 0.0
            except Exception:
                pass

            detections.append({
                "points": poly.flatten().tolist(),
                "transcription": text,
                "confidence": round(conf, 4),
            })

            # 5) 画框与中文标注
            cv2.polylines(annotated, [poly], True, (0, 255, 0), 2)
            pil_img = PILImage.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text(tuple(poly[0]), text[:15], font=FONT, fill=(255, 0, 0))
            annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ann_b64 = base64.b64encode(buf.tobytes()).decode()

        return jsonify({
            "detections": detections,
            "annotated_image": ann_b64,
            "count": len(detections),
            "time_seconds": round(time.time() - t0, 2),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
