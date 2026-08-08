"""特征拼装 + YOLO 可视化绘制。

build_feature_row 是训练/推理共用的核心，任何改动都要两边同步。
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ai_sop.core.constants import (
    CLASS_NAMES,
    YOLO_BOX_THICKNESS,
    YOLO_TEXT_SIZE,
    YOLO_TEXT_OFFSET_Y,
    YOLO_CN_MAP,
)


def build_feature_row(detections, hands, frame_w, frame_h):
    """将一帧的 YOLO 检测 + MediaPipe 关键点拼成 146 维特征向量。

    结构（共 146 维）：
      - YOLO 部分（20 维）：4 个物体类别 × 5 值 [conf, cx, cy, w, h]（归一化到 [0,1]）
      - MediaPipe 部分（126 维）：2 只手 × 21 关键点 × 3 坐标 (x, y, z)（本身已归一化）
    缺失类别/手 用 0 填充。该函数同时用于训练（extract_features.py）和推理，需保持一致。
    """
    # === YOLO 部分：每类取「置信度最高的那一个检测框」的 5 值 ===
    det_feat = {k: [0.0, 0.0, 0.0, 0.0, 0.0] for k in CLASS_NAMES}
    for d in detections:
        cls_name = d.get("cls_name")
        if cls_name not in det_feat:
            continue
        conf = float(d.get("conf", 0.0))
        x1, y1, x2, y2 = d.get("xyxy", [0, 0, 0, 0])
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        cx = float(x1) + w / 2.0
        cy = float(y1) + h / 2.0

        fw = max(float(frame_w), 1.0)
        fh = max(float(frame_h), 1.0)
        # 同一类有多个检测框时只保留置信度最高的（取 max）
        if conf > det_feat[cls_name][0]:
            det_feat[cls_name] = [conf, cx / fw, cy / fh, w / fw, h / fh]

    # 按固定顺序铺平成 4×5 = 20 维
    det_vec = []
    for cname in CLASS_NAMES:
        det_vec.extend(det_feat[cname])

    # === MediaPipe 部分：2 手 × 21 点 × 3 坐标 = 126 维，缺失手补 0 ===
    hand_vec = [0.0] * (2 * 21 * 3)
    for hi, hand in enumerate(hands[:2]):
        lms = hand.get("landmarks", [])
        for li, lm in enumerate(lms[:21]):
            base = hi * 63 + li * 3   # 每只手 63 维起始偏移
            hand_vec[base] = float(lm[0]) if len(lm) > 0 else 0.0
            hand_vec[base + 1] = float(lm[1]) if len(lm) > 1 else 0.0
            hand_vec[base + 2] = float(lm[2]) if len(lm) > 2 else 0.0

    # 最终拼成 20 + 126 = 146 维的 float32 向量
    return np.array(det_vec + hand_vec, dtype=np.float32)


_FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Songti.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    # Linux
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _load_font(font_size: int = YOLO_TEXT_SIZE) -> ImageFont.FreeTypeFont:
    """加载中文字体（跨平台候选），找不到则用 PIL 默认。

    cv2.putText 不支持中文，所以画 YOLO 框标签时要切到 PIL 绘制。
    """
    for fp in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_yolo_boxes_cn(frame: np.ndarray, boxes, names_map: dict):
    """在帧上画 YOLO 检测框 + 中文类别标签。

    返回新的 vis 帧（不修改原 frame）。流程：
      BGR → PIL → 画矩形+文字 → 回 BGR。
    切到 PIL 是为了用 TrueType 字体画中文（cv2.putText 只支持 ASCII）。
    """
    vis = frame.copy()
    if boxes is None:
        return vis

    pil_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(YOLO_TEXT_SIZE)

    for box in boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_name = names_map.get(cls_id, str(cls_id))
        cls_cn = YOLO_CN_MAP.get(cls_name, cls_name)   # 类别名 → 中文
        draw.rectangle([x1, y1, x2, y2], outline=(60, 220, 120), width=YOLO_BOX_THICKNESS)
        draw.text((x1, max(2, y1 - YOLO_TEXT_OFFSET_Y)), cls_cn, fill=(60, 220, 120), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
