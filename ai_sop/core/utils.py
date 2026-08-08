"""杂项工具：图像转换、时间、标签映射。"""
import json
import urllib.request
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from PyQt5.QtGui import QImage

from ai_sop.core.constants import ACTION_CN_MAP


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """OpenCV BGR 帧转 QImage 用于 Qt 显示。

    Qt 不直接吃 OpenCV 的 BGR 排列，需先转 RGB 再用 Format_RGB888 包装。
    `.copy()` 必须：QImage 引用的是 numpy buffer，原始 buffer 释放后图像会乱。
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    bytes_per_line = c * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def to_beijing_time_str() -> str:
    """当前北京时间字符串，用于事件日志的 done_time_beijing 字段。"""
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def action_to_cn(label: str) -> str:
    """LSTM 输出标签转中文显示名（找不到则原样返回）。"""
    return ACTION_CN_MAP.get(label, label)


def fine_label_from_row(step_id, _target_id, _event_type):
    """timeline.csv 的 step_id（D1/D2/D3/D4）→ LSTM 标签名。

    用于从外部 CSV 标注构建 SOP 步骤序列（_build_actions 调用）。
    """
    s = str(step_id).strip().upper()
    if s == "D1":
        return "D1_pick_material"
    if s == "D2":
        return "D2_tear_film"
    if s == "D3":
        return "D3_inspect"
    if s == "D4":
        return "D4_place_material"
    return "background"


def http_post_json(url: str, payload: dict, token: str = "", timeout: float = 5.0) -> bool:
    """POST JSON 到 MES / 中央系统；任何失败都返回 False（不抛异常）。

    用于周期完成事件上报：工位信息 + CT + 各步耗时。失败仅记录日志，不影响推理主流程。
    """
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False
