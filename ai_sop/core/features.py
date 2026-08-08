"""特征拼装 —— 纯 MediaPipe 手部关键点（126 维）。

build_feature_row 是训练/推理共用的核心，任何改动都要两边同步。
"""
import numpy as np


def build_feature_row(hands):
    """将一帧的 MediaPipe 手部关键点拼成 126 维特征向量。

    结构（共 126 维）：2 只手 × 21 关键点 × 3 坐标 (x, y, z)（本身已归一化）。
    缺失的手补 0。该函数同时用于训练（extract_features.py）和推理，需保持一致。
    """
    hand_vec = [0.0] * (2 * 21 * 3)
    for hi, hand in enumerate(hands[:2]):
        lms = hand.get("landmarks", [])
        for li, lm in enumerate(lms[:21]):
            base = hi * 63 + li * 3   # 每只手 63 维起始偏移
            hand_vec[base] = float(lm[0]) if len(lm) > 0 else 0.0
            hand_vec[base + 1] = float(lm[1]) if len(lm) > 1 else 0.0
            hand_vec[base + 2] = float(lm[2]) if len(lm) > 2 else 0.0

    return np.array(hand_vec, dtype=np.float32)
