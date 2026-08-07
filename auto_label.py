"""
自动伪标注脚本：用已标注的 0~18 秒数据作为模板，
通过特征相似度（KNN）自动给未标注部分打标签。

原理：
  1. 从 timeline.csv 已标注区间提取每个动作的 48 帧窗口模板
  2. 对未标注区间的每个 48 帧窗口，找最相似的模板
  3. 相似度超过阈值则分配该标签，否则标为 background
  4. 输出扩展后的 timeline 到 train_data/timeline_auto.csv

用法:
    python auto_label.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_sop_gui import BASE_DIR

FEATURES_PATH = BASE_DIR / "train_data" / "features" / "20260807-154738.npy"
TIMELINE_CSV = BASE_DIR / "train_data" / "timeline.csv"
OUTPUT_CSV = BASE_DIR / "train_data" / "timeline_auto.csv"

WINDOW = 48
STRIDE = 1

STEP_TO_LABEL = {
    "D1": "D1_pick_material",
    "D2": "D2_tear_film",
    "D3": "D3_inspect",
    "D4": "D4_place_material",
}
LABEL_TO_ID = {v: k for k, v in STEP_TO_LABEL.items()}


def load_data():
    feat = np.load(FEATURES_PATH)
    df = pd.read_csv(TIMELINE_CSV)
    fps = 30.0
    return feat, df, fps


def get_frame_labels(feat, df, fps, video_name):
    """根据 timeline 给每帧打标签，未标注的为 None"""
    labels = [None] * len(feat)
    sdf = df[df["video_name"] == video_name]
    for _, row in sdf.iterrows():
        step_id = str(row["step_id"]).strip().upper()
        if step_id not in STEP_TO_LABEL:
            continue
        start_f = int(float(row["start_sec"]) * fps)
        end_f = int(float(row["end_sec"]) * fps)
        for f in range(max(0, start_f), min(end_f, len(feat))):
            labels[f] = step_id
    return labels


def build_templates(feat, frame_labels, window=WINDOW):
    """从已标注区间提取各动作的 48 帧窗口模板"""
    templates = {k: [] for k in STEP_TO_LABEL}
    for start in range(0, len(feat) - window + 1, STRIDE):
        end = start + window
        label = frame_labels[end - 1]
        if label is not None:
            templates[label].append(feat[start:end])
    result = {}
    for label, windows in templates.items():
        if windows:
            arr = np.stack(windows, axis=0)
            result[label] = arr
            print(f"  模板 {label}: {arr.shape}")
    return result


def cosine_sim(a, b):
    """计算两个窗口序列的余弦相似度"""
    fa = a.flatten()
    fb = b.flatten()
    na = np.linalg.norm(fa) + 1e-8
    nb = np.linalg.norm(fb) + 1e-8
    return float(np.dot(fa, fb) / (na * nb))


def auto_label(feat, frame_labels, templates, window=WINDOW, stride=STRIDE):
    """对未标注帧的窗口用 KNN 分配标签"""
    threshold = 0.75
    labeled_ranges = []

    for start in range(0, len(feat) - window + 1, stride):
        end = start + window
        if frame_labels[end - 1] is not None:
            continue

        seg = feat[start:end]
        best_label = None
        best_sim = 0.0
        for label, template_arr in templates.items():
            for i in range(len(template_arr)):
                sim = cosine_sim(seg, template_arr[i])
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

        if best_label and best_sim >= threshold:
            center_frame = end - 1
            labeled_ranges.append((center_frame, best_label, best_sim))

    print(f"  自动标注窗口数: {len(labeled_ranges)}")
    if not labeled_ranges:
        return []

    labeled_ranges.sort(key=lambda x: x[0])

    merged = []
    current_label = None
    range_start = None

    for frame, label, sim in labeled_ranges:
        if label == current_label:
            continue
        if current_label is not None:
            merged.append((range_start, frame - 1, current_label))
        current_label = label
        range_start = frame
    if current_label is not None:
        merged.append((range_start, labeled_ranges[-1][0], current_label))

    return merged


def main():
    print("=== 自动伪标注 ===")
    feat, df, fps = load_data()
    print(f"特征: {feat.shape}, fps={fps}")

    video_name = "20260807-154738.mp4"
    frame_labels = get_frame_labels(feat, df, fps, video_name)

    annotated = sum(1 for l in frame_labels if l is not None)
    print(f"已标注帧: {annotated}/{len(feat)} ({annotated/len(feat)*100:.1f}%)")

    print("\n构建动作模板:")
    templates = build_templates(feat, frame_labels)
    if not templates:
        print("没有已标注数据作为模板!")
        return

    print("\n自动标注未标注区间:")
    ranges = auto_label(feat, frame_labels, templates)

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "video_name": row["video_name"],
            "step_id": row["step_id"],
            "step_cn": row["step_cn"],
            "start_sec": row["start_sec"],
            "end_sec": row["end_sec"],
        })

    for start_f, end_f, label in ranges:
        start_sec = round(start_f / fps, 3)
        end_sec = round(end_f / fps, 3)
        if end_sec - start_sec < 0.2:
            continue
        rows.append({
            "video_name": video_name,
            "step_id": label,
            "step_cn": STEP_TO_LABEL.get(label, label).split("_", 1)[-1] if "_" in STEP_TO_LABEL.get(label, "") else label,
            "start_sec": start_sec,
            "end_sec": end_sec,
        })

    rows.sort(key=lambda r: r["start_sec"])
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n输出: {OUTPUT_CSV}")
    print(f"总标注段: {len(rows)} (原 {len(df)}, 自动新增 {len(rows)-len(df)})")
    print("\n自动标注的新增段:")
    for r in rows[len(df):]:
        print(f"  {r['step_id']} {r['step_cn']}: {r['start_sec']:.1f}~{r['end_sec']:.1f}s")


if __name__ == "__main__":
    main()
