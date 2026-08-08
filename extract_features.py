"""
特征提取脚本：从视频中逐帧提取 126 维特征（MediaPipe 手关键点），
保存为 .npy 文件供 LSTM 训练使用。

用法:
    python extract_features.py                          # 处理 video/ 下所有视频
    python extract_features.py --video video/xxx.mp4    # 处理单个视频
    python extract_features.py --video_dir my_videos    # 处理指定目录

输出:
    train_data/features/<视频名>.npy
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except Exception as e:
    raise ImportError("请先安装 mediapipe: pip install mediapipe") from e

from ai_sop_gui import (
    BASE_DIR,
    build_feature_row,
)

OUTPUT_DIR = BASE_DIR / "train_data" / "features"


def extract_video(video_path: Path, hands):
    """对单个视频逐帧提取 126 维特征，保存为 train_data/features/<视频名>.npy。

    与 ai_sop_gui.py 的 build_feature_row() 共享同一特征定义 —— 训练/推理一致性关键。
    返回输出 .npy 路径；视频无法打开时返回 None。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [跳过] 无法打开: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  处理: {video_path.name}  fps={fps:.1f}  frames={total}")

    feats = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hr = hands.process(rgb)
        hands_out = []
        if hr.multi_hand_landmarks:
            for hidx, hand_lm in enumerate(hr.multi_hand_landmarks):
                lms = [[float(lm.x), float(lm.y), float(lm.z)] for lm in hand_lm.landmark]
                hands_out.append({"hand_index": hidx, "landmarks": lms})

        feat = build_feature_row(hands_out)
        feats.append(feat)

        frame_id += 1
        if frame_id % 200 == 0:
            print(f"    ... {frame_id}/{total} frames")

    cap.release()
    if not feats:
        return None

    arr = np.stack(feats, axis=0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{video_path.stem}.npy"
    np.save(out_path, arr)
    print(f"  保存: {out_path}  shape={arr.shape}")
    return out_path


def main():
    """命令行入口：扫描 video/ 下所有 mp4，逐个调用 extract_video() 提取特征。"""
    parser = argparse.ArgumentParser(description="提取视频特征用于 LSTM 训练")
    parser.add_argument("--video", type=str, help="单个视频路径")
    parser.add_argument("--video_dir", type=str, help="视频目录（默认 video/）")
    args = parser.parse_args()

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if args.video:
        videos = [Path(args.video)]
    else:
        vdir = Path(args.video_dir) if args.video_dir else BASE_DIR / "video"
        videos = sorted(vdir.glob("*.mp4"))

    print(f"共 {len(videos)} 个视频待处理\n")
    for v in videos:
        extract_video(v, hands)

    hands.close()
    print(f"\n完成！特征文件保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
