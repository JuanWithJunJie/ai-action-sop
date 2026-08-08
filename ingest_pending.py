"""误判样本回灌脚本：把 train_data/pending_samples/ 沉淀的样本合并进训练数据。

用法：
    python ingest_pending.py            # 回灌全部待处理样本
    python ingest_pending.py --clean    # 回灌后清空 pending_samples/

流程：
  1. 复制 features/*.npy → train_data/features/
  2. 按 pending_labels.csv 生成 timeline 行（48 帧 @30fps ≈ 1.6s），追加到 train_data/timeline.csv
  3. 自动去重：已存在的同名特征 / 已存在的 timeline 行跳过

回灌完成后执行：python train_lstm.py --epochs 80 重新训练。
"""
import argparse
import csv
import shutil

from ai_sop.core.constants import (
    ACTION_CN_MAP,
    BASE_DIR,
    PENDING_DIR,
    PENDING_FEATURES_DIR,
    PENDING_LABELS_CSV,
)

FEATURES_DST = BASE_DIR / "train_data" / "features"
TIMELINE_CSV = BASE_DIR / "train_data" / "timeline.csv"
WINDOW_FRAMES = 48       # 与 LSTM 输入窗口一致
FPS = 30.0               # 训练脚本读不到视频 fps 时兜底值


def main(clean: bool):
    if not PENDING_LABELS_CSV.exists():
        print("没有待回灌样本（train_data/pending_samples/ 为空）")
        return

    FEATURES_DST.mkdir(parents=True, exist_ok=True)

    # 1. 复制特征文件（同名跳过）
    copied = 0
    for f in PENDING_FEATURES_DIR.glob("*.npy"):
        dst = FEATURES_DST / f.name
        if dst.exists():
            continue
        shutil.copy2(f, dst)
        copied += 1

    # 2. 生成 timeline 行（仅对已回灌成功的特征）
    new_rows = []
    with PENDING_LABELS_CSV.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fname = row.get("file", "")
            label = row.get("action_label", "")
            if not fname or not label or not (FEATURES_DST / fname).exists():
                continue
            stem = fname[: -len(".npy")]
            step_id = label.split("_")[0] if "_" in label else label
            step_cn = ACTION_CN_MAP.get(label, label)
            new_rows.append([stem + ".mp4", step_id, step_cn, 0.0, round(WINDOW_FRAMES / FPS, 3)])

    # 3. 追加到 timeline.csv（video_name 去重）
    existing = set()
    if TIMELINE_CSV.exists():
        with TIMELINE_CSV.open(encoding="utf-8") as f:
            for r in csv.reader(f):
                if r:
                    existing.add(r[0])

    added_rows = 0
    with TIMELINE_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in new_rows:
            if row[0] in existing:
                continue
            writer.writerow(row)
            existing.add(row[0])
            added_rows += 1

    print(f"特征回灌: {copied} 个 | timeline 新增: {added_rows} 行")

    if clean:
        shutil.rmtree(PENDING_DIR)
        print("已清空 pending_samples/")

    print("下一步: python train_lstm.py --epochs 80")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="误判样本回灌训练数据")
    parser.add_argument("--clean", action="store_true", help="回灌后清空 pending_samples/")
    args = parser.parse_args()
    main(args.clean)
