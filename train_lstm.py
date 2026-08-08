"""
LSTM 训练脚本：用提取好的特征 + timeline 标注训练动作识别模型。

前置条件:
    1. 先运行 extract_features.py 生成 train_data/features/*.npy
    2. 编辑 train_data/timeline.csv 标注每个视频的 D1~D4 起止时间

用法:
    python train_lstm.py                         # 默认参数训练
    python train_lstm.py --epochs 100 --lr 0.001 # 自定义参数
    python train_lstm.py --val_video 20260807-154738.mp4  # 指定验证视频

输出:
    lstm_runs_fine/best_lstm_fine.pt   (模型权重)
    lstm_runs_fine/config.json          (配置，GUI 自动读取)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ai_sop_gui import ActionLSTM, BASE_DIR

FEATURES_DIR = BASE_DIR / "train_data" / "features"
TIMELINE_CSV = BASE_DIR / "train_data" / "timeline.csv"
LSTM_OUTPUT_DIR = BASE_DIR / "lstm_runs_fine"

WINDOW = 48
STRIDE = 1
INPUT_SIZE = 146
HIDDEN_SIZE = 128
NUM_LAYERS = 2

LABEL_MAP = [
    {"id": 0, "label": "background"},
    {"id": 1, "label": "D1_pick_material"},
    {"id": 2, "label": "D2_tear_film"},
    {"id": 3, "label": "D3_inspect"},
    {"id": 4, "label": "D4_place_material"},
]
STEP_TO_LABEL = {
    "D1": "D1_pick_material",
    "D2": "D2_tear_film",
    "D3": "D3_inspect",
    "D4": "D4_place_material",
}
LABEL_TO_ID = {item["label"]: item["id"] for item in LABEL_MAP}
NUM_CLASSES = len(LABEL_MAP)


def load_timeline():
    """读取 train_data/timeline.csv，返回 pandas.DataFrame。文件不存在则抛 FileNotFoundError。"""
    if not TIMELINE_CSV.exists():
        raise FileNotFoundError(f"找不到 timeline 标注文件: {TIMELINE_CSV}\n请先编辑该文件标注 D1~D4 起止时间")
    df = pd.read_csv(TIMELINE_CSV)
    return df


def build_frame_labels(feat_arr, fps, df_timeline, video_name):
    """根据 timeline 为每帧生成标签 id

    返回长度 = len(feat_arr) 的 int64 数组，未标注帧默认 background (id=0)。
    时间区间通过 start_sec/end_sec × fps 转换到帧号，并 clamp 到 [0, len-1]。
    """
    labels = np.zeros(len(feat_arr), dtype=np.int64)

    sdf = df_timeline[df_timeline["video_name"] == video_name]
    if sdf.empty:
        print(f"    [警告] timeline 中没有 {video_name} 的标注，全部标记为 background")

    for _, row in sdf.iterrows():
        step_id = str(row["step_id"]).strip().upper()
        if step_id not in STEP_TO_LABEL:
            print(f"    [跳过] 未知 step_id: {step_id}")
            continue
        label_str = STEP_TO_LABEL[step_id]
        label_id = LABEL_TO_ID[label_str]

        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        start_frame = max(0, min(start_frame, len(labels) - 1))
        end_frame = max(0, min(end_frame, len(labels)))

        labels[start_frame:end_frame] = label_id

    return labels


def make_windows(feat_arr, labels, window=WINDOW, stride=STRIDE):
    """滑动窗口切分：每个窗口 48 帧，标签 = 最后一帧的标签

    stride=1 时样本数 = N - window + 1。
    返回 (X: float32 (N, 48, 146), Y: int64 (N,))；帧数不足返回空数组。
    """
    X, Y = [], []
    n = len(feat_arr)
    if n < window:
        return np.array([]), np.array([])

    for start in range(0, n - window + 1, stride):
        end = start + window
        X.append(feat_arr[start:end])
        Y.append(labels[end - 1])

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


def augment_data(X, Y, noise_std=0.02, n_aug=2):
    """数据增强：高斯噪声 + 特征缩放 + 时间扭曲，原始 + n_aug 倍扩充

    种子固定 (RandomState(42)) 保证可复现。返回合并后的 (aug_X, aug_Y)。
    """
    aug_X, aug_Y = list(X), list(Y)
    rng = np.random.RandomState(42)

    for _ in range(n_aug):
        for i in range(len(X)):
            x = X[i].copy()

            x += rng.normal(0, noise_std, x.shape).astype(np.float32)

            scale = rng.uniform(0.9, 1.1)
            x *= scale

            warp = rng.uniform(0, 1, size=WINDOW)
            warp_idx = np.clip(np.cumsum(warp) / warp.sum() * (WINDOW - 1), 0, WINDOW - 1).astype(int)
            x = x[warp_idx]

            aug_X.append(x)
            aug_Y.append(Y[i])

    return np.array(aug_X, dtype=np.float32), np.array(aug_Y, dtype=np.int64)


def load_data(val_video=None):
    """加载所有特征 .npy + timeline 标注，切窗口后返回 (train_X, train_Y, val_X, val_Y)。

    - 指定 val_video：该视频的全部窗口作为验证集，其余作训练集（推荐，避免数据泄漏）
    - 未指定：从训练集随机切 20% 作验证（提示后仍可用，但相邻窗口高度相关）

    数据增强在 train() 里另调 augment_data() 完成。
    """
    df_timeline = load_timeline()

    feat_files = sorted(FEATURES_DIR.glob("*.npy"))
    if not feat_files:
        raise FileNotFoundError(f"找不到特征文件: {FEATURES_DIR}/*.npy\n请先运行 extract_features.py")

    train_X, train_Y = [], []
    val_X, val_Y = [], []

    for fpath in feat_files:
        video_stem = fpath.stem
        video_name = video_stem + ".mp4"
        feat = np.load(fpath)
        print(f"  加载: {fpath.name}  shape={feat.shape}")

        cap_fps = _get_video_fps(video_name)
        frame_labels = build_frame_labels(feat, cap_fps, df_timeline, video_name)

        X, Y = make_windows(feat, frame_labels)
        if len(X) == 0:
            print(f"    [跳过] 帧数不足 {WINDOW}")
            continue

        if val_video and video_name == val_video:
            val_X.append(X)
            val_Y.append(Y)
            print(f"    -> 验证集: {len(X)} 个窗口")
        else:
            train_X.append(X)
            train_Y.append(Y)
            print(f"    -> 训练集: {len(X)} 个窗口")

    if not train_X:
        raise RuntimeError("训练集为空！请检查特征文件和 timeline 标注")

    train_X = np.concatenate(train_X, axis=0)
    train_Y = np.concatenate(train_Y, axis=0)

    if val_X:
        val_X = np.concatenate(val_X, axis=0)
        val_Y = np.concatenate(val_Y, axis=0)
    else:
        print("  [提示] 无指定验证视频，从训练集中切出 20% 作为验证")
        idx = np.random.permutation(len(train_X))
        val_count = max(1, len(idx) // 5)
        val_idx = idx[:val_count]
        train_idx = idx[val_count:]
        val_X, val_Y = train_X[val_idx], train_Y[val_idx]
        train_X, train_Y = train_X[train_idx], train_Y[train_idx]

    print(f"\n训练集: {train_X.shape}  验证集: {val_X.shape}")
    print(f"训练标签分布: {dict(zip(*np.unique(train_Y, return_counts=True)))}")
    print(f"验证标签分布: {dict(zip(*np.unique(val_Y, return_counts=True)))}")

    return train_X, train_Y, val_X, val_Y


def _get_video_fps(video_name):
    """从 video/ 或 train_data/video/ 目录读取视频 fps，找不到返回 30.0 兜底。"""
    import cv2
    for vdir in [BASE_DIR / "video", BASE_DIR / "train_data" / "video"]:
        vpath = vdir / video_name
        if vpath.exists():
            cap = cv2.VideoCapture(str(vpath))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            return fps
    return 30.0


def train(epochs, lr, batch_size, val_video):
    """训练 LSTM：加载特征 → 切窗口 → 数据增强 → 类别权重 → Adam + ReduceLROnPlateau。

    训练完成后保存 best_lstm_fine.pt + config.json 到 lstm_runs_fine/，GUI 直接读取。
    最佳验证准确率的 checkpoint 才会被保存（防止过拟合）。
    """
    # 推理设备：CUDA > MPS(Apple Silicon) > CPU
    device = (
        "cuda:0" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"设备: {device}")

    train_X, train_Y, val_X, val_Y = load_data(val_video)

    print(f"\n增强前: {train_X.shape}")
    train_X, train_Y = augment_data(train_X, train_Y, noise_std=0.02, n_aug=2)
    print(f"增强后: {train_X.shape}")
    print(f"增强后标签分布: {dict(zip(*np.unique(train_Y, return_counts=True)))}")

    train_ds = TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_Y)
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_X), torch.from_numpy(val_Y)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ActionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, dropout=0.4).to(device)

    class_counts = np.bincount(train_Y, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = np.where(class_counts > 0, len(train_Y) / (NUM_CLASSES * class_counts), 1.0)
    print(f"类别权重: {dict(zip([item['label'] for item in LABEL_MAP], np.round(class_weights, 3)))}")
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_acc = 0.0
    LSTM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = LSTM_OUTPUT_DIR / "best_lstm_fine.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)

        val_acc = correct / total if total else 0.0
        val_loss = val_loss / total if total else 0.0
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            star = " *"
        else:
            star = ""

        print(f"Epoch {epoch:3d}/{epochs}  loss={total_loss/len(train_ds):.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{star}")

    config = {
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_classes": NUM_CLASSES,
        "label_map": LABEL_MAP,
        "val_videos": [val_video] if val_video else [],
    }
    config_path = LSTM_OUTPUT_DIR / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型已保存: {best_model_path}")
    print(f"配置已保存: {config_path}")
    print("现在可以直接运行 ai_sop_gui.py 使用新模型")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 LSTM 动作识别模型")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_video", type=str, default=None,
                        help="验证视频文件名（如 20260807-154738.mp4）")
    args = parser.parse_args()

    train(args.epochs, args.lr, args.batch_size, args.val_video)
