from pathlib import Path

import torch
from ultralytics import YOLO


def _select_device():
    """YOLO 训练设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return 0  # GPU 索引
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    """YOLOv8 训练入口：从 yolo/data.yaml 训练，输出到 runs_detect/。"""
    project_root = Path(__file__).resolve().parent
    data_yaml = project_root / "yolo" / "data.yaml"

    model = YOLO("yolov8s.pt")
    model.train(
        data=str(data_yaml),
        epochs=80,
        imgsz=960,
        batch=8,
        workers=4,
        device=_select_device(),
        project=str(project_root / "runs_detect"),
        name="yolov8s_mirror_v14_no_earlystop",
        patience=0,  # 关闭EarlyStopping，完整跑完epochs
        pretrained=True,
        optimizer="auto",
        amp=True,
        cache=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
