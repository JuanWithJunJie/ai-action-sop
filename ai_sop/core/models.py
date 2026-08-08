"""数据结构 + LSTM 模型定义。"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RuntimeParams:
    """推理运行时参数（由 GUI 控件传入）。"""
    lstm_conf: float       # LSTM 动作确认阈值：期望动作概率 ≥ 该值算 1 次命中
    confirm_frames: int   # 连续命中多少帧才算步骤完成
    show_keypoints: bool  # 是否在画面上绘制 MediaPipe 手部关键点
    save_snapshots: bool  # 步骤完成时是否截图留档
    save_log: bool        # 是否导出 result.json + events.csv


@dataclass
class ActionRuntime:
    """单个 SOP 步骤的运行时状态（每个周期内推进一次）。"""
    index: int                                         # 步骤序号（0-based）
    show: str                                          # UI 显示名（如「第一步：取料」）
    fine_label: str                                    # LSTM 标签名（如 D1_pick_material）
    done: bool = False                                 # 该步骤是否已完成
    hit_count: int = 0                                 # 当前累计命中帧数（连续达标才递增，否则清零）
    expected_conf: float = 0.0                         # 最近一帧期望动作的置信度
    start_time_sec: Optional[float] = None             # 步骤开始时间（视频秒，进入该步骤的第一帧）
    duration_sec: Optional[float] = None               # 步骤耗时 = done_time_sec - start_time_sec
    start_time_beijing: Optional[str] = None           # 步骤开始时间（北京时间）
    done_time_sec: Optional[float] = None              # 完成时间（视频内秒数）
    snapshot_path: Optional[str] = None                # 截图保存路径


class ActionLSTM(torch.nn.Module):
    """动作分类 LSTM：输入 (B, T=48, 126)，输出 (B, num_classes) 的 logits。

    many-to-one 结构：取 LSTM 最后一个时间步的隐状态过 Linear 分类。
    126 维输入 = MediaPipe(2手×21点×3坐标=126)。
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # dropout 仅多层 LSTM 生效
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # out: (B, T, hidden) —— 取最后一个时间步 → Linear → (B, num_classes)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
