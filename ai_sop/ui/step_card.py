"""StepCard 控件 —— 单个 SOP 步骤卡片。"""
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from ai_sop.core.constants import (
    C_CYAN_12,
    C_DARK,
    C_DIM,
    C_GREEN,
    C_MUTED,
    C_ORANGE,
    C_PRIMARY,
    C_WHITE_6,
    F_MONO,
)


class StepCard(QFrame):
    """单个 SOP 步骤卡片控件（4 步动作左侧那一列）。

    显示：步骤号 + 中文名 + 英文标签 + 状态 + 置信度条 + 完成截图。
    状态机：pending → 进行中 → 完成 / 超时跳过。
    """
    def __init__(self, idx: int, step_id: str, name: str, en_name: str, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.step_id = step_id
        self.name = name
        self.en_name = en_name
        self.status = "pending"
        self.confidence = 0.0
        self._scale = 1.0

        self.setMinimumSize(240, 200)
        self.setMaximumHeight(240)
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.lbl_num = QLabel(step_id)
        self.lbl_num.setFixedSize(28, 28)
        self.lbl_num.setAlignment(Qt.AlignCenter)
        self.lbl_num.setStyleSheet(f"background: {C_WHITE_6}; border-radius: 5px; font-weight: 700; font-size: 13px; color: {C_MUTED};")

        name_box = QVBoxLayout()
        name_box.setSpacing(0)
        self.lbl_name = QLabel(name)
        self.lbl_name.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {C_PRIMARY};")
        self.lbl_en = QLabel(en_name)
        self.lbl_en.setStyleSheet(f"font-size: 10px; color: {C_MUTED}; font-family: '{F_MONO}';")
        name_box.addWidget(self.lbl_name)
        name_box.addWidget(self.lbl_en)

        header.addWidget(self.lbl_num)
        header.addLayout(name_box)
        header.addStretch()
        layout.addLayout(header)

        self.lbl_status = QLabel("未开始")
        self.lbl_status.setStyleSheet(f"font-size: 12px; color: {C_MUTED}; padding-top: 4px;")
        layout.addWidget(self.lbl_status)

        self.conf_widget = QWidget()
        self.conf_widget.setFixedHeight(20)
        self.conf_layout = QHBoxLayout(self.conf_widget)
        self.conf_layout.setContentsMargins(0, 4, 0, 0)
        self.conf_layout.setSpacing(6)
        self.lbl_conf_text = QLabel("")
        self.lbl_conf_text.setStyleSheet(f"font-size: 11px; font-family: '{F_MONO}'; color: {C_MUTED};")
        self.conf_bar = QProgressBar()
        self.conf_bar.setFixedHeight(6)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setStyleSheet(f"QProgressBar{{background:{C_DARK};border-radius:3px;border:none;}}QProgressBar::chunk{{border-radius:3px;background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {C_ORANGE},stop:1 {C_GREEN});}}")
        self.conf_layout.addWidget(self.lbl_conf_text)
        self.conf_layout.addWidget(self.conf_bar)
        self.conf_widget.setVisible(False)
        layout.addWidget(self.conf_widget)

        self.lbl_snapshot = QLabel()
        self.lbl_snapshot.setAlignment(Qt.AlignCenter)
        self.lbl_snapshot.setStyleSheet(f"background: {C_DARK}; border: 1px solid {C_CYAN_12}; border-radius: 6px; color: {C_DIM}; font-size: 10px;")
        self.lbl_snapshot.setMinimumHeight(60)
        self.lbl_snapshot.setText("截图")
        self.lbl_snapshot.setVisible(False)
        layout.addWidget(self.lbl_snapshot, 1)

    def _apply_style(self):
        """按当前 status 切换边框色 + 背景色（pending 灰 / 进行中 橙 / 完成 绿 / 超时 红）。"""
        border_color = "rgba(0,212,255,0.12)"
        bg = "rgba(13,20,31,0.85)"
        if self.status == "active":
            border_color = "rgba(255,170,0,0.35)"
            bg = "rgba(13,20,31,0.95)"
        elif self.status == "done":
            border_color = "rgba(0,255,136,0.3)"
        elif self.status == "timeout":
            border_color = "rgba(255,68,102,0.3)"

        self.setStyleSheet(
            f"StepCard {{ background: {bg}; border: 1px solid {border_color}; border-radius: 10px; }}"
        )

    def set_scale(self, scale: float):
        """按窗口缩放比例调整字体大小（窗口最大化时文字自动变大）。"""
        self._scale = scale
        s = lambda v: max(1, int(v * scale))
        self.lbl_name.setStyleSheet(f"font-size: {s(18)}px; font-weight: 600; color: {C_PRIMARY};")
        self.lbl_en.setStyleSheet(f"font-size: {s(12)}px; color: {C_MUTED}; font-family: '{F_MONO}';")
        self.lbl_num.setStyleSheet(f"background: {C_WHITE_6}; border-radius: 5px; font-weight: 700; font-size: {s(15)}px; color: {C_MUTED};")
        num_size = s(32)
        self.lbl_num.setFixedSize(num_size, num_size)
        self.lbl_status.setStyleSheet(f"font-size: {s(14)}px; color: {C_MUTED}; padding-top: 4px;")
        self.lbl_conf_text.setStyleSheet(f"font-size: {s(13)}px; font-family: '{F_MONO}'; color: {C_MUTED};")
        self.lbl_snapshot.setStyleSheet(f"background: {C_DARK}; border: 1px solid {C_CYAN_12}; border-radius: 6px; color: {C_DIM}; font-size: {s(12)}px;")
        self.set_status(self.status)

    def set_status(self, status: str, info: str = ""):
        """更新步骤状态并切换颜色 / 显示文案（pending/进行中/完成/超时跳过）。"""
        self.status = status
        self._apply_style()

        status_map = {
            "pending": ("未开始", "#5a6b7d"),
            "进行中": ("进行中", "#ffaa00"),
            "active": ("进行中", "#ffaa00"),
            "完成": ("已完成", "#00ff88"),
            "done": ("已完成", "#00ff88"),
            "超时跳过": ("超时跳过", "#ff4466"),
            "timeout": ("超时跳过", "#ff4466"),
        }
        text, color = status_map.get(status, (status, "#5a6b7d"))
        s = lambda v: max(1, int(v * self._scale))
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"font-size: {s(12)}px; color: {color}; padding-top: 4px;")

        num_color = "#5a6b7d"
        num_bg = "rgba(255,255,255,0.06)"
        if status in ("进行中", "active"):
            num_color = "#ffaa00"; num_bg = "rgba(255,170,0,0.15)"
        elif status in ("完成", "done"):
            num_color = "#00ff88"; num_bg = "rgba(0,255,136,0.15)"
        elif status in ("超时跳过", "timeout"):
            num_color = "#ff4466"; num_bg = "rgba(255,68,102,0.15)"
        self.lbl_num.setStyleSheet(f"background: {num_bg}; border-radius: 5px; font-weight: 700; font-size: {s(13)}px; color: {num_color};")

        self.conf_widget.setVisible(status in ("进行中", "active"))
        if status not in ("完成", "done"):
            self.lbl_snapshot.setVisible(False)
            self.lbl_snapshot.setPixmap(QPixmap())

    def set_confidence(self, conf: float):
        """更新置信度进度条和百分比文字（仅「进行中」状态显示）。"""
        self.confidence = conf
        self.conf_bar.setValue(int(conf * 100))
        self.lbl_conf_text.setText(f"置信度: {conf*100:.1f}%")

    def set_snapshot(self, snapshot_path: Optional[str]):
        """加载步骤完成时保存的截图到卡片下半部分。"""
        if not snapshot_path:
            return
        p = Path(snapshot_path)
        if not p.exists():
            return
        pix = QPixmap(str(p))
        if pix.isNull():
            return
        scaled = pix.scaled(self.lbl_snapshot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_snapshot.setPixmap(scaled)
        self.lbl_snapshot.setText("")
        self.lbl_snapshot.setVisible(True)
