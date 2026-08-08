"""EventItem 控件 —— 事件日志单行。"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ai_sop.core.constants import C_CYAN, C_MUTED, C_PRIMARY, F_MONO


class EventItem(QWidget):
    """事件日志单行控件（底部时间线列表的一项）。

    显示：✓ / ◉ / ! 图标 + 步骤名 + 时间 + 周期编号（C1/C2…）。
    event_type：done=步骤完成，cycle=周期完成（CT），其他=超时跳过。
    """
    def __init__(self, step_name: str, event_type: str, cycle: int, time_str: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        if event_type == "done":
            icon_text, icon_bg, icon_color = "✓", "rgba(0,255,136,0.15)", "#00ff88"
        elif event_type == "cycle":
            icon_text, icon_bg, icon_color = "◉", "rgba(0,212,255,0.15)", "#00d4ff"
        else:
            icon_text, icon_bg, icon_color = "!", "rgba(255,68,102,0.15)", "#ff4466"

        icon = QLabel(icon_text)
        icon.setFixedSize(20, 20)
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet(
            f"background: {icon_bg}; border-radius: 4px; font-size: 11px; font-weight: 700; "
            f"color: {icon_color};"
        )
        layout.addWidget(icon)

        info = QVBoxLayout()
        info.setSpacing(0)
        lbl_action = QLabel(step_name)
        lbl_action.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {C_PRIMARY};")
        lbl_time = QLabel(time_str)
        lbl_time.setStyleSheet(f"font-size: 10px; color: {C_MUTED}; font-family: '{F_MONO}';")
        info.addWidget(lbl_action)
        info.addWidget(lbl_time)
        layout.addLayout(info)
        layout.addStretch()

        lbl_cycle = QLabel(f"C{cycle}")
        lbl_cycle.setStyleSheet(f"font-size: 10px; color: {C_CYAN}; font-family: '{F_MONO}';")
        layout.addWidget(lbl_cycle)
