"""统一样式消息弹窗 —— 与主窗口的深色工业指挥中心风格保持一致。"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ai_sop.core.constants import C_CYAN, C_ORANGE, C_PRIMARY, C_RED
from ai_sop.theme import COLORS


_KIND_STYLE = {
    "error": {
        "accent": C_RED,
        "glyph": "\u2715",                       # ✕
        "bg": "rgba(255,68,102,0.15)",
        "border": "rgba(255,68,102,0.45)",
        "hover_bg": "rgba(255,68,102,0.28)",
    },
    "warning": {
        "accent": C_ORANGE,
        "glyph": "!",
        "bg": "rgba(255,170,0,0.15)",
        "border": "rgba(255,170,0,0.45)",
        "hover_bg": "rgba(255,170,0,0.28)",
    },
    "info": {
        "accent": C_CYAN,
        "glyph": "i",
        "bg": "rgba(0,212,255,0.15)",
        "border": "rgba(0,212,255,0.45)",
        "hover_bg": "rgba(0,212,255,0.28)",
    },
}


def build_message_dialog(parent, kind: str, title: str, text: str) -> QDialog:
    """构建深色主题消息对话框；kind: error / warning / info。"""
    style = _KIND_STYLE.get(kind, _KIND_STYLE["info"])
    accent = style["accent"]

    dlg = QDialog(parent)
    dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
    dlg.setAttribute(Qt.WA_TranslucentBackground)
    dlg.setModal(True)

    root = QVBoxLayout(dlg)
    root.setContentsMargins(0, 0, 0, 0)

    panel = QFrame(dlg)
    panel.setObjectName("dlgPanel")
    panel.setStyleSheet(
        f"""
        QFrame#dlgPanel {{
            background: {COLORS['bg_primary']};
            border: 1px solid {accent}40;
            border-radius: 14px;
        }}
        QLabel#dlgIcon {{
            background: {style['bg']};
            border: 1px solid {style['border']};
            border-radius: 18px;
            color: {accent};
            font-family: 'Microsoft YaHei';
            font-size: 17px;
            font-weight: 700;
        }}
        QLabel#dlgTitle {{
            background: transparent;
            color: {accent};
            font-family: 'Microsoft YaHei';
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 1px;
        }}
        QLabel#dlgBody {{
            background: transparent;
            color: {C_PRIMARY};
            font-family: 'Microsoft YaHei';
            font-size: 14px;
            line-height: 140%;
        }}
        QPushButton#dlgOkBtn {{
            background: {style['bg']};
            border: 1px solid {style['border']};
            border-radius: 8px;
            padding: 8px 28px;
            color: {accent};
            font-family: 'Microsoft YaHei';
            font-size: 14px;
            font-weight: 600;
        }}
        QPushButton#dlgOkBtn:hover {{
            background: {style['hover_bg']};
            border-color: {accent};
        }}
        QPushButton#dlgOkBtn:pressed {{
            background: {style['bg']};
        }}
        """
    )

    panel_layout = QVBoxLayout(panel)
    panel_layout.setContentsMargins(24, 20, 24, 18)
    panel_layout.setSpacing(14)

    # 顶部：图标徽章 + 标题
    top_row = QHBoxLayout()
    top_row.setSpacing(12)

    icon = QLabel(style["glyph"])
    icon.setObjectName("dlgIcon")
    icon.setFixedSize(36, 36)
    icon.setAlignment(Qt.AlignCenter)

    title_lbl = QLabel(title)
    title_lbl.setObjectName("dlgTitle")

    top_row.addWidget(icon)
    top_row.addWidget(title_lbl, 1)
    top_row.addStretch(0)
    panel_layout.addLayout(top_row)

    # 正文
    body = QLabel(text)
    body.setObjectName("dlgBody")
    body.setWordWrap(True)
    body.setMaximumWidth(420)
    body.setMinimumWidth(300)
    body.setTextInteractionFlags(Qt.TextSelectableByMouse)
    panel_layout.addWidget(body)

    # 底部：确认按钮
    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    ok_btn = QPushButton("确定")
    ok_btn.setObjectName("dlgOkBtn")
    ok_btn.setCursor(Qt.PointingHandCursor)
    ok_btn.setDefault(True)
    ok_btn.setMinimumWidth(96)
    ok_btn.clicked.connect(dlg.accept)
    btn_row.addWidget(ok_btn)
    panel_layout.addLayout(btn_row)

    root.addWidget(panel)
    dlg.adjustSize()
    return dlg


def show_message(parent, kind: str, title: str, text: str) -> None:
    """弹出与主界面风格一致的消息框，并居中显示在父窗口上方。"""
    dlg = build_message_dialog(parent, kind, title, text)
    if parent is not None:
        center = parent.frameGeometry().center()
        dlg.move(center - dlg.rect().center())
    dlg.exec_()
