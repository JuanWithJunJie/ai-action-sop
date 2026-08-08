"""MainWindow —— 主窗口，集成所有 UI 区域。"""
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ai_sop.core.constants import (
    BASE_DIR,
    CONFIRM_FRAMES_FIXED,
    F_DISPLAY,
    F_MONO,
    C_BG_PRIMARY,
    C_BG_VIDEO_DARK,
    C_BG_VIDEO_DIM,
    C_BLACK_85,
    C_CYAN,
    C_CYAN_12,
    C_CYAN_DK,
    C_DARK,
    C_DARK_TEXT,
    C_GREEN,
    C_GREEN_10,
    C_GREEN_30,
    C_MUTED,
    C_ORANGE,
    C_ORANGE_10,
    C_ORANGE_30,
    C_PANEL_BG,
    C_PRIMARY,
    C_RED,
    C_RED_30,
)
from ai_sop.core.models import RuntimeParams
from ai_sop.core.utils import bgr_to_qimage
from ai_sop.core.worker import InferenceWorker
from ai_sop.theme import COLORS
from ai_sop.ui.event_item import EventItem
from ai_sop.ui.dialogs import show_message
from ai_sop.ui.stat_ring import StatRing
from ai_sop.ui.step_card import StepCard


class MainWindow(QMainWindow):
    """主窗口：无边框深色工业风格，集成 4 步骤卡片 + 视频区 + 右侧统计 + 底部事件流。

    主要职责：
      1. 加载视频 → 创建 InferenceWorker → 连接信号到槽函数
      2. 接收 sig_frame / sig_status / sig_action 更新 UI
      3. 管理统计计数（pass_count / skip_count）和事件列表
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-SOP Vision | 手机制造智能SOP系统")
        self.setWindowFlags(Qt.FramelessWindowHint)   # 无边框窗口，自己实现拖动/最大化
        self.resize(1920, 1080)
        self.video_source: Optional[str] = None   # 视频源：文件路径 / 摄像头索引(如 "0") / RTSP 地址
        self.worker: Optional[InferenceWorker] = None
        self.last_run_dir: Optional[Path] = None
        self.pass_count = 0     # 完成事件数
        self.skip_count = 0     # 超时跳过事件数

        self.cards: List[StepCard] = []
        self.event_items: List[EventItem] = []
        self._font_scale = 1.0
        self._apply_theme()
        self._build_ui()
        self._apply_font_scale(1.0)

        # 1 秒一次的时钟刷新（标题栏显示当前时间）
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

    def _apply_theme(self):
        """设置全局 QSS 样式表（深色工业指挥中心风格）。"""
        self.setStyleSheet(f"""
            QMainWindow {{ background: {COLORS['bg_app']}; }}
            QWidget {{ color: {C_PRIMARY}; font-family: 'Microsoft YaHei'; font-size: 15px; }}
            QLabel {{ background: transparent; color: {C_PRIMARY}; }}
            QFrame#header {{ background: {C_BG_PRIMARY}; border-bottom: 1px solid {C_CYAN_12}; }}
            QFrame#footer {{ background: {C_BG_PRIMARY}; border-top: 1px solid {C_CYAN_12}; }}
            QFrame#panel {{ background: {C_PANEL_BG}; border: 1px solid {C_CYAN_12}; border-radius: 12px; }}
            QFrame#videoSection {{ background: {C_DARK_TEXT}; border: 1px solid {C_CYAN_12}; border-radius: 12px; }}
            QFrame#titleBar {{ background: {C_BG_PRIMARY}; border-bottom: 1px solid {COLORS['overlay_cyan_35']}; }}
            QLineEdit {{
                background: {C_DARK}; border: 1px solid {C_CYAN_12};
                border-radius: 4px; padding: 4px 8px; color: {C_PRIMARY};
                font-family: 'Consolas'; font-size: 14px; max-width: 60px;
            }}
            QPushButton {{
                background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.35);
                border-radius: 8px; padding: 8px 20px; color: {C_CYAN};
                font-size: 15px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {COLORS['overlay_cyan_35']}; border-color: {C_CYAN}; }}
            QPushButton:pressed {{ background: rgba(0,212,255,0.05); }}
            QPushButton:disabled {{ color: #2a3a4a; border-color: rgba(0,212,255,0.08); background: rgba(0,0,0,0.2); }}
            QPushButton#closeBtn {{ background: rgba(255,68,102,0.08); border-color: {C_RED_30}; color: {C_RED}; border-radius: 0px; border: none; padding: 0px; font-size: 18px; }}
            QPushButton#closeBtn:hover {{ background: rgba(255,68,102,0.2); }}
            QPushButton#winBtn {{ background: transparent; border: none; color: {C_MUTED}; border-radius: 0px; padding: 0px; font-size: 18px; }}
            QPushButton#winBtn:hover {{ background: rgba(0,212,255,0.08); color: {C_CYAN}; }}
            QCheckBox {{ color: {C_PRIMARY}; font-size: 14px; spacing: 6px; }}
            QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 3px; border: 1px solid rgba(0,212,255,0.35); background: {C_DARK}; }}
            QCheckBox::indicator:checked {{ background: {C_CYAN}; border-color: {C_CYAN}; }}
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{ width: 4px; background: transparent; }}
            QScrollBar::handle:vertical {{ background: rgba(0,212,255,0.35); border-radius: 2px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

    def _build_ui(self):
        """构建整个主窗口的控件树。具体子区域见各 _build_* 子方法。"""
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._build_header(root)
        self._build_main_content(root)
        self._build_footer(root)
        self._connect_signals()

    def _build_header(self, root):
        """顶部标题栏：Logo + 系统名 + 工厂信息 + 状态指示 + 时钟 + 窗口控制按钮。"""
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(56)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(24, 8, 24, 8)
        h_layout.setSpacing(20)

        logo = QLabel("AI")
        logo.setFixedSize(32, 32)
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(f"background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 {C_CYAN},stop:1 {C_CYAN_DK}); border-radius: 6px; font-weight: 700; font-size: 16px; color: {C_DARK_TEXT};")
        h_layout.addWidget(logo)

        title = QLabel("SOP·VISION")
        title.setStyleSheet(f"font-size: 18px; font-weight: 700; letter-spacing: 1px; color: {C_PRIMARY};")
        title_sp = QLabel("手机制造智能SOP系统")
        title_sp.setStyleSheet(f"font-size: 12px; color: {C_MUTED};")
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(title_sp)
        h_layout.addLayout(title_box)

        h_layout.addSpacing(30)

        for label, value in [("工厂", "深圳智造工厂"), ("产线", "A线·手机组装"), ("工位", "W-07 镜面贴合"), ("班次", "白班 A组")]:
            info = QVBoxLayout()
            info.setSpacing(1)
            lbl_l = QLabel(label)
            lbl_l.setStyleSheet(f"font-size: 9px; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 1.5px;")
            lbl_v = QLabel(value)
            lbl_v.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {C_PRIMARY};")
            info.addWidget(lbl_l)
            info.addWidget(lbl_v)
            h_layout.addLayout(info)
            h_layout.addSpacing(20)

        h_layout.addStretch()

        self.lbl_status_pill = QLabel("● 系统在线")
        self.lbl_status_pill.setStyleSheet(f"background: {C_GREEN_10}; border: 1px solid {C_GREEN_30}; border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 600; color: {C_GREEN};")
        h_layout.addWidget(self.lbl_status_pill)

        self.lbl_clock = QLabel("--:--:--")
        self.lbl_clock.setStyleSheet(f"font-size: 15px; color: {C_CYAN}; font-family: '{F_MONO}'; letter-spacing: 1px;")
        h_layout.addWidget(self.lbl_clock)

        h_layout.addSpacing(16)

        self.btn_min = QPushButton("—")
        self.btn_min.setObjectName("winBtn")
        self.btn_min.setFixedSize(46, 56)
        self.btn_min.clicked.connect(self.showMinimized)
        h_layout.addWidget(self.btn_min)

        self.btn_max = QPushButton("▢")
        self.btn_max.setObjectName("winBtn")
        self.btn_max.setFixedSize(46, 56)
        self.btn_max.clicked.connect(self._toggle_max)
        h_layout.addWidget(self.btn_max)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("closeBtn")
        self.btn_close.setFixedSize(46, 56)
        self.btn_close.clicked.connect(self.close)
        h_layout.addWidget(self.btn_close)

        # 无边框窗口自定义拖动
        header.mousePressEvent = self._on_header_press
        header.mouseMoveEvent = self._on_header_move
        header.mouseDoubleClickEvent = lambda e: self._toggle_max()
        root.addWidget(header)

    def _build_main_content(self, root):
        """主内容区：左侧视频+步骤卡片 / 右侧统计面板。"""
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        self._build_left_column(main_layout)
        self._build_right_panel(main_layout)

        root.addWidget(main_widget, 1)

    def _build_left_column(self, main_layout):
        """左侧：视频显示区（含底部状态条）+ 4 步骤卡片横排。"""
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # === 视频区 ===
        video_frame = QFrame()
        video_frame.setObjectName("videoSection")
        v_layout = QVBoxLayout(video_frame)
        v_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("点击「导入视频」选择视频文件")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(f"background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 {C_BG_VIDEO_DIM},stop:1 {C_BG_VIDEO_DARK}); color: {C_MUTED}; font-size: 14px; border: none;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v_layout.addWidget(self.video_label)

        self.video_bottom = QFrame()
        self.video_bottom.setStyleSheet(f"background: qlineargradient(x1:0,y1:1,x2:0,y2:0,stop:0 {C_BLACK_85},stop:1 transparent); border: none;")
        self.video_bottom.setFixedHeight(50)
        vb_layout = QHBoxLayout(self.video_bottom)
        vb_layout.setContentsMargins(16, 6, 16, 6)
        vb_layout.setSpacing(16)

        self.bottom_labels = {}
        for key, lbl in [("current", ("当前动作", "#00d4ff")), ("expected", ("期望动作", "#ffaa00")), ("frame", ("帧号", "#5a6b7d")), ("time", ("时间", "#5a6b7d")), ("hit", ("命中", "#00ff88"))]:
            box = QVBoxLayout()
            box.setSpacing(0)
            l = QLabel(lbl[0])
            l.setStyleSheet(f"font-size: 9px; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 1px; border: none;")
            v = QLabel("--")
            v.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {lbl[1]}; font-family: '{F_MONO if key in ('frame','time','hit') else F_DISPLAY}'; border: none;")
            box.addWidget(l)
            box.addWidget(v)
            self.bottom_labels[key] = v
            vb_layout.addLayout(box)

        self.video_bottom.setVisible(False)
        v_layout.addWidget(self.video_bottom)

        left_col.addWidget(video_frame, 1)

        # === 步骤卡片 ===
        steps_frame = QFrame()
        steps_layout = QHBoxLayout(steps_frame)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(10)

        step_defs = [
            ("D1", "取料", "D1_pick_material"),
            ("D2", "撕膜", "D2_tear_film"),
            ("D3", "检测", "D3_inspect"),
            ("D4", "放料", "D4_place_material"),
        ]
        for i, (sid, name, en) in enumerate(step_defs):
            card = StepCard(i, sid, name, en)
            self.cards.append(card)
            steps_layout.addWidget(card)
        steps_layout.addStretch()

        left_col.addWidget(steps_frame)

        main_layout.addLayout(left_col, 1)

    def _build_right_panel(self, main_layout):
        """右侧：周期计数 + 实时推理指标 + 生产统计 + 事件日志。"""
        right_frame = QFrame()
        right_frame.setFixedWidth(380)
        right_col = QVBoxLayout(right_frame)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(12)

        # === 周期计数面板 ===
        cycle_panel = QFrame()
        cycle_panel.setObjectName("panel")
        cp_layout = QVBoxLayout(cycle_panel)
        cp_layout.setContentsMargins(14, 14, 14, 14)
        cp_layout.setSpacing(8)

        lbl_title = QLabel("生产周期")
        lbl_title.setStyleSheet(f"font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: {C_CYAN};")
        cp_layout.addWidget(lbl_title)

        cycle_row = QHBoxLayout()
        cycle_left = QVBoxLayout()
        cycle_left.setSpacing(0)
        self.lbl_cycle = QLabel("1")
        self.lbl_cycle.setStyleSheet(f"font-size: 42px; font-weight: 700; font-family: '{F_MONO}'; color: {C_CYAN};")
        lbl_cycle_l = QLabel("当前轮次")
        lbl_cycle_l.setStyleSheet(f"font-size: 11px; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 1px;")
        cycle_left.addWidget(self.lbl_cycle)
        cycle_left.addWidget(lbl_cycle_l)
        cycle_row.addLayout(cycle_left)
        cycle_row.addStretch()

        cycle_right = QVBoxLayout()
        cycle_right.setSpacing(0)
        cycle_right.setAlignment(Qt.AlignRight)
        self.lbl_completed = QLabel("0")
        self.lbl_completed.setStyleSheet(f"font-size: 24px; font-weight: 700; font-family: '{F_MONO}'; color: {C_GREEN};")
        lbl_completed_l = QLabel("已完成")
        lbl_completed_l.setStyleSheet(f"font-size: 11px; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 1px;")
        cycle_right.addWidget(self.lbl_completed)
        cycle_right.addWidget(lbl_completed_l)
        cycle_row.addLayout(cycle_right)
        cp_layout.addLayout(cycle_row)

        right_col.addWidget(cycle_panel)

        # === 实时推理指标面板 ===
        rt_panel = QFrame()
        rt_panel.setObjectName("panel")
        rt_layout = QVBoxLayout(rt_panel)
        rt_layout.setContentsMargins(14, 14, 14, 14)
        rt_layout.setSpacing(8)

        rt_title = QLabel("实时推理")
        rt_title.setStyleSheet(f"font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: {C_CYAN};")
        rt_layout.addWidget(rt_title)

        metric_grid = QGridLayout()
        metric_grid.setSpacing(8)
        for i, (key, label) in enumerate([("current", "当前动作"), ("expected", "期望动作"), ("conf", "置信度"), ("hit", "命中帧")]):
            box = QFrame()
            box.setStyleSheet(f"background: {C_DARK}; border: 1px solid {C_CYAN_12}; border-radius: 8px; padding: 10px 12px;")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(12, 10, 12, 10)
            bl.setSpacing(2)
            l = QLabel(label)
            l.setStyleSheet(f"font-size: 9px; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 1px; border: none;")
            v = QLabel("--")
            v.setStyleSheet(f"font-size: 20px; font-weight: 700; font-family: '{F_MONO if key in ('conf','hit') else F_DISPLAY}'; color: {C_CYAN if key == 'current' else C_ORANGE if key == 'expected' else C_PRIMARY}; border: none;")
            bl.addWidget(l)
            bl.addWidget(v)
            self.bottom_labels[f"rt_{key}"] = v
            metric_grid.addWidget(box, i // 2, i % 2)

        rt_layout.addLayout(metric_grid)

        lbl_top3_title = QLabel("Top3 候选")
        lbl_top3_title.setStyleSheet(f"font-size: 10px; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 1px; padding-top: 4px; border: none;")
        rt_layout.addWidget(lbl_top3_title)
        self.lbl_top3 = QLabel("等待数据...")
        self.lbl_top3.setWordWrap(True)
        self.lbl_top3.setStyleSheet(f"font-size: 12px; color: {C_PRIMARY}; border: none;")
        rt_layout.addWidget(self.lbl_top3)

        right_col.addWidget(rt_panel)

        # === 生产统计面板 ===
        stats_panel = QFrame()
        stats_panel.setObjectName("panel")
        sp_layout = QVBoxLayout(stats_panel)
        sp_layout.setContentsMargins(14, 14, 14, 14)
        sp_layout.setSpacing(8)

        stats_title = QLabel("生产统计")
        stats_title.setStyleSheet(f"font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: {C_CYAN};")
        sp_layout.addWidget(stats_title)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(14)

        self.stat_ring = StatRing()
        stats_row.addWidget(self.stat_ring)

        stats_info = QVBoxLayout()
        stats_info.setSpacing(4)
        self.lbl_pass = QLabel("合格: 0")
        self.lbl_pass.setStyleSheet(f"font-size: 14px; font-weight: 600; font-family: '{F_MONO}'; color: {C_GREEN}; border: none;")
        self.lbl_skip = QLabel("超时: 0")
        self.lbl_skip.setStyleSheet(f"font-size: 14px; font-weight: 600; font-family: '{F_MONO}'; color: {C_RED}; border: none;")
        self.lbl_total = QLabel("总计: 0")
        self.lbl_total.setStyleSheet(f"font-size: 14px; font-weight: 600; font-family: '{F_MONO}'; color: {C_PRIMARY}; border: none;")
        self.lbl_ct = QLabel("CT: --")
        self.lbl_ct.setStyleSheet(f"font-size: 14px; font-weight: 600; font-family: '{F_MONO}'; color: {C_CYAN}; border: none;")
        self.lbl_avg_ct = QLabel("均CT: --")
        self.lbl_avg_ct.setStyleSheet(f"font-size: 14px; font-weight: 600; font-family: '{F_MONO}'; color: {C_MUTED}; border: none;")
        stats_info.addWidget(self.lbl_pass)
        stats_info.addWidget(self.lbl_skip)
        stats_info.addWidget(self.lbl_total)
        stats_info.addWidget(self.lbl_ct)
        stats_info.addWidget(self.lbl_avg_ct)
        stats_row.addLayout(stats_info)
        stats_row.addStretch()
        sp_layout.addLayout(stats_row)

        right_col.addWidget(stats_panel)

        # === 事件日志面板 ===
        log_panel = QFrame()
        log_panel.setObjectName("panel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(14, 14, 14, 14)
        log_layout.setSpacing(8)

        log_title = QLabel("事件日志")
        log_title.setStyleSheet(f"font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: {C_CYAN};")
        log_layout.addWidget(log_title)

        self.event_scroll = QScrollArea()
        self.event_scroll.setWidgetResizable(True)
        self.event_container = QWidget()
        self.event_container.setStyleSheet("background: transparent;")
        self.event_container_layout = QVBoxLayout(self.event_container)
        self.event_container_layout.setContentsMargins(0, 0, 0, 0)
        self.event_container_layout.setSpacing(0)
        self.event_container_layout.addStretch()
        self.event_scroll.setWidget(self.event_container)
        log_layout.addWidget(self.event_scroll)

        right_col.addWidget(log_panel, 1)

        main_layout.addWidget(right_frame)

    def _build_footer(self, root):
        """底部：导入/开始/暂停/停止/导出按钮 + LSTM 阈值 + 显示开关。"""
        footer = QFrame()
        footer.setObjectName("footer")
        footer.setFixedHeight(48)
        f_layout = QHBoxLayout(footer)
        f_layout.setContentsMargins(24, 8, 24, 8)
        f_layout.setSpacing(8)

        self.btn_import = QPushButton("导入视频")
        self.btn_camera = QPushButton("摄像头")
        self.btn_start = QPushButton("开始分析")
        self.btn_pause = QPushButton("暂停/继续")
        self.btn_stop = QPushButton("停止")
        self.btn_export = QPushButton("导出结果")
        f_layout.addWidget(self.btn_import)
        f_layout.addWidget(self.btn_camera)
        f_layout.addWidget(self.btn_start)
        f_layout.addWidget(self.btn_pause)
        f_layout.addWidget(self.btn_stop)
        f_layout.addWidget(self.btn_export)

        f_layout.addStretch()

        self.edit_rtsp = QLineEdit("")
        self.edit_rtsp.setPlaceholderText("RTSP 地址 (rtsp://...)")
        self.edit_rtsp.setFixedWidth(230)
        self.btn_rtsp = QPushButton("连接RTSP")
        f_layout.addWidget(self.edit_rtsp)
        f_layout.addWidget(self.btn_rtsp)
        f_layout.addSpacing(10)

        self.edit_lstm = QLineEdit("0.15")

        f_layout.addWidget(QLabel("LSTM阈值"))
        f_layout.addWidget(self.edit_lstm)
        f_layout.addSpacing(10)

        self.chk_keypoints = QCheckBox("显示关键点")
        self.chk_snapshots = QCheckBox("保存截图")
        self.chk_log = QCheckBox("保存日志")
        for cb in [self.chk_keypoints, self.chk_snapshots, self.chk_log]:
            cb.setChecked(True)
            f_layout.addWidget(cb)

        root.addWidget(footer)

    def _connect_signals(self):
        """按钮点击信号 → 对应槽函数。"""
        self.btn_import.clicked.connect(self.on_import_video)
        self.btn_camera.clicked.connect(self.on_camera)
        self.btn_rtsp.clicked.connect(self.on_rtsp)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_export.clicked.connect(self.on_export)

    def _toggle_max(self):
        """切换最大化 / 还原（标题栏按钮触发）。"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _on_header_press(self, event):
        """记录鼠标按下点位置，用于无边框窗口自定义拖动。"""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos()

    def _on_header_move(self, event):
        """拖动标题栏移动整个窗口（最大化状态下不允许拖动）。"""
        if hasattr(self, '_drag_pos') and self._drag_pos is not None:
            if event.buttons() & Qt.LeftButton:
                if self.isMaximized():
                    return
                delta = event.globalPos() - self._drag_pos
                self._drag_pos = event.globalPos()
                self.move(self.pos() + delta)

    def _update_clock(self):
        """每秒刷新标题栏的当前时间显示。"""
        now = datetime.now()
        self.lbl_clock.setText(now.strftime("%H:%M:%S"))

    def on_import_video(self):
        """「导入视频」按钮槽：弹文件对话框选 mp4，预览首帧，校验可解码后保存路径。"""
        f, _ = QFileDialog.getOpenFileName(self, "选择视频", str(BASE_DIR / "video"), "Video (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if not f:
            return

        path = Path(f)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            show_message(self, "warning", "导入失败", "该视频无法打开")
            return

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            show_message(self, "warning", "导入失败", "该视频无法读取帧")
            return

        h, w = frame.shape[:2]
        cap.release()

        self.video_source = str(path)
        self.video_label.setText("")
        pix = QPixmap.fromImage(bgr_to_qimage(frame))
        self._set_video_preview_pixmap(pix)

    def on_camera(self):
        """「摄像头」按钮槽：使用默认摄像头（索引 0）作为视频源。"""
        self.video_source = "0"
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("摄像头已就绪，点击「开始分析」")

    def on_rtsp(self):
        """「连接RTSP」按钮槽：使用输入框中的 RTSP 地址作为视频源。"""
        url = self.edit_rtsp.text().strip()
        if not url:
            show_message(self, "warning", "提示", "请输入 RTSP 地址")
            return
        self.video_source = url
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText(f"RTSP: {url}")

    def _reset_cards(self, names: List[str]):
        """多周期循环重置时把所有步骤卡片重置为 pending 状态、刷新名称。"""
        for i, card in enumerate(self.cards):
            if i < len(names):
                card.lbl_name.setText(names[i].split("：")[-1] if "：" in names[i] else names[i])
                card.set_status("pending")
            else:
                card.set_status("pending")

    def _params(self) -> RuntimeParams:
        """从 GUI 控件读取当前 RuntimeParams（LSTM 阈值/各开关）。"""
        return RuntimeParams(
            lstm_conf=float(self.edit_lstm.text().strip()),
            confirm_frames=CONFIRM_FRAMES_FIXED,
            show_keypoints=self.chk_keypoints.isChecked(),
            save_snapshots=self.chk_snapshots.isChecked(),
            save_log=self.chk_log.isChecked(),
        )

    def on_start(self):
        """「开始分析」按钮槽：校验前置条件 → 重置统计 → 创建并启动 InferenceWorker。"""
        if self.worker and self.worker.isRunning():
            show_message(self, "info", "提示", "正在分析中")
            return
        if not self.video_source:
            show_message(self, "warning", "提示", "请先选择视频源（导入视频 / 摄像头 / RTSP）")
            return

        try:
            params = self._params()
        except Exception:
            show_message(self, "warning", "参数错误", "请检查阈值参数格式")
            return

        self.pass_count = 0
        self.skip_count = 0
        self.lbl_pass.setText("合格: 0")
        self.lbl_skip.setText("超时: 0")
        self.lbl_total.setText("总计: 0")
        self.lbl_ct.setText("CT: --")
        self.lbl_avg_ct.setText("均CT: --")
        self.stat_ring.set_rate(100)
        for c in self.cards:
            c.set_status("pending")
        self.video_bottom.setVisible(True)
        self.lbl_status_pill.setText("● 推理中")
        self.lbl_status_pill.setStyleSheet(f"background: {C_ORANGE_10}; border: 1px solid {C_ORANGE_30}; border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 600; color: {C_ORANGE};")

        self._clear_events()

        self.worker = InferenceWorker(self.video_source, params)
        self.worker.sig_frame.connect(self.on_frame)
        self.worker.sig_status.connect(self.on_status)
        self.worker.sig_action.connect(self.on_action)
        self.worker.sig_finished.connect(self.on_finished)
        self.worker.sig_error.connect(self.on_error)
        self.worker.start()

    def on_pause(self):
        """「暂停」按钮槽：切换 worker._pause 状态（不读帧、不推进状态机）。"""
        if self.worker and self.worker.isRunning():
            self.worker.toggle_pause()

    def on_stop(self):
        """「停止」按钮槽：设置 _stop=True 通知 run() 退出，最多等 1.5 秒。"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        self.lbl_status_pill.setText("● 系统在线")
        self.lbl_status_pill.setStyleSheet(f"background: {C_GREEN_10}; border: 1px solid {C_GREEN_30}; border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 600; color: {C_GREEN};")
        self.video_bottom.setVisible(False)

    def on_export(self):
        """「导出结果」按钮槽：把上次 run_dir 复制到用户选的目录（截图 + JSON + CSV）。"""
        if not self.last_run_dir or not self.last_run_dir.exists():
            show_message(self, "info", "提示", "暂无可导出结果")
            return
        dst = QFileDialog.getExistingDirectory(self, "选择导出目录", str(BASE_DIR))
        if not dst:
            return
        out = Path(dst) / self.last_run_dir.name
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(self.last_run_dir, out)
        show_message(self, "info", "导出完成", f"已导出到: {out}")

    def _set_video_preview_pixmap(self, pix: QPixmap):
        """把 QPixmap 按比例缩放到 video_label 尺寸并显示（导入视频时显示首帧）。"""
        if pix.isNull():
            return
        scaled = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)
        self.video_label.setText("")

    def _apply_font_scale(self, scale: float = None):
        """按窗口宽度计算字体缩放比例（最大化时文字自动放大），并应用到所有卡片。"""
        if scale is None:
            scale = max(1.0, min(2.5, self.width() / 1600.0))
        self._font_scale = scale
        s = lambda v: max(1, int(v * scale))

        for card in self.cards:
            card.set_scale(scale)

        if hasattr(self, 'lbl_cycle'):
            self.lbl_cycle.setStyleSheet(f"font-size: {s(52)}px; font-weight: 700; font-family: '{F_MONO}'; color: {C_CYAN};")
        if hasattr(self, 'lbl_completed'):
            self.lbl_completed.setStyleSheet(f"font-size: {s(30)}px; font-weight: 700; font-family: '{F_MONO}'; color: {C_GREEN};")
        if hasattr(self, 'lbl_pass'):
            self.lbl_pass.setStyleSheet(f"font-size: {s(18)}px; font-weight: 600; font-family: '{F_MONO}'; color: {C_GREEN}; border: none;")
        if hasattr(self, 'lbl_skip'):
            self.lbl_skip.setStyleSheet(f"font-size: {s(18)}px; font-weight: 600; font-family: '{F_MONO}'; color: {C_RED}; border: none;")
        if hasattr(self, 'lbl_total'):
            self.lbl_total.setStyleSheet(f"font-size: {s(18)}px; font-weight: 600; font-family: '{F_MONO}'; color: {C_PRIMARY}; border: none;")
        if hasattr(self, 'lbl_ct'):
            self.lbl_ct.setStyleSheet(f"font-size: {s(18)}px; font-weight: 600; font-family: '{F_MONO}'; color: {C_CYAN}; border: none;")
        if hasattr(self, 'lbl_avg_ct'):
            self.lbl_avg_ct.setStyleSheet(f"font-size: {s(18)}px; font-weight: 600; font-family: '{F_MONO}'; color: {C_MUTED}; border: none;")

        if hasattr(self, 'bottom_labels'):
            for key, lbl in self.bottom_labels.items():
                if key in ('current', 'expected'):
                    lbl.setStyleSheet(f"font-size: {s(20)}px; font-weight: 700; color: {C_CYAN if key == 'current' else C_ORANGE}; border: none;")
                elif key in ('frame', 'time', 'hit'):
                    lbl.setStyleSheet(f"font-size: {s(20)}px; font-weight: 700; font-family: '{F_MONO}'; color: {C_GREEN if key == 'hit' else C_MUTED}; border: none;")
                elif key == 'rt_current':
                    lbl.setStyleSheet(f"font-size: {s(26)}px; font-weight: 700; color: {C_CYAN}; border: none;")
                elif key == 'rt_expected':
                    lbl.setStyleSheet(f"font-size: {s(26)}px; font-weight: 700; color: {C_ORANGE}; border: none;")
                elif key == 'rt_conf':
                    lbl.setStyleSheet(f"font-size: {s(26)}px; font-weight: 700; font-family: '{F_MONO}'; color: {C_GREEN}; border: none;")
                elif key == 'rt_hit':
                    lbl.setStyleSheet(f"font-size: {s(26)}px; font-weight: 700; font-family: '{F_MONO}'; color: {C_PRIMARY}; border: none;")

    def resizeEvent(self, event):
        """窗口尺寸变化时重新计算字体缩放、刷新视频预览比例。"""
        super().resizeEvent(event)
        self._apply_font_scale()
        pix = self.video_label.pixmap()
        if pix is not None and not pix.isNull():
            self._set_video_preview_pixmap(pix)

    def on_frame(self, qimg: QImage):
        """sig_frame 槽：收到推理线程发来的画面 → 转 QPixmap → 显示到视频区。"""
        pix = QPixmap.fromImage(qimg)
        self._set_video_preview_pixmap(pix)

    def _clear_events(self):
        """清空底部事件流（开始新一轮推理前调用）。"""
        while self.event_container_layout.count() > 1:
            item = self.event_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.event_items.clear()

    def _add_event(self, step_name: str, event_type: str, cycle: int, time_str: str):
        """新增一条事件到事件流顶部（最新的在上方）。"""
        item = EventItem(step_name, event_type, cycle, time_str)
        self.event_container_layout.insertWidget(0, item)
        self.event_items.append(item)

    def on_status(self, st: dict):
        """sig_status 槽：每帧状态更新 —— 处理两种事件：
          1) action_defs 字段：多周期循环重置，刷新卡片名称 + cycle 计数
          2) 普通帧状态：更新当前预测/期望步骤/置信度/命中数等
        """
        if "action_defs" in st:
            self._reset_cards(st["action_defs"])
            cycle = st.get("cycle", 1)
            self.lbl_cycle.setText(str(cycle))
            self.lbl_completed.setText(str(cycle - 1))
            for c in self.cards:
                c.set_status("pending")
            if self.cards:
                self.cards[0].set_status("进行中")
            return

        self.bottom_labels["frame"].setText(str(st.get("frame_id", 0)))
        self.bottom_labels["time"].setText(f"{st.get('time_sec', 0.0):.2f}s")
        self.bottom_labels["current"].setText(st.get('current_pred', '-'))
        self.bottom_labels["expected"].setText(st.get("expected_show", "-"))
        self.bottom_labels["hit"].setText(f"{st.get('hit', 0)}/{st.get('confirm_frames', 0)}")

        self.bottom_labels["rt_current"].setText(st.get('current_pred', '-'))
        self.bottom_labels["rt_expected"].setText(st.get("expected_show", "-"))
        conf = st.get('expected_conf', 0.0)
        self.bottom_labels["rt_conf"].setText(f"{conf*100:.1f}%")
        self.bottom_labels["rt_conf"].setStyleSheet(f"font-size: 20px; font-weight: 700; font-family: '{F_MONO}'; color: {C_GREEN if conf > 0.5 else C_ORANGE}; border: none;")
        self.bottom_labels["rt_hit"].setText(f"{st.get('hit', 0)}/{st.get('confirm_frames', 0)}")

        top3 = st.get('top3', '-')
        self.lbl_top3.setText(top3 if top3 != '-' else '等待数据...')

        p = st.get("progress", 0)
        cycle = st.get("cycle", 1)
        self.lbl_cycle.setText(str(cycle))
        self.lbl_completed.setText(str(cycle - 1))

        for i, card in enumerate(self.cards):
            if i < p:
                if card.status not in ("done", "timeout"):
                    card.set_status("完成")
            elif i == p:
                card.set_status("进行中")
                card.set_confidence(conf)
                elapsed = max(0.0, st.get("time_sec", 0.0) - st.get("stage_start_sec", 0.0))
                card.set_active_elapsed(elapsed)
            else:
                if card.status not in ("done", "timeout"):
                    card.set_status("pending")

    def on_action(self, ev: dict):
        """sig_action 槽：步骤完成或超时跳过时被调用。

        做三件事：
          1) 更新对应 StepCard 状态（绿/红边框）+ 截图
          2) 累加 pass_count / skip_count，重算合格率环形图
          3) 在事件流顶部新增一条记录
        """
        idx = int(ev.get("index", -1))
        status = ev.get("status", "完成")

        # 周期完成事件（index=-1）：更新 CT 统计 + 事件流，不参与合格率统计
        if status == "周期完成":
            ct = ev.get("cycle_time_sec", 0.0)
            avg = ev.get("avg_cycle_time_sec", 0.0)
            self.lbl_ct.setText(f"CT: {ct:.2f}s")
            self.lbl_avg_ct.setText(f"均CT: {avg:.2f}s")
            cycle = int(self.lbl_cycle.text())
            time_str = datetime.now().strftime("%H:%M:%S")
            self._add_event(f"周期 {cycle} 完成", "cycle", cycle, f"CT: {ct:.2f}s · {time_str}")
            return

        if 0 <= idx < len(self.cards):
            self.cards[idx].set_status(status, ev.get("info", ""))
            self.cards[idx].set_snapshot(ev.get("snapshot"))

        if status == "完成":
            self.pass_count += 1
        elif status == "超时跳过":
            self.skip_count += 1

        total = self.pass_count + self.skip_count
        rate = round(self.pass_count / total * 100) if total > 0 else 100
        self.lbl_pass.setText(f"合格: {self.pass_count}")
        self.lbl_skip.setText(f"超时: {self.skip_count}")
        self.lbl_total.setText(f"总计: {total}")
        self.stat_ring.set_rate(rate)

        if idx < len(self.cards):
            step = self.cards[idx]
            time_str = datetime.now().strftime("%H:%M:%S")
            self._add_event(step.name, "done" if status == "完成" else "skip",
                           int(self.lbl_cycle.text()), time_str)

    def on_finished(self, result: dict):
        """sig_finished 槽：推理线程正常退出，保存 run_dir 供导出使用。"""
        run_dir = result.get("run_dir", "")
        self.last_run_dir = Path(run_dir) if run_dir else None
        self.lbl_status_pill.setText("● 系统在线")
        self.lbl_status_pill.setStyleSheet(f"background: {C_GREEN_10}; border: 1px solid {C_GREEN_30}; border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 600; color: {C_GREEN};")
        show_message(self, "info", "完成", "视频分析完成")

    def on_error(self, msg: str):
        """sig_error 槽：推理线程抛异常时弹错误对话框。"""
        show_message(self, "error", "错误", msg)
        self.lbl_status_pill.setText("● 系统在线")
        self.lbl_status_pill.setStyleSheet(f"background: {C_GREEN_10}; border: 1px solid {C_GREEN_30}; border-radius: 20px; padding: 5px 14px; font-size: 12px; font-weight: 600; color: {C_GREEN};")


def main():
    """程序入口：创建 QApplication + MainWindow，进入 Qt 事件循环。"""
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
