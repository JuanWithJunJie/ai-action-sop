"""AI-SOP Vision —— 手机制造智能SOP系统。

包结构：
  ai_sop.theme          颜色/字体常量
  ai_sop.core           推理核心层（constants/models/features/utils/worker）
  ai_sop.ui             PyQt5 控件层（StepCard/StatRing/EventItem/MainWindow）
"""
# 公共 API 重导出（向后兼容 ai_sop_gui.py 旧导入路径）
from ai_sop.theme import COLORS, FONTS, SIZES  # noqa: F401
from ai_sop.core import (  # noqa: F401
    BASE_DIR,
    LSTM_MODEL_PATH,
    LSTM_CONFIG_PATH,
    RUNS_GUI_DIR,
    TIMELINE_CSV,
    SOURCE_WINDOWS,
    CONFIRM_FRAMES_FIXED,
    STEP_TIMEOUT_SEC,
    STEP_MIN_STAGE_SEC,
    DEFAULT_ACTION_DEFS,
    ACTION_CN_MAP,
    ActionLSTM,
    ActionRuntime,
    RuntimeParams,
    InferenceWorker,
    build_feature_row,
    bgr_to_qimage,
    to_beijing_time_str,
    action_to_cn,
    fine_label_from_row,
)
from ai_sop.ui import MainWindow, main  # noqa: F401

__all__ = [
    "COLORS", "FONTS", "SIZES",
    "BASE_DIR", "LSTM_MODEL_PATH", "LSTM_CONFIG_PATH",
    "RUNS_GUI_DIR", "TIMELINE_CSV", "SOURCE_WINDOWS",
    "CONFIRM_FRAMES_FIXED", "STEP_TIMEOUT_SEC", "STEP_MIN_STAGE_SEC",
    "DEFAULT_ACTION_DEFS", "ACTION_CN_MAP",
    "ActionLSTM", "ActionRuntime", "RuntimeParams", "InferenceWorker",
    "build_feature_row", "bgr_to_qimage",
    "to_beijing_time_str", "action_to_cn", "fine_label_from_row",
    "MainWindow", "main",
]
