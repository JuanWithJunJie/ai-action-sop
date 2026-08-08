"""ai_sop.core —— 推理核心层（常量 / 模型 / 特征 / 工具 / Worker）。"""
from ai_sop.core.constants import (  # noqa: F401
    ACTION_CN_MAP,
    BASE_DIR,
    CONFIRM_FRAMES_FIXED,
    DEFAULT_ACTION_DEFS,
    DEFAULT_RTSP_URL,
    DEFAULT_VIDEO_SOURCE,
    F_DISPLAY,
    F_MONO,
    HAND_CIRCLE_RADIUS,
    HAND_CONNECTION_THICKNESS,
    HAND_LANDMARK_THICKNESS,
    LSTM_CONFIG_PATH,
    LSTM_CONF_DEFAULT,
    LSTM_MODEL_PATH,
    MAX_RECONNECT_ATTEMPTS,
    PENDING_DIR,
    PENDING_FEATURES_DIR,
    PENDING_LABELS_CSV,
    RECONNECT_DELAY_SEC,
    RUNS_GUI_DIR,
    SITE_INFO,
    SOURCE_WINDOWS,
    STEP_MIN_STAGE_SEC,
    STEP_TIMEOUT_SEC,
    TIMELINE_CSV,
    UI_BG_PATH,
    WATCHDOG_TIMEOUT_SEC,
    # 主题色快捷引用
    C_BLACK_85,
    C_BG_PRIMARY,
    C_BG_VIDEO_DARK,
    C_BG_VIDEO_DIM,
    C_CYAN,
    C_CYAN_12,
    C_CYAN_DK,
    C_DARK,
    C_DARK_TEXT,
    C_DIM,
    C_GREEN,
    C_GREEN_10,
    C_GREEN_15,
    C_GREEN_30,
    C_MUTED,
    C_ORANGE,
    C_ORANGE_10,
    C_ORANGE_30,
    C_PANEL_BG,
    C_PRIMARY,
    C_RED,
    C_RED_15,
    C_RED_30,
    C_WHITE_6,
)
from ai_sop.core.models import ActionLSTM, ActionRuntime, RuntimeParams  # noqa: F401
from ai_sop.core.utils import (  # noqa: F401
    action_to_cn,
    bgr_to_qimage,
    fine_label_from_row,
    to_beijing_time_str,
)
from ai_sop.core.features import build_feature_row  # noqa: F401
from ai_sop.core.worker import InferenceWorker  # noqa: F401

__all__ = [
    # constants
    "ACTION_CN_MAP", "BASE_DIR", "CONFIRM_FRAMES_FIXED",
    "DEFAULT_ACTION_DEFS", "F_DISPLAY", "F_MONO", "HAND_CIRCLE_RADIUS",
    "HAND_CONNECTION_THICKNESS", "HAND_LANDMARK_THICKNESS", "LSTM_CONFIG_PATH",
    "LSTM_MODEL_PATH", "RUNS_GUI_DIR", "SOURCE_WINDOWS", "STEP_MIN_STAGE_SEC",
    "STEP_TIMEOUT_SEC", "TIMELINE_CSV", "UI_BG_PATH",
    # 主题色
    "C_BLACK_85", "C_BG_PRIMARY", "C_BG_VIDEO_DARK", "C_BG_VIDEO_DIM",
    "C_CYAN", "C_CYAN_12", "C_CYAN_DK", "C_DARK", "C_DARK_TEXT", "C_DIM",
    "C_GREEN", "C_GREEN_10", "C_GREEN_15", "C_GREEN_30", "C_MUTED",
    "C_ORANGE", "C_ORANGE_10", "C_ORANGE_30", "C_PANEL_BG", "C_PRIMARY",
    "C_RED", "C_RED_15", "C_RED_30", "C_WHITE_6",
    # models
    "ActionLSTM", "ActionRuntime", "RuntimeParams",
    # utils
    "action_to_cn", "bgr_to_qimage", "fine_label_from_row", "to_beijing_time_str",
    # features
    "build_feature_row",
    # worker
    "InferenceWorker",
]
