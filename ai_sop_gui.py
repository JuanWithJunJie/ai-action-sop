"""AI-SOP Vision —— 桌面 GUI 入口。

历史: 这个文件原本是 1700+ 行的单文件实现，现在已重构为 ai_sop/ 包。
保留本文件作为兼容入口，让旧脚本（extract_features.py / train_lstm.py / auto_label.py）的
`from ai_sop_gui import ...` 仍可工作。
"""
from ai_sop import main  # noqa: F401  重导出 main()
from ai_sop import (  # noqa: F401  兼容旧导入路径
    BASE_DIR,
    LSTM_MODEL_PATH,
    ActionLSTM,
    RuntimeParams,
    ActionRuntime,
    InferenceWorker,
    build_feature_row,
    bgr_to_qimage,
    to_beijing_time_str,
    action_to_cn,
    fine_label_from_row,
    SOURCE_WINDOWS,
    CONFIRM_FRAMES_FIXED,
    STEP_TIMEOUT_SEC,
    STEP_MIN_STAGE_SEC,
    DEFAULT_ACTION_DEFS,
    ACTION_CN_MAP,
)

__all__ = [
    "main",
    "BASE_DIR", "LSTM_MODEL_PATH",
    "ActionLSTM", "RuntimeParams", "ActionRuntime", "InferenceWorker",
    "build_feature_row", "bgr_to_qimage",
    "to_beijing_time_str", "action_to_cn", "fine_label_from_row",
    "SOURCE_WINDOWS", "CONFIRM_FRAMES_FIXED",
    "STEP_TIMEOUT_SEC", "STEP_MIN_STAGE_SEC",
    "DEFAULT_ACTION_DEFS", "ACTION_CN_MAP",
]


if __name__ == "__main__":
    main()
