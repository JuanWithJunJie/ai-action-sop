"""路径 + 全局常量。被所有模块依赖。"""
from pathlib import Path

from ai_sop.theme import COLORS, FONTS

# ===== 路径 =====
# constants.py 在 ai_sop/core/，需上溯 3 层到项目根
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TIMELINE_CSV = BASE_DIR / "full_timeline_10videos_template.csv"
LSTM_MODEL_PATH = BASE_DIR / "lstm_runs_fine/best_lstm_fine.pt"
LSTM_CONFIG_PATH = BASE_DIR / "lstm_runs_fine/config.json"
RUNS_GUI_DIR = BASE_DIR / "runs_gui"
UI_BG_PATH = BASE_DIR / "背景图片.jpg"

# ===== 推理核心常量 =====
SOURCE_WINDOWS = [48, 72, 96]                          # 多尺度时序窗口：短=响应快、长=更稳，三者投票融合
CONFIRM_FRAMES_FIXED = 4                              # 命中确认帧数：连续 N 帧达标才算步骤完成（反单帧误触）
STEP_TIMEOUT_SEC = 6.0                                # 步骤超时：某步 6 秒未完成则强制跳过，避免卡住产线
STEP_MIN_STAGE_SEC = 0.8                             # 步骤最小持续时间：切换后 0.8 秒内强制置信度=0（防尾巴误判）

HAND_LANDMARK_THICKNESS = 4
HAND_CONNECTION_THICKNESS = 4
HAND_CIRCLE_RADIUS = 5

# ===== SOP 动作定义（4 步循环） =====
# (展示名, LSTM 标签) —— 注意 LSTM 标签需与 train_lstm.py 的 LABEL_MAP 完全一致
DEFAULT_ACTION_DEFS = [
    ("第一步：取料", "D1_pick_material"),
    ("第二步：撕膜", "D2_tear_film"),
    ("第三步：检测", "D3_inspect"),
    ("第四步：放料", "D4_place_material"),
]
ACTION_CN_MAP = {                                     # LSTM 输出标签 → 中文显示名
    "D1_pick_material": "取料",
    "D2_tear_film": "撕膜",
    "D3_inspect": "检测",
    "D4_place_material": "放料",
    "background": "背景",
}

# ===== 主题色快捷引用（详见 theme.py）—— 用于 f-string 内的 QSS 颜色插值 =====
C_PRIMARY = COLORS["text_primary"]
C_MUTED = COLORS["text_muted"]
C_DIM = COLORS["text_dim"]
C_DARK_TEXT = COLORS["text_dark"]
C_CYAN = COLORS["accent_cyan"]
C_CYAN_DK = COLORS["accent_cyan_dk"]
C_GREEN = COLORS["status_online"]
C_ORANGE = COLORS["status_warning"]
C_RED = COLORS["status_danger"]
C_BG_PRIMARY = COLORS["bg_primary"]
C_BG_VIDEO_DARK = COLORS["bg_video_dark"]
C_BG_VIDEO_DIM = COLORS["bg_video_dim"]
C_WHITE_6 = COLORS["overlay_white_6"]
C_DARK = COLORS["overlay_dark"]
C_BLACK_85 = COLORS["overlay_black_85"]
C_CYAN_12 = COLORS["overlay_cyan_12"]
C_GREEN_10 = COLORS["overlay_green_10"]
C_GREEN_15 = COLORS["overlay_green_15"]
C_GREEN_30 = COLORS["overlay_green_30"]
C_ORANGE_10 = COLORS["overlay_orange_10"]
C_ORANGE_30 = COLORS["overlay_orange_30"]
C_RED_15 = COLORS["overlay_red_15"]
C_RED_30 = COLORS["overlay_red_30"]
C_PANEL_BG = COLORS["overlay_panel"]
F_MONO = FONTS["mono"]
F_DISPLAY = FONTS["display"]
