"""路径 + 全局常量。被所有模块依赖。"""
import json
from pathlib import Path

from ai_sop.theme import COLORS, FONTS

# ===== 路径 =====
# constants.py 在 ai_sop/core/，需上溯 3 层到项目根
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.json"


def _load_config() -> dict:
    """读取 config.json（不存在或损坏时返回空 dict，全部走默认值）。"""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


CONFIG = _load_config()


def _cfg_path(key: str, default: str) -> Path:
    """从 CONFIG.model 取相对路径并拼到项目根。"""
    return BASE_DIR / CONFIG.get("model", {}).get(key, default)


TIMELINE_CSV = BASE_DIR / "full_timeline_10videos_template.csv"
LSTM_MODEL_PATH = _cfg_path("lstm_model", "lstm_runs_fine/best_lstm_fine.pt")
LSTM_CONFIG_PATH = _cfg_path("lstm_config", "lstm_runs_fine/config.json")
RUNS_GUI_DIR = BASE_DIR / "runs_gui"
UI_BG_PATH = BASE_DIR / "背景图片.jpg"
PENDING_DIR = BASE_DIR / "train_data" / "pending_samples"   # 误判样本沉淀目录（超时自动 + 手动标记）
PENDING_FEATURES_DIR = PENDING_DIR / "features"              # 沉淀的 48 帧特征 .npy
PENDING_LABELS_CSV = PENDING_DIR / "pending_labels.csv"      # 沉淀样本标注（回灌 ingest_pending.py 用）

# ===== 推理核心常量 =====
SOURCE_WINDOWS = [48, 72, 96]                          # 多尺度时序窗口：短=响应快、长=更稳，三者投票融合
_INF = CONFIG.get("inference", {})
CONFIRM_FRAMES_FIXED = int(_INF.get("confirm_frames", 4))          # 命中确认帧数：连续 N 帧达标才算步骤完成
STEP_TIMEOUT_SEC = float(_INF.get("step_timeout_sec", 6.0))        # 步骤超时：某步 N 秒未完成则强制跳过
STEP_MIN_STAGE_SEC = float(_INF.get("step_min_stage_sec", 0.8))    # 步骤最小持续时间（防尾巴误判）
LSTM_CONF_DEFAULT = float(_INF.get("lstm_conf_default", 0.15))     # GUI 阈值输入框默认值
MAX_RECONNECT_ATTEMPTS = int(_INF.get("max_reconnect_attempts", 5))  # 实时源断线最大重连次数
RECONNECT_DELAY_SEC = float(_INF.get("reconnect_delay_sec", 2.0))   # 每次重连间隔（秒）
WATCHDOG_TIMEOUT_SEC = float(_INF.get("watchdog_timeout_sec", 8.0)) # 推理无心跳判定异常的超时（秒）

# ===== 工位信息（config.json 可覆盖）=====
_SITE = CONFIG.get("site", {})
SITE_INFO = {
    "factory": _SITE.get("factory", "深圳智造工厂"),
    "line": _SITE.get("line", "A线·手机组装"),
    "station": _SITE.get("station", "W-07 镜面贴合"),
    "shift": _SITE.get("shift", "白班 A组"),
}

# ===== 视频源默认值（config.json 可覆盖）=====
DEFAULT_VIDEO_SOURCE = CONFIG.get("video", {}).get("default_source", "")
DEFAULT_RTSP_URL = CONFIG.get("video", {}).get("rtsp_url", "")

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
