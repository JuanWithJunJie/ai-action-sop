"""主题常量模块 —— 集中管理 ai_sop_gui.py 中所有颜色 / 字体 / 通用尺寸。

设计意图：换主题或改字体时只改这一个文件，不必在 1700 行的 GUI 代码里翻找。

颜色命名规则：
  - *_primary / *_muted：背景与文字层级
  - accent_*：装饰色（边框、标题、强调）
  - status_*：状态语义（成功 / 警告 / 危险）
  - panel_*：面板背景与边框
"""

# ===== 颜色 =====
COLORS = {
    # 文字层级
    "text_primary":   "#d8e4f0",  # 主文字（白偏蓝）
    "text_muted":     "#5a6b7d",  # 弱文字 / 副标签
    "text_dim":       "#3a4a5a",  # 极弱文字（占位符）
    "text_dark":      "#000000",  # 用于亮色 Logo 内文字

    # 背景层级
    "bg_app":         "#060a12",  # 主窗口背景
    "bg_primary":     "#0d141f",  # 卡片/面板背景
    "bg_video_dark":  "#060a10",  # 视频区渐变终点
    "bg_video_dim":   "#0a0f18",  # 视频区渐变起点

    # 装饰色
    "accent_cyan":    "#00d4ff",  # 主强调色（标题、Logo 渐变、监控指标）
    "accent_cyan_dk": "#0088aa",  # Logo 渐变终点

    # 状态语义
    "status_online":  "#00ff88",  # 在线 / 完成 / 合格
    "status_warning": "#ffaa00",  # 进行中 / 期望
    "status_danger":  "#ff4466",  # 超时 / 错误

    # 通用半透明（rgba 形式，QSS 用）
    "overlay_panel":  "rgba(13,20,31,0.85)",     # 卡片背景
    "overlay_dark":   "rgba(0,0,0,0.3)",        # 子盒背景
    "overlay_black_85":"rgba(0,0,0,0.85)",      # 视频底部渐变起点
    "overlay_cyan_12":"rgba(0,212,255,0.12)",   # 边框淡 cyan
    "overlay_cyan_35":"rgba(0,212,255,0.35)",
    "overlay_green_10":"rgba(0,255,136,0.1)",   # 在线 pill 背景
    "overlay_green_15":"rgba(0,255,136,0.15)",
    "overlay_green_30":"rgba(0,255,136,0.3)",
    "overlay_orange_10":"rgba(255,170,0,0.1)",  # 推理中 pill 背景
    "overlay_orange_30":"rgba(255,170,0,0.3)",
    "overlay_red_15": "rgba(255,68,102,0.15)",
    "overlay_red_30": "rgba(255,68,102,0.3)",
    "overlay_white_6":"rgba(255,255,255,0.06)",
}

# ===== 字体 =====
FONTS = {
    "mono":    "Consolas",          # 等宽：数字、时间、置信度
    "display": "Microsoft YaHei",   # 中文显示
}

# ===== 通用尺寸 =====
SIZES = {
    "header_height":    56,
    "footer_height":    48,
    "video_bottom_h":   50,
    "right_panel_w":    380,
    "card_min_w":       240,
    "card_max_h":       240,
    "logo_size":        32,
    "win_btn_w":        46,
    "stat_ring_size":   64,
}
