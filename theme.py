"""向后兼容 shim —— 转发到 ai_sop.theme。"""
from ai_sop.theme import COLORS, FONTS, SIZES  # noqa: F401

__all__ = ["COLORS", "FONTS", "SIZES"]
