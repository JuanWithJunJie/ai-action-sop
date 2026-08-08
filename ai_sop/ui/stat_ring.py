"""StatRing 控件 —— 合格率环形图。"""
from PyQt5.QtWidgets import QWidget


class StatRing(QWidget):
    """合格率环形图控件（右侧统计区显示完成百分比）。

    自绘 QPainter 实现：背景灰圈 + 进度彩圈 + 中心百分比文字。
    颜色按合格率分级：≥90 绿 / ≥70 橙 / <70 红。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rate = 100
        self.setFixedSize(64, 64)

    def set_rate(self, rate: int):
        """设置合格率并触发重绘（0-100 自动 clamp）。"""
        self.rate = max(0, min(100, rate))
        self.update()

    def paintEvent(self, event):
        """Qt 重绘事件：用 QPainter 画环 + 文字。"""
        from PyQt5.QtGui import QPainter, QPen, QColor, QFont as QFont2

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect_size = 50
        x = (self.width() - rect_size) // 2
        y = (self.height() - rect_size) // 2

        pen = QPen(QColor(0, 212, 255, 25))
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawArc(x, y, rect_size, rect_size, 0, 360 * 16)

        color = QColor(0, 255, 136) if self.rate >= 90 else QColor(255, 170, 0) if self.rate >= 70 else QColor(255, 68, 102)
        pen = QPen(color)
        pen.setWidth(4)
        painter.setPen(pen)
        span = int(self.rate / 100 * 360 * 16)
        painter.drawArc(x, y, rect_size, rect_size, 90 * 16, -span)

        painter.setPen(QColor(216, 228, 240))
        font = QFont2("Consolas", 10, QFont2.Bold)
        painter.setFont(font)
        text = f"{self.rate}%"
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        painter.drawText((self.width() - tw) // 2, self.height() // 2 + 5, text)

        painter.end()
