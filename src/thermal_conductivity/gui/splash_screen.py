"""Splash screen shown at application startup - Dual Mode Ready."""

from PyQt6.QtWidgets import QSplashScreen, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter, QLinearGradient, QBrush, QPolygonF
from PyQt6.QtCore import QPointF


class SplashScreen(QSplashScreen):
    """Professional splash screen with mode selection."""

    # ===== ADD THIS SIGNAL =====
    finished = pyqtSignal()  # This was missing!

    mode_selected = pyqtSignal(str)  # Emits 'nanothermite' or 'polymer'

    def __init__(self):
        # Create a pixmap for the splash
        self.pixmap = QPixmap(700, 500)
        self.pixmap.fill(Qt.GlobalColor.transparent)

        super().__init__(self.pixmap)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        self.progress_value = 0
        self.mode = None
        self.main_window = None
        self._draw_splash()

    def _draw_splash(self, show_mode_buttons=False):
        """Draw the splash screen content."""
        painter = QPainter(self.pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background gradient
        gradient = QLinearGradient(0, 0, 700, 500)
        gradient.setColorAt(0, QColor("#1a5490"))
        gradient.setColorAt(0.5, QColor("#1a6bb5"))
        gradient.setColorAt(1, QColor("#0d2b4e"))
        painter.fillRect(self.pixmap.rect(), gradient)

        # Decorative top bar
        painter.fillRect(0, 0, 700, 6, QColor("#e74c3c"))

        # Decorative bottom bar
        painter.fillRect(0, 494, 700, 6, QColor("#f39c12"))

        # Logo/Icon (simplified - a hexagon)
        painter.setBrush(QBrush(QColor("#e74c3c")))
        painter.setPen(Qt.PenStyle.NoPen)
        
        size = 40
        center_x = 350
        center_y = 100
        
        points = []
        for i in range(6):
            angle = (i * 60 - 30) * 3.14159 / 180
            x = center_x + size * 0.8 * (angle)
            y = center_y + size * 0.8 * (angle)
            points.append(QPointF(x, y))
        
        polygon = QPolygonF(points)
        painter.drawPolygon(polygon)

        # Title
        title_font = QFont("Arial", 28, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(0, 80, 700, 50, Qt.AlignmentFlag.AlignCenter,
                        "Thermal Conductivity")
        painter.drawText(0, 130, 700, 50, Qt.AlignmentFlag.AlignCenter,
                        "Modeling Suite")

        # Subtitle
        sub_font = QFont("Arial", 14)
        painter.setFont(sub_font)
        painter.setPen(QColor("#bdc3c7"))
        painter.drawText(0, 180, 700, 30, Qt.AlignmentFlag.AlignCenter,
                        "v2.0 - Dual Mode Analysis")

        # Version
        ver_font = QFont("Arial", 10)
        painter.setFont(ver_font)
        painter.setPen(QColor("#95a5a6"))
        painter.drawText(0, 205, 700, 25, Qt.AlignmentFlag.AlignCenter,
                        "Version 3.0.0")

        # Author
        painter.setPen(QColor("#ecf0f1"))
        painter.drawText(0, 230, 700, 25, Qt.AlignmentFlag.AlignCenter,
                        "Developed by Oukil Khaled ibn El-walid")

        if show_mode_buttons:
            # Draw mode selection area
            painter.setPen(QColor("#ffffff"))
            painter.drawText(0, 275, 700, 30, Qt.AlignmentFlag.AlignCenter,
                            "Select Analysis Mode:")

            # Nanothermite button
            painter.setBrush(QBrush(QColor("#c0392b")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(120, 310, 200, 80, 10, 10)
            
            # Polymer button
            painter.setBrush(QBrush(QColor("#27ae60")))
            painter.drawRoundedRect(380, 310, 200, 80, 10, 10)

            # Button text
            painter.setPen(QColor("#ffffff"))
            font = QFont("Arial", 14, QFont.Weight.Bold)
            painter.setFont(font)
            
            painter.drawText(120, 310, 200, 40, Qt.AlignmentFlag.AlignCenter,
                            "🔥 Nanothermite")
            painter.drawText(120, 350, 200, 40, Qt.AlignmentFlag.AlignCenter,
                            "Composites")
            
            painter.drawText(380, 310, 200, 40, Qt.AlignmentFlag.AlignCenter,
                            "🧪 Polymer")
            painter.drawText(380, 350, 200, 40, Qt.AlignmentFlag.AlignCenter,
                            "Composites")

            # Description text
            desc_font = QFont("Arial", 9)
            painter.setFont(desc_font)
            painter.setPen(QColor("#bdc3c7"))
            
            painter.drawText(120, 395, 200, 20, Qt.AlignmentFlag.AlignCenter,
                            "Al/CuO & Thermite Systems")
            painter.drawText(380, 395, 200, 20, Qt.AlignmentFlag.AlignCenter,
                            "Epoxy/Graphene & Polymers")

            # Small instruction
            inst_font = QFont("Arial", 9)
            painter.setFont(inst_font)
            painter.setPen(QColor("#95a5a6"))
            painter.drawText(0, 440, 700, 25, Qt.AlignmentFlag.AlignCenter,
                            "Click on a mode to start the application")

        else:
            # Loading text
            painter.setPen(QColor("#f39c12"))
            painter.drawText(0, 290, 700, 30, Qt.AlignmentFlag.AlignCenter,
                            "Initializing components...")

            # Progress bar background
            painter.fillRect(150, 330, 400, 8, QColor("#34495e"))
            # Progress bar fill
            bar_width = int(400 * self.progress_value / 100)
            painter.fillRect(150, 330, bar_width, 8, QColor("#e74c3c"))

        painter.end()
        self.setPixmap(self.pixmap)

    def update_progress(self, value, message=""):
        """Update progress bar and optional message."""
        self.progress_value = min(value, 100)
        
        if message:
            painter = QPainter(self.pixmap)
            painter.setPen(QColor("#f39c12"))
            painter.drawText(0, 290, 700, 30, Qt.AlignmentFlag.AlignCenter, message)
            painter.end()
        
        self._draw_splash(show_mode_buttons=False)
        self.show()
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

    def show_mode_selection(self):
        """Show mode selection buttons on the splash screen."""
        self._draw_splash(show_mode_buttons=True)
        self.show()
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

    def mousePressEvent(self, event):
        """Handle mouse click on splash screen for mode selection."""
        if not self.pixmap:
            return
        
        pos = event.position()
        x = pos.x()
        y = pos.y()

        # Check if click is within Nanothermite button area
        if 120 <= x <= 320 and 310 <= y <= 390:
            self.mode = "nanothermite"
            self.mode_selected.emit("nanothermite")
            painter = QPainter(self.pixmap)
            painter.setPen(QColor("#27ae60"))
            painter.drawText(0, 470, 700, 25, Qt.AlignmentFlag.AlignCenter,
                            "✅ Nanothermite Mode Selected - Loading...")
            painter.end()
            self.setPixmap(self.pixmap)
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            return

        # Check if click is within Polymer button area
        if 380 <= x <= 580 and 310 <= y <= 390:
            self.mode = "polymer"
            self.mode_selected.emit("polymer")
            painter = QPainter(self.pixmap)
            painter.setPen(QColor("#27ae60"))
            painter.drawText(0, 470, 700, 25, Qt.AlignmentFlag.AlignCenter,
                            "✅ Polymer Composite Mode Selected - Loading...")
            painter.end()
            self.setPixmap(self.pixmap)
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            return

    def finish_splash(self, main_window):
        """Finish splash and show main window."""
        self.finished.emit()  # Now this works because finished is defined
        self.finish(main_window)