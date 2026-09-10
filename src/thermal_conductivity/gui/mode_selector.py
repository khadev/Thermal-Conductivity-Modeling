
"""Mode Selection Screen - Initial application mode picker."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QGridLayout, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette, QPainter, QLinearGradient, QBrush


class ModeCard(QFrame):
    """Clickable mode card widget."""
    
    clicked = pyqtSignal(str)
    
    def __init__(self, mode_id, mode_data, parent=None):
        super().__init__(parent)
        self.mode_id = mode_id
        self.mode_data = mode_data
        self.setFrameStyle(QFrame.Shape.Box)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self._build_ui()
        self._apply_style()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        icon_label = QLabel(self.mode_data.get("icon", "🔬"))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 48px;")
        layout.addWidget(icon_label)
        
        title_label = QLabel(self.mode_data.get("name", "Unknown Mode"))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title_label)
        
        desc_label = QLabel(self.mode_data.get("description", ""))
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        layout.addWidget(desc_label)
        
        if self.mode_id == "nanothermite":
            examples = "Al/CuO · Mg/Fe₂O₃ · Ti/MnO₂"
        else:
            examples = "Epoxy/Graphene · PE/CNT · PP/Al₂O₃"
        
        examples_label = QLabel(examples)
        examples_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        examples_label.setWordWrap(True)
        examples_label.setStyleSheet("font-size: 10px; color: #95a5a6; font-style: italic;")
        layout.addWidget(examples_label)
        
        indicator = QFrame()
        indicator.setFrameStyle(QFrame.Shape.HLine)
        indicator.setStyleSheet(f"background-color: {self.mode_data.get('color_primary', '#1a5490')};")
        indicator.setFixedHeight(3)
        layout.addWidget(indicator)
    
    def _apply_style(self):
        primary = self.mode_data.get("color_primary", "#1a5490")
        bg = self.mode_data.get("surface", "#ffffff")
        text = self.mode_data.get("text", "#2c3e50")
        
        self.setStyleSheet(f"""
            ModeCard {{
                background-color: {bg};
                border: 2px solid #bdc3c7;
                border-radius: 12px;
                padding: 10px;
            }}
            ModeCard:hover {{
                border: 2px solid {primary};
                background-color: {bg}dd;
            }}
            ModeCard:pressed {{
                background-color: {bg}aa;
            }}
            ModeCard QLabel {{
                color: {text};
            }}
        """)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.mode_id)
        super().mousePressEvent(event)


class ModeSelectorDialog(QDialog):
    """Mode selection dialog shown at application startup."""
    
    mode_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Thermal Conductivity Modeling Suite")
        self.setMinimumSize(700, 450)
        self.setMaximumSize(900, 600)
        self.setModal(True)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        
        self.dragPosition = None
        
        self._build_ui()
        self._apply_styles()
    
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        header_layout = QVBoxLayout()
        header_layout.setSpacing(5)
        
        title = QLabel("🔬 Thermal Conductivity Modeling Suite")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1a5490;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("Select Analysis Mode")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 16px; color: #7f8c8d;")
        header_layout.addWidget(subtitle)
        
        version = QLabel("Version 2.0.0")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("font-size: 11px; color: #95a5a6;")
        header_layout.addWidget(version)
        
        main_layout.addLayout(header_layout)
        
        sep = QFrame()
        sep.setFrameStyle(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #bdc3c7;")
        sep.setFixedHeight(2)
        main_layout.addWidget(sep)
        
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(20)
        cards_layout.setContentsMargins(20, 20, 20, 20)
        
        try:
            from ..core.material_constants import MODES
        except ImportError:
            MODES = {
                "nanothermite": {
                    "name": "Nanothermite",
                    "icon": "🔥",
                    "description": "Energetic Material Composites",
                    "color_primary": "#c0392b",
                    "surface": "#2d2d44",
                    "text": "#ffffff"
                },
                "polymer": {
                    "name": "Polymer Composite",
                    "icon": "🧪",
                    "description": "Polymer Matrix Composites",
                    "color_primary": "#1a5490",
                    "surface": "#ffffff",
                    "text": "#2c3e50"
                }
            }
        
        for mode_id, mode_data in MODES.items():
            card = ModeCard(mode_id, mode_data)
            card.clicked.connect(self._on_mode_selected)
            cards_layout.addWidget(card)
        
        main_layout.addLayout(cards_layout)
        
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()
        developer = QLabel("Developed by Oukil Khaled ibn El-walid")
        developer.setStyleSheet("font-size: 10px; color: #95a5a6;")
        footer_layout.addWidget(developer)
        footer_layout.addStretch()
        main_layout.addLayout(footer_layout)
    
    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                          stop: 0 #ecf0f1,
                                          stop: 1 #ffffff);
                border-radius: 15px;
            }
        """)
    
    def _on_mode_selected(self, mode_id):
        self.mode_selected.emit(mode_id)
        self.accept()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragPosition = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.dragPosition is not None:
            self.move(event.globalPosition().toPoint() - self.dragPosition)
            event.accept()
        else:
            super().mouseMoveEvent(event)
