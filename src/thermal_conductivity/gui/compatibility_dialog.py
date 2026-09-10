"""
Professional Compatibility Dialog
==================================
Custom Qt dialog that replaces the default QMessageBox for
model compatibility warnings.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QFrame, QSizePolicy,
    QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor


class CompatibilityDialog(QDialog):
    """
    Professional, reactive dialog for reporting model-compatibility issues.

    Two modes:
        - 'error'  : models are incompatible → Yes/No to continue
        - 'warning': models have warnings    → OK to acknowledge
    """

    def __init__(
        self,
        title,
        subtitle,
        data_trend,
        incompatible_models,
        warned_models,
        compatibility_results,
        recommended_models=None,
        mode="polymer",
        show_continue=True,
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setMinimumSize(780, 640)
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint
        )

        self.mode = mode
        self.data_trend = data_trend
        self.incompatible_models = incompatible_models
        self.warned_models = warned_models
        self.compatibility_results = compatibility_results
        self.recommended_models = recommended_models or []
        self.show_continue = show_continue
        self.user_choice = False

        self._build_ui()
        self._apply_styles()

    # ========================================================================
    # UI CONSTRUCTION
    # ========================================================================

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---------- HEADER BANNER ----------
        header = QFrame()
        header.setObjectName("header")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(28, 22, 28, 22)
        header_layout.setSpacing(6)

        icon_lbl = QLabel("⚠" if self.incompatible_models else "ℹ")
        icon_lbl.setObjectName("headerIcon")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(icon_lbl)

        title_lbl = QLabel(self.windowTitle())
        title_lbl.setObjectName("headerTitle")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_lbl)

        subtitle_lbl = QLabel(self._build_subtitle_text())
        subtitle_lbl.setObjectName("headerSubtitle")
        subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_lbl.setWordWrap(True)
        header_layout.addWidget(subtitle_lbl)

        root.addWidget(header)

        # ---------- BODY (scrollable) ----------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("scrollArea")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        body = QWidget()
        body.setObjectName("body")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(28, 22, 28, 22)
        body_layout.setSpacing(18)

        # Data summary card
        body_layout.addWidget(self._build_data_card())

        # Incompatible models
        if self.incompatible_models:
            body_layout.addWidget(
                self._build_models_section(
                    "Incompatible Models",
                    self.incompatible_models,
                    "sectionTitleError",
                    "modelCardError",
                    "❌",
                )
            )

        # Warned models
        if self.warned_models:
            body_layout.addWidget(
                self._build_models_section(
                    "Models with Warnings",
                    self.warned_models,
                    "sectionTitleWarn",
                    "modelCardWarn",
                    "⚠",
                )
            )

        # Recommended models
        if self.incompatible_models and self.recommended_models:
            body_layout.addWidget(self._build_recommended_card())

        body_layout.addStretch()
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        # ---------- FOOTER BUTTONS ----------
        footer = QFrame()
        footer.setObjectName("footer")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(28, 16, 28, 16)
        footer_layout.setSpacing(12)

        if self.show_continue:
            self.btn_cancel = QPushButton("✕  Cancel Analysis")
            self.btn_cancel.setObjectName("btnCancel")
            self.btn_cancel.clicked.connect(self._on_cancel)
            footer_layout.addWidget(self.btn_cancel)

            footer_layout.addStretch()

            self.btn_continue = QPushButton("▶  Continue Anyway")
            self.btn_continue.setObjectName("btnContinue")
            self.btn_continue.clicked.connect(self._on_continue)
            footer_layout.addWidget(self.btn_continue)
        else:
            footer_layout.addStretch()
            self.btn_ok = QPushButton("✓  OK, I Understand")
            self.btn_ok.setObjectName("btnOk")
            self.btn_ok.clicked.connect(self._on_continue)
            self.btn_ok.setDefault(True)
            footer_layout.addWidget(self.btn_ok)

        root.addWidget(footer)

    # ------------------------------------------------------------------------
    # Card builders
    # ------------------------------------------------------------------------

    def _build_subtitle_text(self):
        if self.incompatible_models:
            n = len(self.incompatible_models)
            return (
                f"{n} model{'s' if n > 1 else ''} "
                f"{'are' if n > 1 else 'is'} not physically compatible "
                f"with your data."
            )
        n = len(self.warned_models)
        return (
            f"{n} model{'s' if n > 1 else ''} may give less reliable results."
        )

    def _build_data_card(self):
        card = QFrame()
        card.setObjectName("dataCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)

        title = QLabel("Data Summary")
        title.setObjectName("cardTitle")
        layout.addWidget(title)

        dt = self.data_trend
        mode_label = "Polymer Composite" if self.mode == "polymer" else "Nanothermite"

        trend_symbol = {
            "increasing": "⬆  Increasing",
            "decreasing": "⬇  Decreasing",
            "flat": "→  Flat",
        }.get(dt["trend"], dt["trend"])

        rows = [
            ("Mode", mode_label),
            ("φ range", f"{dt['phi_start']:.3f} → {dt['phi_end']:.3f}"),
            ("k range", f"{dt['k_start']:.3f} → {dt['k_end']:.3f} W/m·K"),
            ("Trend", trend_symbol),
            ("Magnitude", f"{dt['magnitude_range']:.2f}×"),
        ]

        for label, value in rows:
            row = QHBoxLayout()
            row.setSpacing(10)
            lbl = QLabel(f"<b>{label}:</b>")
            lbl.setObjectName("dataLabel")
            lbl.setFixedWidth(110)
            val = QLabel(value)
            val.setObjectName("dataValue")
            val.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred
            )
            row.addWidget(lbl)
            row.addWidget(val)
            layout.addLayout(row)

        return card

    def _build_models_section(self, section_title, model_names,
                              title_obj_name, card_obj_name, icon):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel(section_title)
        title.setObjectName(title_obj_name)
        layout.addWidget(title)

        for name in model_names:
            res = self.compatibility_results.get(name, {})
            layout.addWidget(
                self._build_model_card(name, res.get("details", []),
                                       card_obj_name, icon)
            )

        return container

    def _build_model_card(self, model_name, reasons, obj_name, icon):
        """Build a single model card with auto-height reason labels."""
        card = QFrame()
        card.setObjectName(obj_name)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Model name row
        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        icon_lbl = QLabel(icon)
        icon_lbl.setObjectName("modelIcon")
        name_lbl = QLabel(model_name)
        name_lbl.setObjectName("modelName")
        name_row.addWidget(icon_lbl)
        name_row.addWidget(name_lbl)
        name_row.addStretch()
        layout.addLayout(name_row)

        # Reason paragraphs (auto-height QLabels)
        for reason in reasons:
            text = QLabel()
            text.setObjectName("reasonText")
            text.setTextFormat(Qt.TextFormat.RichText)
            text.setWordWrap(True)
            text.setText(reason)
            text.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )
            text.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Minimum
            )
            layout.addWidget(text)

        return card

    def _build_recommended_card(self):
        card = QFrame()
        card.setObjectName("recommendCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)

        title = QLabel("✅  Recommended Models for This Data")
        title.setObjectName("cardTitleGreen")
        layout.addWidget(title)

        chips_row = QHBoxLayout()
        chips_row.setSpacing(8)
        for name in self.recommended_models:
            chip = QLabel(name)
            chip.setObjectName("chip")
            chips_row.addWidget(chip)
        chips_row.addStretch()
        layout.addLayout(chips_row)

        return card

    # ========================================================================
    # STYLES
    # ========================================================================

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog { background-color: #f5f6fa; }

            QFrame#header {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #c0392b, stop:1 #e74c3c
                );
                border-bottom: 3px solid #8b1e12;
            }
            QLabel#headerIcon {
                font-size: 42px; color: white; background: transparent;
            }
            QLabel#headerTitle {
                font-size: 20px; font-weight: bold;
                color: white; background: transparent;
            }
            QLabel#headerSubtitle {
                font-size: 12px; color: #fadbd8; background: transparent;
            }

            QWidget#body { background-color: #f5f6fa; }
            QScrollArea#scrollArea { background-color: #f5f6fa; border: none; }
            QScrollBar:vertical {
                background: #ecf0f1; width: 10px; border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #bdc3c7; border-radius: 5px; min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background: #95a5a6; }

            QFrame#dataCard {
                background-color: white;
                border: 1px solid #d5d8dc;
                border-radius: 8px;
            }
            QLabel#cardTitle {
                font-size: 13px; font-weight: bold; color: #2c3e50;
            }
            QLabel#dataLabel { color: #7f8c8d; font-size: 11px; }
            QLabel#dataValue { color: #2c3e50; font-size: 11px; }

            QLabel#sectionTitleError {
                font-size: 14px; font-weight: bold;
                color: #c0392b; padding: 4px 0;
            }
            QLabel#sectionTitleWarn {
                font-size: 14px; font-weight: bold;
                color: #d68910; padding: 4px 0;
            }

            QFrame#modelCardError {
                background-color: #fdedec;
                border-left: 4px solid #c0392b;
                border-radius: 6px;
            }
            QFrame#modelCardWarn {
                background-color: #fef5e7;
                border-left: 4px solid #d68910;
                border-radius: 6px;
            }
            QLabel#modelIcon {
                font-size: 14px; background: transparent;
            }
            QLabel#modelName {
                font-size: 12px; font-weight: bold; color: #2c3e50;
                background: transparent;
            }
            QLabel#reasonText {
                background: transparent;
                border: none;
                font-size: 11px;
                color: #2c3e50;
                padding: 0;
                margin: 0;
            }

            QFrame#recommendCard {
                background-color: #eafaf1;
                border: 1px solid #a9dfbf;
                border-radius: 8px;
            }
            QLabel#cardTitleGreen {
                font-size: 13px; font-weight: bold; color: #1e8449;
            }
            QLabel#chip {
                background-color: #abebc6;
                color: #145a32;
                border-radius: 10px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: bold;
            }

            QFrame#footer {
                background-color: #ffffff;
                border-top: 1px solid #d5d8dc;
            }

            QPushButton#btnCancel {
                background-color: #ecf0f1;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                padding: 10px 22px;
                font-weight: bold;
                font-size: 12px;
                min-width: 160px;
            }
            QPushButton#btnCancel:hover {
                background-color: #dfe6e9;
                border-color: #95a5a6;
            }
            QPushButton#btnCancel:pressed { background-color: #cfd8dc; }

            QPushButton#btnContinue {
                background-color: #c0392b;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 22px;
                font-weight: bold;
                font-size: 12px;
                min-width: 180px;
            }
            QPushButton#btnContinue:hover { background-color: #e74c3c; }
            QPushButton#btnContinue:pressed { background-color: #a93226; }

            QPushButton#btnOk {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 26px;
                font-weight: bold;
                font-size: 12px;
                min-width: 180px;
            }
            QPushButton#btnOk:hover { background-color: #2ecc71; }
            QPushButton#btnOk:pressed { background-color: #1e8449; }
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 6)
        self.setGraphicsEffect(shadow)

    # ========================================================================
    # HANDLERS
    # ========================================================================

    def _on_continue(self):
        self.user_choice = True
        self.accept()

    def _on_cancel(self):
        self.user_choice = False
        self.reject()

    def user_wants_to_continue(self):
        return self.user_choice
