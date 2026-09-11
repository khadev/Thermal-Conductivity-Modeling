"""About dialog for the application - Mode-aware using material_constants."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont

from ..core.material_constants import get_mode_info, get_mode_style


class AboutDialog(QDialog):
    """About dialog showing application information with mode awareness."""

    def __init__(self, mode="nanothermite", parent=None):
        super().__init__(parent)
        self.mode = mode
        mode_info = get_mode_info(mode)
        self.mode_label = mode_info.get("name", "Nanothermite")
        self.mode_icon = mode_info.get("icon", "🔥")
        self.setWindowTitle(f"About Thermal Conductivity Modeling Suite - {self.mode_label}")
        self.setMinimumSize(550, 480)
        self.setModal(True)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        styles = get_mode_style(self.mode)

        # Title
        title = QLabel("Thermal Conductivity Modeling Suite")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1a5490;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Version 3.0.0")
        subtitle.setStyleSheet("font-size: 14px; color: #7f8c8d;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Mode indicator
        mode_text = f"{self.mode_icon} {self.mode_label} Mode"
        mode_label = QLabel(mode_text)
        mode_label.setStyleSheet(f"""
            font-size: 14px; 
            font-weight: bold; 
            color: {styles['primary']};
            padding: 8px;
            background-color: {styles['background']};
            border-radius: 5px;
            border: 2px solid {styles['primary']};
        """)
        mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(mode_label)

        # Separator
        line = QLabel("─" * 50)
        line.setStyleSheet("color: #bdc3c7;")
        line.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(line)

        # Description - Mode specific
        if self.mode == "nanothermite":
            desc_text = (
                "Professional desktop application for experimental data analysis and modeling\n"
                "of thermal conductivity of energetic material composites (Al/CuO nanothermites)."
            )
        else:
            desc_text = (
                "Professional desktop application for experimental data analysis and modeling\n"
                "of thermal conductivity of polymer matrix composites (Epoxy/Graphene, etc.)."
            )

        desc = QLabel(desc_text)
        desc.setStyleSheet("font-size: 12px; color: #2c3e50;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Features group
        features_group = QGroupBox("Features")
        features_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        features_layout = QVBoxLayout(features_group)

        features = [
            "• 8+ Thermal Conductivity Models (Parallel, Series, Maxwell-Eucken, Bruggeman, Agari, Lewis-Nielsen, Percolation, etc.)",
            "• Nonlinear Least-Squares Fitting with Custom Parameter Control",
            "• Statistical Metrics: R², RMSE, MAE, MAPE",
            "• Publication-Ready Plots with Journal Presets",
            "• Export to PNG, SVG, PDF, TIFF, Excel, CSV",
            "• Professional PDF Report Generation",
            "• Dual Mode Support: Nanothermite & Polymer Composites"
        ]

        for feature in features:
            label = QLabel(feature)
            label.setStyleSheet("font-size: 11px; color: #2c3e50; padding: 2px;")
            label.setWordWrap(True)
            features_layout.addWidget(label)

        layout.addWidget(features_group)

        # Supported materials - Mode specific
        material_group = QGroupBox("Supported Materials")
        material_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        material_layout = QVBoxLayout(material_group)

        if self.mode == "nanothermite":
            materials = (
                "• Reductants: Al, Mg, Ti, Zr, B, Si, Fe, Ni, Zn, Sn, Cu, AlH₃, MgH₂, LiAlH₄, NaBH₄\n"
                "• Oxidizers: CuO, Fe₂O₃, MnO₂, MoO₃, WO₃, Bi₂O₃, SnO₂, PbO₂, Cr₂O₃, Co₃O₄, NiO, V₂O₅\n"
                "• Oxidizers (Halogenated): KClO₄, NH₄ClO₄, KNO₃, NaNO₃, NH₄NO₃, Teflon, PTFE, PFOA"
            )
        else:
            materials = (
                "• Polymer Matrices: Epoxy, PE, PP, PS, Polyimide, Silicone, PMMA, Nylon, PET, PEEK, PTFE\n"
                "• Conductive Fillers: Graphene, CNT, Al₂O₃, SiO₂, BN, SiC, Diamond, Cu, Al, Ag, Graphite, Carbon Black, ZnO, TiO₂"
            )

        mat_label = QLabel(materials)
        mat_label.setStyleSheet("font-size: 11px; color: #2c3e50; padding: 4px;")
        mat_label.setWordWrap(True)
        material_layout.addWidget(mat_label)

        layout.addWidget(material_group)

        # Developer info
        dev_group = QGroupBox("Developer")
        dev_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        dev_layout = QVBoxLayout(dev_group)

        dev_name = QLabel("Oukil Khaled ibn El-walid")
        dev_name.setStyleSheet("font-size: 13px; font-weight: bold; color: #1a5490;")
        dev_layout.addWidget(dev_name)

        dev_email = QLabel("Email: oukil.khaled@gmail.com")
        dev_email.setStyleSheet("font-size: 11px; color: #7f8c8d;")
        dev_layout.addWidget(dev_email)

        layout.addWidget(dev_group)

        # Acknowledgments
        ack_group = QGroupBox("Acknowledgments")
        ack_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        ack_layout = QVBoxLayout(ack_group)

        ack_text = QLabel(
            "This software was developed using:\n"
            "• PyQt6 - GUI Framework\n"
            "• NumPy/SciPy - Numerical Computing\n"
            "• Matplotlib - Plotting\n"
            "• Pandas - Data Management\n"
            "• ReportLab - PDF Generation"
        )
        ack_text.setStyleSheet("font-size: 11px; color: #2c3e50;")
        ack_text.setWordWrap(True)
        ack_layout.addWidget(ack_text)

        layout.addWidget(ack_group)

        # License
        license_label = QLabel("© 2026 Oukil Khaled ibn El-walid - Research. All Rights Reserved.")
        license_label.setStyleSheet("font-size: 10px; color: #95a5a6;")
        license_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(license_label)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(f"""
            QPushButton {{
                background-color: {styles['primary']};
                color: white;
                padding: 10px 30px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {styles['accent']}; }}
        """)
        btn_close.clicked.connect(self.accept)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)