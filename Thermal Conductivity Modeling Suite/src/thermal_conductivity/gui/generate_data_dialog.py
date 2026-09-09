"""Generate Data Dialog - Mode-aware synthetic data generation using material_constants."""

import numpy as np
import pandas as pd
import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QGroupBox, QWidget, QCheckBox, QDoubleSpinBox, QSpinBox,
    QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QTextEdit, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor

from ..core.models import (
    parallel_model, series_model, maxwell_eucken,
    bruggeman_model, agari_model, lewis_nielsen_model,
    percolation_model, agari_percolation_model
)
from ..core.material_constants import (
    get_mode_info, get_materials_for_mode,
    get_default_materials, get_k_value,
    get_mode_style
)


class GenerateDataDialog(QDialog):
    """Dialog for generating synthetic data for research and teaching."""

    def __init__(self, mode="nanothermite", parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.mode = mode
        
        # Get mode info
        mode_info = get_mode_info(mode)
        self.mode_label = mode_info.get("name", "Nanothermite")
        self.mode_icon = mode_info.get("icon", "🔬")
        
        materials = get_materials_for_mode(mode)
        self.matrix_list = materials.get("matrix_list", [])
        self.filler_list = materials.get("filler_list", [])
        self.matrix_label = mode_info.get("matrix_label", "Matrix")
        self.filler_label = mode_info.get("filler_label", "Filler")
        
        defaults = get_default_materials(mode)
        self.default_matrix = defaults.get("matrix", "Al")
        self.default_filler = defaults.get("filler", "CuO")
        
        self.generated_data = None
        self.main_tabs = None

        self.setWindowTitle(f"📊 Generate Synthetic Data - {self.mode_label}")
        self.setMinimumSize(950, 750)
        self.setModal(True)

        self._build_ui()
        self._setup_defaults()

    def _build_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title with mode
        title = QLabel(f"📊 Generate Synthetic Data for {self.mode_label}")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1a5490;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel(
            f"Generate realistic thermal conductivity data for {self.mode_label} composites.\n"
            "Use this tool to create synthetic datasets for teaching, research, and model validation."
        )
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        layout.addWidget(subtitle)

        # Main content
        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #bdc3c7; background: white; border-radius: 4px; }
            QTabBar::tab { padding: 8px 16px; font-weight: bold; }
            QTabBar::tab:selected { background: #1a5490; color: white; }
        """)

        # Tab 1: Parameters
        self.tab_params = QWidget()
        self._build_parameters_tab()
        self.main_tabs.addTab(self.tab_params, "📐 Parameters")

        # Tab 2: Presets
        self.tab_presets = QWidget()
        self._build_presets_tab()
        self.main_tabs.addTab(self.tab_presets, "⚡ Presets")

        # Tab 3: Preview
        self.tab_preview = QWidget()
        self._build_preview_tab()
        self.main_tabs.addTab(self.tab_preview, "👁️ Preview")

        # Tab 4: Advanced
        self.tab_advanced = QWidget()
        self._build_advanced_tab()
        self.main_tabs.addTab(self.tab_advanced, "⚙️ Advanced")

        layout.addWidget(self.main_tabs, 1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        btn_generate = QPushButton("🚀 Generate Data")
        btn_generate.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 10px 30px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2ecc71; }
        """)
        btn_generate.clicked.connect(self.generate_data)
        btn_layout.addWidget(btn_generate)

        btn_save = QPushButton("💾 Save Data")
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px 30px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        btn_save.clicked.connect(self.save_data)
        btn_save.setEnabled(False)
        btn_layout.addWidget(btn_save)

        btn_save_multi = QPushButton("📁 Save Multiple")
        btn_save_multi.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                color: white;
                padding: 10px 30px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #9b59b6; }
        """)
        btn_save_multi.clicked.connect(self.save_multiple)
        btn_save_multi.setEnabled(False)
        btn_layout.addWidget(btn_save_multi)

        btn_layout.addStretch()

        btn_close = QPushButton("✕ Close")
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 10px 25px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def _build_parameters_tab(self):
        """Build the parameters tab."""
        layout = QVBoxLayout(self.tab_params)
        layout.setSpacing(15)

        # Material selection
        material_group = QGroupBox(f"🧪 Material Parameters - {self.mode_label}")
        material_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        material_layout = QFormLayout(material_group)

        # Matrix selection
        self.matrix_combo = QComboBox()
        self.matrix_combo.addItems(self.matrix_list)
        if self.default_matrix in self.matrix_list:
            self.matrix_combo.setCurrentText(self.default_matrix)
        self.matrix_combo.setToolTip(f"Select the {self.matrix_label}")
        self.matrix_combo.currentTextChanged.connect(self._update_material_info)
        material_layout.addRow(f"{self.matrix_label}:", self.matrix_combo)

        # Filler selection
        self.filler_combo = QComboBox()
        self.filler_combo.addItems(self.filler_list)
        if self.default_filler in self.filler_list:
            self.filler_combo.setCurrentText(self.default_filler)
        self.filler_combo.setToolTip(f"Select the {self.filler_label}")
        self.filler_combo.currentTextChanged.connect(self._update_material_info)
        material_layout.addRow(f"{self.filler_label}:", self.filler_combo)

        # Material info
        self.material_info_label = QLabel("")
        self.material_info_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        material_layout.addRow("", self.material_info_label)

        layout.addWidget(material_group)

        # Volume fraction parameters
        phi_group = QGroupBox(f"📊 Volume Fraction Parameters")
        phi_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        phi_layout = QFormLayout(phi_group)

        # Number of points
        self.num_points_spin = QSpinBox()
        self.num_points_spin.setRange(3, 50)
        self.num_points_spin.setValue(10)
        phi_layout.addRow("Number of Points:", self.num_points_spin)

        # Phi range
        self.phi_min_spin = QDoubleSpinBox()
        self.phi_min_spin.setRange(0.0, 0.99)
        self.phi_min_spin.setValue(0.0)
        self.phi_min_spin.setSingleStep(0.01)
        self.phi_min_spin.setDecimals(3)
        phi_layout.addRow("φ Min:", self.phi_min_spin)

        self.phi_max_spin = QDoubleSpinBox()
        self.phi_max_spin.setRange(0.01, 1.0)
        if self.mode == "polymer":
            self.phi_max_spin.setValue(0.3)  # Polymer composites typically up to 30%
        else:
            self.phi_max_spin.setValue(1.0)
        self.phi_max_spin.setSingleStep(0.01)
        self.phi_max_spin.setDecimals(3)
        phi_layout.addRow("φ Max:", self.phi_max_spin)

        # Distribution type
        self.distribution_combo = QComboBox()
        self.distribution_combo.addItems(["Linear", "Logarithmic", "Random"])
        phi_layout.addRow("Distribution:", self.distribution_combo)

        layout.addWidget(phi_group)

        # Model selection
        model_group = QGroupBox("📐 Generation Model")
        model_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        model_layout = QFormLayout(model_group)

        # For Polymer mode, show models that give INCREASING conductivity
        if self.mode == "polymer":
            model_list = [
                "Agari (Recommended for Polymer)",
                "Maxwell-Eucken",
                "Bruggeman",
                "Lewis-Nielsen",
                "Percolation",
                "Agari-Percolation"
            ]
        else:
            model_list = [
                "Agari-Percolation (Recommended)",
                "Agari",
                "Bruggeman",
                "Maxwell-Eucken",
                "Parallel",
                "Series",
                "Lewis-Nielsen",
                "Percolation"
            ]
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(model_list)
        self.model_combo.setCurrentIndex(0)
        model_layout.addRow("Base Model:", self.model_combo)

        layout.addWidget(model_group)

        # Add note about conductivity trend
        note_label = QLabel()
        if self.mode == "polymer":
            note_label.setText("✅ Polymer Mode: Conductivity INCREASES with filler fraction (φ)")
            note_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px; background-color: #d4edda; border-radius: 4px;")
        else:
            note_label.setText("🔥 Nanothermite Mode: Conductivity DECREASES with filler fraction (φ)")
            note_label.setStyleSheet("color: #c0392b; font-weight: bold; padding: 5px; background-color: #f8d7da; border-radius: 4px;")
        layout.addWidget(note_label)

        layout.addStretch()

    def _build_presets_tab(self):
        """Build the presets tab with mode-specific presets."""
        layout = QVBoxLayout(self.tab_presets)
        layout.setSpacing(15)

        desc = QLabel(
            f"Select a preset to automatically configure noise and outlier settings.\n"
            f"Each preset is designed for a specific research or teaching purpose."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #7f8c8d; padding: 5px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        presets_grid = QGridLayout()
        presets_grid.setSpacing(15)

        if self.mode == "polymer":
            presets = self._get_polymer_presets()
        else:
            presets = self._get_nanothermite_presets()

        row = 0
        col = 0
        for preset in presets:
            btn = QPushButton()
            btn.setText(f"{preset['icon']}\n{preset['name']}")
            btn.setToolTip(
                f"{preset['name']}\n\n"
                f"{preset['desc']}\n\n"
                f"Settings:\n"
                f"• Noise Type: {preset['noise_type']}\n"
                f"• Noise Level: {preset['noise_std'] if preset['noise_std'] > 0 else preset['noise_percent']}\n"
                f"• Outliers: {preset['outlier_count']} at {preset['outlier_magnitude']}%"
            )
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f4f8;
                    border: 2px solid #bdc3c7;
                    border-radius: 8px;
                    padding: 15px;
                    font-weight: bold;
                    font-size: 14px;
                    min-height: 80px;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #e8f0fe;
                    border-color: #1a5490;
                }
                QPushButton:pressed {
                    background-color: #d4e0f0;
                }
            """)
            btn.clicked.connect(lambda checked, p=preset: self._apply_preset(p))
            presets_grid.addWidget(btn, row, col)

            col += 1
            if col > 2:
                col = 0
                row += 1

        layout.addLayout(presets_grid)

        self.preset_info_group = QGroupBox("📋 Current Preset Configuration")
        self.preset_info_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        preset_info_layout = QVBoxLayout(self.preset_info_group)

        self.preset_info_label = QLabel("No preset selected. Configure manually in the Advanced tab.")
        self.preset_info_label.setWordWrap(True)
        self.preset_info_label.setStyleSheet("font-size: 11px; color: #7f8c8d; padding: 5px;")
        preset_info_layout.addWidget(self.preset_info_label)

        layout.addWidget(self.preset_info_group)

        layout.addStretch()

    def _get_polymer_presets(self):
        """Get polymer-specific presets with INCREASING conductivity."""
        return [
            {
                "name": "📚 Teaching",
                "desc": "Ideal data for teaching\nNo noise, no outliers\nConductivity INCREASES with filler",
                "icon": "📚",
                "noise_type": "Gaussian (Normal)",
                "noise_std": 0.0,
                "noise_percent": 0.0,
                "outlier_count": 0,
                "outlier_magnitude": 0.0
            },
            {
                "name": "⭐ Ideal Data",
                "desc": "Perfect theoretical data\nNo noise, no outliers\nConductivity INCREASES with filler",
                "icon": "⭐",
                "noise_type": "Gaussian (Normal)",
                "noise_std": 0.0,
                "noise_percent": 0.0,
                "outlier_count": 0,
                "outlier_magnitude": 0.0
            },
            {
                "name": "📊 Realistic Data",
                "desc": "Typical experimental data\nSmall noise, 1 outlier\nConductivity INCREASES with filler",
                "icon": "📊",
                "noise_type": "Gaussian (Normal)",
                "noise_std": 0.02,
                "noise_percent": 0.0,
                "outlier_count": 1,
                "outlier_magnitude": 20.0
            },
            {
                "name": "🔬 Research Grade",
                "desc": "High-quality research data\nVery low noise, minimal outliers\nConductivity INCREASES with filler",
                "icon": "🔬",
                "noise_type": "Gaussian (Normal)",
                "noise_std": 0.005,
                "noise_percent": 0.0,
                "outlier_count": 0,
                "outlier_magnitude": 0.0
            },
            {
                "name": "📈 Percolation Effect",
                "desc": "Shows percolation threshold\nConductivity INCREASES sharply\nTypical for CNT/polymer",
                "icon": "📈",
                "noise_type": "Gaussian (Normal)",
                "noise_std": 0.01,
                "noise_percent": 0.0,
                "outlier_count": 0,
                "outlier_magnitude": 0.0
            },
            {
                "name": "⚠️ Worst Case",
                "desc": "Extreme conditions\nHigh noise, 2 outliers\nConductivity still INCREASES",
                "icon": "⚠️",
                "noise_type": "Percentage",
                "noise_std": 0.0,
                "noise_percent": 15.0,
                "outlier_count": 2,
                "outlier_magnitude": 50.0
            }
        ]

    def _get_nanothermite_presets(self):
        """Get nanothermite-specific presets with DECREASING conductivity."""
        return [
            {
                "name": "📚 Teaching",
                "desc": "Ideal data for teaching\nNo noise, no outliers",
                "icon": "📚",
                "noise_type": "Gaussian (Normal)",
                "noise_std": 0.0,
                "noise_percent": 0.0,
                "outlier_count": 0,
                "outlier_magnitude": 0.0
            },
            {
                "name": "⭐ Ideal Data",
                "desc": "Perfect theoretical data\nNo noise, no outliers",
                "icon": "⭐",
                "noise_type": "Gaussian (Normal)",
                "noise_std": 0.0,
                "noise_percent": 0.0,
                "outlier_count": 0,
                "outlier_magnitude": 0.0
            },
            {
                "name": "📊 Realistic Data",
                "desc": "Typical experimental data\nGaussian noise, 1 outlier",
                "icon": "📊",
                "noise_type": "Gaussian (Normal)",
                "noise_std": 2.0,
                "noise_percent": 0.0,
                "outlier_count": 1,
                "outlier_magnitude": 20.0
            },
            {
                "name": "🔬 Research Grade",
                "desc": "High-quality research data\nLow noise, minimal outliers",
                "icon": "🔬",
                "noise_type": "Gaussian (Normal)",
                "noise_std": 1.0,
                "noise_percent": 0.0,
                "outlier_count": 1,
                "outlier_magnitude": 10.0
            },
            {
                "name": "🛡️ Robustness Test",
                "desc": "Test model robustness\nHigher noise, 2-3 outliers",
                "icon": "🛡️",
                "noise_type": "Gaussian (Normal)",
                "noise_std": 4.0,
                "noise_percent": 0.0,
                "outlier_count": 3,
                "outlier_magnitude": 30.0
            },
            {
                "name": "⚠️ Worst Case",
                "desc": "Extreme conditions\nPercentage noise, 2 outliers",
                "icon": "⚠️",
                "noise_type": "Percentage",
                "noise_std": 0.0,
                "noise_percent": 15.0,
                "outlier_count": 2,
                "outlier_magnitude": 50.0
            }
        ]

    def _apply_preset(self, preset):
        self.noise_enabled.setChecked(preset['noise_std'] > 0 or preset['noise_percent'] > 0)

        if preset['noise_type'] == "Gaussian (Normal)":
            self.noise_type_combo.setCurrentIndex(0)
        elif preset['noise_type'] == "Uniform":
            self.noise_type_combo.setCurrentIndex(1)
        else:
            self.noise_type_combo.setCurrentIndex(2)

        self.noise_std_spin.setValue(preset['noise_std'])
        self.noise_percent_spin.setValue(preset['noise_percent'])

        self.outlier_enabled.setChecked(preset['outlier_count'] > 0)
        self.outlier_count_spin.setValue(preset['outlier_count'])
        self.outlier_magnitude_spin.setValue(preset['outlier_magnitude'])

        self.preset_info_label.setText(
            f"✅ Applied: {preset['name']}\n\n"
            f"{preset['desc']}\n\n"
            f"📊 Settings:\n"
            f"• Noise Type: {preset['noise_type']}\n"
            f"• Noise Level: {preset['noise_std'] if preset['noise_std'] > 0 else preset['noise_percent']}\n"
            f"• Outliers: {preset['outlier_count']} at {preset['outlier_magnitude']}%"
        )
        self.preset_info_label.setStyleSheet("font-size: 11px; color: #27ae60; padding: 5px; font-weight: bold;")

        if self.main_tabs:
            self.main_tabs.setCurrentIndex(3)

        QMessageBox.information(self, "Preset Applied",
            f"✅ Preset '{preset['name']}' applied successfully!\n\n"
            f"{preset['desc']}\n\n"
            f"Switch to the 'Advanced' tab to see the applied settings.")

    def _build_preview_tab(self):
        layout = QVBoxLayout(self.tab_preview)

        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(4)
        if self.mode == "nanothermite":
            headers = ["φ (Filler)", "1-φ (Matrix)", "k_meas (W/m·K)", "k_std_dev (±)"]
        else:
            headers = ["φ (Filler)", "1-φ (Matrix)", "k_meas (W/m·K)", "k_std_dev (±)"]
        self.preview_table.setHorizontalHeaderLabels(headers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setStyleSheet("""
            QTableWidget { font-size: 11px; }
            QHeaderView::section { background-color: #1a5490; color: white; padding: 6px; font-weight: bold; }
        """)
        layout.addWidget(self.preview_table)

        self.preview_stats = QLabel("")
        self.preview_stats.setStyleSheet("font-size: 12px; color: #7f8c8d; padding: 5px;")
        self.preview_stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview_stats)

    def _build_advanced_tab(self):
        layout = QVBoxLayout(self.tab_advanced)
        layout.setSpacing(15)

        # Noise group
        noise_group = QGroupBox("📈 Noise & Uncertainty")
        noise_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        noise_layout = QFormLayout(noise_group)

        self.noise_enabled = QCheckBox("Add Measurement Noise")
        self.noise_enabled.setToolTip("Enable to add random errors to the generated data.")
        self.noise_enabled.setChecked(True)
        self.noise_enabled.toggled.connect(self._update_noise_widgets)
        noise_layout.addRow("", self.noise_enabled)

        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(["Gaussian (Normal)", "Uniform", "Percentage"])
        self.noise_type_combo.setCurrentIndex(0)
        noise_layout.addRow("Noise Type:", self.noise_type_combo)

        self.noise_std_spin = QDoubleSpinBox()
        self.noise_std_spin.setRange(0.0, 100.0)
        if self.mode == "polymer":
            self.noise_std_spin.setValue(0.02)  # Small noise for polymer
        else:
            self.noise_std_spin.setValue(2.0)
        self.noise_std_spin.setSingleStep(0.01)
        self.noise_std_spin.setDecimals(3)
        noise_layout.addRow("Noise Level (σ):", self.noise_std_spin)

        self.noise_percent_spin = QDoubleSpinBox()
        self.noise_percent_spin.setRange(0.0, 50.0)
        self.noise_percent_spin.setValue(5.0)
        self.noise_percent_spin.setSingleStep(1.0)
        self.noise_percent_spin.setDecimals(1)
        noise_layout.addRow("Noise (%):", self.noise_percent_spin)

        layout.addWidget(noise_group)

        # Outlier group
        outlier_group = QGroupBox("🎯 Outliers")
        outlier_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        outlier_layout = QFormLayout(outlier_group)

        self.outlier_enabled = QCheckBox("Add Outliers")
        self.outlier_enabled.setToolTip("Enable to add extreme values (outliers) to the data.")
        self.outlier_enabled.setChecked(False)
        self.outlier_enabled.toggled.connect(self._update_outlier_widgets)
        outlier_layout.addRow("", self.outlier_enabled)

        self.outlier_count_spin = QSpinBox()
        self.outlier_count_spin.setRange(0, 5)
        self.outlier_count_spin.setValue(1)
        outlier_layout.addRow("Number of Outliers:", self.outlier_count_spin)

        self.outlier_magnitude_spin = QDoubleSpinBox()
        self.outlier_magnitude_spin.setRange(0.0, 100.0)
        self.outlier_magnitude_spin.setValue(20.0)
        self.outlier_magnitude_spin.setSingleStep(5.0)
        self.outlier_magnitude_spin.setDecimals(1)
        outlier_layout.addRow("Outlier Magnitude (%):", self.outlier_magnitude_spin)

        layout.addWidget(outlier_group)

        # Save options
        save_group = QGroupBox("💾 Save Options")
        save_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        save_layout = QFormLayout(save_group)

        self.filename_prefix = QLineEdit()
        if self.mode == "polymer":
            self.filename_prefix.setText("Polymer_Composite")
        else:
            self.filename_prefix.setText("Nanothermite")
        save_layout.addRow("Filename Prefix:", self.filename_prefix)

        self.save_folder = QLineEdit()
        self.save_folder.setText(os.getcwd())
        save_layout.addRow("Save Folder:", self.save_folder)

        btn_folder = QPushButton("Browse...")
        btn_folder.clicked.connect(self._browse_folder)
        save_layout.addRow("", btn_folder)

        layout.addWidget(save_group)

        layout.addStretch()

    def _update_noise_widgets(self, enabled):
        self.noise_type_combo.setEnabled(enabled)
        self.noise_std_spin.setEnabled(enabled)
        self.noise_percent_spin.setEnabled(enabled and self.noise_type_combo.currentIndex() == 2)

    def _update_outlier_widgets(self, enabled):
        self.outlier_count_spin.setEnabled(enabled)
        self.outlier_magnitude_spin.setEnabled(enabled)

    def _setup_defaults(self):
        self._update_material_info()
        self._update_noise_widgets(True)
        self._update_outlier_widgets(False)

    def _update_material_info(self):
        matrix = self.matrix_combo.currentText()
        filler = self.filler_combo.currentText()

        k_matrix = get_k_value(self.mode, "matrix", matrix) if matrix else 0
        k_filler = get_k_value(self.mode, "filler", filler) if filler else 0

        if self.mode == "polymer":
            trend = "⬆️ INCREASING"
            trend_color = "#27ae60"
        else:
            trend = "⬇️ DECREASING"
            trend_color = "#c0392b"

        self.material_info_label.setText(
            f"k_matrix ({matrix}) = {k_matrix:.2f} W/m·K | "
            f"k_filler ({filler}) = {k_filler:.2f} W/m·K | "
            f"<span style='color: {trend_color}; font-weight: bold;'>Trend: {trend}</span>"
        )
        self.material_info_label.setTextFormat(Qt.TextFormat.RichText)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder", self.save_folder.text())
        if folder:
            self.save_folder.setText(folder)

    def _get_k_matrix(self, material):
        return get_k_value(self.mode, "matrix", material)

    def _get_k_filler(self, material):
        return get_k_value(self.mode, "filler", material)

    def generate_data(self):
        try:
            num_points = self.num_points_spin.value()
            phi_min = self.phi_min_spin.value()
            phi_max = self.phi_max_spin.value()
            distribution = self.distribution_combo.currentText()
            model_name = self.model_combo.currentText()

            if phi_min >= phi_max:
                QMessageBox.warning(self, "Warning", "φ Min must be less than φ Max.")
                return

            if distribution == "Linear":
                phi = np.linspace(phi_min, phi_max, num_points)
            elif distribution == "Logarithmic":
                if phi_min <= 0:
                    phi_min = 0.001
                phi = np.logspace(np.log10(phi_min), np.log10(phi_max), num_points)
                phi = np.clip(phi, 0, 0.99)
            else:
                phi = np.random.uniform(phi_min, phi_max, num_points)
                phi = np.sort(phi)

            matrix = self.matrix_combo.currentText()
            filler = self.filler_combo.currentText()
            k_matrix = self._get_k_matrix(matrix)
            k_filler = self._get_k_filler(filler)

            # Store the noise standard deviation for saving
            noise_std_value = self.noise_std_spin.value()
            noise_type = self.noise_type_combo.currentText()

            model_func = self._get_model_function(model_name)
            k_meas = model_func(phi, k_matrix=k_matrix, k_filler=k_filler)

            # ===== IMPORTANT: Use the same noise for preview and saving =====
            if self.noise_enabled.isChecked():
                k_meas = self._add_noise(k_meas, noise_type)
                # Use the actual noise std from the spinbox
                actual_noise_std = self.noise_std_spin.value()
            else:
                actual_noise_std = 0.0

            if self.outlier_enabled.isChecked():
                k_meas = self._add_outliers(k_meas, phi)

            # ===== ENSURE CORRECT TREND FOR POLYMER MODE =====
            if self.mode == "polymer":
                # Verify conductivity is increasing
                if k_meas[-1] < k_meas[0]:
                    # If decreasing, use Agari model to force increasing
                    log_k = phi * 0.85 * np.log10(k_filler) + (1 - phi) * np.log10(1.2 * k_matrix)
                    k_meas = 10 ** log_k
                    if self.noise_enabled.isChecked():
                        k_meas = self._add_noise(k_meas, noise_type)

            # ===== CREATE DATAFRAME WITH CONSISTENT DATA =====
            self.generated_data = pd.DataFrame({
                "phi": phi,
                "Filler_volume_fraction": phi,
                "Matrix_volume_fraction": 1.0 - phi,
                "k_meas": k_meas,
                "k_std_dev": np.full(num_points, actual_noise_std)  # Use the actual noise value
            })

            # Legacy columns
            self.generated_data["Red_volume_fraction"] = phi
            self.generated_data["Oxy_volume_fraction"] = 1.0 - phi

            self._update_preview()

            for btn in self.findChildren(QPushButton):
                if btn.text() == "💾 Save Data":
                    btn.setEnabled(True)
                elif btn.text() == "📁 Save Multiple":
                    btn.setEnabled(True)

            QMessageBox.information(self, "Success",
                f"✅ Generated {len(self.generated_data)} data points successfully!\n\n"
                f"φ range: {phi.min():.3f} - {phi.max():.3f}\n"
                f"k range: {k_meas.min():.2f} - {k_meas.max():.2f} W/m·K\n"
                f"k_std_dev: {actual_noise_std:.3f}\n"
                f"Trend: {'INCREASING ✓' if k_meas[-1] > k_meas[0] else 'DECREASING ✗'}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate data:\n{str(e)}")

    def _get_model_function(self, model_name):
        """Get the appropriate model function based on mode."""
        if self.mode == "polymer":
            # Polymer mode: use models that give INCREASING conductivity
            if "Agari" in model_name and "Percolation" not in model_name:
                def func(phi, k_matrix, k_filler):
                    return agari_model(phi, 1.2, 0.85, k_matrix, k_filler, mode="polymer")
                return func
            elif "Agari-Percolation" in model_name:
                def func(phi, k_matrix, k_filler):
                    return agari_percolation_model(phi, 1.2, 0.85, 50.0, 0.08, 2.0, k_matrix, k_filler, mode="polymer")
                return func
            elif "Maxwell-Eucken" in model_name:
                return maxwell_eucken
            elif "Bruggeman" in model_name:
                return bruggeman_model
            elif "Lewis-Nielsen" in model_name:
                def func(phi, k_matrix, k_filler):
                    return lewis_nielsen_model(phi, 2.0, 0.6, k_matrix, k_filler, mode="polymer")
                return func
            elif "Percolation" in model_name and "Agari" not in model_name:
                def func(phi, k_matrix, k_filler):
                    return percolation_model(phi, 50.0, 0.08, 2.0, k_matrix, mode="polymer")
                return func
            else:
                def func(phi, k_matrix, k_filler):
                    return agari_model(phi, 1.2, 0.85, k_matrix, k_filler, mode="polymer")
                return func
        else:
            # Nanothermite mode: use original models
            if "Agari-Percolation" in model_name:
                def func(phi, k_matrix, k_filler):
                    return agari_percolation_model(phi, 1.0, 1.0, 50.0, 0.3, 2.0, k_matrix, k_filler, mode="nanothermite")
                return func
            elif "Agari" in model_name:
                def func(phi, k_matrix, k_filler):
                    return agari_model(phi, 1.0, 1.0, k_matrix, k_filler, mode="nanothermite")
                return func
            elif "Bruggeman" in model_name:
                return bruggeman_model
            elif "Maxwell-Eucken" in model_name:
                return maxwell_eucken
            elif "Parallel" in model_name:
                return parallel_model
            elif "Series" in model_name:
                return series_model
            elif "Lewis-Nielsen" in model_name:
                def func(phi, k_matrix, k_filler):
                    return lewis_nielsen_model(phi, 2.0, 0.6, k_matrix, k_filler, mode="nanothermite")
                return func
            elif "Percolation" in model_name:
                def func(phi, k_matrix, k_filler):
                    return percolation_model(phi, 50.0, 0.3, 2.0, k_matrix, mode="nanothermite")
                return func
            else:
                return parallel_model

    def _add_noise(self, k_meas, noise_type):
        std = self.noise_std_spin.value()
        percent = self.noise_percent_spin.value() / 100.0

        if std == 0 and percent == 0:
            return k_meas

        if noise_type == "Gaussian (Normal)":
            noise = np.random.normal(0, std, len(k_meas))
        elif noise_type == "Uniform":
            noise = np.random.uniform(-std, std, len(k_meas))
        else:
            noise = np.random.normal(0, percent * np.abs(k_meas), len(k_meas))

        result = k_meas + noise
        return np.maximum(result, 0.01)

    def _add_outliers(self, k_meas, phi):
        n_outliers = self.outlier_count_spin.value()
        magnitude = self.outlier_magnitude_spin.value() / 100.0

        if n_outliers > 0 and len(k_meas) > n_outliers:
            indices = np.random.choice(len(k_meas), n_outliers, replace=False)
            for idx in indices:
                # For polymer, outliers should be positive (higher conductivity)
                direction = np.random.choice([-1, 1])
                if self.mode == "polymer":
                    # Prefer positive outliers for polymer
                    direction = 1 if np.random.random() > 0.3 else -1
                k_meas[idx] = k_meas[idx] * (1 + direction * magnitude)
                k_meas[idx] = max(k_meas[idx], 0.01)

        return k_meas

    def _update_preview(self):
        if self.generated_data is None:
            return

        data = self.generated_data
        n_rows = len(data)

        self.preview_table.setRowCount(n_rows)
        for i in range(n_rows):
            self.preview_table.setItem(i, 0, QTableWidgetItem(f"{data['phi'].iloc[i]:.4f}"))
            self.preview_table.setItem(i, 1, QTableWidgetItem(f"{data['Matrix_volume_fraction'].iloc[i]:.4f}"))
            self.preview_table.setItem(i, 2, QTableWidgetItem(f"{data['k_meas'].iloc[i]:.2f}"))
            self.preview_table.setItem(i, 3, QTableWidgetItem(f"{data['k_std_dev'].iloc[i]:.3f}"))

        k_mean = data['k_meas'].mean()
        k_std = data['k_meas'].std()
        k_min = data['k_meas'].min()
        k_max = data['k_meas'].max()
        k_start = data['k_meas'].iloc[0]
        k_end = data['k_meas'].iloc[-1]

        trend = "⬆️ INCREASING" if k_end > k_start else "⬇️ DECREASING"
        trend_color = "#27ae60" if k_end > k_start else "#c0392b"

        self.preview_stats.setText(
            f"📊 {n_rows} points  |  k_mean = {k_mean:.2f} W/m·K  |  "
            f"k_std = {k_std:.2f}  |  k_min = {k_min:.2f}  |  k_max = {k_max:.2f}  |  "
            f"<span style='color: {trend_color}; font-weight: bold;'>Trend: {trend}</span>"
        )
        self.preview_stats.setTextFormat(Qt.TextFormat.RichText)

    def save_data(self):
        if self.generated_data is None:
            QMessageBox.warning(self, "Warning", "Please generate data first.")
            return

        prefix = self.filename_prefix.text().strip()
        if not prefix:
            prefix = "Polymer_Composite" if self.mode == "polymer" else "Nanothermite"

        folder = self.save_folder.text().strip()
        if not folder:
            folder = os.getcwd()

        # Make sure folder exists
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except:
                QMessageBox.critical(self, "Error", f"Cannot create folder: {folder}")
                return

        index = 1
        while True:
            filename = f"{prefix}_{index}.txt"
            filepath = os.path.join(folder, filename)
            if not os.path.exists(filepath):
                break
            index += 1

        try:
            # ===== SAVE THE EXACT SAME DATA AS SHOWN IN PREVIEW =====
            save_df = self.generated_data[['phi', 'k_meas', 'k_std_dev']].copy()
            save_df.to_csv(filepath, sep='\t', index=False, float_format='%.6f')

            QMessageBox.information(self, "Success",
                f"✅ Data saved successfully!\n\n"
                f"File: {filename}\n"
                f"Location: {folder}\n"
                f"Points: {len(self.generated_data)}\n"
                f"k_std_dev: {self.generated_data['k_std_dev'].iloc[0]:.3f}\n"
                f"Trend: {'INCREASING' if save_df['k_meas'].iloc[-1] > save_df['k_meas'].iloc[0] else 'DECREASING'}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data:\n{str(e)}")

    def save_multiple(self):
        if self.generated_data is None:
            QMessageBox.warning(self, "Warning", "Please generate data first.")
            return

        from PyQt6.QtWidgets import QInputDialog
        n_files, ok = QInputDialog.getInt(
            self, "Save Multiple Files",
            "How many files to generate?\n(Each file will have slight variations)",
            min=1, max=20, value=3
        )

        if not ok or n_files < 1:
            return

        prefix = self.filename_prefix.text().strip()
        if not prefix:
            prefix = "Polymer_Composite" if self.mode == "polymer" else "Nanothermite"

        folder = self.save_folder.text().strip()
        if not folder:
            folder = os.getcwd()

        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except:
                QMessageBox.critical(self, "Error", f"Cannot create folder: {folder}")
                return

        saved_files = []
        for i in range(n_files):
            variation = 1.0 + (i - n_files/2) * 0.05
            data_copy = self.generated_data.copy()
            data_copy['k_meas'] = data_copy['k_meas'] * variation
            # Keep std_dev consistent
            data_copy['k_std_dev'] = self.generated_data['k_std_dev']

            # Ensure trend remains increasing for polymer
            if self.mode == "polymer" and data_copy['k_meas'].iloc[-1] < data_copy['k_meas'].iloc[0]:
                # Swap to ensure increasing
                data_copy['k_meas'] = data_copy['k_meas'].iloc[::-1].values

            index = 1
            while True:
                filename = f"{prefix}_{index}.txt"
                filepath = os.path.join(folder, filename)
                if not os.path.exists(filepath):
                    break
                index += 1

            save_df = data_copy[['phi', 'k_meas', 'k_std_dev']].copy()
            save_df.to_csv(filepath, sep='\t', index=False, float_format='%.6f')
            saved_files.append(filename)

        QMessageBox.information(self, "Success",
            f"✅ Saved {len(saved_files)} files successfully!\n\n"
            f"Files:\n{chr(10).join(saved_files)}\n\n"
            f"Location: {folder}")