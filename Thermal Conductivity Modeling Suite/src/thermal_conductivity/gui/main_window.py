"""Main PyQt6 application window - Professional UI with full responsiveness."""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
import traceback
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QTableWidget, QTableWidgetItem, QCheckBox,
    QLabel, QLineEdit, QFileDialog, QMessageBox, QProgressBar, QGroupBox,
    QGridLayout, QHeaderView, QSplitter, QTextEdit, QComboBox,
    QStatusBar, QToolBar, QMenuBar, QMenu, QSizePolicy, QDialog,
    QScrollArea, QDoubleSpinBox, QDialogButtonBox, QScrollBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QAction, QFont, QIcon, QColor

from ..core.models import get_model_registry, get_model_category_names
from ..core.optimizer import fit_all_models
from ..core.statistics import calculate_statistics
from ..core.data_loader import load_data
from ..core.material_constants import (
    MODES, get_mode_info, get_materials_for_mode,
    get_k_value, get_rho_value, get_default_materials,
    get_mode_style, get_available_modes
)
from ..visualization.plotter import MplCanvas, create_main_plot, create_residuals_plot
from ..visualization.styles import PLOT_STYLES
from ..export.exporter import export_plot, export_excel, export_csv
from ..export.report_generator import generate_pdf_report
from .parameter_dialog import ModelParameterDialog
from .about_dialog import AboutDialog
from .data_dialog import DataLoadDialog
from .best_r2_tab import BestR2Tab
from .generate_data_dialog import GenerateDataDialog


class FitWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, phi, k, selected, custom, mode="nanothermite"):
        super().__init__()
        self.phi = phi
        self.k = k
        self.selected = selected
        self.custom = custom
        self.mode = mode

    def run(self):
        try:
            results = fit_all_models(self.phi, self.k, self.selected, self.custom, mode=self.mode)
            for name, res in results.items():
                if res["success"] and res.get("y_pred") is not None:
                    stats = calculate_statistics(self.k, res["y_pred"])
                    res.update(stats)
                    res["residuals"] = self.k - res["y_pred"]
                else:
                    res.update({"r2": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan})
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(traceback.format_exc())


class MultiFileWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, files_data, selected_models, custom_params=None, mode="nanothermite"):
        super().__init__()
        self.files_data = files_data
        self.selected_models = selected_models
        self.custom_params = custom_params or {}
        self.mode = mode

    def run(self):
        try:
            results = {}
            total = len(self.files_data)

            for idx, (filename, phi, k, df) in enumerate(self.files_data):
                self.progress.emit(int((idx + 1) / total * 100), f"Analyzing {filename}...")

                file_results = fit_all_models(phi, k, self.selected_models, self.custom_params, mode=self.mode)

                for name, res in file_results.items():
                    if res["success"] and res.get("y_pred") is not None:
                        stats = calculate_statistics(k, res["y_pred"])
                        res.update(stats)
                        res["residuals"] = k - res["y_pred"]
                    else:
                        res.update({"r2": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan})

                results[filename] = file_results

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self, mode="nanothermite"):
        super().__init__()
        
        # ===== MODE SETUP =====
        self.current_mode = mode
        
        mode_info = get_mode_info(mode)
        self.mode_label = mode_info.get("name", "Nanothermite")
        self.mode_icon = mode_info.get("icon", "🔬")
        
        materials = get_materials_for_mode(mode)
        self.matrix_list = materials.get("matrix_list", [])
        self.filler_list = materials.get("filler_list", [])
        self.matrix_label = mode_info.get("matrix_label", "Matrix")
        self.filler_label = mode_info.get("filler_label", "Filler")
        
        defaults = get_default_materials(mode)
        self.matrix_name = defaults.get("matrix", "Al")
        self.filler_name = defaults.get("filler", "CuO")
        
        self.k_matrix = get_k_value(mode, "matrix", self.matrix_name)
        self.k_filler = get_k_value(mode, "filler", self.filler_name)
        self.rho_matrix = get_rho_value(mode, "matrix", self.matrix_name)
        self.rho_filler = get_rho_value(mode, "filler", self.filler_name)
        
        self.material_info = f"{self.matrix_name}/{self.filler_name}"
        
        # ===== WINDOW SETUP =====
        self.setWindowTitle(f"{self.mode_icon} Thermal Conductivity Modeling Suite v1.0")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(self._get_global_style())

        self.df = None
        self.fit_results = {}
        self.combined_fit_results = {}
        self.current_style = "Thermochimica Acta"
        self.log_scale = True
        self.custom_params = {}
        self.show_legend = True
        self.legend_format = "compact"

        self.file_path = None
        self.multi_files = []
        self.multi_file_results = {}
        self.current_file_index = 0
        self.current_display_file = "Combined Data"

        self.model_checks = {}
        self.current_category = "All Models"
        self.worker = None
        self.multi_worker = None
        self.analysis_running = False
        self.best_r2_tab_index = -1
        self.models_tab_index = -1

        # Get model registry for current mode
        self.MODEL_REGISTRY = get_model_registry(mode)
        self.MODEL_CATEGORIES = get_model_category_names(mode)

        self._build_ui()
        self._create_menus()
        self._create_toolbar()
        self._populate_models_by_category()
        self._update_mode_ui()

    # ========================================================================
    # GLOBAL STYLESHEET
    # ========================================================================

    def _get_global_style(self):
        return """
            QMainWindow {
                background-color: #f5f6fa;
            }
            
            QMenuBar {
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 4px 10px;
                font-weight: bold;
                border: none;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background-color: transparent;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #34495e;
            }
            QMenu {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 6px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 30px 8px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #34495e;
            }
            QMenu::separator {
                height: 1px;
                background-color: #34495e;
                margin: 4px 10px;
            }
            
            QToolBar {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 #2c3e50, stop: 1 #34495e);
                border: none;
                padding: 6px 10px;
                spacing: 6px;
            }
            QToolBar QLabel {
                color: #ecf0f1;
                font-weight: bold;
            }
            QToolBar QPushButton {
                background-color: rgba(255, 255, 255, 0.1);
                color: #ecf0f1;
                border: none;
                border-radius: 5px;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 11px;
            }
            QToolBar QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
            QToolBar QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.05);
            }
            QToolBar QComboBox {
                background-color: #ecf0f1;
                color: #2c3e50;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                min-width: 150px;
            }
            QToolBar QComboBox::drop-down {
                border: none;
            }
            QToolBar QComboBox QAbstractItemView {
                background-color: #ecf0f1;
                color: #2c3e50;
                selection-background-color: #27ae60;
                selection-color: white;
            }
            
            QTabWidget::pane {
                background: white;
                border: 1px solid #d5d8dc;
                border-radius: 8px;
                margin-top: -1px;
            }
            QTabBar::tab {
                background: #ecf0f1;
                color: #7f8c8d;
                padding: 10px 24px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #2c3e50;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #d5d8dc;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d5d8dc;
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 10px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                color: #2c3e50;
            }
            
            QTableWidget {
                gridline-color: #d5d8dc;
                background-color: white;
                alternate-background-color: #f8f9fa;
                selection-background-color: #27ae60;
                selection-color: white;
                font-size: 11px;
                border: none;
            }
            QHeaderView::section {
                background-color: #2c3e50;
                color: white;
                padding: 8px;
                font-weight: bold;
                font-size: 11px;
                border: none;
            }
            
            QCheckBox {
                spacing: 8px;
                color: #2c3e50;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 22px;
                height: 22px;
                border-radius: 4px;
                border: 2px solid #bdc3c7;
                background: white;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #3498db;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #27ae60;
                background: #27ae60;
            }
            QCheckBox::indicator:checked:hover {
                border: 2px solid #2ecc71;
                background: #2ecc71;
            }
            
            QComboBox {
                padding: 6px 12px;
                border: 2px solid #d5d8dc;
                border-radius: 6px;
                background: white;
                color: #2c3e50;
                font-size: 11px;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background: white;
                color: #2c3e50;
                selection-background-color: #27ae60;
                selection-color: white;
                border: 2px solid #3498db;
                border-radius: 6px;
            }
            
            QLineEdit {
                padding: 6px 10px;
                border: 2px solid #d5d8dc;
                border-radius: 6px;
                background: white;
                color: #2c3e50;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
            
            QDoubleSpinBox, QSpinBox {
                padding: 6px 10px;
                border: 2px solid #d5d8dc;
                border-radius: 6px;
                background: white;
                color: #2c3e50;
                font-size: 11px;
                min-height: 20px;
            }
            QDoubleSpinBox:focus, QSpinBox:focus {
                border-color: #3498db;
            }
            
            QStatusBar {
                background: #2c3e50;
                color: #ecf0f1;
                border-top: none;
            }
            QStatusBar QLabel {
                color: #ecf0f1;
                font-size: 11px;
            }
            
            QProgressBar {
                border: none;
                border-radius: 4px;
                background: #34495e;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 10px;
                height: 18px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 #3498db, stop: 1 #2ecc71);
                border-radius: 4px;
            }
            
            QScrollArea {
                border: none;
                background: transparent;
            }
            
            QTextEdit {
                background: white;
                color: #2c3e50;
                border: 1px solid #d5d8dc;
                border-radius: 6px;
                padding: 10px;
                font-size: 12px;
                line-height: 1.6;
            }
            
            QPushButton {
                border: none;
                border-radius: 5px;
                font-weight: bold;
                padding: 8px 16px;
                font-size: 11px;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QPushButton:pressed {
                opacity: 0.7;
            }
            QPushButton:disabled {
                background-color: #bdc3c7 !important;
                color: #7f8c8d !important;
            }
            
            QScrollBar:vertical {
                background: #ecf0f1;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #bdc3c7;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #95a5a6;
            }
            QScrollBar:horizontal {
                background: #ecf0f1;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #bdc3c7;
                border-radius: 6px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #95a5a6;
            }
            
            QSplitter::handle {
                background: #d5d8dc;
                width: 4px;
            }
            QSplitter::handle:hover {
                background: #27ae60;
            }
            
            QLabel {
                color: #2c3e50;
            }
        """

    # ========================================================================
    # BUILD UI
    # ========================================================================

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setTextVisible(True)
        self.status.addPermanentWidget(QLabel("Status:"))
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("Ready")

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        main_layout.addWidget(self.tabs)

        self._build_data_tab()
        self._build_fit_tab()
        self._build_stats_tab()
        self._build_best_r2_tab()
        self._build_report_tab()
        self._build_models_tab()

    # ========================================================================
    # TOOLBAR
    # ========================================================================

    def _create_toolbar(self):
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(self.toolbar)

        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("color: #ecf0f1; font-weight: bold;")
        self.toolbar.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        available_modes = get_available_modes()
        for mode_id in available_modes:
            mode_info = get_mode_info(mode_id)
            self.mode_combo.addItem(f"{mode_info.get('icon', '🔬')} {mode_info.get('name', mode_id)}")
        
        self.mode_combo.setCurrentIndex(available_modes.index(self.current_mode) if self.current_mode in available_modes else 0)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.toolbar.addWidget(self.mode_combo)
        self.toolbar.addSeparator()

        buttons = [
            ("📊 Generate", self.open_generate_data, "#9b59b6"),
            ("📂 Load", self.load_data, "#3498db"),
            ("📂 Load Multi", self.load_multiple_files, "#2980b9"),
            ("⚙️ Params", self.open_parameter_dialog, "#9b59b6"),
            ("▶ Run", self.run_analysis, "#27ae60"),
            ("🧠 Smart", self.open_smart_fit, "#8e44ad"),
            ("🔍 Discover", self.open_model_discovery, "#e67e22"),
            ("💾 Save", self.save_graph, "#e67e22"),
            ("📤 Export", self.export_results, "#1abc9c"),
            ("📄 Report", self.generate_report, "#e74c3c"),
        ]

        for text, callback, color in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 6px 14px;
                    font-weight: bold;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: {color}dd;
                }}
                QPushButton:pressed {{
                    background-color: {color}aa;
                }}
            """)
            btn.clicked.connect(callback)
            self.toolbar.addWidget(btn)
            self.toolbar.addSeparator()

        about_btn = QPushButton("❓ About")
        about_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #2c3e50; }
        """)
        about_btn.clicked.connect(self.show_about)
        self.toolbar.addWidget(about_btn)

    # ========================================================================
    # DATA TAB
    # ========================================================================

    def _build_data_tab(self):
        self.tab_data = QWidget()
        layout = QVBoxLayout(self.tab_data)
        layout.setSpacing(12)

        self.data_info_label = QLabel()
        self.data_info_label.setWordWrap(True)
        self.data_info_label.setStyleSheet("""
            padding: 12px 16px;
            border-radius: 8px;
            background: #f0f4f8;
            border: 1px solid #d5d8dc;
            font-size: 12px;
        """)
        layout.addWidget(self.data_info_label)

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(5)
        self.data_table.setColumnWidth(0, 40)
        self.data_table.setColumnWidth(1, 120)
        self.data_table.setColumnWidth(2, 150)
        self.data_table.setColumnWidth(3, 100)
        self.data_table.setColumnWidth(4, 120)

        self.data_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.data_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.data_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.data_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self.data_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)

        self.data_table.setAlternatingRowColors(True)
        self.data_table.cellChanged.connect(self.on_data_cell_changed)

        layout.addWidget(self.data_table)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("+ Add Manual Row")
        btn_add.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        btn_add.clicked.connect(self.add_manual_row)
        btn_row.addWidget(btn_add)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.tabs.addTab(self.tab_data, "📊 Data")

    # ========================================================================
    # FIT TAB
    # ========================================================================

    def _build_fit_tab(self):
        self.tab_fit = QWidget()
        layout = QVBoxLayout(self.tab_fit)
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        vertical_splitter.setHandleWidth(5)
        vertical_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #d5d8dc;
                height: 5px;
                margin: 2px;
            }
            QSplitter::handle:hover {
                background: #27ae60;
            }
        """)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setSpacing(15)
        top_layout.setContentsMargins(0, 0, 0, 0)

        group_models = QGroupBox("📐 Model Selection")
        group_models.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d5d8dc;
                border-radius: 8px;
                padding-top: 14px;
                background: white;
            }
            QGroupBox::title {
                color: #2c3e50;
                padding: 0 8px;
            }
        """)
        models_layout = QVBoxLayout(group_models)
        models_layout.setSpacing(4)
        models_layout.setContentsMargins(8, 8, 8, 8)

        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(["All Models"] + list(self.MODEL_CATEGORIES.keys()))
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        cat_layout.addWidget(self.category_combo)
        cat_layout.addStretch()
        models_layout.addLayout(cat_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(160)
        scroll_widget = QWidget()
        self.models_grid = QGridLayout(scroll_widget)
        self.models_grid.setSpacing(3)
        scroll.setWidget(scroll_widget)
        models_layout.addWidget(scroll)

        btn_layout = QHBoxLayout()
        select_all = QPushButton("✅ Select All")
        select_all.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 3px 10px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #2ecc71; }
        """)
        select_all.clicked.connect(lambda: self.set_all_models(True))
        btn_layout.addWidget(select_all)

        deselect_all = QPushButton("❌ Deselect All")
        deselect_all.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 3px 10px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        deselect_all.clicked.connect(lambda: self.set_all_models(False))
        btn_layout.addWidget(deselect_all)
        btn_layout.addStretch()
        models_layout.addLayout(btn_layout)

        top_layout.addWidget(group_models, 2)

        group_style = QGroupBox("🎨 Plot Settings")
        group_style.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d5d8dc;
                border-radius: 8px;
                padding-top: 14px;
                background: white;
            }
            QGroupBox::title {
                color: #2c3e50;
                padding: 0 8px;
            }
        """)
        vlay = QVBoxLayout(group_style)
        vlay.setSpacing(4)
        vlay.setContentsMargins(8, 8, 8, 8)

        style_label = QLabel("Journal Style:")
        style_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        vlay.addWidget(style_label)
        self.style_combo = QComboBox()
        self.style_combo.addItems(list(PLOT_STYLES.keys()))
        self.style_combo.currentTextChanged.connect(self.change_style)
        vlay.addWidget(self.style_combo)

        self.log_check = QCheckBox("📈 Logarithmic Y-Axis")
        self.log_check.setChecked(True)
        self.log_check.stateChanged.connect(self.toggle_log)
        vlay.addWidget(self.log_check)

        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X-Axis:"))
        self.x_var_combo = QComboBox()
        self.x_var_combo.addItems(["Filler (φ)", "Matrix (1-φ)"])
        self.x_var_combo.currentTextChanged.connect(self.on_x_var_changed)
        x_layout.addWidget(self.x_var_combo)
        x_layout.addStretch()
        vlay.addLayout(x_layout)

        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Range:"))
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(-0.5, 1.0)
        self.x_min_spin.setSingleStep(0.01)
        self.x_min_spin.setValue(0.0)
        self.x_min_spin.setDecimals(3)
        self.x_min_spin.valueChanged.connect(self.on_x_range_changed)
        range_layout.addWidget(self.x_min_spin)

        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(-0.5, 1.0)
        self.x_max_spin.setSingleStep(0.01)
        self.x_max_spin.setValue(1.0)
        self.x_max_spin.setDecimals(3)
        self.x_max_spin.valueChanged.connect(self.on_x_range_changed)
        range_layout.addWidget(self.x_max_spin)

        reset_btn = QPushButton("↺ Reset")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 2px 10px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #95a5a6; }
        """)
        reset_btn.clicked.connect(self.reset_x_range)
        range_layout.addWidget(reset_btn)
        range_layout.addStretch()
        vlay.addLayout(range_layout)

        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(True)
        self.legend_check.stateChanged.connect(self.toggle_legend)
        vlay.addWidget(self.legend_check)

        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("Format:"))
        self.legend_format_combo = QComboBox()
        self.legend_format_combo.addItems(["Compact", "Full"])
        self.legend_format_combo.currentTextChanged.connect(self.change_legend_format)
        legend_layout.addWidget(self.legend_format_combo)
        legend_layout.addStretch()
        vlay.addLayout(legend_layout)

        top_layout.addWidget(group_style, 1)

        vertical_splitter.addWidget(top_widget)

        graph_splitter = QSplitter(Qt.Orientation.Horizontal)
        graph_splitter.setHandleWidth(5)
        graph_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #d5d8dc;
                width: 5px;
                margin: 2px;
            }
            QSplitter::handle:hover {
                background: #27ae60;
            }
        """)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        try:
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
            self.canvas_main = MplCanvas(self, width=9, height=7, dpi=150)
            self.canvas_main.setMinimumSize(600, 500)
            self.canvas_main.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.toolbar_main = NavigationToolbar2QT(self.canvas_main, self)
            self.toolbar_main.setStyleSheet("background: #ecf0f1; border: none;")
            ll.addWidget(self.toolbar_main)
            ll.addWidget(self.canvas_main)
        except Exception as e:
            print(f"Error creating main canvas: {e}")
            self.canvas_main = MplCanvas(self, width=9, height=7, dpi=150)
            self.canvas_main.setMinimumSize(600, 500)
            self.canvas_main.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            ll.addWidget(self.canvas_main)

        graph_splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        try:
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
            self.canvas_resid = MplCanvas(self, width=9, height=7, dpi=150)
            self.canvas_resid.setMinimumSize(600, 500)
            self.canvas_resid.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.toolbar_resid = NavigationToolbar2QT(self.canvas_resid, self)
            self.toolbar_resid.setStyleSheet("background: #ecf0f1; border: none;")
            rl.addWidget(self.toolbar_resid)
            rl.addWidget(self.canvas_resid)
        except Exception as e:
            print(f"Error creating residual canvas: {e}")
            self.canvas_resid = MplCanvas(self, width=9, height=7, dpi=150)
            self.canvas_resid.setMinimumSize(600, 500)
            self.canvas_resid.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            rl.addWidget(self.canvas_resid)

        graph_splitter.addWidget(right)
        graph_splitter.setStretchFactor(0, 1)
        graph_splitter.setStretchFactor(1, 1)

        vertical_splitter.addWidget(graph_splitter)
        vertical_splitter.setSizes([200, 800])

        layout.addWidget(vertical_splitter)

        self.tabs.addTab(self.tab_fit, "📈 Model Fit")

    # ========================================================================
    # MODELS REFERENCE TAB
    # ========================================================================

    def _build_models_tab(self):
        self.tab_models = QWidget()
        layout = QVBoxLayout(self.tab_models)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.addWidget(QLabel("📚 Models Reference"))
        header.addStretch()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        refresh_btn.clicked.connect(self._update_models_reference)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        self.models_text = QTextEdit()
        self.models_text.setReadOnly(True)
        self.models_text.setStyleSheet("""
            QTextEdit {
                font-size: 12px;
                line-height: 1.8;
                background-color: white;
                border: 1px solid #d5d8dc;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        layout.addWidget(self.models_text)

        self.tabs.addTab(self.tab_models, "📚 Models Reference")
        self.models_tab_index = self.tabs.indexOf(self.tab_models)
        self._update_models_reference()

    def _update_models_reference(self):
        html = self._build_models_html()
        self.models_text.setHtml(html)

    def _build_models_html(self):
        if self.current_mode == "nanothermite":
            mode_badge = "🔥 Nanothermite Mode"
            mode_color = "#e74c3c"
            matrix_name = "Reductant"
            filler_name = "Oxidizer"
        else:
            mode_badge = "🧪 Polymer Mode"
            mode_color = "#27ae60"
            matrix_name = "Polymer Matrix"
            filler_name = "Filler"

        html = f"""
        <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">📐 Thermal Conductivity Models Reference</h2>
            <p style="color: #bdc3c7; margin: 5px 0 0 0;">
                <span style="background: {mode_color}; padding: 2px 12px; border-radius: 12px; color: white; font-size: 12px;">{mode_badge}</span>
                &nbsp; • &nbsp; k<sub>matrix</sub> = {self.k_matrix:.2f} W/m·K ({self.matrix_name}) &nbsp; • &nbsp; k<sub>filler</sub> = {self.k_filler:.2f} W/m·K ({self.filler_name})
            </p>
        </div>
        """

        if self.current_mode == "nanothermite":
            html += f"""
            <div style="background: #f8f9fa; border-left: 4px solid {mode_color}; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                <h4 style="margin: 0 0 10px 0; color: #2c3e50;">📋 Nomenclature (φ = Filler)</h4>
                <table style="width:100%; border-collapse: collapse;">
                    <tr><td style="padding: 4px 10px; width:140px;"><b>k<sub>eff</sub></b></td>
                        <td style="padding: 4px 10px;">Effective thermal conductivity</td></tr>
                    <tr><td style="padding: 4px 10px;"><b>k<sub>filler</sub></b></td>
                        <td style="padding: 4px 10px;">Thermal conductivity of <b style="color: {mode_color};">oxidizer</b> (filler)</td></tr>
                    <tr><td style="padding: 4px 10px;"><b>k<sub>matrix</sub></b></td>
                        <td style="padding: 4px 10px;">Thermal conductivity of <b style="color: #3498db;">reductant</b> (matrix)</td></tr>
                    <tr><td style="padding: 4px 10px;"><b>φ</b></td>
                        <td style="padding: 4px 10px;"><b style="color: {mode_color};">Filler volume fraction</b></td></tr>
                    <tr><td style="padding: 4px 10px;"><b>(1-φ)</b></td>
                        <td style="padding: 4px 10px;"><b style="color: #3498db;">Matrix volume fraction</b></td></tr>
                </table>
            </div>
            """
        else:
            html += f"""
            <div style="background: #f8f9fa; border-left: 4px solid {mode_color}; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                <h4 style="margin: 0 0 10px 0; color: #2c3e50;">📋 Nomenclature (φ = Filler)</h4>
                <table style="width:100%; border-collapse: collapse;">
                    <tr><td style="padding: 4px 10px; width:140px;"><b>k<sub>eff</sub></b></td>
                        <td style="padding: 4px 10px;">Effective thermal conductivity</td></tr>
                    <tr><td style="padding: 4px 10px;"><b>k<sub>filler</sub></b></td>
                        <td style="padding: 4px 10px;">Thermal conductivity of <b style="color: {mode_color};">conductive filler</b></td></tr>
                    <tr><td style="padding: 4px 10px;"><b>k<sub>matrix</sub></b></td>
                        <td style="padding: 4px 10px;">Thermal conductivity of <b style="color: #3498db;">polymer matrix</b></td></tr>
                    <tr><td style="padding: 4px 10px;"><b>φ</b></td>
                        <td style="padding: 4px 10px;"><b style="color: {mode_color};">Filler volume fraction</b></td></tr>
                    <tr><td style="padding: 4px 10px;"><b>(1-φ)</b></td>
                        <td style="padding: 4px 10px;"><b style="color: #3498db;">Matrix volume fraction</b></td></tr>
                </table>
            </div>
            """

        html += f"<h3 style='color: #2c3e50;'>📊 Model List ({len(self.MODEL_REGISTRY)} models)</h3><hr style='border: 1px solid #d5d8dc;'>"

        equations = {
            "Parallel": "k<sub>eff</sub> = φ·k<sub>filler</sub> + (1-φ)·k<sub>matrix</sub>",
            "Series": "1/k<sub>eff</sub> = φ/k<sub>filler</sub> + (1-φ)/k<sub>matrix</sub>",
            "Maxwell-Eucken": "k<sub>eff</sub> = k<sub>matrix</sub> · [k<sub>filler</sub> + 2k<sub>matrix</sub> + 2φ(k<sub>filler</sub> - k<sub>matrix</sub>)] / [k<sub>filler</sub> + 2k<sub>matrix</sub> - φ(k<sub>filler</sub> - k<sub>matrix</sub>)]",
            "Bruggeman": "φ = (k<sub>filler</sub> - k<sub>eff</sub>)/(k<sub>filler</sub> - k<sub>matrix</sub>) · (k<sub>matrix</sub>/k<sub>eff</sub>)<sup>1/3</sup>",
            "Agari": "log(k<sub>eff</sub>) = φ·C<sub>2</sub>·log(k<sub>filler</sub>) + (1-φ)·log(C<sub>1</sub>·k<sub>matrix</sub>)",
            "Lewis-Nielsen": "k<sub>eff</sub> = k<sub>matrix</sub> · (1 + A·B·φ) / (1 - B·ψ·φ)",
            "Percolation": "k<sub>eff</sub> = k₀·(φ - φ<sub>c</sub>)<sup>t</sup>",
            "Agari-Percolation": "log(k<sub>eff</sub>) = φ·C<sub>2</sub>·log(k<sub>filler</sub>) + (1-φ)·log(C<sub>1</sub>·k<sub>matrix</sub>) + k₀·(φ - φ<sub>c</sub>)<sup>t</sup>",
        }

        for category, model_names in self.MODEL_CATEGORIES.items():
            if category == "Discovered Models" and not model_names:
                continue
            
            html += f"""
            <div style="margin-top: 15px;">
                <h4 style="color: #2c3e50; border-bottom: 2px solid {mode_color}; padding-bottom: 5px;">{category}</h4>
            """
            
            for name in model_names:
                if name in self.MODEL_REGISTRY:
                    meta = self.MODEL_REGISTRY[name]
                    color = meta.get("color", "#2c3e50")
                    is_discovered = meta.get("discovered", False)
                    
                    html += f"""
                    <div style="background: {'#fff5f5' if is_discovered else '#fafafa'}; 
                                border-left: 3px solid {color}; 
                                padding: 10px 15px; 
                                margin: 8px 0; 
                                border-radius: 4px;">
                        <b style="color: {color};">{name}</b>
                        <span style="color: #7f8c8d; font-size: 11px; margin-left: 10px;">
                            {'' if meta.get('fixed', True) else '📐 Fitted'}
                            {'' if not is_discovered else '🔍 Discovered'}
                        </span>
                        <br>
                        <span style="font-size: 11px; color: #555;">{meta.get('description', '')}</span>
                        <br>
                        <span style="font-size: 12px; font-family: monospace; color: #2c3e50;">
                            {equations.get(name, '')}
                        </span>
                        <br>
                        <span style="font-size: 10px; color: #95a5a6;">
                            {f"Params: {', '.join(meta['params'])}" if meta.get('params') else "Fixed model"}
                            {f" | Ref: {meta['reference']}" if meta.get('reference') else ""}
                        </span>
                    </div>
                    """
            
            html += "</div>"

        html += f"""
        <div style="margin-top: 20px; padding: 12px; background: #f8f9fa; border-radius: 6px; text-align: center; font-size: 11px; color: #95a5a6;">
            Thermal Conductivity Modeling Suite v1.0 • {self.mode_label} Mode • Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        </div>
        """

        return html

    # ========================================================================
    # STATISTICS TAB
    # ========================================================================

    def _build_stats_tab(self):
        self.tab_stats = QWidget()
        layout = QVBoxLayout(self.tab_stats)
        layout.setSpacing(12)

        file_selector_layout = QHBoxLayout()
        file_selector_layout.addWidget(QLabel("Data Set:"))
        self.file_selector = QComboBox()
        self.file_selector.addItem("Combined Data")
        self.file_selector.currentTextChanged.connect(self.on_file_selected)
        file_selector_layout.addWidget(self.file_selector)
        file_selector_layout.addStretch()
        layout.addLayout(file_selector_layout)

        stats_label = QLabel("📊 Statistical Metrics")
        stats_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        layout.addWidget(stats_label)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(6)
        self.stats_table.setHorizontalHeaderLabels(["Model", "R²", "RMSE", "MAE", "MAPE (%)", "Status"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.stats_table)

        params_label = QLabel("📐 Fitted Parameters")
        params_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50; margin-top: 10px;")
        layout.addWidget(params_label)

        self.param_table = QTableWidget()
        self.param_table.setColumnCount(4)
        self.param_table.setHorizontalHeaderLabels(["Model", "Parameter", "Value", "Std. Error"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.param_table)

        self.tabs.addTab(self.tab_stats, "📊 Statistics")

    # ========================================================================
    # BEST R² TAB
    # ========================================================================

    def _build_best_r2_tab(self):
        self.tab_best_r2 = BestR2Tab(self)
        self.tabs.addTab(self.tab_best_r2, "🏆 Best R²")
        self.best_r2_tab_index = self.tabs.indexOf(self.tab_best_r2)
        self.tabs.setTabEnabled(self.best_r2_tab_index, False)

    def _enable_best_r2_tab(self, enabled):
        if hasattr(self, 'best_r2_tab_index') and self.best_r2_tab_index >= 0:
            self.tabs.setTabEnabled(self.best_r2_tab_index, enabled)

    # ========================================================================
    # REPORT TAB
    # ========================================================================

    def _build_report_tab(self):
        self.tab_report = QWidget()
        layout = QVBoxLayout(self.tab_report)
        layout.setSpacing(12)

        report_label = QLabel("📄 Report Preview")
        report_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        layout.addWidget(report_label)

        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        self.report_preview.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #d5d8dc;
                border-radius: 6px;
                padding: 15px;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.report_preview)

        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        refresh_btn.clicked.connect(self.update_report_preview)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.tabs.addTab(self.tab_report, "📄 Report")

    # ========================================================================
    # MENU BAR
    # ========================================================================

    def _create_menus(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 4px 10px;
                font-weight: bold;
                border: none;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background-color: transparent;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #34495e;
            }
            QMenu {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 6px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 30px 8px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #34495e;
            }
            QMenu::separator {
                height: 1px;
                background-color: #34495e;
                margin: 4px 10px;
            }
        """)
        
        file_menu = menubar.addMenu("File")
        act_load = QAction("Load Data...", self)
        act_load.setShortcut("Ctrl+O")
        act_load.triggered.connect(self.load_data)
        file_menu.addAction(act_load)
        
        act_load_multi = QAction("Load Multiple...", self)
        act_load_multi.setShortcut("Ctrl+Shift+O")
        act_load_multi.triggered.connect(self.load_multiple_files)
        file_menu.addAction(act_load_multi)
        
        file_menu.addSeparator()
        act_save = QAction("Save Graph...", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.save_graph)
        file_menu.addAction(act_save)
        
        act_export = QAction("Export Results...", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self.export_results)
        file_menu.addAction(act_export)
        
        file_menu.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        analysis_menu = menubar.addMenu("Analysis")
        act_params = QAction("Configure Parameters...", self)
        act_params.triggered.connect(self.open_parameter_dialog)
        analysis_menu.addAction(act_params)
        
        act_run = QAction("Run Analysis", self)
        act_run.setShortcut("Ctrl+R")
        act_run.triggered.connect(self.run_analysis)
        analysis_menu.addAction(act_run)
        
        act_smart = QAction("Smart Fit...", self)
        act_smart.setShortcut("Ctrl+F")
        act_smart.triggered.connect(self.open_smart_fit)
        analysis_menu.addAction(act_smart)
        
        act_discovery = QAction("Discover Model...", self)
        act_discovery.setShortcut("Ctrl+D")
        act_discovery.triggered.connect(self.open_model_discovery)
        analysis_menu.addAction(act_discovery)

        export_menu = menubar.addMenu("Export")
        act_excel = QAction("Export to Excel...", self)
        act_excel.triggered.connect(lambda: self._do_export("excel"))
        export_menu.addAction(act_excel)
        
        act_csv = QAction("Export to CSV...", self)
        act_csv.triggered.connect(lambda: self._do_export("csv"))
        export_menu.addAction(act_csv)
        
        act_pdf = QAction("Generate PDF Report...", self)
        act_pdf.triggered.connect(self.generate_report)
        export_menu.addAction(act_pdf)

        help_menu = menubar.addMenu("Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self.show_about)
        help_menu.addAction(act_about)

    # ========================================================================
    # MODE SWITCHING
    # ========================================================================

    def _on_mode_changed(self, index):
        available_modes = get_available_modes()
        if index >= len(available_modes):
            return
            
        new_mode = available_modes[index]
        
        if new_mode == self.current_mode:
            return
        
        if self.df is not None:
            mode_info = get_mode_info(new_mode)
            reply = QMessageBox.question(
                self, 
                "Switch Mode", 
                f"This will clear all current data and results.\n\n"
                f"Current: {self.mode_label}\n"
                f"New: {mode_info.get('name', new_mode)}\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.mode_combo.blockSignals(True)
                self.mode_combo.setCurrentIndex(available_modes.index(self.current_mode))
                self.mode_combo.blockSignals(False)
                return
        
        self._switch_mode(new_mode)

    def _switch_mode(self, new_mode):
        self.current_mode = new_mode
        
        mode_info = get_mode_info(new_mode)
        self.mode_label = mode_info.get("name", "Nanothermite")
        self.mode_icon = mode_info.get("icon", "🔬")
        
        materials = get_materials_for_mode(new_mode)
        self.matrix_list = materials.get("matrix_list", [])
        self.filler_list = materials.get("filler_list", [])
        self.matrix_label = mode_info.get("matrix_label", "Matrix")
        self.filler_label = mode_info.get("filler_label", "Filler")
        
        defaults = get_default_materials(new_mode)
        self.matrix_name = defaults.get("matrix", "Al")
        self.filler_name = defaults.get("filler", "CuO")
        
        self.k_matrix = get_k_value(new_mode, "matrix", self.matrix_name)
        self.k_filler = get_k_value(new_mode, "filler", self.filler_name)
        self.rho_matrix = get_rho_value(new_mode, "matrix", self.matrix_name)
        self.rho_filler = get_rho_value(new_mode, "filler", self.filler_name)
        
        self.material_info = f"{self.matrix_name}/{self.filler_name}"
        
        self.MODEL_REGISTRY = get_model_registry(new_mode)
        self.MODEL_CATEGORIES = get_model_category_names(new_mode)
        
        # ===== UPDATE GLOBAL REGISTRY =====
        from ..core.models import set_global_registry
        set_global_registry(new_mode)
        
        self._clear_all_data()
        self._populate_models_by_category()
        self._update_mode_ui()
        self._update_models_reference()
        self.update_report_preview()
        
        self.status.showMessage(f"Switched to {self.mode_label} Mode")

    def _clear_all_data(self):
        self.df = None
        self.fit_results = {}
        self.combined_fit_results = {}
        self.multi_files = []
        self.multi_file_results = {}
        self.file_path = None
        self.custom_params = {}
        
        self.data_table.setRowCount(0)
        self.stats_table.setRowCount(0)
        self.param_table.setRowCount(0)
        self.report_preview.clear()
        self.tab_best_r2.clear()
        self._enable_best_r2_tab(False)
        
        if hasattr(self, 'canvas_main'):
            self.canvas_main.axes.clear()
            self.canvas_main.draw()
        if hasattr(self, 'canvas_resid'):
            self.canvas_resid.axes.clear()
            self.canvas_resid.draw()

    def _update_mode_ui(self):
        if self.current_mode == "nanothermite":
            header_labels = ["#", "φ (Filler)", "k_meas (W/m·K)", "k_std_dev", "1-φ (Matrix)"]
            info_text = f"🔥 <b>Nanothermite Mode</b> — φ = Oxidizer (filler), (1-φ) = Reductant (matrix)"
        else:
            header_labels = ["#", "φ (Filler)", "k_meas (W/m·K)", "k_std_dev", "1-φ (Matrix)"]
            info_text = f"🧪 <b>Polymer Mode</b> — φ = Filler, (1-φ) = Polymer Matrix"
        
        self.data_table.setHorizontalHeaderLabels(header_labels)
        self.data_info_label.setText(info_text)

    # ========================================================================
    # MODEL POPULATION - FIXED: Don't reload registry, use existing
    # ========================================================================

    def _populate_models_by_category(self):
        """Populate the model selection grid with green checkmarks."""
        # ===== FIX: Don't reload the registry - use what's already there =====
        # Only reload if the registry is empty or doesn't have discovered models
        if not self.MODEL_REGISTRY or len(self.MODEL_REGISTRY) <= 26:  # 26 is the number of built-in models
            self.MODEL_REGISTRY = get_model_registry(self.current_mode)
            self.MODEL_CATEGORIES = get_model_category_names(self.current_mode)
        
        # Ensure "Discovered Models" category exists
        if "Discovered Models" not in self.MODEL_CATEGORIES:
            self.MODEL_CATEGORIES["Discovered Models"] = []
        
        # Find all discovered models from the current registry
        discovered_models = [name for name, meta in self.MODEL_REGISTRY.items() 
                            if meta.get("discovered", False)]
        self.MODEL_CATEGORIES["Discovered Models"] = discovered_models
        
        # Clear existing checkboxes
        for widget in self.model_checks.values():
            self.models_grid.removeWidget(widget)
            widget.deleteLater()
        self.model_checks.clear()

        category = self.category_combo.currentText()
        
        if category == "All Models":
            model_names = list(self.MODEL_REGISTRY.keys())
        else:
            model_names = self.MODEL_CATEGORIES.get(category, [])

        if not model_names:
            model_names = list(self.MODEL_REGISTRY.keys())

        # Default models to check
        default_models = ["Agari", "Bruggeman", "Maxwell-Eucken", "Percolation"]
        if "Agari-Percolation" in model_names:
            default_models.append("Agari-Percolation")

        def sort_key(name):
            meta = self.MODEL_REGISTRY.get(name, {})
            return (0 if meta.get("discovered", False) else 1, name)
        
        model_names.sort(key=sort_key, reverse=False)

        row, col = 0, 0
        for name in model_names:
            if name in self.MODEL_REGISTRY:
                cb = QCheckBox(name)
                cb.setChecked(name in default_models)
                meta = self.MODEL_REGISTRY[name]
                
                if meta.get("description"):
                    cb.setToolTip(meta["description"])
                
                if meta.get("discovered", False):
                    cb.setStyleSheet("""
                        QCheckBox {
                            font-style: italic;
                            color: #27ae60;
                            font-weight: bold;
                        }
                    """)
                else:
                    cb.setStyleSheet("")
                
                self.model_checks[name] = cb
                self.models_grid.addWidget(cb, row, col)
                col += 1
                if col > 2:
                    col = 0
                    row += 1
        
        # Print debug info
        print(f"[DEBUG] Populated {len(self.model_checks)} models in UI")
        discovered = [name for name, cb in self.model_checks.items() 
                      if self.MODEL_REGISTRY.get(name, {}).get("discovered", False)]
        if discovered:
            print(f"[DEBUG] Discovered models in UI: {discovered}")
        else:
            print(f"[DEBUG] No discovered models in UI")

    def set_all_models(self, checked):
        for cb in self.model_checks.values():
            cb.setChecked(checked)

    def on_category_changed(self, category_name):
        self._populate_models_by_category()

    # ========================================================================
    # TOGGLE FUNCTIONS
    # ========================================================================

    def toggle_legend(self, state):
        self.show_legend = bool(state)
        if self.df is not None and self.fit_results:
            self.update_plots()

    def change_legend_format(self, format_name):
        self.legend_format = "compact" if format_name == "Compact" else "full"
        if self.df is not None and self.fit_results:
            self.update_plots()

    def change_style(self, style_name):
        self.current_style = style_name
        if self.df is not None and self.fit_results:
            self.update_plots()

    def toggle_log(self, state):
        self.log_scale = bool(state)
        if self.df is not None and self.fit_results:
            self.update_plots()

    def reset_x_range(self):
        if self.df is not None:
            x_var = self.get_x_var()
            if x_var == "Filler (φ)":
                x_col = "Filler_volume_fraction"
            else:
                x_col = "Matrix_volume_fraction"
            xdata = self.df[x_col].values
            x_min = xdata.min()
            x_max = xdata.max()
            padding = (x_max - x_min) * 0.05
            self.x_min_spin.setValue(max(-0.5, x_min - padding))
            self.x_max_spin.setValue(min(1.0, x_max + padding))
        else:
            self.x_min_spin.setValue(0.0)
            self.x_max_spin.setValue(1.0)
        self.update_plots()

    def get_x_range(self):
        return (self.x_min_spin.value(), self.x_max_spin.value())

    def on_x_var_changed(self):
        if self.df is not None and self.fit_results:
            self.update_plots()

    def get_x_var(self):
        return self.x_var_combo.currentText()

    def on_x_range_changed(self):
        if self.df is not None and self.fit_results:
            self.update_plots()

    def _on_tab_changed(self, index):
        if index < 0:
            return
        tab_text = self.tabs.tabText(index)
        if tab_text == "Models Reference" or tab_text == "📚 Models Reference":
            self._update_models_reference()

    # ========================================================================
    # UPDATE PLOTS
    # ========================================================================

    def update_plots(self):
        if self.df is None:
            self.canvas_main.axes.clear()
            self.canvas_main.axes.text(0.5, 0.5, "Load data first", 
                                      ha='center', va='center', transform=self.canvas_main.axes.transAxes, fontsize=14)
            self.canvas_main.draw()
            self.canvas_resid.axes.clear()
            self.canvas_resid.draw()
            return

        current_file = self.file_selector.currentText() if hasattr(self, 'file_selector') else "Combined Data"

        if current_file != "Combined Data" and current_file in self.multi_file_results:
            results = self.multi_file_results[current_file]
            for filename, df in self.multi_files:
                if filename == current_file:
                    display_df = df
                    break
            else:
                display_df = self.df
        else:
            results = self.combined_fit_results if self.combined_fit_results else self.fit_results
            display_df = self.df

        if not results:
            self.canvas_main.axes.clear()
            self.canvas_main.axes.text(0.5, 0.5, "No fit results. Run analysis.", 
                                      ha='center', va='center', transform=self.canvas_main.axes.transAxes, fontsize=14)
            self.canvas_main.draw()
            self.canvas_resid.axes.clear()
            self.canvas_resid.draw()
            return

        self.legend_format = "compact" if self.legend_format_combo.currentText() == "Compact" else "full"
        top_models = self._get_top_models(results, max_models=8)

        if not top_models:
            self.canvas_main.axes.clear()
            self.canvas_main.axes.text(0.5, 0.5, "No valid model fits", 
                                      ha='center', va='center', transform=self.canvas_main.axes.transAxes, fontsize=14)
            self.canvas_main.draw()
            self.canvas_resid.axes.clear()
            self.canvas_resid.draw()
            return

        x_min, x_max = self.get_x_range()
        x_var = self.get_x_var()
        
        if x_var == "Filler (φ)":
            x_col = "Filler_volume_fraction"
            x_label = f"{self.filler_name} Volume Fraction, φ"
        else:
            x_col = "Matrix_volume_fraction"
            x_label = f"{self.matrix_name} Volume Fraction, (1-φ)"

        try:
            create_main_plot(
                self.canvas_main, display_df, top_models,
                self.current_style, self.log_scale,
                show_legend=self.legend_check.isChecked(),
                legend_format=self.legend_format,
                material=f"{self.material_info} - {current_file}",
                matrix=self.matrix_name, filler=self.filler_name,
                x_col=x_col, x_label=x_label,
                phi_col="Filler_volume_fraction",
                x_min=x_min, x_max=x_max,
                max_models_to_show=8, mode=self.current_mode
            )
            create_residuals_plot(
                self.canvas_resid, display_df, top_models,
                self.current_style,
                show_legend=self.legend_check.isChecked(),
                matrix=self.matrix_name, filler=self.filler_name,
                x_col=x_col, x_label=x_label,
                x_min=x_min, x_max=x_max,
                max_models_to_show=8, mode=self.current_mode
            )
        except Exception as e:
            print(f"Error updating plots: {traceback.format_exc()}")

    def _get_top_models(self, results, max_models=8):
        if not results:
            return {}

        valid_results = {}
        for name, result in results.items():
            if result.get("success", False) and result.get("y_pred") is not None:
                r2 = result.get("r2", -1)
                if np.isfinite(r2):
                    valid_results[name] = result

        if not valid_results:
            return {}

        sorted_results = sorted(
            valid_results.items(),
            key=lambda x: x[1].get("r2", -1),
            reverse=True
        )

        discovered = []
        standard = []
        for name, result in sorted_results:
            if self.MODEL_REGISTRY.get(name, {}).get("discovered", False):
                discovered.append((name, result))
            else:
                standard.append((name, result))

        selected = standard[:]
        selected.extend(discovered[:5])

        if len(selected) > max_models:
            return dict(selected[:max_models])

        return dict(selected)

    # ========================================================================
    # STATISTICS UPDATE FUNCTIONS
    # ========================================================================

    def on_file_selected(self, filename):
        self.current_display_file = filename
        self.update_stats_table()
        self.update_param_table()
        self.update_plots()
        self.update_report_preview()

    def update_stats_table(self):
        try:
            current_file = self.file_selector.currentText() if hasattr(self, 'file_selector') else "Combined Data"

            if current_file != "Combined Data" and current_file in self.multi_file_results:
                results = self.multi_file_results[current_file]
            else:
                results = self.combined_fit_results if self.combined_fit_results else self.fit_results

            if not results:
                self.stats_table.setRowCount(1)
                msg_item = QTableWidgetItem("No results available. Run analysis first.")
                msg_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.stats_table.setSpan(0, 0, 1, 6)
                self.stats_table.setItem(0, 0, msg_item)
                return

            filtered_results = {}
            for name, res in results.items():
                if res.get("success", False):
                    r2 = res.get("r2", -1)
                    if np.isfinite(r2):
                        filtered_results[name] = res

            if not filtered_results:
                self.stats_table.setRowCount(1)
                msg_item = QTableWidgetItem("No valid model fits available.")
                msg_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.stats_table.setSpan(0, 0, 1, 6)
                self.stats_table.setItem(0, 0, msg_item)
                return

            sorted_results = sorted(
                filtered_results.items(),
                key=lambda x: x[1].get("r2", -1),
                reverse=True
            )

            self.stats_table.setRowCount(len(sorted_results))
            self.stats_table.clearSpans()

            best_r2 = -1
            best_name = None
            for name, res in sorted_results:
                if res.get("success", False):
                    r2 = res.get("r2", -1)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_name = name

            for row, (name, res) in enumerate(sorted_results):
                name_item = QTableWidgetItem(name)
                if name == best_name and res.get("success", False):
                    name_item.setForeground(QColor("#27ae60"))
                    font = name_item.font()
                    font.setBold(True)
                    name_item.setFont(font)
                    name_item.setText(f"🏆 {name}")
                self.stats_table.setItem(row, 0, name_item)

                if res.get("success", False):
                    r2 = res.get('r2', 0)
                    r2_item = QTableWidgetItem(f"{r2:.4f}")
                    if name == best_name:
                        r2_item.setBackground(QColor("#d4edda"))
                        r2_item.setForeground(QColor("#155724"))
                        font = r2_item.font()
                        font.setBold(True)
                        r2_item.setFont(font)
                    elif r2 >= 0.95:
                        r2_item.setBackground(QColor("#d4edda"))
                        r2_item.setForeground(QColor("#155724"))
                    elif r2 >= 0.85:
                        r2_item.setBackground(QColor("#fff3cd"))
                        r2_item.setForeground(QColor("#856404"))
                    elif r2 >= 0.70:
                        r2_item.setBackground(QColor("#ffeaa7"))
                        r2_item.setForeground(QColor("#6c5200"))
                    else:
                        r2_item.setBackground(QColor("#f8d7da"))
                        r2_item.setForeground(QColor("#721c24"))
                    self.stats_table.setItem(row, 1, r2_item)

                    self.stats_table.setItem(row, 2, QTableWidgetItem(f"{res.get('rmse', 0):.4f}"))
                    self.stats_table.setItem(row, 3, QTableWidgetItem(f"{res.get('mae', 0):.4f}"))
                    self.stats_table.setItem(row, 4, QTableWidgetItem(f"{res.get('mape', 0):.2f}"))

                    status_item = QTableWidgetItem("✅ Success")
                    if name == best_name:
                        status_item.setForeground(QColor("#27ae60"))
                        font = status_item.font()
                        font.setBold(True)
                        status_item.setFont(font)
                    else:
                        status_item.setForeground(QColor("#27ae60"))
                    self.stats_table.setItem(row, 5, status_item)
                else:
                    for c in range(1, 6):
                        self.stats_table.setItem(row, c, QTableWidgetItem("-"))

            self.stats_table.resizeColumnsToContents()
        except Exception as e:
            print(f"[ERROR] update_stats_table: {e}")

    def update_param_table(self):
        try:
            self.param_table.setRowCount(0)

            current_file = self.file_selector.currentText() if hasattr(self, 'file_selector') else "Combined Data"

            if current_file != "Combined Data" and current_file in self.multi_file_results:
                results = self.multi_file_results[current_file]
            else:
                results = self.combined_fit_results if self.combined_fit_results else self.fit_results

            if not results:
                self.param_table.setRowCount(1)
                msg_item = QTableWidgetItem("No parameters available. Run analysis first.")
                msg_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.param_table.setSpan(0, 0, 1, 4)
                self.param_table.setItem(0, 0, msg_item)
                return

            best_name = None
            best_r2 = -1
            for name, res in results.items():
                if res.get("success", False):
                    r2 = res.get("r2", -1)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_name = name

            has_params = False
            for name, res in results.items():
                if res.get("params") and res.get("success", False):
                    has_params = True
                    break

            if not has_params:
                self.param_table.setRowCount(1)
                msg_item = QTableWidgetItem("No fitted parameters available (fixed models only).")
                msg_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.param_table.setSpan(0, 0, 1, 4)
                self.param_table.setItem(0, 0, msg_item)
                return

            for name, res in results.items():
                if res.get("params") and res.get("success", False):
                    perr = res.get("perr", [None] * len(res["params"]))
                    for (pname, pval), std_err in zip(res["params"].items(), perr):
                        row = self.param_table.rowCount()
                        self.param_table.insertRow(row)

                        name_item = QTableWidgetItem(name)
                        if name == best_name:
                            name_item.setForeground(QColor("#27ae60"))
                            font = name_item.font()
                            font.setBold(True)
                            name_item.setFont(font)
                        self.param_table.setItem(row, 0, name_item)

                        self.param_table.setItem(row, 1, QTableWidgetItem(str(pname)))

                        value_item = QTableWidgetItem(f"{pval:.4f}")
                        if name == best_name:
                            font = value_item.font()
                            font.setBold(True)
                            value_item.setFont(font)
                        self.param_table.setItem(row, 2, value_item)

                        if std_err is not None:
                            self.param_table.setItem(row, 3, QTableWidgetItem(f"{std_err:.4f}"))
                        else:
                            self.param_table.setItem(row, 3, QTableWidgetItem("-"))
        except Exception as e:
            print(f"[ERROR] update_param_table: {e}")

    def update_report_preview(self):
        try:
            current_file = self.file_selector.currentText() if hasattr(self, 'file_selector') else "Combined Data"

            if current_file != "Combined Data" and current_file in self.multi_file_results:
                results = self.multi_file_results[current_file]
            else:
                results = self.combined_fit_results if self.combined_fit_results else self.fit_results

            if not results:
                self.report_preview.setHtml("<p style='color:#7f8c8d; text-align:center;'>No results available. Run analysis first.</p>")
                return

            filtered_results = {}
            for name, res in results.items():
                if res.get("success", False):
                    r2 = res.get("r2", -1)
                    if np.isfinite(r2):
                        filtered_results[name] = res

            if not filtered_results:
                self.report_preview.setHtml("<p style='color:#7f8c8d; text-align:center;'>No valid model fits available.</p>")
                return

            sorted_results = sorted(
                filtered_results.items(),
                key=lambda x: x[1].get("r2", -1),
                reverse=True
            )

            best_name = None
            best_r2 = -1
            for name, res in sorted_results:
                if res.get("success", False):
                    r2 = res.get("r2", -1)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_name = name

            if self.current_mode == "nanothermite":
                convention = f"φ = {self.filler_name} (filler), (1-φ) = {self.matrix_name} (matrix)"
            else:
                convention = f"φ = {self.filler_name} (filler), (1-φ) = {self.matrix_name} (polymer matrix)"

            html = f"""<h2 style='color:#2c3e50;'>Thermal Conductivity Modeling Report - {self.mode_label}</h2>
            <hr><p><b>Material System:</b> {self.material_info}</p>
            <p><b>Data Set:</b> {current_file}</p>
            <p><b>Data Points:</b> {len(self.df) if self.df is not None else 0}</p>
            <p><b>Models Evaluated:</b> {len(filtered_results)}</p>
            <p><b>Convention:</b> {convention}</p>
            <hr><h3>Statistical Comparison</h3>
            <table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;width:100%;'>
            <tr style='background:#2c3e50;color:white;font-weight:bold;'>
            <th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th><th>MAPE %</th><th>Status</th></tr>"""

            for name, res in sorted_results:
                if res["success"]:
                    r2 = res.get('r2', 0)
                    if name == best_name:
                        style_color = "color: #27ae60; font-weight: bold; background-color: #d4edda;"
                        name_display = f"🏆 {name}"
                    else:
                        style_color = ""
                        name_display = name
                    html += f"<tr><td style='{style_color}'><b>{name_display}</b></td>"
                    html += f"<td style='{style_color}'>{r2:.4f}</td>"
                    html += f"<td style='{style_color}'>{res.get('rmse', 0):.4f}</td>"
                    html += f"<td style='{style_color}'>{res.get('mae', 0):.4f}</td>"
                    html += f"<td style='{style_color}'>{res.get('mape', 0):.2f}</td>"
                    html += "<td style='color:green;'>✅ Success</td></tr>"
                else:
                    html += f"<tr><td>{name}</td><td>-</td><td>-</td><td>-</td><td>-</td><td style='color:red;'>❌ Failed</td></tr>"

            html += "</table>"

            if best_name and best_r2 > 0:
                html += f"""<hr><h3 style='color:#27ae60;'>🏆 Best Performing Model: {best_name}</h3>
                <p style='color:#27ae60; font-weight:bold;'>
                R² = {best_r2:.4f} | RMSE = {res.get('rmse', 0):.4f} W/m·K | MAE = {res.get('mae', 0):.4f} W/m·K
                </p>"""

            html += f"<hr><p style='color:#7f8c8d;font-size:11px;'>Generated by Thermal Conductivity Modeling Suite v1.0 - {self.mode_label} Mode | Oukil Khaled</p>"
            self.report_preview.setHtml(html)
        except Exception as e:
            print(f"[ERROR] update_report_preview: {e}")
            self.report_preview.setHtml(f"<p style='color:red;'>Error loading report: {str(e)}</p>")

    # ========================================================================
    # SET BUTTONS ENABLED
    # ========================================================================

    def _set_buttons_enabled(self, enabled):
        for child in self.findChildren(QPushButton):
            if child.text() in ["▶ Run", "⚙️ Params", "📂 Load", "📂 Load Multi", "🧠 Smart", "🔍 Discover"]:
                child.setEnabled(enabled)

    def _update_progress(self, value, message):
        self.progress.setValue(value)
        self.status.showMessage(message)

    # ========================================================================
    # ACTIONS
    # ========================================================================

    def load_data(self):
        try:
            from .data_dialog import DataLoadDialog
            dialog = DataLoadDialog(mode=self.current_mode, parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                result = dialog.get_data()
                if result:
                    self.df = result['data']
                    self.material_info = result['material']
                    self.matrix_name = result['matrix']
                    self.filler_name = result['filler']
                    self.file_path = result.get('file_path', '')
                    
                    self.k_matrix = result.get('k_matrix', get_k_value(self.current_mode, "matrix", self.matrix_name))
                    self.k_filler = result.get('k_filler', get_k_value(self.current_mode, "filler", self.filler_name))

                    if 'Filler_volume_fraction' not in self.df.columns:
                        if 'phi' in self.df.columns:
                            self.df['Filler_volume_fraction'] = self.df['phi']
                            self.df['Matrix_volume_fraction'] = 1.0 - self.df['phi']
                        elif 'Red_volume_fraction' in self.df.columns:
                            self.df['Filler_volume_fraction'] = self.df['Red_volume_fraction']
                            self.df['Matrix_volume_fraction'] = 1.0 - self.df['Red_volume_fraction']

                    self.multi_files = []
                    self.multi_file_results = {}
                    self.combined_fit_results = {}
                    self.fit_results = {}

                    self.populate_data_table()
                    self._populate_models_by_category()
                    self.status.showMessage(f"Loaded {len(self.df)} data points from {os.path.basename(self.file_path)}")
                    self._enable_best_r2_tab(False)
                    self.update_stats_table()
                    self.update_param_table()
                    self.update_report_preview()
                    
                    if dialog.auto_plot_check.isChecked():
                        self.update_plots()

                    QMessageBox.information(self, "Success", f"Data loaded successfully!\n\nMaterial: {self.material_info}\nData points: {len(self.df)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n\n{str(e)}")

    def load_multiple_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Multiple Data Files", "",
            "Data Files (*.txt *.csv *.xlsx *.xls);;All Files (*.*)"
        )

        if not paths:
            return

        self.multi_files = []
        self.multi_file_results = {}
        self.current_file_index = 0
        self.current_display_file = "Combined Data"

        all_data = []
        for path in paths:
            try:
                df = load_data(path, mode=self.current_mode)
                filename = os.path.basename(path)
                self.multi_files.append((filename, df.copy()))
                all_data.append(df)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load {path}:\n{str(e)}")

        if not all_data:
            return

        combined_df = pd.concat(all_data, ignore_index=True)
        self.df = combined_df

        if 'Filler_volume_fraction' not in self.df.columns:
            if 'phi' in self.df.columns:
                self.df['Filler_volume_fraction'] = self.df['phi']
                self.df['Matrix_volume_fraction'] = 1.0 - self.df['phi']

        first_df = all_data[0]
        self.material_info = f"Multiple Datasets ({len(paths)} files)"
        self.matrix_name = first_df.attrs.get('matrix', self.matrix_name)
        self.filler_name = first_df.attrs.get('filler', self.filler_name)
        self.file_path = ", ".join([os.path.basename(p) for p in paths])

        self.fit_results = {}
        self.combined_fit_results = {}
        self.multi_file_results = {}

        self.populate_data_table()
        self._populate_models_by_category()
        self.status.showMessage(f"Loaded {len(self.df)} data points from {len(paths)} files")

        if hasattr(self, 'file_selector'):
            self.file_selector.clear()
            self.file_selector.addItem("Combined Data")
            for filename, _ in self.multi_files:
                self.file_selector.addItem(filename)

        self.tab_best_r2.clear()
        self._enable_best_r2_tab(True)
        self.tab_best_r2.interpretation_text.setText(
            f"📋 {len(paths)} files loaded successfully!\n\n"
            "To populate this tab:\n"
            "1. Make sure models are selected in the 'Model Selection' panel\n"
            "2. Click 'Run Analysis' and select 'Yes' when prompted for multi-file analysis\n"
            "3. Wait for the analysis to complete\n"
            "4. Results will appear here automatically"
        )

        self.update_stats_table()
        self.update_param_table()
        self.update_report_preview()
        self.update_plots()

        QMessageBox.information(self, "Success", f"✅ Loaded {len(paths)} files successfully!\nTotal data points: {len(self.df)}")

    def populate_data_table(self):
        try:
            if self.df is None:
                return

            self.data_table.blockSignals(True)
            self.data_table.setRowCount(0)
            self.data_table.setUpdatesEnabled(False)

            if 'Filler_volume_fraction' not in self.df.columns:
                if 'phi' in self.df.columns:
                    self.df['Filler_volume_fraction'] = self.df['phi']
                    self.df['Matrix_volume_fraction'] = 1.0 - self.df['phi']

            if self.multi_files and len(self.multi_files) > 0:
                for file_idx, (filename, df) in enumerate(self.multi_files):
                    try:
                        display_name = os.path.splitext(filename)[0]
                        header_row = self.data_table.rowCount()
                        self.data_table.insertRow(header_row)
                        header_item = QTableWidgetItem(f"📁 {display_name}")
                        header_item.setBackground(QColor("#2c3e50"))
                        header_item.setForeground(QColor("white"))
                        header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        font = header_item.font()
                        font.setBold(True)
                        font.setPointSize(11)
                        header_item.setFont(font)
                        self.data_table.setItem(header_row, 0, header_item)
                        self.data_table.setSpan(header_row, 0, 1, 5)
                        self.data_table.setRowHeight(header_row, 32)

                        for i, row in df.iterrows():
                            data_row = self.data_table.rowCount()
                            self.data_table.insertRow(data_row)

                            phi_val = row.get('Filler_volume_fraction', 0)
                            matrix_val = 1.0 - phi_val

                            self.data_table.setItem(data_row, 0, QTableWidgetItem(str(i + 1)))
                            self.data_table.setItem(data_row, 1, QTableWidgetItem(f"{phi_val:.6f}"))
                            self.data_table.setItem(data_row, 2, QTableWidgetItem(f"{row.get('k_meas', 0):.4f}"))
                            std_val = row.get("k_std_dev", "")
                            if pd.notna(std_val):
                                self.data_table.setItem(data_row, 3, QTableWidgetItem(f"{std_val:.4f}"))
                            else:
                                self.data_table.setItem(data_row, 3, QTableWidgetItem(""))
                            self.data_table.setItem(data_row, 4, QTableWidgetItem(f"{matrix_val:.6f}"))
                            self.data_table.setRowHeight(data_row, 24)

                        if file_idx < len(self.multi_files) - 1:
                            sep_row = self.data_table.rowCount()
                            self.data_table.insertRow(sep_row)
                            sep_item = QTableWidgetItem("─" * 80)
                            sep_item.setBackground(QColor("#ecf0f1"))
                            sep_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                            self.data_table.setItem(sep_row, 0, sep_item)
                            self.data_table.setSpan(sep_row, 0, 1, 5)
                            self.data_table.setRowHeight(sep_row, 6)
                    except Exception as e:
                        print(f"[ERROR] Error processing file {filename}: {e}")
                        continue
            else:
                try:
                    display_name = os.path.splitext(os.path.basename(self.file_path))[0] if self.file_path else "Data"
                    header_row = self.data_table.rowCount()
                    self.data_table.insertRow(header_row)
                    header_item = QTableWidgetItem(f"📁 {display_name}")
                    header_item.setBackground(QColor("#2c3e50"))
                    header_item.setForeground(QColor("white"))
                    header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    font = header_item.font()
                    font.setBold(True)
                    header_item.setFont(font)
                    self.data_table.setItem(header_row, 0, header_item)
                    self.data_table.setSpan(header_row, 0, 1, 5)
                    self.data_table.setRowHeight(header_row, 32)

                    for i, row in self.df.iterrows():
                        data_row = self.data_table.rowCount()
                        self.data_table.insertRow(data_row)

                        phi_val = row.get('Filler_volume_fraction', 0)
                        matrix_val = 1.0 - phi_val

                        self.data_table.setItem(data_row, 0, QTableWidgetItem(str(i + 1)))
                        self.data_table.setItem(data_row, 1, QTableWidgetItem(f"{phi_val:.6f}"))
                        self.data_table.setItem(data_row, 2, QTableWidgetItem(f"{row.get('k_meas', 0):.4f}"))
                        std_val = row.get("k_std_dev", "")
                        if pd.notna(std_val):
                            self.data_table.setItem(data_row, 3, QTableWidgetItem(f"{std_val:.4f}"))
                        else:
                            self.data_table.setItem(data_row, 3, QTableWidgetItem(""))
                        self.data_table.setItem(data_row, 4, QTableWidgetItem(f"{matrix_val:.6f}"))
                        self.data_table.setRowHeight(data_row, 24)
                except Exception as e:
                    print(f"[ERROR] Error processing single file: {e}")

            self.data_table.setUpdatesEnabled(True)
            self.data_table.blockSignals(False)
        except Exception as e:
            print(f"[ERROR] populate_data_table: {e}")
            self.data_table.setUpdatesEnabled(True)
            self.data_table.blockSignals(False)
            QMessageBox.critical(self, "Error", f"Failed to display data:\n\n{str(e)}")

    def add_manual_row(self):
        try:
            if self.multi_files and len(self.multi_files) > 0:
                last_file_idx = len(self.multi_files) - 1
                filename, df = self.multi_files[last_file_idx]
                
                new_row = pd.DataFrame({
                    'Filler_volume_fraction': [0.0],
                    'Matrix_volume_fraction': [1.0],
                    'k_meas': [0.0],
                    'k_std_dev': [0.0],
                    'phi': [0.0]
                })
                df_updated = pd.concat([df, new_row], ignore_index=True)
                self.multi_files[last_file_idx] = (filename, df_updated)
                self.df = pd.concat([df_updated for _, df in self.multi_files], ignore_index=True)
            else:
                if self.df is not None:
                    new_row = pd.DataFrame({
                        'Filler_volume_fraction': [0.0],
                        'Matrix_volume_fraction': [1.0],
                        'k_meas': [0.0],
                        'k_std_dev': [0.0],
                        'phi': [0.0]
                    })
                    self.df = pd.concat([self.df, new_row], ignore_index=True)
                else:
                    self.df = pd.DataFrame({
                        'Filler_volume_fraction': [0.0],
                        'Matrix_volume_fraction': [1.0],
                        'k_meas': [0.0],
                        'k_std_dev': [0.0],
                        'phi': [0.0]
                    })
                    self.file_path = "Manual Data"

            self.populate_data_table()
            self.status.showMessage("Added manual row")
        except Exception as e:
            print(f"[ERROR] add_manual_row: {e}")
            QMessageBox.warning(self, "Error", f"Failed to add row: {str(e)}")

    def on_data_cell_changed(self, row, column):
        if self.df is None:
            return

        if self.multi_files and len(self.multi_files) > 0:
            header_rows = []
            current_row = 0
            for filename, df in self.multi_files:
                header_rows.append(current_row)
                current_row += len(df) + 1
                current_row += 1
            if row in header_rows:
                return

        if column == 1:
            try:
                item = self.data_table.item(row, column)
                if item is not None:
                    phi_val = float(item.text())
                    phi_val = max(0, min(1, phi_val))
                    
                    self.data_table.blockSignals(True)
                    self.data_table.setItem(row, column, QTableWidgetItem(f"{phi_val:.6f}"))
                    matrix_val = 1.0 - phi_val
                    self.data_table.setItem(row, 4, QTableWidgetItem(f"{matrix_val:.6f}"))
                    self.data_table.blockSignals(False)
                    
                    self._update_dataframe_from_table(row, phi_val, matrix_val)
            except ValueError:
                pass
            except Exception as e:
                print(f"[ERROR] on_data_cell_changed: {e}")

        elif column == 2:
            try:
                item = self.data_table.item(row, column)
                if item is not None:
                    k_val = float(item.text())
                    self.data_table.blockSignals(True)
                    self.data_table.setItem(row, column, QTableWidgetItem(f"{k_val:.4f}"))
                    self.data_table.blockSignals(False)
                    self._update_dataframe_k_meas(row, k_val)
            except ValueError:
                pass
            except Exception as e:
                print(f"[ERROR] on_data_cell_changed: {e}")

        elif column == 4:
            try:
                item = self.data_table.item(row, column)
                if item is not None:
                    matrix_val = float(item.text())
                    matrix_val = max(0, min(1, matrix_val))
                    
                    self.data_table.blockSignals(True)
                    self.data_table.setItem(row, column, QTableWidgetItem(f"{matrix_val:.6f}"))
                    phi_val = 1.0 - matrix_val
                    self.data_table.setItem(row, 1, QTableWidgetItem(f"{phi_val:.6f}"))
                    self.data_table.blockSignals(False)
                    
                    self._update_dataframe_from_table(row, phi_val, matrix_val)
            except ValueError:
                pass
            except Exception as e:
                print(f"[ERROR] on_data_cell_changed: {e}")

    def _update_dataframe_from_table(self, row, phi_val, matrix_val):
        try:
            if self.multi_files and len(self.multi_files) > 0:
                file_idx, data_idx = self._get_file_and_index(row)
                if file_idx >= 0 and data_idx >= 0:
                    self.multi_files[file_idx][1].at[data_idx, 'Filler_volume_fraction'] = phi_val
                    self.multi_files[file_idx][1].at[data_idx, 'Matrix_volume_fraction'] = matrix_val
                    self.multi_files[file_idx][1].at[data_idx, 'phi'] = phi_val
            else:
                if row < len(self.df):
                    self.df.at[row, 'Filler_volume_fraction'] = phi_val
                    self.df.at[row, 'Matrix_volume_fraction'] = matrix_val
                    self.df.at[row, 'phi'] = phi_val
        except Exception as e:
            print(f"[ERROR] _update_dataframe_from_table: {e}")

    def _update_dataframe_k_meas(self, row, k_val):
        try:
            if self.multi_files and len(self.multi_files) > 0:
                file_idx, data_idx = self._get_file_and_index(row)
                if file_idx >= 0 and data_idx >= 0:
                    self.multi_files[file_idx][1].at[data_idx, 'k_meas'] = k_val
            else:
                if row < len(self.df):
                    self.df.at[row, 'k_meas'] = k_val
        except Exception as e:
            print(f"[ERROR] _update_dataframe_k_meas: {e}")

    def _get_file_and_index(self, row):
        current_row = 0
        for file_idx, (filename, df) in enumerate(self.multi_files):
            current_row += 1
            if row == current_row - 1:
                return -1, -1
            if row < current_row + len(df):
                return file_idx, row - current_row
            current_row += len(df)
            current_row += 1
        return -1, -1

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    def run_analysis(self):
        if self.df is None or len(self.df) < 2:
            QMessageBox.warning(self, "Warning", "Please load experimental data first (minimum 2 points).")
            return

        selected = [name for name, cb in self.model_checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one model.")
            return

        if hasattr(self, 'multi_files') and self.multi_files and len(self.multi_files) > 0:
            reply = QMessageBox.question(self, "Multi-File Analysis",
                f"You have {len(self.multi_files)} files loaded.\n\n"
                "Do you want to run analysis on individual files?\n"
                "• Yes: Analysis on each file individually (for Best R² tab)\n"
                "• No: Analysis on combined data only",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                self.run_multi_file_analysis()
                return

        phi = self.df["Filler_volume_fraction"].values
        k = self.df["k_meas"].values

        self.progress.setValue(0)
        self.status.showMessage("Running analysis... fitting models")

        self._set_buttons_enabled(False)
        self.analysis_running = True
        # ===== PASS MODE =====
        self.worker = FitWorker(phi, k, selected, self.custom_params, mode=self.current_mode)
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.start()

    def _on_analysis_finished(self, results):
        self.combined_fit_results = results
        self.fit_results = results

        self.progress.setValue(100)
        self.analysis_running = False
        successes = sum(1 for r in results.values() if r["success"])
        self.status.showMessage(f"Analysis complete: {successes}/{len(results)} models fitted successfully")

        self.legend_format = "compact" if self.legend_format_combo.currentText() == "Compact" else "full"

        if self.df is not None:
            self.reset_x_range()

        if hasattr(self, 'file_selector') and self.multi_files:
            self.file_selector.clear()
            self.file_selector.addItem("Combined Data")
            for filename, _ in self.multi_files:
                self.file_selector.addItem(filename)

        self.update_plots()
        self.update_stats_table()
        self.update_param_table()
        self.update_report_preview()
        self.tabs.setCurrentIndex(1)
        self._set_buttons_enabled(True)

        if hasattr(self, 'tab_best_r2'):
            if self.multi_file_results and len(self.multi_file_results) > 0:
                self.tab_best_r2.update_data(self.multi_file_results, self.fit_results, self.df)
                self._enable_best_r2_tab(True)
                self.tabs.setCurrentIndex(3)
            elif self.multi_files and len(self.multi_files) > 0:
                self.tab_best_r2.clear()
                self._enable_best_r2_tab(True)
                self.tab_best_r2.interpretation_text.setText(
                    "📋 Multiple files are loaded but no multi-file analysis has been run.\n\n"
                    "To populate this tab:\n"
                    "1. Make sure models are selected\n"
                    "2. Click 'Run Analysis' and select 'Yes' for multi-file analysis\n"
                    "3. Wait for the analysis to complete"
                )
            else:
                self.tab_best_r2.clear()
                self._enable_best_r2_tab(False)

        best = max(results.items(), key=lambda x: x[1].get("r2", -1) if x[1]["success"] else -1)
        if best[1]["success"]:
            QMessageBox.information(self, "Analysis Complete",
                f"<b>{successes}</b> models fitted successfully.<br><br>"
                f"<b>Best Model:</b> {best[0]}<br>"
                f"R² = {best[1].get('r2', 0):.4f}<br>"
                f"RMSE = {best[1].get('rmse', 0):.4f} W/m·K")

    def _on_analysis_error(self, error_msg):
        self._set_buttons_enabled(True)
        self.progress.setValue(0)
        self.analysis_running = False
        self.status.showMessage("Analysis failed")
        QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n\n{error_msg}")

    def run_multi_file_analysis(self):
        if not self.multi_files:
            QMessageBox.warning(self, "Warning", "No multiple files loaded.")
            return

        selected = [name for name, cb in self.model_checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one model.")
            return

        discovered_selected = [name for name in selected if self.MODEL_REGISTRY.get(name, {}).get("discovered", False)]
        if discovered_selected:
            reply = QMessageBox.question(self, "Discovered Models",
                f"You have {len(discovered_selected)} discovered model(s) selected.\n"
                "Do you want to include them in the analysis?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                selected = [name for name in selected if name not in discovered_selected]
                if not selected:
                    QMessageBox.warning(self, "Warning", "No models selected.")
                    return

        self.status.showMessage("Running analysis on individual files...")
        self.progress.setValue(0)
        self._set_buttons_enabled(False)
        self.analysis_running = True

        files_data = []
        for filename, df in self.multi_files:
            phi = df["Filler_volume_fraction"].values
            k = df["k_meas"].values
            files_data.append((filename, phi, k, df))

        # ===== PASS MODE =====
        self.multi_worker = MultiFileWorker(files_data, selected, self.custom_params, mode=self.current_mode)
        self.multi_worker.progress.connect(self._update_progress)
        self.multi_worker.finished.connect(self._on_multi_file_finished)
        self.multi_worker.error.connect(self._on_multi_file_error)
        self.multi_worker.start()

    def _on_multi_file_finished(self, results):
        self.multi_file_results = results
        self.progress.setValue(100)
        self._set_buttons_enabled(True)
        self.analysis_running = False

        if hasattr(self, 'file_selector'):
            self.file_selector.clear()
            self.file_selector.addItem("Combined Data")
            for filename in results.keys():
                self.file_selector.addItem(filename)

        selected = [name for name, cb in self.model_checks.items() if cb.isChecked()]
        if selected and self.df is not None:
            phi = self.df["Filler_volume_fraction"].values
            k = self.df["k_meas"].values
            # ===== PASS MODE =====
            combined_results = fit_all_models(phi, k, selected, self.custom_params, mode=self.current_mode)
            for name, res in combined_results.items():
                if res["success"] and res.get("y_pred") is not None:
                    stats = calculate_statistics(k, res["y_pred"])
                    res.update(stats)
                    res["residuals"] = k - res["y_pred"]
                else:
                    res.update({"r2": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan})
            self.combined_fit_results = combined_results
            self.fit_results = combined_results

        self.update_stats_table()
        self.update_param_table()
        self._show_scrollable_results_dialog(results)

        if hasattr(self, 'tab_best_r2'):
            self.tab_best_r2.update_data(results, self.fit_results, self.df)
            self._enable_best_r2_tab(True)
            self.tabs.setCurrentIndex(3)

        self.update_plots()
        self.update_report_preview()
        self.status.showMessage(f"Multi-file analysis complete: {len(results)} files analyzed")

    def _on_multi_file_error(self, error_msg):
        self._set_buttons_enabled(True)
        self.progress.setValue(0)
        self.analysis_running = False
        self.status.showMessage("Multi-file analysis failed")
        QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n\n{error_msg}")

    def _show_scrollable_results_dialog(self, results):
        dialog = QDialog(self)
        dialog.setWindowTitle("Multi-File Analysis Results")
        dialog.setMinimumSize(700, 500)
        dialog.setMaximumSize(900, 700)

        layout = QVBoxLayout(dialog)
        title = QLabel(f"📊 Multi-File Analysis Results ({len(results)} files)")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        summary = self._generate_multi_file_summary(results)
        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet("font-family: 'Courier New', monospace; font-size: 11px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;")
        scroll_layout.addWidget(summary_label)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dialog.close)
        layout.addWidget(btn_box)

        dialog.exec()

    def _generate_multi_file_summary(self, results):
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append(f"MULTI-FILE ANALYSIS SUMMARY - {self.mode_label}")
        summary_lines.append("=" * 60)

        best_models = {}
        for filename, file_results in results.items():
            best_name = None
            best_r2 = -1
            for model_name, result in file_results.items():
                if result.get("success", False):
                    r2 = result.get("r2", -1)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_name = model_name

            if best_name:
                best_models[filename] = (best_name, best_r2)
                summary_lines.append(f"\n📁 {filename}")
                summary_lines.append(f"   Best Model: {best_name}")
                summary_lines.append(f"   R² = {best_r2:.4f}")

        model_scores = {}
        for filename, file_results in results.items():
            for model_name, result in file_results.items():
                if result.get("success", False):
                    r2 = result.get("r2", -1)
                    if model_name not in model_scores:
                        model_scores[model_name] = []
                    model_scores[model_name].append(r2)

        if model_scores:
            summary_lines.append("\n" + "=" * 60)
            summary_lines.append("OVERALL MODEL PERFORMANCE (Average R²)")
            summary_lines.append("=" * 60)

            avg_scores = []
            for model_name, scores in model_scores.items():
                avg_r2 = np.mean(scores) if scores else 0
                std_r2 = np.std(scores) if scores else 0
                avg_scores.append((model_name, avg_r2, std_r2, len(scores)))

            avg_scores.sort(key=lambda x: x[1], reverse=True)

            for i, (model_name, avg_r2, std_r2, n) in enumerate(avg_scores[:10], 1):
                marker = "🏆" if i == 1 else "  "
                summary_lines.append(f"{marker} {i:2d}. {model_name:<35} R² = {avg_r2:.4f} ± {std_r2:.4f} (n={n})")

        return "\n".join(summary_lines)

    # ========================================================================
    # ACTIONS (continued)
    # ========================================================================

    def open_parameter_dialog(self):
        selected = [name for name, cb in self.model_checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one model first.")
            return
        dialog = ModelParameterDialog(selected, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            custom = dialog.get_custom_params()
            if custom is not None:
                self.custom_params = custom
                self.status.showMessage(f"Parameters configured for {len(custom)} model(s)")
                QMessageBox.information(self, "Parameters Saved",
                                       f"Custom parameters configured for {len(custom)} model(s).\nClick 'Run Analysis' to execute fitting.")

    def open_smart_fit(self):
        if self.df is None or len(self.df) < 2:
            QMessageBox.warning(self, "Warning", "Please load experimental data first (minimum 2 points).")
            return
        try:
            from .smart_fit_dialog import SmartFitDialog
            dialog = SmartFitDialog(self.df, self, mode=self.current_mode)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Smart Fit:\n{str(e)}")

    def open_model_discovery(self):
        if self.df is None or len(self.df) < 2:
            QMessageBox.warning(self, "Warning", "Please load experimental data first (minimum 2 points).")
            return
        try:
            from .model_discovery_dialog import ModelDiscoveryDialog
            
            # Create and show the dialog
            dialog = ModelDiscoveryDialog(self.df, self, mode=self.current_mode)
            
            # Connect the models_added signal to refresh the UI
            dialog.models_added.connect(self._refresh_after_discovery)
            
            # Show the dialog
            result = dialog.exec()
            
            # If the dialog was accepted (models were added), refresh
            if result == QDialog.DialogCode.Accepted:
                self._refresh_after_discovery()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open model discovery:\n{str(e)}\n{traceback.format_exc()}")

    def _refresh_after_discovery(self):
        """Refresh the UI after discovering models."""
        print("[DEBUG] _refresh_after_discovery called")
        
        # ===== FIX: Don't reload the registry, just refresh the UI =====
        # The models are already in self.MODEL_REGISTRY from the dialog
        
        # Ensure "Discovered Models" category exists
        if "Discovered Models" not in self.MODEL_CATEGORIES:
            self.MODEL_CATEGORIES["Discovered Models"] = []
        
        # Find all discovered models
        discovered_models = [name for name, meta in self.MODEL_REGISTRY.items() 
                            if meta.get("discovered", False)]
        self.MODEL_CATEGORIES["Discovered Models"] = discovered_models
        print(f"[DEBUG] Discovered models in main window: {discovered_models}")
        
        # Repopulate the model selection grid
        self._populate_models_by_category()
        
        # Update the category combo box
        self.category_combo.clear()
        self.category_combo.addItems(["All Models"] + list(self.MODEL_CATEGORIES.keys()))
        self.category_combo.setCurrentText("All Models")
        
        # Update the models reference tab
        self._update_models_reference()
        
        # Count discovered models
        count = len(discovered_models)
        if count > 0:
            self.status.showMessage(f"✅ {count} new model(s) discovered and added to the registry!")
            # Show the Models Reference tab to let the user see the new models
            self.tabs.setCurrentIndex(self.models_tab_index)
            QMessageBox.information(self, "Models Added",
                f"✅ {count} new model(s) have been added to the registry.\n"
                "They now appear in the 'Model Selection' panel.\n"
                "Look for the 'Discovered Models' category (in green).")

    def open_generate_data(self):
        try:
            dialog = GenerateDataDialog(mode=self.current_mode, parent=self)
            dialog.exec()
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Failed to open Generate Data dialog:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")

    def save_graph(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Please load data and run analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", "thermal_conductivity_plot.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tiff)"
        )
        if not path:
            return
        try:
            export_plot(self.canvas_main, path)
            self.status.showMessage(f"Graph saved to {os.path.basename(path)}")
            QMessageBox.information(self, "Success", f"Graph saved successfully.\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save graph:\n{str(e)}")

    def export_results(self):
        if not self.fit_results:
            QMessageBox.warning(self, "Warning", "No results to export. Run analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "results.xlsx",
            "Excel (*.xlsx);;CSV (*.csv)"
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".xlsx":
                export_excel(path, self.df, self.fit_results)
            elif ext == ".csv":
                export_csv(path, self.df, self.fit_results)
            self.status.showMessage(f"Results exported to {os.path.basename(path)}")
            QMessageBox.information(self, "Success", f"Results exported successfully.\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def _do_export(self, fmt):
        if fmt == "excel":
            path, _ = QFileDialog.getSaveFileName(self, "Export to Excel", "results.xlsx", "Excel (*.xlsx)")
            if path:
                export_excel(path, self.df, self.fit_results)
                self.status.showMessage(f"Excel exported: {os.path.basename(path)}")
        elif fmt == "csv":
            path, _ = QFileDialog.getSaveFileName(self, "Export to CSV", "results.csv", "CSV (*.csv)")
            if path:
                export_csv(path, self.df, self.fit_results)
                self.status.showMessage(f"CSV exported: {os.path.basename(path)}")

    def generate_report(self):
        if not self.fit_results:
            QMessageBox.warning(self, "Warning", "No results to report. Run analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", "report.pdf", "PDF (*.pdf)")
        if not path:
            return
        try:
            tmpdir = tempfile.mkdtemp()
            fig_main = os.path.join(tmpdir, "main.png")
            fig_resid = os.path.join(tmpdir, "resid.png")
            self.canvas_main.fig.savefig(fig_main, dpi=300, bbox_inches="tight")
            self.canvas_resid.fig.savefig(fig_resid, dpi=300, bbox_inches="tight")

            meta = {
                "Material": self.material_info,
                "Matrix": self.matrix_name,
                "Filler": self.filler_name,
                "k_matrix": f"{self.k_matrix:.2f} W/m·K",
                "k_filler": f"{self.k_filler:.2f} W/m·K",
                "Data Points": str(len(self.df)) if self.df is not None else "0",
                "Software": f"Thermal Conductivity Modeling Suite v1.0 - {self.mode_label} Mode",
                "Developer": "Oukil Khaled",
                "Mode": self.mode_label
            }

            generate_pdf_report(path, self.df, self.fit_results, meta, [fig_main, fig_resid], mode=self.current_mode)
            self.status.showMessage(f"PDF report saved to {os.path.basename(path)}")
            QMessageBox.information(self, "Success", f"Professional PDF report generated.\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Report generation failed:\n{str(e)}")

    def show_about(self):
        dialog = AboutDialog(mode=self.current_mode, parent=self)
        dialog.exec()