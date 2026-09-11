"""Smart Fit Dialog - Interactive model fitting interface."""

import numpy as np
import pandas as pd
import traceback

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QComboBox, QPushButton, QLabel, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QSplitter, QScrollArea, QCheckBox,
    QSpinBox, QDoubleSpinBox, QMessageBox, QProgressBar,
    QFileDialog, QToolTip
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QColor, QAction

from ..core.models import get_model_registry
from ..core.statistics import calculate_statistics
from ..smart_fit.fit_engine import SmartFitEngine
from ..visualization.plotter import MplCanvas


# ============================================================================
# WORKER THREAD
# ============================================================================

class FitWorkerThread(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, engine, model_name, p0=None, bounds=None,
                 method='trf', mode="nanothermite",
                 k_matrix=237.0, k_filler=33.0):
        super().__init__()
        self.engine = engine
        self.model_name = model_name
        self.p0 = p0
        self.bounds = bounds
        self.method = method
        self.mode = mode
        self.k_matrix = k_matrix
        self.k_filler = k_filler

    def run(self):
        try:
            self.progress.emit(5, f"Preparing {self.model_name}...")

            # Inject mode + material constants into the engine
            self.engine.mode = self.mode
            self.engine.k_matrix = float(self.k_matrix)
            self.engine.k_filler = float(self.k_filler)

            self.progress.emit(30, f"Fitting {self.model_name}...")

            result = self.engine.fit_model(
                self.model_name,
                p0=self.p0,
                bounds=self.bounds,
                method=self.method,
            )

            self.progress.emit(100, "Fitting completed")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())


# ============================================================================
# SMART FIT DIALOG
# ============================================================================

class SmartFitDialog(QDialog):
    """Main Smart Fit dialog."""

    def __init__(self, df, parent=None, mode="nanothermite"):
        super().__init__(parent)
        self.df = df
        self.mode = mode
        self.fit_engine = None
        self.fit_results = {}
        self.current_model = None
        self.worker = None
        self.optimize_timer = None

        self.setWindowTitle(f"Smart Fit - Advanced Model Fitting ({mode})")
        self.setMinimumSize(1200, 800)
        self.setModal(False)

        self._build_ui()
        self._load_models()

    # ========================================================================
    # UI BUILDING
    # ========================================================================

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # ---------- TOP CONTROLS ----------
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)

        top_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.setToolTip("Select the model to fit")
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        top_layout.addWidget(self.model_combo)

        top_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(['trf', 'lm', 'dogbox'])
        self.method_combo.setMinimumWidth(100)
        self.method_combo.setToolTip(
            "Optimization algorithm:\n"
            "• trf: Trust Region Reflective (recommended, handles bounds)\n"
            "• lm: Levenberg-Marquardt (fast, ignores bounds)\n"
            "• dogbox: Dogleg with box constraints"
        )
        top_layout.addWidget(self.method_combo)

        self.method_info_label = QLabel("")
        self.method_info_label.setStyleSheet(
            "color: #7f8c8d; font-size: 10px; font-style: italic;"
        )
        self.method_info_label.setWordWrap(True)
        self.method_info_label.setMaximumWidth(350)
        top_layout.addWidget(self.method_info_label)
        self.method_combo.currentIndexChanged.connect(self._update_method_info)

        self.run_button = QPushButton("▶ Run Fit")
        self.run_button.setToolTip("Run fitting with the selected model")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; color: white;
                padding: 8px 20px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.run_button.clicked.connect(self.run_fit)
        top_layout.addWidget(self.run_button)

        self.optimize_button = QPushButton("⚡ Auto-Optimize")
        self.optimize_button.setToolTip("Find optimal parameters via global optimization")
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #f39c12; color: white;
                padding: 8px 20px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #f1c40f; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.optimize_button.clicked.connect(self.auto_optimize)
        top_layout.addWidget(self.optimize_button)

        self.sens_button = QPushButton("📊 Sensitivity")
        self.sens_button.setToolTip("Analyze parameter sensitivity")
        self.sens_button.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad; color: white;
                padding: 8px 20px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #9b59b6; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.sens_button.clicked.connect(self.analyze_sensitivity)
        top_layout.addWidget(self.sens_button)

        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        # ====================================================================
        # MODEL RECOMMENDATION HINT (mode-aware)
        # ====================================================================
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setTextFormat(Qt.TextFormat.RichText)
        self.hint_label.setStyleSheet("""
            QLabel {
                background-color: #eaf2f8;
                border-left: 4px solid #2980b9;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 11px;
                color: #2c3e50;
            }
        """)
        self._update_hint()
        main_layout.addWidget(self.hint_label)

        # ---------- X-AXIS CONTROLS ----------
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X-Axis:"))
        self.x_var_combo = QComboBox()
        self.x_var_combo.addItems(["Filler (φ)", "Matrix (1-φ)"])
        self.x_var_combo.currentTextChanged.connect(self.update_plots_smart)
        x_layout.addWidget(self.x_var_combo)

        x_layout.addWidget(QLabel("Min:"))
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(0.0, 1.0)
        self.x_min_spin.setSingleStep(0.01)
        self.x_min_spin.setValue(0.0)
        self.x_min_spin.setDecimals(3)
        self.x_min_spin.valueChanged.connect(self.update_plots_smart)
        x_layout.addWidget(self.x_min_spin)

        x_layout.addWidget(QLabel("Max:"))
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(0.0, 1.0)
        self.x_max_spin.setSingleStep(0.01)
        self.x_max_spin.setValue(1.0)
        self.x_max_spin.setDecimals(3)
        self.x_max_spin.valueChanged.connect(self.update_plots_smart)
        x_layout.addWidget(self.x_max_spin)

        btn_reset = QPushButton("Reset to Data Range")
        btn_reset.setStyleSheet(
            "background-color: #7f8c8d; color: white; "
            "padding: 4px 12px; border-radius: 4px;"
        )
        btn_reset.clicked.connect(self.reset_x_range_smart)
        x_layout.addWidget(btn_reset)

        x_layout.addStretch()
        main_layout.addLayout(x_layout)

        # ---------- PROGRESS BAR ----------
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7; border-radius: 5px;
                text-align: center; height: 20px;
            }
            QProgressBar::chunk { background-color: #27ae60; border-radius: 5px; }
        """)
        main_layout.addWidget(self.progress_bar)

        # ---------- MAIN SPLITTER ----------
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        self.param_group = QGroupBox("📐 Parameters")
        self.param_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 12px; }"
        )
        param_layout = QVBoxLayout(self.param_group)
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(3)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Std. Error"])
        self.param_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.param_table.setAlternatingRowColors(True)
        param_layout.addWidget(self.param_table)
        left_layout.addWidget(self.param_group)

        self.stats_group = QGroupBox("📊 Fit Statistics")
        self.stats_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 12px; }"
        )
        stats_layout = QVBoxLayout(self.stats_group)
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.stats_table.setAlternatingRowColors(True)
        stats_layout.addWidget(self.stats_table)
        left_layout.addWidget(self.stats_group)

        self.quality_label = QLabel("Select a model and run fit")
        self.quality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quality_label.setStyleSheet("""
            font-size: 16px; font-weight: bold; padding: 15px;
            background-color: #ecf0f1; border-radius: 5px; color: #7f8c8d;
        """)
        left_layout.addWidget(self.quality_label)

        splitter.addWidget(left_panel)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_tabs = QTabWidget()
        self.plot_tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #bdc3c7; }"
        )

        self.main_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.plot_tabs.addTab(self.main_canvas, "Fit Plot")

        self.resid_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.plot_tabs.addTab(self.resid_canvas, "Residuals")

        self.sens_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.plot_tabs.addTab(self.sens_canvas, "Sensitivity")

        right_layout.addWidget(self.plot_tabs)
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 800])
        main_layout.addWidget(splitter)

        # ---------- BOTTOM BUTTONS ----------
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)

        self.compare_button = QPushButton("📊 Compare All Models")
        self.compare_button.setStyleSheet("""
            QPushButton {
                background-color: #1a5490; color: white;
                padding: 8px 16px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1f6cb0; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.compare_button.clicked.connect(self.compare_models)
        bottom_layout.addWidget(self.compare_button)

        self.save_graph_button = QPushButton("💾 Save Graph")
        self.save_graph_button.setStyleSheet("""
            QPushButton {
                background-color: #e67e22; color: white;
                padding: 8px 16px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #d35400; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.save_graph_button.clicked.connect(self.save_graph)
        bottom_layout.addWidget(self.save_graph_button)

        self.export_button = QPushButton("💾 Export Results")
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; color: white;
                padding: 8px 16px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.export_button.clicked.connect(self.export_results)
        bottom_layout.addWidget(self.export_button)

        bottom_layout.addStretch()

        self.close_button = QPushButton("✕ Close")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c; color: white;
                padding: 8px 20px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        self.close_button.clicked.connect(self.close)
        bottom_layout.addWidget(self.close_button)

        main_layout.addLayout(bottom_layout)

        self._update_method_info(0)

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _update_method_info(self, index):
        info = {
            0: "TRF: Most robust, handles bounds. Recommended.",
            1: "LM: Fastest but ignores bounds.",
            2: "Dogbox: Fast with bounds."
        }
        self.method_info_label.setText(info.get(index, ""))

    def _update_hint(self):
        """Update the recommended-model hint based on mode."""
        if self.mode == "nanothermite":
            self.hint_label.setText(
                "<b>🔥 Nanothermite Mode</b> — conductivity <b>decreases</b> "
                "with oxidizer fraction.<br>"
                "<b>Recommended models:</b> "
                "<span style='color:#27ae60; font-weight:bold;'>"
                "Maxwell-Eucken</span>, "
                "<span style='color:#27ae60; font-weight:bold;'>"
                "Cheng-Vachon</span>, "
                "<span style='color:#27ae60; font-weight:bold;'>"
                "Baschirow-Selenew</span>, "
                "Series, Bruggeman, Tsao.<br>"
                "<b>Avoid:</b> "
                "<span style='color:#c0392b; font-weight:bold;'>"
                "Parallel</span>, "
                "<span style='color:#c0392b; font-weight:bold;'>"
                "Lewis-Nielsen</span>, "
                "<span style='color:#c0392b; font-weight:bold;'>"
                "Maxwell-Eucken Upper</span>."
            )
        else:
            self.hint_label.setText(
                "<b>🧪 Polymer Composite Mode</b> — conductivity <b>increases</b> "
                "with filler fraction.<br>"
                "<b>Recommended models:</b> "
                "<span style='color:#27ae60; font-weight:bold;'>"
                "Agari</span>, "
                "<span style='color:#27ae60; font-weight:bold;'>"
                "Lewis-Nielsen</span>, "
                "<span style='color:#27ae60; font-weight:bold;'>"
                "Maxwell-Eucken</span>, "
                "Bruggeman, Percolation.<br>"
                "<b>Avoid:</b> "
                "<span style='color:#c0392b; font-weight:bold;'>"
                "Series</span>, "
                "<span style='color:#c0392b; font-weight:bold;'>"
                "Maxwell-Eucken Lower</span>."
            )

    def _load_models(self):
        registry = get_model_registry(self.mode)
        model_names = list(registry.keys())
        self.model_combo.addItems(model_names)
        for i, name in enumerate(model_names):
            meta = registry.get(name, {})
            desc = meta.get("description", "")
            params = meta.get("params", [])
            param_str = f"Params: {', '.join(params)}" if params else "Fixed model"
            tooltip = f"{name}\n\n{desc}\n\n{param_str}"
            self.model_combo.setItemData(i, tooltip, Qt.ItemDataRole.ToolTipRole)
        if model_names:
            self.model_combo.setCurrentIndex(0)
            self.current_model = model_names[0]
            self.update_parameter_table(model_names[0])

    def _get_k_matrix(self):
        parent = self.parent()
        if parent and hasattr(parent, "k_matrix"):
            return float(parent.k_matrix)
        return 237.0 if self.mode == "nanothermite" else 0.2

    def _get_k_filler(self):
        parent = self.parent()
        if parent and hasattr(parent, "k_filler"):
            return float(parent.k_filler)
        return 33.0 if self.mode == "nanothermite" else 300.0

    # ========================================================================
    # MODEL CHANGE
    # ========================================================================

    def on_model_changed(self, model_name):
        self.current_model = model_name
        self.clear_results()
        self.update_parameter_table(model_name)

    def update_parameter_table(self, model_name):
        registry = get_model_registry(self.mode)
        meta = registry.get(model_name, {})
        params = meta.get('params', [])
        self.param_table.setRowCount(0)

        if meta.get('fixed', True):
            self.param_table.setRowCount(1)
            self.param_table.setItem(0, 0, QTableWidgetItem("Fixed Model"))
            self.param_table.setItem(0, 1, QTableWidgetItem("No parameters"))
            self.param_table.setItem(0, 2, QTableWidgetItem("-"))
            self.run_button.setEnabled(False)
            self.optimize_button.setEnabled(False)
            self.sens_button.setEnabled(False)
        else:
            self.param_table.setRowCount(len(params))
            for i, param in enumerate(params):
                self.param_table.setItem(i, 0, QTableWidgetItem(param))
                self.param_table.setItem(i, 1, QTableWidgetItem("-"))
                self.param_table.setItem(i, 2, QTableWidgetItem("-"))
            self.run_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
            self.sens_button.setEnabled(True)

    def clear_results(self):
        self.fit_results = {}
        self.param_table.setRowCount(0)
        self.stats_table.setRowCount(0)
        self.quality_label.setText("Select a model and run fit")
        self.quality_label.setStyleSheet("""
            font-size: 16px; font-weight: bold; padding: 15px;
            background-color: #ecf0f1; border-radius: 5px; color: #7f8c8d;
        """)
        self.main_canvas.axes.clear()
        self.main_canvas.draw()
        self.resid_canvas.axes.clear()
        self.resid_canvas.draw()
        self.sens_canvas.axes.clear()
        self.sens_canvas.draw()

    def reset_x_range_smart(self):
        if self.df is not None:
            x_var = self.x_var_combo.currentText()
            x_col = "Filler_volume_fraction" if x_var == "Filler (φ)" else "Matrix_volume_fraction"
            if x_col in self.df.columns:
                xdata = self.df[x_col].values
                x_min = float(xdata.min())
                x_max = float(xdata.max())
                padding = (x_max - x_min) * 0.05
                self.x_min_spin.setValue(max(0, x_min - padding))
                self.x_max_spin.setValue(min(1, x_max + padding))
        else:
            self.x_min_spin.setValue(0.0)
            self.x_max_spin.setValue(1.0)
        self.update_plots_smart()

    def update_plots_smart(self):
        if self.df is not None and self.fit_results and self.current_model in self.fit_results:
            self._update_plots(self.fit_results[self.current_model])

    # ========================================================================
    # FIT WORKFLOW
    # ========================================================================

    def run_fit(self):
        if self.df is None or len(self.df) < 2:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return

        # ---- 1. SANITIZE DATA ----
        phi_raw = self.df["Filler_volume_fraction"].values.astype(float)
        k_raw = self.df["k_meas"].values.astype(float)

        mask = np.isfinite(phi_raw) & np.isfinite(k_raw)
        phi = phi_raw[mask]
        k = k_raw[mask]

        if len(phi) < 2:
            QMessageBox.critical(
                self, "Data Error",
                f"Not enough valid data points.\n\n"
                f"Original: {len(phi_raw)}\nValid: {len(phi)}"
            )
            return

        if len(phi) < len(phi_raw):
            removed = len(phi_raw) - len(phi)
            reply = QMessageBox.question(
                self, "Data Cleaned",
                f"{removed} invalid point(s) removed.\n"
                f"Proceed with {len(phi)} valid points?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # ---- 2. GET SAFE p0 / BOUNDS ----
        registry = get_model_registry(self.mode)
        meta = registry.get(self.current_model, {})

        k_matrix = self._get_k_matrix()
        k_filler = self._get_k_filler()

        if not (np.isfinite(k_matrix) and k_matrix > 0):
            k_matrix = 237.0 if self.mode == "nanothermite" else 0.2
        if not (np.isfinite(k_filler) and k_filler > 0):
            k_filler = 33.0 if self.mode == "nanothermite" else 300.0

        p0 = meta.get("p0", None)
        bounds = meta.get("bounds", None)

        if p0 is not None:
            p0 = [float(v) if np.isfinite(v) else 1.0 for v in p0]
        if bounds is not None:
            try:
                low, high = bounds
                low = [float(x) if np.isfinite(x) else -1e6 for x in low]
                high = [float(x) if np.isfinite(x) else 1e6 for x in high]
                bounds = (low, high)
            except Exception:
                bounds = None

        # Agari smart init
        if self.current_model == "Agari" and p0 is None:
            C1_guess = max(k[0] / max(k_matrix, 1e-6), 1e-3)
            p0 = [C1_guess, 1.0]
            bounds = ([1e-6, -10.0], [1e6, 10.0])

        if self.current_model == "Agari-Percolation" and p0 is None:
            p0 = [1.0, 1.0, max(float(k.mean()), 1.0), 0.1, 2.0]
            bounds = ([1e-6, -10.0, 1e-6, 1e-6, 0.1],
                      [1e6, 10.0, 1e6, 0.95, 10.0])

        # ---- 3. BUILD ENGINE + WORKER ----
        try:
            self.fit_engine = SmartFitEngine(
                phi, k, None,
                mode=self.mode,
                k_matrix=k_matrix,
                k_filler=k_filler
            )
        except Exception as e:
            QMessageBox.critical(self, "Engine Error", f"Failed to create fit engine:\n{e}")
            return

        self._set_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting fit...")

        method = self.method_combo.currentText()
        self.worker = FitWorkerThread(
            self.fit_engine,
            self.current_model,
            p0=p0,
            bounds=bounds,
            method=method,
            mode=self.mode,
            k_matrix=k_matrix,
            k_filler=k_filler,
        )
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._on_fit_finished)
        self.worker.error.connect(self._on_fit_error)
        self.worker.start()

    def _update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{message}")

    def _on_fit_finished(self, result):
        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)

        if not result.get('success', False):
            QMessageBox.critical(
                self, "Fit Failed",
                result.get('message', 'Unknown error')
            )
            return

        self.fit_results[self.current_model] = result

        # Update parameters table
        if not result.get('is_fixed', True):
            params = result.get('params', {})
            param_errors = result.get('param_errors', {})
            self.param_table.setRowCount(len(params))
            for i, (name, value) in enumerate(params.items()):
                self.param_table.setItem(i, 0, QTableWidgetItem(name))
                self.param_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
                error = param_errors.get(name, np.nan)
                self.param_table.setItem(
                    i, 2,
                    QTableWidgetItem(f"{error:.4f}" if np.isfinite(error) else "-")
                )

        # Update statistics table
        stats = result.get('stats', {})
        self.stats_table.setRowCount(0)
        metrics = [
            ('R²', stats.get('r2', 0), '{:.4f}'),
            ('RMSE (W/m·K)', stats.get('rmse', 0), '{:.4f}'),
            ('MAE (W/m·K)', stats.get('mae', 0), '{:.4f}'),
            ('MAPE (%)', stats.get('mape', 0), '{:.2f}'),
            ('Data Points', len(self.fit_engine.phi), '{:d}'),
        ]
        for i, (label, value, fmt) in enumerate(metrics):
            self.stats_table.insertRow(i)
            self.stats_table.setItem(i, 0, QTableWidgetItem(label))
            try:
                self.stats_table.setItem(i, 1, QTableWidgetItem(fmt.format(value)))
            except Exception:
                self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # Quality indicator
        r2 = stats.get('r2', 0)
        if r2 >= 0.95:
            quality, color, bg = "⭐ Excellent", "#27ae60", "#d4edda"
        elif r2 >= 0.85:
            quality, color, bg = "✓ Good", "#856404", "#fff3cd"
        elif r2 >= 0.70:
            quality, color, bg = "⚠ Fair", "#6c5200", "#ffeaa7"
        else:
            quality, color, bg = "✗ Poor", "#721c24", "#f8d7da"

        self.quality_label.setText(f"Fit Quality: {quality} (R² = {r2:.4f})")
        self.quality_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; padding: 15px; "
            f"background-color: {bg}; border-radius: 5px; color: {color};"
        )

        self._update_plots(result)

        QMessageBox.information(
            self, "Fit Complete",
            f"Model {self.current_model} fitted successfully!\n\n"
            f"R² = {r2:.4f}\n"
            f"RMSE = {stats.get('rmse', 0):.4f} W/m·K"
        )

    def _on_fit_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)

        friendly = error_msg
        if "not finite" in error_msg.lower():
            friendly = (
                "The fit failed because the initial parameter guess produced "
                "infinite or NaN values.\n\n"
                "This usually means:\n"
                "• The selected model is not compatible with your data trend\n"
                "• Material constants (k_matrix, k_filler) are wrong\n"
                "• The model has a singularity at low φ\n\n"
                "Try:\n"
                "• Selecting a different model\n"
                "• Using Auto-Optimize instead\n"
                "• Checking your data for outliers"
            )

        QMessageBox.critical(self, "Fit Failed", friendly)

    def _set_buttons_enabled(self, enabled):
        self.run_button.setEnabled(enabled)
        self.optimize_button.setEnabled(enabled)
        self.sens_button.setEnabled(enabled)
        self.compare_button.setEnabled(enabled)
        self.save_graph_button.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.method_combo.setEnabled(enabled)

    # ========================================================================
    # PLOTS
    # ========================================================================

    def _update_plots(self, result):
        if 'y_pred' not in result:
            return

        x_var = self.x_var_combo.currentText()
        if x_var == "Filler (φ)":
            x_col = "Filler_volume_fraction"
            x_label = "Filler Volume Fraction, φ"
        else:
            x_col = "Matrix_volume_fraction"
            x_label = "Matrix Volume Fraction, (1-φ)"

        x_min = self.x_min_spin.value()
        x_max = self.x_max_spin.value()

        # ---- MAIN PLOT ----
        ax = self.main_canvas.axes
        ax.clear()

        xdata = self.df[x_col].values
        k_meas = self.df["k_meas"].values
        k_pred = result['y_pred']
        k_err = self.df.get("k_std_dev", None)

        mask = (xdata >= x_min) & (xdata <= x_max)
        xdata_filtered = xdata[mask]
        k_meas_filtered = k_meas[mask]
        k_err_filtered = k_err[mask] if k_err is not None else None

        if k_err_filtered is not None and not np.all(np.isnan(k_err_filtered)):
            ax.errorbar(xdata_filtered, k_meas_filtered, yerr=k_err_filtered,
                        fmt='o', color='black', markersize=8, capsize=4,
                        label='Experimental', zorder=5)
        else:
            ax.scatter(xdata_filtered, k_meas_filtered, color='black',
                       s=60, label='Experimental', zorder=5)

        x_smooth = np.linspace(x_min, x_max, 200)
        phi_smooth = x_smooth if x_col == "Filler_volume_fraction" else (1.0 - x_smooth)

        registry = get_model_registry(self.mode)
        meta = registry[self.current_model]
        model_func = meta['func']

        try:
            if result.get('popt') is not None:
                y_smooth = model_func(
                    phi_smooth, *result['popt'],
                    mode=self.mode,
                    k_matrix=self._get_k_matrix(),
                    k_filler=self._get_k_filler()
                )
            else:
                y_smooth = model_func(
                    phi_smooth,
                    mode=self.mode,
                    k_matrix=self._get_k_matrix(),
                    k_filler=self._get_k_filler()
                )
            ax.plot(x_smooth, y_smooth, color='#1a5490',
                    linewidth=2.5, label='Predicted')
        except Exception as e:
            print(f"[SmartFit] plotting error: {e}")

        ax.set_xlabel(x_label)
        ax.set_ylabel('Thermal Conductivity, k (W/m·K)')
        material = getattr(self.parent(), 'material_info', 'Composite') if self.parent() else 'Composite'
        ax.set_title(f'{material} - {self.current_model} Fit')
        ax.set_xlim(x_min, x_max)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        self.main_canvas.draw()

        # ---- RESIDUALS ----
        ax = self.resid_canvas.axes
        ax.clear()
        residuals = k_meas - k_pred
        residuals_filtered = residuals[mask]
        ax.scatter(xdata_filtered, residuals_filtered, color='blue', s=50, alpha=0.7)
        ax.axhline(0, color='black', linewidth=1.5, linestyle='--')
        if len(residuals_filtered) > 0:
            mean_r = float(np.mean(residuals_filtered))
            std_r = float(np.std(residuals_filtered))
            ax.text(0.02, 0.98, f'Mean: {mean_r:.3f}\nStd: {std_r:.3f}',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(x_label)
        ax.set_ylabel('Residuals (W/m·K)')
        ax.set_title('Residuals')
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3)
        self.resid_canvas.draw()

    # ========================================================================
    # AUTO-OPTIMIZE / SENSITIVITY
    # ========================================================================

    def auto_optimize(self):
        if self.df is None or len(self.df) < 2:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return

        registry = get_model_registry(self.mode)
        meta = registry.get(self.current_model, {})
        if meta.get('fixed', True):
            QMessageBox.information(self, "Info", f"{self.current_model} is a fixed model.")
            return

        phi_raw = self.df["Filler_volume_fraction"].values.astype(float)
        k_raw = self.df["k_meas"].values.astype(float)
        mask = np.isfinite(phi_raw) & np.isfinite(k_raw)
        phi = phi_raw[mask]
        k = k_raw[mask]

        self.fit_engine = SmartFitEngine(
            phi, k, None,
            mode=self.mode,
            k_matrix=self._get_k_matrix(),
            k_filler=self._get_k_filler()
        )

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Optimizing parameters...")
        self._set_buttons_enabled(False)

        self.optimize_timer = QTimer()
        self.optimize_timer.setSingleShot(True)
        self.optimize_timer.timeout.connect(self._run_optimization)
        self.optimize_timer.start(100)

    def _run_optimization(self):
        try:
            result = self.fit_engine.optimize_parameters(self.current_model)
            self._on_optimize_complete(result)
        except Exception as e:
            self.progress_bar.setVisible(False)
            self._set_buttons_enabled(True)
            QMessageBox.critical(self, "Optimization Error", f"An error occurred:\n{e}")

    def _on_optimize_complete(self, result):
        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)

        if not result.get('success', False):
            QMessageBox.critical(self, "Optimization Failed",
                                 result.get('message', 'Unknown error'))
            return

        params = result.get('params', {})
        self.param_table.setRowCount(len(params))
        for i, (name, value) in enumerate(params.items()):
            self.param_table.setItem(i, 0, QTableWidgetItem(name))
            self.param_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
            self.param_table.setItem(i, 2, QTableWidgetItem("(optimized)"))

        QMessageBox.information(
            self, "Optimization Complete",
            f"Optimization completed in {result.get('iterations', 0)} iterations.\n"
            f"Objective value (RMSE): {result.get('objective', 0):.4f}\n\n"
            f"Optimized parameters have been set. Click 'Run Fit' to use them."
        )

    def analyze_sensitivity(self):
        if self.df is None or len(self.df) < 2:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return

        registry = get_model_registry(self.mode)
        meta = registry.get(self.current_model, {})
        if meta.get('fixed', True):
            QMessageBox.information(self, "Info", f"{self.current_model} is a fixed model.")
            return

        phi_raw = self.df["Filler_volume_fraction"].values.astype(float)
        k_raw = self.df["k_meas"].values.astype(float)
        mask = np.isfinite(phi_raw) & np.isfinite(k_raw)
        phi = phi_raw[mask]
        k = k_raw[mask]

        self.fit_engine = SmartFitEngine(
            phi, k, None,
            mode=self.mode,
            k_matrix=self._get_k_matrix(),
            k_filler=self._get_k_filler()
        )

        fit_result = self.fit_engine.fit_model(self.current_model)
        if not fit_result.get('success', False):
            QMessageBox.warning(self, "Error", "Please run a fit first.")
            return

        sensitivity = self.fit_engine.parameter_sensitivity(self.current_model)
        if not sensitivity:
            QMessageBox.warning(self, "Error", "Failed to run sensitivity analysis.")
            return

        self._plot_sensitivity(sensitivity)
        QMessageBox.information(
            self, "Sensitivity Analysis",
            "Sensitivity analysis completed.\n"
            "The Sensitivity tab shows how RMSE changes with parameter variations."
        )

    def _plot_sensitivity(self, sensitivity):
        ax = self.sens_canvas.axes
        ax.clear()
        colors = ['#1a5490', '#e74c3c', '#27ae60', '#f39c12',
                  '#8e44ad', '#1abc9c', '#2c3e50', '#e67e22']
        has_data = False

        for idx, (param_name, data) in enumerate(sensitivity.items()):
            if param_name == 'message':
                continue
            variations = data.get('variations', [])
            if not variations:
                continue
            has_data = True
            factors = [v['factor'] for v in variations]
            rmse_values = [v['rmse'] for v in variations]
            color = colors[idx % len(colors)]
            ax.plot(factors, rmse_values, marker='o', color=color,
                    linewidth=2, markersize=8, label=param_name)
            min_idx = int(np.argmin(rmse_values))
            ax.plot(factors[min_idx], rmse_values[min_idx], 's',
                    color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=1)

        if not has_data:
            ax.text(0.5, 0.5, 'No sensitivity data',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='#7f8c8d')
        else:
            ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='Optimal')
            ax.legend(loc='best')

        ax.set_xlabel('Parameter Variation Factor')
        ax.set_ylabel('RMSE (W/m·K)')
        ax.set_title('Parameter Sensitivity Analysis')
        ax.grid(True, alpha=0.3)
        self.sens_canvas.draw()
        self.plot_tabs.setCurrentIndex(2)

    # ========================================================================
    # COMPARE / EXPORT
    # ========================================================================

    def compare_models(self):
        if not self.fit_results:
            QMessageBox.warning(self, "Warning", "No fit results to compare.")
            return
        try:
            from .model_comparison_widget import ModelComparisonDialog
            dialog = ModelComparisonDialog(self.fit_results, self.parent())
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open comparison:\n{e}")

    def save_graph(self):
        if not self.fit_results or self.current_model not in self.fit_results:
            QMessageBox.warning(self, "Warning", "Please run a fit first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", f"smart_fit_{self.current_model}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tiff)"
        )
        if not path:
            return
        try:
            self.main_canvas.fig.savefig(path, dpi=300, bbox_inches="tight")
            QMessageBox.information(self, "Success", f"Graph saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save graph:\n{e}")

    def export_results(self):
        if not self.fit_results:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "fit_results.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if not path:
            return
        try:
            data = []
            for model_name, result in self.fit_results.items():
                row = {
                    'Model': model_name,
                    'Success': result.get('success', False),
                    'Message': result.get('message', '')
                }
                stats = result.get('stats', {})
                for key, value in stats.items():
                    row[f'Stat_{key}'] = value
                params = result.get('params', {})
                for key, value in params.items():
                    row[f'Param_{key}'] = value
                data.append(row)

            df = pd.DataFrame(data)
            if path.endswith('.csv'):
                df.to_csv(path, index=False)
            else:
                df.to_excel(path, index=False)
            QMessageBox.information(self, "Success", f"Results exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")
