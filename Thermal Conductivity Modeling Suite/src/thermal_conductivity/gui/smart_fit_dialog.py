"""Smart Fit Dialog - Interactive model fitting interface."""

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

import numpy as np
import pandas as pd
import traceback

from ..core.models import get_model_registry
from ..core.statistics import calculate_statistics
from ..smart_fit.fit_engine import SmartFitEngine
from ..visualization.plotter import MplCanvas


class FitWorkerThread(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    
    def __init__(self, engine, model_name, p0=None, bounds=None, method='trf'):
        super().__init__()
        self.engine = engine
        self.model_name = model_name
        self.p0 = p0
        self.bounds = bounds
        self.method = method
    
    def run(self):
        try:
            self.progress.emit(0, f"Fitting {self.model_name}...")
            result = self.engine.fit_model(self.model_name, self.p0, self.bounds, self.method)
            self.progress.emit(100, "Fitting completed")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())


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
        
        self.setWindowTitle("Smart Fit - Advanced Model Fitting")
        self.setMinimumSize(1200, 800)
        self.setModal(False)
        
        self._build_ui()
        self._load_models()
    
    def _build_ui(self):
        """Build the dialog UI with tooltips."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # ===== TOP CONTROLS =====
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)
        
        # Model selection
        top_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.setToolTip("Select the model to fit to the experimental data")
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        top_layout.addWidget(self.model_combo)
        
        # Method selection
        top_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(['trf', 'lm', 'dogbox'])
        self.method_combo.setMinimumWidth(100)
        self.method_combo.setToolTip(
            "Optimization algorithm for fitting:\n\n"
            "• TRF (Trust Region Reflective): Most robust, handles bounds. Recommended for most cases.\n"
            "• LM (Levenberg-Marquardt): Fastest, but ignores bounds. Use with good initial guesses.\n"
            "• Dogbox: Fast with bounds. Good alternative to TRF when speed is needed."
        )
        self.method_combo.setItemData(0, "TRF (Trust Region Reflective)\n\n" "Best for: Problems with parameter bounds\n" "Pros: Most robust, handles bounds well\n" "Cons: Slower than LM\n" "Recommended: Default choice for most cases", Qt.ItemDataRole.ToolTipRole)
        self.method_combo.setItemData(1, "LM (Levenberg-Marquardt)\n\n" "Best for: Unbounded problems\n" "Pros: Fastest convergence\n" "Cons: Cannot handle bounds, sensitive to initial guesses\n" "Recommended: When you have good initial guesses and no bounds", Qt.ItemDataRole.ToolTipRole)
        self.method_combo.setItemData(2, "Dogbox (Dogleg with Box Constraints)\n\n" "Best for: Bounded problems where speed is needed\n" "Pros: Fast with bounds\n" "Cons: Less robust than TRF for complex models\n" "Recommended: When TRF is too slow with bounds", Qt.ItemDataRole.ToolTipRole)
        top_layout.addWidget(self.method_combo)
        
        # Method info label
        self.method_info_label = QLabel("")
        self.method_info_label.setStyleSheet("color: #7f8c8d; font-size: 10px; font-style: italic;")
        self.method_info_label.setWordWrap(True)
        self.method_info_label.setMaximumWidth(350)
        top_layout.addWidget(self.method_info_label)
        self.method_combo.currentIndexChanged.connect(self._update_method_info)
        
        # Run button
        self.run_button = QPushButton("▶ Run Fit")
        self.run_button.setToolTip("Run the fitting algorithm with the selected model and method.")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; 
                color: white; 
                padding: 8px 20px; 
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.run_button.clicked.connect(self.run_fit)
        top_layout.addWidget(self.run_button)
        
        # Optimize button
        self.optimize_button = QPushButton("⚡ Auto-Optimize")
        self.optimize_button.setToolTip("Automatically find optimal parameter values using global optimization.")
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #f39c12; 
                color: white; 
                padding: 8px 20px; 
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #f1c40f; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.optimize_button.clicked.connect(self.auto_optimize)
        top_layout.addWidget(self.optimize_button)
        
        # Sensitivity button
        self.sens_button = QPushButton("📊 Sensitivity")
        self.sens_button.setToolTip("Analyze how parameter variations affect the fit quality.")
        self.sens_button.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad; 
                color: white; 
                padding: 8px 20px; 
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #9b59b6; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.sens_button.clicked.connect(self.analyze_sensitivity)
        top_layout.addWidget(self.sens_button)
        
        top_layout.addStretch()
        main_layout.addLayout(top_layout)
        
        # ===== X-AXIS CONTROLS =====
        x_controls_layout = QHBoxLayout()
        x_controls_layout.addWidget(QLabel("X-Axis:"))
        self.x_var_combo = QComboBox()
        self.x_var_combo.addItems(["Filler (φ)", "Matrix (1-φ)"])
        self.x_var_combo.setToolTip("Select which variable to plot on the X-axis")
        self.x_var_combo.currentTextChanged.connect(self.update_plots_smart)
        x_controls_layout.addWidget(self.x_var_combo)
        
        x_controls_layout.addWidget(QLabel("Min:"))
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(0.0, 1.0)
        self.x_min_spin.setSingleStep(0.01)
        self.x_min_spin.setValue(0.0)
        self.x_min_spin.setDecimals(3)
        self.x_min_spin.setToolTip("Minimum X-axis value")
        self.x_min_spin.valueChanged.connect(self.update_plots_smart)
        x_controls_layout.addWidget(self.x_min_spin)
        
        x_controls_layout.addWidget(QLabel("Max:"))
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(0.0, 1.0)
        self.x_max_spin.setSingleStep(0.01)
        self.x_max_spin.setValue(1.0)
        self.x_max_spin.setDecimals(3)
        self.x_max_spin.setToolTip("Maximum X-axis value")
        self.x_max_spin.valueChanged.connect(self.update_plots_smart)
        x_controls_layout.addWidget(self.x_max_spin)
        
        btn_reset = QPushButton("Reset to Data Range")
        btn_reset.setToolTip("Reset X-axis to the range of the experimental data")
        btn_reset.setStyleSheet("background-color: #7f8c8d; color: white; padding: 4px 12px; border-radius: 4px;")
        btn_reset.clicked.connect(self.reset_x_range_smart)
        x_controls_layout.addWidget(btn_reset)
        
        x_controls_layout.addStretch()
        main_layout.addLayout(x_controls_layout)
        
        # ===== PROGRESS BAR =====
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
        main_layout.addWidget(self.progress_bar)
        
        # ===== MAIN CONTENT SPLITTER =====
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Parameters and Statistics
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        self.param_group = QGroupBox("📐 Parameters")
        self.param_group.setToolTip("Fitted parameter values and their standard errors")
        self.param_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        param_layout = QVBoxLayout(self.param_group)
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(3)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Std. Error"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.param_table.setAlternatingRowColors(True)
        param_layout.addWidget(self.param_table)
        left_layout.addWidget(self.param_group)
        
        self.stats_group = QGroupBox("📊 Fit Statistics")
        self.stats_group.setToolTip("Statistical metrics to evaluate the quality of the fit")
        self.stats_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        stats_layout = QVBoxLayout(self.stats_group)
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.setAlternatingRowColors(True)
        stats_layout.addWidget(self.stats_table)
        left_layout.addWidget(self.stats_group)
        
        self.quality_label = QLabel("Select a model and run fit")
        self.quality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quality_label.setToolTip("Visual indicator of the fit quality based on R² value")
        self.quality_label.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            padding: 15px; 
            background-color: #ecf0f1; 
            border-radius: 5px;
            color: #7f8c8d;
        """)
        left_layout.addWidget(self.quality_label)
        
        splitter.addWidget(left_panel)
        
        # Right panel: Plots
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setToolTip("Switch between different plots: Fit, Residuals, and Sensitivity")
        self.plot_tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #bdc3c7; }")
        
        self.main_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.main_canvas.setToolTip("Main fit plot showing experimental data and fitted model curve")
        self.plot_tabs.addTab(self.main_canvas, "Fit Plot")
        
        self.resid_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.resid_canvas.setToolTip("Residuals plot showing the difference between experimental and predicted values")
        self.plot_tabs.addTab(self.resid_canvas, "Residuals")
        
        self.sens_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.sens_canvas.setToolTip("Parameter sensitivity analysis showing how RMSE changes with parameter variations")
        self.plot_tabs.addTab(self.sens_canvas, "Sensitivity")
        
        right_layout.addWidget(self.plot_tabs)
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 800])
        main_layout.addWidget(splitter)
        
        # ===== BOTTOM BUTTONS =====
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)
        
        self.compare_button = QPushButton("📊 Compare All Models")
        self.compare_button.setToolTip("Open a dashboard comparing all fitted models")
        self.compare_button.setStyleSheet("""
            QPushButton {
                background-color: #1a5490; 
                color: white; 
                padding: 8px 16px; 
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1f6cb0; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.compare_button.clicked.connect(self.compare_models)
        bottom_layout.addWidget(self.compare_button)
        
        self.save_graph_button = QPushButton("💾 Save Graph")
        self.save_graph_button.setToolTip("Save the current plot as an image file (PNG, PDF, SVG, TIFF)")
        self.save_graph_button.setStyleSheet("""
            QPushButton {
                background-color: #e67e22; 
                color: white; 
                padding: 8px 16px; 
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #d35400; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.save_graph_button.clicked.connect(self.save_graph)
        bottom_layout.addWidget(self.save_graph_button)
        
        self.export_button = QPushButton("💾 Export Results")
        self.export_button.setToolTip("Export all fit results to CSV or Excel file")
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; 
                color: white; 
                padding: 8px 16px; 
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.export_button.clicked.connect(self.export_results)
        bottom_layout.addWidget(self.export_button)
        
        bottom_layout.addStretch()
        
        self.close_button = QPushButton("✕ Close")
        self.close_button.setToolTip("Close the Smart Fit dialog")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c; 
                color: white; 
                padding: 8px 20px; 
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        self.close_button.clicked.connect(self.close)
        bottom_layout.addWidget(self.close_button)
        
        main_layout.addLayout(bottom_layout)
        
        self._update_method_info(0)
    
    def _update_method_info(self, index):
        info = {
            0: "TRF: Most robust, handles bounds. Recommended for most cases.",
            1: "LM: Fastest but ignores bounds. Use with good initial guesses.",
            2: "Dogbox: Fast with bounds. Good when TRF is too slow."
        }
        self.method_info_label.setText(info.get(index, ""))
    
    def _load_models(self):
        """Populate model dropdown with tooltips."""
        registry = get_model_registry(self.mode)
        model_names = list(registry.keys())
        self.model_combo.addItems(model_names)
        for i, name in enumerate(model_names):
            meta = registry.get(name, {})
            desc = meta.get("description", "No description available")
            params = meta.get("params", [])
            param_str = f"Parameters: {', '.join(params)}" if params else "Fixed model - no parameters"
            fixed = "Fixed model" if meta.get("fixed", True) else "Fitted model"
            tooltip = f"{name}\n\n{desc}\n\n{param_str}\n\nType: {fixed}"
            self.model_combo.setItemData(i, tooltip, Qt.ItemDataRole.ToolTipRole)
        if model_names:
            self.model_combo.setCurrentIndex(0)
            self.current_model = model_names[0]
            self.update_parameter_table(model_names[0])
    
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
            font-size: 16px; 
            font-weight: bold; 
            padding: 15px; 
            background-color: #ecf0f1; 
            border-radius: 5px;
            color: #7f8c8d;
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
            if x_var == "Filler (φ)":
                x_col = "Filler_volume_fraction"
            else:
                x_col = "Matrix_volume_fraction"
            xdata = self.df[x_col].values
            x_min = xdata.min()
            x_max = xdata.max()
            padding = (x_max - x_min) * 0.05
            self.x_min_spin.setValue(max(0, x_min - padding))
            self.x_max_spin.setValue(min(1, x_max + padding))
        else:
            self.x_min_spin.setValue(0.0)
            self.x_max_spin.setValue(1.0)
        self.update_plots_smart()
    
    def update_plots_smart(self):
        if self.df is not None and self.fit_results:
            self._update_plots(self.fit_results[self.current_model])
    
    def run_fit(self):
        if self.df is None or len(self.df) < 2:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
        
        # Use filler fraction as phi
        phi = self.df["Filler_volume_fraction"].values
        k = self.df["k_meas"].values
        k_std = self.df.get("k_std_dev", None)
        
        self.fit_engine = SmartFitEngine(phi, k, k_std)
        
        self._set_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting fit...")
        
        self.worker = FitWorkerThread(
            self.fit_engine,
            self.current_model,
            method=self.method_combo.currentText()
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
        
        if not result['success']:
            QMessageBox.critical(self, "Fit Failed", result.get('message', 'Unknown error'))
            return
        
        self.fit_results[self.current_model] = result
        
        # Update parameter table
        if not result.get('is_fixed', True):
            params = result.get('params', {})
            param_errors = result.get('param_errors', {})
            self.param_table.setRowCount(len(params))
            for i, (name, value) in enumerate(params.items()):
                self.param_table.setItem(i, 0, QTableWidgetItem(name))
                self.param_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
                error = param_errors.get(name, np.nan)
                self.param_table.setItem(i, 2, QTableWidgetItem(f"{error:.4f}" if not np.isnan(error) else "-"))
        
        # Update statistics table
        stats = result.get('stats', {})
        self.stats_table.setRowCount(0)
        metrics = [
            ('R²', stats.get('r2', 0), '{:.4f}'),
            ('RMSE (W/m·K)', stats.get('rmse', 0), '{:.4f}'),
            ('MAE (W/m·K)', stats.get('mae', 0), '{:.4f}'),
            ('MAPE (%)', stats.get('mape', 0), '{:.2f}'),
            ('Data Points', len(self.df), '{:d}'),
        ]
        for i, (label, value, fmt) in enumerate(metrics):
            self.stats_table.insertRow(i)
            self.stats_table.setItem(i, 0, QTableWidgetItem(label))
            try:
                self.stats_table.setItem(i, 1, QTableWidgetItem(fmt.format(value)))
            except:
                self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))
        
        # Update quality indicator
        r2 = stats.get('r2', 0)
        if r2 >= 0.95:
            quality = "⭐ Excellent"
            color = "#27ae60"
            bg = "#d4edda"
        elif r2 >= 0.85:
            quality = "✓ Good"
            color = "#856404"
            bg = "#fff3cd"
        elif r2 >= 0.70:
            quality = "⚠ Fair"
            color = "#6c5200"
            bg = "#ffeaa7"
        else:
            quality = "✗ Poor"
            color = "#721c24"
            bg = "#f8d7da"
        
        self.quality_label.setText(f"Fit Quality: {quality} (R² = {r2:.4f})")
        self.quality_label.setStyleSheet(f"""
            font-size: 16px; 
            font-weight: bold; 
            padding: 15px; 
            background-color: {bg}; 
            border-radius: 5px;
            color: {color};
        """)
        
        self._update_plots(result)
        QMessageBox.information(self, "Fit Complete", 
            f"Model {self.current_model} fitted successfully!\n"
            f"R² = {r2:.4f}\n"
            f"RMSE = {stats.get('rmse', 0):.4f} W/m·K")
    
    def _on_fit_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)
        QMessageBox.critical(self, "Fit Error", f"An error occurred:\n\n{error_msg}")
    
    def _set_buttons_enabled(self, enabled):
        self.run_button.setEnabled(enabled)
        self.optimize_button.setEnabled(enabled)
        self.sens_button.setEnabled(enabled)
        self.compare_button.setEnabled(enabled)
        self.save_graph_button.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.method_combo.setEnabled(enabled)
    
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
        
        # Main plot
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
            ax.errorbar(xdata_filtered, k_meas_filtered, yerr=k_err_filtered, fmt='o', color='black', 
                       markersize=8, capsize=4, label='Experimental', zorder=5)
        else:
            ax.scatter(xdata_filtered, k_meas_filtered, color='black', s=60, label='Experimental', zorder=5)
        
        x_smooth = np.linspace(x_min, x_max, 200)
        if x_col == "Filler_volume_fraction":
            phi_oxy_smooth = x_smooth
        else:
            phi_oxy_smooth = 1.0 - x_smooth
        
        registry = get_model_registry(self.mode)
        meta = registry[self.current_model]
        if result.get('popt') is not None:
            y_smooth = meta['func'](phi_oxy_smooth, *result['popt'])
        else:
            y_smooth = meta['func'](phi_oxy_smooth)
        
        ax.plot(x_smooth, y_smooth, color='#1a5490', linewidth=2.5, label='Predicted')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Thermal Conductivity, k (W/m·K)')
        material = getattr(self.parent(), 'material_info', 'Composite') if self.parent() else 'Composite'
        ax.set_title(f'{material} - {self.current_model} Fit')
        ax.set_xlim(x_min, x_max)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        self.main_canvas.draw()
        
        # Residuals plot
        ax = self.resid_canvas.axes
        ax.clear()
        residuals = k_meas - k_pred
        residuals_filtered = residuals[mask]
        ax.scatter(xdata_filtered, residuals_filtered, color='blue', s=50, alpha=0.7)
        ax.axhline(0, color='black', linewidth=1.5, linestyle='--')
        mean_resid = np.mean(residuals_filtered) if len(residuals_filtered) > 0 else 0
        std_resid = np.std(residuals_filtered) if len(residuals_filtered) > 0 else 0
        ax.text(0.02, 0.98, f'Mean: {mean_resid:.2f}\nStd: {std_resid:.2f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(x_label)
        ax.set_ylabel('Residuals (W/m·K)')
        ax.set_title('Residuals')
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3)
        self.resid_canvas.draw()
    
    def auto_optimize(self):
        if self.df is None or len(self.df) < 2:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
        
        registry = get_model_registry(self.mode)
        meta = registry.get(self.current_model, {})
        if meta.get('fixed', True):
            QMessageBox.information(self, "Info", f"{self.current_model} is a fixed model.\nNo optimization needed.")
            return
        
        if self.current_model not in self.fit_results or not self.fit_results[self.current_model].get('success', False):
            QMessageBox.warning(self, "Warning", "Please run a fit first before optimizing.")
            return
        
        phi = self.df["Filler_volume_fraction"].values
        k = self.df["k_meas"].values
        self.fit_engine = SmartFitEngine(phi, k)
        
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
            QMessageBox.critical(self, "Optimization Error", f"An error occurred:\n{str(e)}")
    
    def _on_optimize_complete(self, result):
        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)
        if not result.get('success', False):
            QMessageBox.critical(self, "Optimization Failed", result.get('message', 'Unknown error'))
            return
        params = result.get('params', {})
        self.param_table.setRowCount(len(params))
        for i, (name, value) in enumerate(params.items()):
            self.param_table.setItem(i, 0, QTableWidgetItem(name))
            self.param_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
            self.param_table.setItem(i, 2, QTableWidgetItem("(optimized)"))
        QMessageBox.information(self, "Optimization Complete", 
            f"Optimization completed in {result.get('iterations', 0)} iterations.\n"
            f"Objective value (RMSE): {result.get('objective', 0):.4f}\n\n"
            f"Optimized parameters have been set as initial guesses.\n"
            f"Click 'Run Fit' again to use the optimized parameters.")
    
    def analyze_sensitivity(self):
        if self.df is None or len(self.df) < 2:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
        
        registry = get_model_registry(self.mode)
        meta = registry.get(self.current_model, {})
        if meta.get('fixed', True):
            QMessageBox.information(self, "Info", f"{self.current_model} is a fixed model.\nNo sensitivity analysis available.")
            return
        
        phi = self.df["Filler_volume_fraction"].values
        k = self.df["k_meas"].values
        self.fit_engine = SmartFitEngine(phi, k)
        
        fit_result = self.fit_engine.fit_model(self.current_model)
        if not fit_result.get('success', False):
            QMessageBox.warning(self, "Error", "Please run a fit first before sensitivity analysis.")
            return
        
        sensitivity = self.fit_engine.parameter_sensitivity(self.current_model)
        if not sensitivity:
            QMessageBox.warning(self, "Error", "Failed to run sensitivity analysis.")
            return
        
        self._plot_sensitivity(sensitivity)
        param_names = list(sensitivity.keys())
        QMessageBox.information(self, "Sensitivity Analysis", 
            f"Sensitivity analysis completed for parameters: {', '.join(param_names)}\n"
            "The Sensitivity tab shows how RMSE changes when each parameter varies.\n"
            "A steeper slope means the model is more sensitive to that parameter.")
    
    def _plot_sensitivity(self, sensitivity):
        ax = self.sens_canvas.axes
        ax.clear()
        colors = ['#1a5490', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad', '#1abc9c', '#2c3e50', '#e67e22']
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
            min_idx = np.argmin(rmse_values)
            ax.plot(factors[min_idx], rmse_values[min_idx], 's', color=color, 
                   markersize=10, markeredgecolor='black', markeredgewidth=1)
        if not has_data:
            ax.text(0.5, 0.5, 'No sensitivity data available\nRun a fit first', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='#7f8c8d')
            ax.set_xlabel('Parameter Variation Factor')
            ax.set_ylabel('RMSE (W/m·K)')
            ax.set_title('Parameter Sensitivity Analysis')
            ax.grid(True, alpha=0.3)
            self.sens_canvas.draw()
            self.plot_tabs.setCurrentIndex(2)
            return
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='Optimal')
        ax.set_xlabel('Parameter Variation Factor')
        ax.set_ylabel('RMSE (W/m·K)')
        ax.set_title('Parameter Sensitivity Analysis')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        self.sens_canvas.draw()
        self.plot_tabs.setCurrentIndex(2)
    
    def compare_models(self):
        if not self.fit_results:
            QMessageBox.warning(self, "Warning", "No fit results to compare.\nRun fits for models first.")
            return
        try:
            from .model_comparison_widget import ModelComparisonDialog
            dialog = ModelComparisonDialog(self.fit_results, self.parent())
            dialog.exec()
        except ImportError as e:
            QMessageBox.critical(self, "Feature Unavailable", 
                f"The Model Comparison feature is not available.\n\n"
                f"Error: {str(e)}\n\n"
                f"Please ensure all Smart Fit files are installed correctly.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open comparison:\n{str(e)}")
    
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
            QMessageBox.critical(self, "Error", f"Failed to save graph:\n{str(e)}")
    
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
            QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")
