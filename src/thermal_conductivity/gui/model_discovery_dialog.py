"""
Model Discovery Dialog - Auto-discover new models.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QGroupBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox,
    QProgressBar, QLineEdit, QComboBox,
    QTabWidget, QSplitter, QCheckBox, QFileDialog,
    QTextEdit, QScrollArea, QApplication, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QClipboard

import numpy as np
import re
import sys
import os

from ..core.model_discovery import ModelDiscovery
from ..core.models import MODEL_REGISTRY, MODEL_CATEGORIES
from ..visualization.plotter import MplCanvas


class EquationPopup(QDialog):
    """Popup dialog to show full equation with copy button."""
    
    def __init__(self, model_name, equation, r2, rmse, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Full Equation - {model_name}")
        self.setMinimumSize(700, 450)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        name_label = QLabel(f"📐 {model_name}")
        name_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1a5490;")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name_label)
        
        eq_display = QTextEdit()
        clean_eq = self._clean_equation(equation)
        eq_display.setPlainText(f"k = {clean_eq}")
        eq_display.setReadOnly(True)
        eq_display.setFont(QFont("Courier New", 14))
        eq_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 2px solid #1a5490;
                border-radius: 8px;
                padding: 20px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                color: #2c3e50;
            }
        """)
        eq_display.setMinimumHeight(120)
        layout.addWidget(eq_display)
        
        stats_layout = QHBoxLayout()
        r2_label = QLabel(f"R² = {r2:.4f}")
        r2_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1a5490;")
        stats_layout.addWidget(r2_label)
        stats_layout.addSpacing(30)
        rmse_label = QLabel(f"RMSE = {rmse:.2f} W/m·K")
        rmse_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        stats_layout.addWidget(rmse_label)
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        if r2 >= 0.95:
            quality = "⭐ Excellent Fit"
            color = "#27ae60"
            bg = "#d4edda"
        elif r2 >= 0.85:
            quality = "✓ Good Fit"
            color = "#856404"
            bg = "#fff3cd"
        elif r2 >= 0.70:
            quality = "⚠️ Fair Fit"
            color = "#6c5200"
            bg = "#ffeaa7"
        else:
            quality = "✗ Poor Fit"
            color = "#721c24"
            bg = "#f8d7da"
        
        quality_label = QLabel(f"Quality: {quality}")
        quality_label.setStyleSheet(f"font-size: 13px; color: {color}; font-weight: bold; padding: 8px; background-color: {bg}; border-radius: 5px;")
        quality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(quality_label)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        copy_btn = QPushButton("📋 Copy Equation")
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px 25px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        copy_btn.clicked.connect(lambda: self._copy_equation(clean_eq))
        btn_layout.addWidget(copy_btn)
        close_btn = QPushButton("✕ Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 10px 25px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
    
    def _clean_equation(self, equation):
        try:
            eq = equation
            eq = eq.replace('6xp(', 'exp(')
            eq = eq.replace('6xp', 'exp')
            eq = eq.replace('xp(', 'exp(')
            eq = eq.replace('xp', 'exp')
            while 'ee' in eq:
                eq = eq.replace('ee', 'e')
            eq = eq.replace('eexp', 'exp')
            eq = eq.replace('exxp', 'exp')
            eq = re.sub(r'(\d+\.?\d*)\s*·\s*\1\s*·', r'\1·', eq)
            eq = eq.replace('· ·', '·')
            eq = eq.replace('·  ·', '·')
            eq = eq.replace('**', '^')
            eq = eq.replace('*', '·')
            eq = eq.replace('phi', 'φ')
            eq = re.sub(r'exp\s*\(\s*([^)]+)\s*\)', r'e^(\1)', eq)
            eq = eq.replace('++', '+')
            eq = eq.replace('--', '+')
            eq = eq.replace('+-', '-')
            eq = eq.replace('-+', '-')
            eq = re.sub(r'e\^\(\s*([^)]*?)$', r'e^(\1)', eq)
            open_parens = eq.count('(')
            close_parens = eq.count(')')
            if open_parens > close_parens:
                eq += ')' * (open_parens - close_parens)
            eq = eq.replace('()', '')
            eq = re.sub(r'0\.0+\s*·\s*[^+\s]+', '', eq)
            eq = re.sub(r'0\.0+\s*·\s*e\^\([^)]+\)', '', eq)
            eq = eq.replace('  ', ' ')
            eq = eq.replace('· ', '·')
            eq = eq.replace(' ·', '·')
            eq = re.sub(r'\s*\+\s*$', '', eq)
            eq = re.sub(r'\s*-\s*$', '', eq)
            eq = re.sub(r'\s*·\s*$', '', eq)
            return eq
        except Exception:
            return equation.replace('*', '·').replace('phi', 'φ')
    
    def _copy_equation(self, equation):
        clipboard = QApplication.clipboard()
        clipboard.setText(f"k = {equation}")
        QMessageBox.information(self, "Copied", "✅ Equation copied to clipboard!")


class DiscoveryWorker(QThread):
    finished = pyqtSignal(list)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    
    def __init__(self, phi, k, k_red, k_oxy):
        super().__init__()
        self.phi = phi
        self.k = k
        self.k_red = k_red
        self.k_oxy = k_oxy
    
    def run(self):
        try:
            self.progress.emit(0, "Initializing model discovery...")
            discovery = ModelDiscovery(self.phi, self.k, self.k_red, self.k_oxy)
            self.progress.emit(30, "Testing model families...")
            models = discovery.discover_models(target_r2_min=0.6, target_r2_max=1.0, max_models=15)
            self.progress.emit(100, "Discovery complete!")
            self.finished.emit(models)
        except Exception as e:
            self.error.emit(str(e))


class ModelDiscoveryDialog(QDialog):
    """Dialog for discovering new models."""
    
    # Signal emitted when models are added
    models_added = pyqtSignal()
    
    def __init__(self, df, parent=None, mode="nanothermite"):
        super().__init__(parent)
        self.df = df
        self.mode = mode
        self.discovered_models = []
        self.selected_model = None
        self.selected_row = -1
        self.worker = None
        self.main_window = parent
        self.added_models = []
        
        self.setWindowTitle("🔍 Auto-Discover New Model")
        self.setMinimumSize(1100, 750)
        self.setModal(True)
        
        self._build_ui()
    
    def _format_equation_display(self, equation):
        try:
            eq = equation
            eq = eq.replace('6xp(', 'exp(')
            eq = eq.replace('6xp', 'exp')
            eq = eq.replace('xp(', 'exp(')
            eq = eq.replace('xp', 'exp')
            while 'ee' in eq:
                eq = eq.replace('ee', 'e')
            eq = eq.replace('eexp', 'exp')
            eq = eq.replace('exxp', 'exp')
            eq = re.sub(r'(\d+\.?\d*)\s*·\s*\1\s*·', r'\1·', eq)
            eq = eq.replace('· ·', '·')
            eq = eq.replace('·  ·', '·')
            eq = eq.replace('**', '^')
            eq = eq.replace('*', '·')
            eq = eq.replace('phi', 'φ')
            eq = re.sub(r'exp\s*\(\s*([^)]+)\s*\)', r'e^(\1)', eq)
            eq = eq.replace('++', '+')
            eq = eq.replace('--', '+')
            eq = eq.replace('+-', '-')
            eq = eq.replace('-+', '-')
            eq = re.sub(r'e\^\(\s*([^)]*?)$', r'e^(\1)', eq)
            open_parens = eq.count('(')
            close_parens = eq.count(')')
            if open_parens > close_parens:
                eq += ')' * (open_parens - close_parens)
            eq = eq.replace('()', '')
            eq = re.sub(r'0\.0+\s*·\s*[^+\s]+', '', eq)
            eq = re.sub(r'0\.0+\s*·\s*e\^\([^)]+\)', '', eq)
            eq = eq.replace('  ', ' ')
            eq = eq.replace('· ', '·')
            eq = eq.replace(' ·', '·')
            eq = re.sub(r'\s*\+\s*$', '', eq)
            eq = re.sub(r'\s*-\s*$', '', eq)
            eq = re.sub(r'\s*·\s*$', '', eq)
            return eq
        except Exception:
            return equation.replace('*', '·').replace('phi', 'φ')
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        title = QLabel("🔍 Auto-Discover New Model")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1a5490;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        desc = QLabel(
            "The application will search for new mathematical models that fit your experimental data.\n"
            "Discovered models with R² between 0.6 and 1.0 will be displayed.\n"
            "Double-click on the equation cell to view the full equation with copy option."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        layout.addWidget(desc)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        self.discover_button = QPushButton("🚀 Start Discovery")
        self.discover_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 10px 25px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.discover_button.clicked.connect(self.start_discovery)
        layout.addWidget(self.discover_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setHandleWidth(8)
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #bdc3c7;
                height: 4px;
            }
            QSplitter::handle:hover {
                background-color: #1a5490;
            }
        """)
        
        results_group = QGroupBox("Discovered Models")
        results_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Select", "Model Name", "Equation", "R²", "RMSE", "MAPE (%)"
        ])
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setStyleSheet("""
            QTableWidget { 
                font-size: 11px; 
                gridline-color: #dcdde1;
            }
            QHeaderView::section { 
                background-color: #1a5490; 
                color: white; 
                padding: 6px; 
                font-weight: bold;
            }
            QTableWidget::item { padding: 5px; }
        """)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setColumnWidth(0, 60)
        self.results_table.setColumnWidth(1, 150)
        self.results_table.setColumnWidth(2, 350)
        self.results_table.setColumnWidth(3, 80)
        self.results_table.setColumnWidth(4, 80)
        self.results_table.setColumnWidth(5, 80)
        self.results_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.results_table.itemSelectionChanged.connect(self.on_selection_changed)
        results_layout.addWidget(self.results_table)
        main_splitter.addWidget(results_group)
        
        plot_group = QGroupBox("Model Preview")
        plot_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        plot_layout = QVBoxLayout(plot_group)
        
        self.plot_canvas = MplCanvas(self, width=6, height=4.5, dpi=100)
        self.plot_canvas.setMinimumSize(400, 300)
        self.plot_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_layout.addWidget(self.plot_canvas)
        
        main_splitter.addWidget(plot_group)
        main_splitter.setSizes([400, 300])
        layout.addWidget(main_splitter, 1)
        
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)
        
        self.add_all_button = QPushButton("➕ Add All Models")
        self.add_all_button.setStyleSheet("""
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
        self.add_all_button.clicked.connect(self.add_all_to_registry)
        self.add_all_button.setEnabled(False)
        bottom_layout.addWidget(self.add_all_button)
        
        bottom_layout.addWidget(QLabel("Model Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter model name...")
        self.name_edit.setMinimumWidth(180)
        self.name_edit.setEnabled(False)
        bottom_layout.addWidget(self.name_edit)
        
        self.add_button = QPushButton("➕ Add Selected")
        self.add_button.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                color: white;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #9b59b6; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.add_button.clicked.connect(self.add_to_registry)
        self.add_button.setEnabled(False)
        bottom_layout.addWidget(self.add_button)
        
        self.save_graph_button = QPushButton("💾 Save Graph")
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
        self.save_graph_button.clicked.connect(self.save_preview_graph)
        self.save_graph_button.setEnabled(False)
        bottom_layout.addWidget(self.save_graph_button)
        
        bottom_layout.addStretch()
        close_button = QPushButton("✕ Close")
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 8px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        close_button.clicked.connect(self.close)
        bottom_layout.addWidget(close_button)
        layout.addLayout(bottom_layout)
    
    def _on_cell_double_clicked(self, row, column):
        if column == 2 and row < len(self.discovered_models):
            model = self.discovered_models[row]
            formatted_eq = self._format_equation_display(model['equation'])
            popup = EquationPopup(
                model['name'],
                formatted_eq,
                model['r2'],
                model['rmse'],
                self
            )
            popup.exec()
    
    def start_discovery(self):
        if self.df is None or len(self.df) < 2:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
        
        if "Filler_volume_fraction" in self.df.columns:
            phi = self.df["Filler_volume_fraction"].values
        elif "phi" in self.df.columns:
            phi = self.df["phi"].values
        else:
            QMessageBox.warning(self, "Warning", "No phi column found in data.")
            return
            
        k = self.df["k_meas"].values
        
        if self.main_window:
            k_red = getattr(self.main_window, 'k_matrix', 237.0)
            k_oxy = getattr(self.main_window, 'k_filler', 33.0)
        else:
            k_red = 237.0
            k_oxy = 33.0
        
        self.discover_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.results_table.setRowCount(0)
        self.discovered_models = []
        self.selected_model = None
        self.selected_row = -1
        self.add_button.setEnabled(False)
        self.add_all_button.setEnabled(False)
        self.save_graph_button.setEnabled(False)
        self.name_edit.setEnabled(False)
        self.name_edit.setText("")
        
        self.worker = DiscoveryWorker(phi, k, k_red, k_oxy)
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._on_discovery_finished)
        self.worker.error.connect(self._on_discovery_error)
        self.worker.start()
    
    def _update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{message}")
    
    def _on_discovery_finished(self, models):
        self.progress_bar.setVisible(False)
        self.discover_button.setEnabled(True)
        
        if not models:
            self.results_table.setRowCount(1)
            msg_item = QTableWidgetItem("No models with R² between 0.6 and 1.0 were found.")
            msg_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setSpan(0, 0, 1, 6)
            self.results_table.setItem(0, 0, msg_item)
            QMessageBox.information(self, "No Models Found", 
                "No models with R² between 0.6 and 1.0 were found.\n"
                "Try adjusting your data or model parameters.")
            return
        
        self.discovered_models = models
        
        self.results_table.setRowCount(len(models))
        self.results_table.clearSpans()
        
        for i, model in enumerate(models):
            cb = QCheckBox()
            cb.setChecked(False)
            cb.stateChanged.connect(lambda state, row=i: self._on_checkbox_changed(row, state))
            self.results_table.setCellWidget(i, 0, cb)
            self.results_table.setItem(i, 1, QTableWidgetItem(model['name']))
            eq_formatted = self._format_equation_display(model['equation'])
            eq_display = eq_formatted[:80] + "..." if len(eq_formatted) > 80 else eq_formatted
            eq_item = QTableWidgetItem(eq_display)
            eq_item.setToolTip("Double-click to view full equation")
            eq_item.setForeground(QColor("#2980b9"))
            self.results_table.setItem(i, 2, eq_item)
            r2_item = QTableWidgetItem(f"{model['r2']:.4f}")
            if model['r2'] >= 0.95:
                r2_item.setBackground(QColor("#d4edda"))
                r2_item.setForeground(QColor("#155724"))
            elif model['r2'] >= 0.85:
                r2_item.setBackground(QColor("#fff3cd"))
                r2_item.setForeground(QColor("#856404"))
            else:
                r2_item.setBackground(QColor("#f8d7da"))
                r2_item.setForeground(QColor("#721c24"))
            self.results_table.setItem(i, 3, r2_item)
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{model['rmse']:.2f}"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{model['mape']:.2f}"))
        
        self.results_table.resizeColumnsToContents()
        
        if models:
            self.add_all_button.setEnabled(True)
        
        best_idx = np.argmax([m['r2'] for m in models])
        cb = self.results_table.cellWidget(best_idx, 0)
        if cb:
            cb.setChecked(True)
        
        QMessageBox.information(self, "Discovery Complete", 
            f"Found {len(models)} models with R² ≥ 0.6!\n"
            "Select a model to preview and add to the registry.\n"
            "Click 'Add All Models' to add all discovered models.\n"
            "Double-click on an equation to view the full equation with copy option.")
    
    def _on_discovery_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.discover_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Discovery failed:\n{error_msg}")
    
    def _on_checkbox_changed(self, row, state):
        if state == 2:
            for i in range(self.results_table.rowCount()):
                if i != row:
                    cb = self.results_table.cellWidget(i, 0)
                    if cb:
                        cb.setChecked(False)
            self._select_model(row)
        else:
            if self.selected_row == row:
                self.selected_model = None
                self.selected_row = -1
                self.name_edit.setEnabled(False)
                self.add_button.setEnabled(False)
                self.save_graph_button.setEnabled(False)
                self.name_edit.setText("")
                self.plot_canvas.axes.clear()
                self.plot_canvas.draw()
    
    def on_selection_changed(self):
        selected = self.results_table.selectedItems()
        if selected:
            row = selected[0].row()
            cb = self.results_table.cellWidget(row, 0)
            if cb and not cb.isChecked():
                cb.setChecked(True)
    
    def _select_model(self, row):
        if row < len(self.discovered_models):
            self.selected_model = self.discovered_models[row]
            self.selected_row = row
            self.name_edit.setEnabled(True)
            self.add_button.setEnabled(True)
            self.save_graph_button.setEnabled(True)
            default_name = f"Discovered_{row+1}_{self.selected_model['name'].replace(' ', '_')}"
            self.name_edit.setText(default_name)
            self._update_plot()
    
    def _update_plot(self):
        if self.selected_model is None:
            return
        ax = self.plot_canvas.axes
        ax.clear()
        
        if "Filler_volume_fraction" in self.df.columns:
            phi = self.df["Filler_volume_fraction"].values
        else:
            phi = self.df["phi"].values
            
        k_meas = self.df["k_meas"].values
        ax.scatter(phi, k_meas, color='black', s=80, label='Experimental', zorder=5, marker='o', edgecolors='white', linewidth=1)
        if len(np.unique(phi)) < 3:
            ax.text(0.5, 0.5, "⚠️ Not enough data variation\nAdd more phi values",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='red')
            ax.set_xlabel('Filler Volume Fraction, φ')
            ax.set_ylabel('Thermal Conductivity, k (W/m·K)')
            ax.set_title(f'Preview: {self.selected_model["name"]} (R²={self.selected_model["r2"]:.4f})')
            ax.grid(True, alpha=0.3)
            self.plot_canvas.draw()
            return
        phi_min = max(0, phi.min() - 0.05)
        phi_max = min(1, phi.max() + 0.05)
        phi_smooth = np.linspace(phi_min, phi_max, 500)
        expr = self.selected_model['expr']
        param_names = self.selected_model['param_names']
        param_values = self.selected_model['param_values']
        y_smooth = []
        for p in phi_smooth:
            local_vars = {'phi': p, 'np': np, 'exp': np.exp}
            for name, value in zip(param_names, param_values):
                local_vars[name] = value
            try:
                val = eval(expr, {"__builtins__": {}}, local_vars)
                if np.isnan(val) or np.isinf(val):
                    y_smooth.append(np.nan)
                else:
                    y_smooth.append(float(val))
            except:
                y_smooth.append(np.nan)
        y_smooth = np.array(y_smooth)
        valid_mask = ~np.isnan(y_smooth)
        if np.any(valid_mask):
            ax.plot(phi_smooth[valid_mask], y_smooth[valid_mask], 
                    color='#e74c3c', linewidth=2.5, label='Discovered Model', zorder=4)
        ax.set_xlim(phi_min, phi_max)
        if np.any(~np.isnan(y_smooth)):
            y_min = min(k_meas.min(), np.nanmin(y_smooth))
            y_max = max(k_meas.max(), np.nanmax(y_smooth))
            y_padding = (y_max - y_min) * 0.15 if y_max > y_min else 20
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.set_xlabel('Filler Volume Fraction, φ', fontsize=12)
        ax.set_ylabel('Thermal Conductivity, k (W/m·K)', fontsize=12)
        r2 = self.selected_model['r2']
        rmse = self.selected_model['rmse']
        ax.text(0.02, 0.98, f'R² = {r2:.4f}\nRMSE = {rmse:.2f} W/m·K',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, family='monospace')
        ax.set_title(f'Preview: {self.selected_model["name"]} (R²={r2:.4f})', fontsize=13)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        self.plot_canvas.draw()
    
    def add_to_registry(self):
        if self.selected_model is None:
            return
        model_name = self.name_edit.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Warning", "Please enter a model name.")
            return
        
        print(f"[DEBUG] Adding model: {model_name}")
        
        # Check if model already exists
        if model_name in MODEL_REGISTRY:
            reply = QMessageBox.question(self, "Model Exists", 
                f"Model '{model_name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Create the registry entry
        phi_data = self.df["Filler_volume_fraction"].values if "Filler_volume_fraction" in self.df.columns else self.df["phi"].values
        k_data = self.df["k_meas"].values
        k_red = getattr(self.main_window, 'k_matrix', 237.0) if self.main_window else 237.0
        k_oxy = getattr(self.main_window, 'k_filler', 33.0) if self.main_window else 33.0
        
        discovery = ModelDiscovery(phi_data, k_data, k_red, k_oxy)
        registry_entry = discovery.add_to_registry(model_name, self.selected_model)
        
        print(f"[DEBUG] Registry entry created")
        
        # ===== ADD TO GLOBAL REGISTRY =====
        MODEL_REGISTRY[model_name] = registry_entry
        if "Discovered Models" not in MODEL_CATEGORIES:
            MODEL_CATEGORIES["Discovered Models"] = []
        if model_name not in MODEL_CATEGORIES["Discovered Models"]:
            MODEL_CATEGORIES["Discovered Models"].append(model_name)
        
        print(f"[DEBUG] Added to MODEL_REGISTRY. Total models: {len(MODEL_REGISTRY)}")
        print(f"[DEBUG] Discovered models: {MODEL_CATEGORIES.get('Discovered Models', [])}")
        
        # ===== UPDATE MAIN WINDOW =====
        if self.main_window:
            print("[DEBUG] Updating main window...")
            self.main_window.MODEL_REGISTRY[model_name] = registry_entry
            if "Discovered Models" not in self.main_window.MODEL_CATEGORIES:
                self.main_window.MODEL_CATEGORIES["Discovered Models"] = []
            if model_name not in self.main_window.MODEL_CATEGORIES["Discovered Models"]:
                self.main_window.MODEL_CATEGORIES["Discovered Models"].append(model_name)
            
            # Also ensure the main window's categories include discovered models
            discovered_models = [name for name, meta in self.main_window.MODEL_REGISTRY.items() 
                                if meta.get("discovered", False)]
            self.main_window.MODEL_CATEGORIES["Discovered Models"] = discovered_models
            print(f"[DEBUG] Main window discovered models: {discovered_models}")
        
        # Store the added model name
        self.added_models.append(model_name)
        
        # ===== REFRESH UI =====
        self._update_main_window_ui()
        
        formatted_eq = self._format_equation_display(self.selected_model['equation'])
        print(f"[DEBUG] Model added successfully: {model_name}")
        
        # Emit signal
        self.models_added.emit()
        
        QMessageBox.information(self, "Success", 
            f"✅ Model '{model_name}' has been added to the registry!\n\n"
            f"📐 Equation: k = {formatted_eq}\n"
            f"📊 R² = {self.selected_model['r2']:.4f}\n"
            f"📊 RMSE = {self.selected_model['rmse']:.2f} W/m·K\n\n"
            f"The model is now available in the 'Model Selection' panel under 'Discovered Models' category.")
        
        self.accept()
    
    def add_all_to_registry(self):
        if not self.discovered_models:
            QMessageBox.warning(self, "Warning", "No models to add.")
            return
        
        print(f"[DEBUG] Adding all {len(self.discovered_models)} models...")
        
        added_count = 0
        phi_data = self.df["Filler_volume_fraction"].values if "Filler_volume_fraction" in self.df.columns else self.df["phi"].values
        k_data = self.df["k_meas"].values
        k_red = getattr(self.main_window, 'k_matrix', 237.0) if self.main_window else 237.0
        k_oxy = getattr(self.main_window, 'k_filler', 33.0) if self.main_window else 33.0
        
        for model in self.discovered_models:
            base_name = model['name'].replace(' ', '_')
            model_name = f"Discovered_{base_name}"
            counter = 1
            while model_name in MODEL_REGISTRY:
                model_name = f"Discovered_{base_name}_{counter}"
                counter += 1
            
            print(f"[DEBUG] Adding model: {model_name}")
            
            discovery = ModelDiscovery(phi_data, k_data, k_red, k_oxy)
            registry_entry = discovery.add_to_registry(model_name, model)
            
            # Add to GLOBAL registry
            MODEL_REGISTRY[model_name] = registry_entry
            if "Discovered Models" not in MODEL_CATEGORIES:
                MODEL_CATEGORIES["Discovered Models"] = []
            if model_name not in MODEL_CATEGORIES["Discovered Models"]:
                MODEL_CATEGORIES["Discovered Models"].append(model_name)
            
            # Update main window
            if self.main_window:
                self.main_window.MODEL_REGISTRY[model_name] = registry_entry
                if "Discovered Models" not in self.main_window.MODEL_CATEGORIES:
                    self.main_window.MODEL_CATEGORIES["Discovered Models"] = []
                if model_name not in self.main_window.MODEL_CATEGORIES["Discovered Models"]:
                    self.main_window.MODEL_CATEGORIES["Discovered Models"].append(model_name)
            
            self.added_models.append(model_name)
            added_count += 1
        
        print(f"[DEBUG] Added {added_count} models to registry")
        print(f"[DEBUG] Total models in registry: {len(MODEL_REGISTRY)}")
        
        # Also ensure the main window's categories include discovered models
        if self.main_window:
            discovered_models = [name for name, meta in self.main_window.MODEL_REGISTRY.items() 
                                if meta.get("discovered", False)]
            self.main_window.MODEL_CATEGORIES["Discovered Models"] = discovered_models
            print(f"[DEBUG] Main window discovered models: {discovered_models}")
        
        # ===== REFRESH UI =====
        self._update_main_window_ui()
        
        # Emit signal
        self.models_added.emit()
        
        QMessageBox.information(self, "Success", 
            f"✅ Added {added_count} models to the registry!\n\n"
            f"They are available in the 'Model Selection' panel under 'Discovered Models' category.")
        
        self.accept()
    
    def _update_main_window_ui(self):
        """Update the main window UI with the new models."""
        if not self.main_window:
            print("[DEBUG] No main window to update")
            return
        
        print("[DEBUG] Updating main window UI...")
        
        # Get discovered models from the main window's registry
        discovered_models = [name for name, meta in self.main_window.MODEL_REGISTRY.items() 
                            if meta.get("discovered", False)]
        self.main_window.MODEL_CATEGORIES["Discovered Models"] = discovered_models
        print(f"[DEBUG] Discovered models in main window: {discovered_models}")
        
        # Update the model selection grid - this will repopulate everything
        if hasattr(self.main_window, '_populate_models_by_category'):
            print("[DEBUG] Calling _populate_models_by_category")
            self.main_window._populate_models_by_category()
        
        # Update the category combo box
        if hasattr(self.main_window, 'category_combo'):
            current_text = self.main_window.category_combo.currentText()
            self.main_window.category_combo.clear()
            categories = list(self.main_window.MODEL_CATEGORIES.keys())
            self.main_window.category_combo.addItems(["All Models"] + categories)
            index = self.main_window.category_combo.findText(current_text)
            if index >= 0:
                self.main_window.category_combo.setCurrentIndex(index)
            else:
                self.main_window.category_combo.setCurrentText("All Models")
            print(f"[DEBUG] Category combo updated with {len(categories)} categories")
        
        # Update the models reference tab
        if hasattr(self.main_window, '_update_models_reference'):
            print("[DEBUG] Calling _update_models_reference")
            self.main_window._update_models_reference()
        
        # Force update
        if hasattr(self.main_window, 'update'):
            self.main_window.update()
        
        print("[DEBUG] Main window UI update complete")
    
    def save_preview_graph(self):
        if self.selected_model is None:
            QMessageBox.warning(self, "Warning", "Please select a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", f"discovered_model_{self.selected_model['name']}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tiff)"
        )
        if not path:
            return
        try:
            self.plot_canvas.fig.savefig(path, dpi=300, bbox_inches="tight")
            QMessageBox.information(self, "Success", f"Graph saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save graph:\n{str(e)}")