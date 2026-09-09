"""Model Comparison Widget - Compare all fitted models."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import numpy as np
import pandas as pd


class ModelComparisonDialog(QDialog):
    """Dialog to compare all fitted models."""

    def __init__(self, fit_results, parent=None):
        super().__init__(parent)
        self.fit_results = fit_results

        self.setWindowTitle("Model Comparison Dashboard")
        self.setMinimumSize(1100, 700)
        self.setModal(True)

        self._build_ui()
        self._populate_table()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel("📊 Model Comparison Dashboard")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1a5490;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Convention note
        convention = QLabel("φ = Reductant (matrix) volume fraction | (1-φ) = Oxidizer (filler) volume fraction")
        convention.setStyleSheet("font-size: 11px; color: #7f8c8d; font-style: italic;")
        convention.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(convention)

        # Subtitle with stats
        total_models = len(self.fit_results)
        successful = sum(1 for r in self.fit_results.values() if r.get('success', False))
        stats_label = QLabel(f"Comparing {total_models} models ({successful} successful)")
        stats_label.setStyleSheet("font-size: 13px; color: #7f8c8d;")
        stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(stats_label)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Rank", "Model", "R²", "RMSE", "MAE", "MAPE (%)",
            "Parameters", "Status", "Category"
        ])
        
        # Enable interactive column resizing
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                gridline-color: #dcdde1;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #1a5490;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QTableWidget::item { padding: 5px; }
        """)
        
        # Set initial column widths
        self.table.setColumnWidth(0, 60)   # Rank
        self.table.setColumnWidth(1, 250)  # Model
        self.table.setColumnWidth(2, 80)   # R²
        self.table.setColumnWidth(3, 80)   # RMSE
        self.table.setColumnWidth(4, 80)   # MAE
        self.table.setColumnWidth(5, 80)   # MAPE
        self.table.setColumnWidth(6, 200)  # Parameters
        self.table.setColumnWidth(7, 80)   # Status
        self.table.setColumnWidth(8, 150)  # Category
        
        layout.addWidget(self.table)

        # Best model indicator
        self.best_label = QLabel("")
        self.best_label.setStyleSheet("font-size: 15px; font-weight: bold; padding: 15px; border-radius: 8px;")
        self.best_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.best_label)

        # Buttons
        btn_layout = QHBoxLayout()

        export_btn = QPushButton("📥 Export Comparison")
        export_btn.setStyleSheet("background-color: #3498db; color: white; padding: 8px 16px; border-radius: 4px;")
        export_btn.clicked.connect(self.export_comparison)
        btn_layout.addWidget(export_btn)

        btn_layout.addStretch()

        close_btn = QPushButton("✕ Close")
        close_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 8px 20px; border-radius: 4px; font-weight: bold;")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _populate_table(self):
        """Populate the table with model data."""
        if not self.fit_results:
            return

        try:
            from ..core.models import MODEL_REGISTRY
        except:
            MODEL_REGISTRY = {}

        # Get all results with R² - check both 'stats' and top-level
        sorted_models = []
        for name, result in self.fit_results.items():
            if result.get('success', False):
                # Try to get stats from 'stats' key or top-level
                stats = result.get('stats', {})
                
                # If stats is empty, try to get values directly from result
                if not stats:
                    stats = {
                        'r2': result.get('r2', 0),
                        'rmse': result.get('rmse', 0),
                        'mae': result.get('mae', 0),
                        'mape': result.get('mape', 0)
                    }
                
                r2 = stats.get('r2', 0)
                # Only include models with valid R² > 0
                if r2 > 0:
                    sorted_models.append((name, result, r2, stats))
                else:
                    sorted_models.append((name, result, -1, stats))
            else:
                sorted_models.append((name, result, -1, {}))
        
        # Sort by R² descending (valid R² first)
        sorted_models.sort(key=lambda x: x[2], reverse=True)

        self.table.setRowCount(len(sorted_models))

        for i, (model_name, result, r2, stats) in enumerate(sorted_models):
            params = result.get('params', {})
            success = result.get('success', False)

            meta = MODEL_REGISTRY.get(model_name, {})
            category = meta.get('category', 'Unknown')

            # Rank
            if success and r2 > 0:
                if i == 0:
                    rank_text = "🏆 #1"
                    rank_color = QColor("#27ae60")
                elif i == 1:
                    rank_text = "🥈 #2"
                    rank_color = QColor("#f39c12")
                elif i == 2:
                    rank_text = "🥉 #3"
                    rank_color = QColor("#e67e22")
                else:
                    rank_text = f"#{i+1}"
                    rank_color = QColor("#95a5a6")
            else:
                rank_text = "❌"
                rank_color = QColor("#e74c3c")

            rank_item = QTableWidgetItem(rank_text)
            rank_item.setBackground(rank_color)
            rank_item.setForeground(QColor("white"))
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 0, rank_item)

            # Model name
            model_item = QTableWidgetItem(model_name)
            if success and i == 0 and r2 > 0:
                model_item.setForeground(QColor("#27ae60"))
                font = model_item.font()
                font.setBold(True)
                model_item.setFont(font)
            elif not success:
                model_item.setForeground(QColor("#e74c3c"))
            self.table.setItem(i, 1, model_item)

            # R²
            if success and r2 > 0:
                r2_item = QTableWidgetItem(f"{r2:.4f}")
                if r2 >= 0.95:
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
            else:
                r2_item = QTableWidgetItem("-")
            r2_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 2, r2_item)

            # RMSE
            rmse = stats.get('rmse', np.nan)
            rmse_item = QTableWidgetItem(f"{rmse:.4f}" if not np.isnan(rmse) and success and r2 > 0 else "-")
            rmse_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 3, rmse_item)

            # MAE
            mae = stats.get('mae', np.nan)
            mae_item = QTableWidgetItem(f"{mae:.4f}" if not np.isnan(mae) and success and r2 > 0 else "-")
            mae_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 4, mae_item)

            # MAPE
            mape = stats.get('mape', np.nan)
            mape_item = QTableWidgetItem(f"{mape:.2f}" if not np.isnan(mape) and success and r2 > 0 else "-")
            mape_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 5, mape_item)

            # Parameters
            if success and params:
                param_str = ", ".join([f"{k}={v:.3f}" for k, v in params.items()])
                param_item = QTableWidgetItem(param_str)
            else:
                param_item = QTableWidgetItem("(fixed)" if success else "-")
            self.table.setItem(i, 6, param_item)

            # Status
            status_text = "✅ Success" if success else "❌ Failed"
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QColor("#27ae60") if success else QColor("#e74c3c"))
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 7, status_item)

            # Category
            cat_item = QTableWidgetItem(category if category else "Discovered Models")
            cat_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 8, cat_item)

        self.table.resizeColumnsToContents()

        # Best model
        if sorted_models:
            # Find the best model with valid R²
            best_entry = None
            for entry in sorted_models:
                if entry[2] > 0:
                    best_entry = entry
                    break
            
            if best_entry:
                best_name, best_result, best_r2, best_stats = best_entry
                best_rmse = best_stats.get('rmse', 0)
                best_params = best_result.get('params', {})

                if best_r2 >= 0.95:
                    emoji = "⭐"
                    color = "#27ae60"
                elif best_r2 >= 0.85:
                    emoji = "✓"
                    color = "#f39c12"
                else:
                    emoji = "📊"
                    color = "#e67e22"

                param_str = ", ".join([f"{k}={v:.3f}" for k, v in best_params.items()]) if best_params else "(no parameters)"

                self.best_label.setText(
                    f"{emoji} Best Performing Model: {best_name}  |  "
                    f"R² = {best_r2:.4f}  |  "
                    f"RMSE = {best_rmse:.4f} W/m·K  |  "
                    f"Parameters: {param_str}"
                )
                self.best_label.setStyleSheet(
                    f"font-size: 14px; font-weight: bold; padding: 12px; "
                    f"border-radius: 8px; background-color: {color}; color: white;"
                )
            else:
                self.best_label.setText("⚠️ No models with valid R²")
                self.best_label.setStyleSheet(
                    "font-size: 14px; font-weight: bold; padding: 12px; "
                    "border-radius: 8px; background-color: #e74c3c; color: white;"
                )

    def export_comparison(self):
        """Export comparison to CSV or Excel."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Comparison", "model_comparison.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if not path:
            return

        try:
            data = []
            for model_name, result in self.fit_results.items():
                stats = result.get('stats', {})
                # If stats is empty, try to get values directly from result
                if not stats:
                    stats = {
                        'r2': result.get('r2', np.nan),
                        'rmse': result.get('rmse', np.nan),
                        'mae': result.get('mae', np.nan),
                        'mape': result.get('mape', np.nan)
                    }
                params = result.get('params', {})
                row = {
                    'Model': model_name,
                    'R2': stats.get('r2', np.nan),
                    'RMSE': stats.get('rmse', np.nan),
                    'MAE': stats.get('mae', np.nan),
                    'MAPE': stats.get('mape', np.nan),
                    'Success': result.get('success', False),
                    'Message': result.get('message', '')
                }
                for param_name, param_value in params.items():
                    row[f'Param_{param_name}'] = param_value
                data.append(row)

            df = pd.DataFrame(data)

            if path.endswith('.csv'):
                df.to_csv(path, index=False)
            else:
                df.to_excel(path, index=False)

            QMessageBox.information(self, "Export Complete", f"Comparison exported to:\n{path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")
