"""Best R² Tab - Multi-file model performance comparison and ranking."""

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
    QTableWidgetItem, QHeaderView, QLabel, QPushButton,
    QGroupBox, QFileDialog, QMessageBox, QProgressBar,
    QSplitter, QTextEdit, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont

from ..core.models import MODEL_REGISTRY


class BestR2Tab(QWidget):
    """Tab for displaying best R² results across multiple datasets."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.results_data = {}
        self.fit_results = {}
        self.df = None
        self.best_results = {}
        self.average_results = {}
        self.model_names = []
        self.is_initialized = False
        self.has_data = False
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the Best R² tab UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # ============================================================
        # TOP: Summary Bar
        # ============================================================
        self.summary_group = QGroupBox("📊 Summary")
        summary_layout = QHBoxLayout(self.summary_group)
        
        self.summary_layout = QHBoxLayout()
        
        # Summary labels
        self.total_files_label = QLabel("📁 0 Datasets Analyzed")
        self.total_files_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #1a5490;")
        self.summary_layout.addWidget(self.total_files_label)
        
        self.summary_layout.addWidget(self._create_separator())
        
        self.best_model_label = QLabel("🏆 Best Model: None")
        self.best_model_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #27ae60;")
        self.summary_layout.addWidget(self.best_model_label)
        
        self.summary_layout.addWidget(self._create_separator())
        
        self.avg_r2_label = QLabel("📈 Avg R²: N/A")
        self.avg_r2_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #2c3e50;")
        self.summary_layout.addWidget(self.avg_r2_label)
        
        self.summary_layout.addWidget(self._create_separator())
        
        self.quality_label = QLabel("🟢 Excellent: 0/0 (0%)")
        self.quality_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #2c3e50;")
        self.summary_layout.addWidget(self.quality_label)
        
        self.summary_layout.addStretch()
        
        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.refresh_btn.clicked.connect(self.refresh_data)
        self.refresh_btn.setEnabled(False)
        self.summary_layout.addWidget(self.refresh_btn)
        
        summary_layout.addLayout(self.summary_layout)
        layout.addWidget(self.summary_group)
        
        # ============================================================
        # MIDDLE: Table with Interactive Column Resizing
        # ============================================================
        self.table_group = QGroupBox("📊 Model Performance Matrix")
        table_layout = QVBoxLayout(self.table_group)
        
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        
        # Enable interactive column resizing
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        
        self.table.setStyleSheet("""
            QTableWidget {
                font-size: 11px;
                gridline-color: #dcdde1;
            }
            QHeaderView::section {
                background-color: #1a5490;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
            QTableWidget::item { padding: 5px; }
        """)
        table_layout.addWidget(self.table)
        
        layout.addWidget(self.table_group, 2)
        
        # ============================================================
        # BOTTOM: Interpretation and Buttons
        # ============================================================
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Interpretation
        self.interpretation_group = QGroupBox("📝 Interpretation")
        interp_layout = QVBoxLayout(self.interpretation_group)
        
        self.interpretation_text = QTextEdit()
        self.interpretation_text.setReadOnly(True)
        self.interpretation_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 10px;
                font-size: 11px;
                font-family: 'Consolas', monospace;
            }
        """)
        interp_layout.addWidget(self.interpretation_text)
        bottom_splitter.addWidget(self.interpretation_group)
        
        # Buttons
        button_group = QGroupBox("Actions")
        button_layout = QVBoxLayout(button_group)
        
        btn_export = QPushButton("📥 Export to CSV")
        btn_export.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2ecc71; }
        """)
        btn_export.clicked.connect(self.export_results)
        button_layout.addWidget(btn_export)
        
        btn_export_excel = QPushButton("📊 Export to Excel")
        btn_export_excel.setStyleSheet("""
            QPushButton {
                background-color: #1abc9c;
                color: white;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #16a085; }
        """)
        btn_export_excel.clicked.connect(self.export_excel)
        button_layout.addWidget(btn_export_excel)
        
        btn_chart = QPushButton("📊 Show Bar Chart")
        btn_chart.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                color: white;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #9b59b6; }
        """)
        btn_chart.clicked.connect(self.show_bar_chart)
        button_layout.addWidget(btn_chart)
        
        btn_details = QPushButton("🔍 View Details")
        btn_details.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #d35400; }
        """)
        btn_details.clicked.connect(self.show_details)
        button_layout.addWidget(btn_details)
        
        button_layout.addStretch()
        bottom_splitter.addWidget(button_group)
        
        bottom_splitter.setSizes([600, 200])
        layout.addWidget(bottom_splitter, 1)
        
        # ============================================================
        # Progress Bar
        # ============================================================
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
        
        self.is_initialized = True
        
        # Show initial message
        self.interpretation_text.setText(
            "📋 No data available yet.\n\n"
            "To populate this tab:\n"
            "1. Click 'Load Multiple' to load multiple data files\n"
            "2. Select models in the 'Model Selection' panel\n"
            "3. Click 'Run Analysis' and select 'Yes' for multi-file analysis\n"
            "4. Results will appear here automatically"
        )
    
    def _create_separator(self):
        """Create a vertical separator."""
        sep = QLabel("|")
        sep.setStyleSheet("color: #bdc3c7; font-size: 18px;")
        return sep
    
    def update_data(self, multi_file_results, fit_results, df):
        """Update the Best R² tab with new data."""
        print(f"[BestR2Tab] update_data called with {len(multi_file_results) if multi_file_results else 0} files")
        
        self.results_data = multi_file_results
        self.fit_results = fit_results
        self.df = df
        
        if not multi_file_results or len(multi_file_results) == 0:
            print("[BestR2Tab] No multi-file results to display")
            self.clear()
            self.has_data = False
            self.interpretation_text.setText(
                "📋 No multi-file results available.\n\n"
                "Make sure you:\n"
                "1. Loaded multiple files using 'Load Multiple'\n"
                "2. Ran multi-file analysis (select 'Yes' when prompted)\n"
                "3. Analysis completed successfully"
            )
            return
        
        self.refresh_btn.setEnabled(True)
        self.has_data = True
        self._build_best_results()
        self._populate_table()
        self._update_summary()
        self._generate_interpretation()
        
        print(f"[BestR2Tab] Data updated successfully with {len(self.average_results)} models")
    
    def clear(self):
        """Clear all data from the tab."""
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.total_files_label.setText("📁 0 Datasets Analyzed")
        self.best_model_label.setText("🏆 Best Model: None")
        self.avg_r2_label.setText("📈 Avg R²: N/A")
        self.quality_label.setText("🟢 Excellent: 0/0 (0%)")
        self.refresh_btn.setEnabled(False)
        self.average_results = []
        self.best_results = {}
        self.has_data = False
        print("[BestR2Tab] Cleared all data")
    
    def refresh_data(self):
        """Refresh the displayed data."""
        print("[BestR2Tab] Refresh button clicked")
        if self.results_data and len(self.results_data) > 0:
            self._build_best_results()
            self._populate_table()
            self._update_summary()
            self._generate_interpretation()
            QMessageBox.information(self, "Refreshed", "Data refreshed successfully!")
            print("[BestR2Tab] Data refreshed")
        else:
            QMessageBox.warning(self, "Warning", "No data to refresh. Load multiple files and run analysis first.")
    
    def _build_best_results(self):
        """Build the best results per dataset and model averages."""
        if not self.results_data:
            print("[BestR2Tab] No results data to build")
            return
        
        # Get all model names from the first file
        first_file = list(self.results_data.keys())[0]
        model_names = list(self.results_data[first_file].keys())
        print(f"[BestR2Tab] Found {len(model_names)} models: {model_names}")
        
        # Build R² matrix
        r2_matrix = {}
        for model_name in model_names:
            r2_matrix[model_name] = []
        
        best_per_file = {}
        
        for filename, file_results in self.results_data.items():
            best_model = None
            best_r2 = -1
            
            print(f"[BestR2Tab] Processing file: {filename}")
            
            for model_name, result in file_results.items():
                if result.get("success", False):
                    # Get R² from stats or direct
                    stats = result.get('stats', {})
                    r2 = stats.get('r2', result.get('r2', -1))
                    
                    if r2 > 0:  # Only add valid R² values
                        r2_matrix[model_name].append(r2)
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_model = model_name
                        print(f"[BestR2Tab]   {model_name}: R² = {r2:.4f}")
                    else:
                        print(f"[BestR2Tab]   {model_name}: invalid R² = {r2}")
                else:
                    print(f"[BestR2Tab]   {model_name}: FAILED")
            
            best_per_file[filename] = {
                "best_model": best_model,
                "best_r2": best_r2,
                "all_results": file_results
            }
            print(f"[BestR2Tab]   Best: {best_model} (R² = {best_r2:.4f})")
        
        # Calculate averages and statistics
        avg_results = []
        for model_name, r2_values in r2_matrix.items():
            if r2_values and len(r2_values) > 0:
                avg_r2 = np.mean(r2_values)
                std_r2 = np.std(r2_values)
                count_best = sum(1 for data in best_per_file.values() 
                                if data["best_model"] == model_name)
                avg_results.append({
                    "model": model_name,
                    "avg_r2": avg_r2,
                    "std_r2": std_r2,
                    "count_best": count_best,
                    "n_datasets": len(r2_values),
                    "r2_values": r2_values
                })
                print(f"[BestR2Tab] Model {model_name}: Avg R² = {avg_r2:.4f}, Best in {count_best} files")
            else:
                print(f"[BestR2Tab] Model {model_name} had no valid R² values")
        
        # Sort by average R² descending
        avg_results.sort(key=lambda x: x["avg_r2"], reverse=True)
        
        self.best_results = best_per_file
        self.average_results = avg_results
        self.model_names = model_names
        
        print(f"[BestR2Tab] Built results for {len(self.average_results)} models")
    
    def _populate_table(self):
        """Populate the table with R² values with interactive column resizing."""
        if not self.average_results:
            print("[BestR2Tab] No average results to populate table")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        
        # Get all dataset names
        filenames = list(self.best_results.keys())
        n_datasets = len(filenames)
        n_models = len(self.average_results)
        
        print(f"[BestR2Tab] Populating table with {n_models} models and {n_datasets} datasets")
        
        # Set up table
        self.table.setRowCount(n_models)
        self.table.setColumnCount(n_datasets + 3)  # +3 for Rank, Model, Avg R²
        
        # Set headers
        headers = ["Rank", "Model"] + filenames + ["Avg R²"]
        self.table.setHorizontalHeaderLabels(headers)
        
        # Enable interactive column resizing
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        
        # Set initial column widths
        self.table.setColumnWidth(0, 60)   # Rank
        self.table.setColumnWidth(1, 250)  # Model
        
        # Allow all columns to be resized with cursor
        for i in range(2, self.table.columnCount()):
            self.table.setColumnWidth(i, 120)  # Initial width
        
        # Populate data
        for row, data in enumerate(self.average_results):
            model_name = data["model"]
            avg_r2 = data["avg_r2"]
            r2_values = data.get("r2_values", [])
            
            # Rank
            rank_item = QTableWidgetItem(f"#{row + 1}")
            if row == 0:
                rank_item.setText("🏆 #1")
                rank_item.setForeground(QColor("#27ae60"))
                font = rank_item.font()
                font.setBold(True)
                rank_item.setFont(font)
            elif row == 1:
                rank_item.setText("🥈 #2")
                rank_item.setForeground(QColor("#f39c12"))
            elif row == 2:
                rank_item.setText("🥉 #3")
                rank_item.setForeground(QColor("#e67e22"))
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, rank_item)
            
            # Model name
            model_item = QTableWidgetItem(model_name)
            if row == 0:
                font = model_item.font()
                font.setBold(True)
                model_item.setFont(font)
                model_item.setForeground(QColor("#27ae60"))
            self.table.setItem(row, 1, model_item)
            
            # R² values for each dataset
            for col, filename in enumerate(filenames):
                if col < len(r2_values):
                    r2_val = r2_values[col]
                    r2_item = QTableWidgetItem(f"{r2_val:.4f}")
                    
                    # Color code based on R² value
                    if r2_val >= 0.95:
                        r2_item.setBackground(QColor("#d4edda"))
                        r2_item.setForeground(QColor("#155724"))
                    elif r2_val >= 0.85:
                        r2_item.setBackground(QColor("#fff3cd"))
                        r2_item.setForeground(QColor("#856404"))
                    elif r2_val >= 0.70:
                        r2_item.setBackground(QColor("#ffeaa7"))
                        r2_item.setForeground(QColor("#6c5200"))
                    else:
                        r2_item.setBackground(QColor("#f8d7da"))
                        r2_item.setForeground(QColor("#721c24"))
                    
                    # Bold if this is the best model for this dataset
                    if self.best_results[filename]["best_model"] == model_name:
                        font = r2_item.font()
                        font.setBold(True)
                        r2_item.setFont(font)
                    
                    r2_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(row, col + 2, r2_item)
                else:
                    self.table.setItem(row, col + 2, QTableWidgetItem("-"))
            
            # Average R²
            avg_item = QTableWidgetItem(f"{avg_r2:.4f}")
            if avg_r2 >= 0.95:
                avg_item.setBackground(QColor("#d4edda"))
                avg_item.setForeground(QColor("#155724"))
            elif avg_r2 >= 0.85:
                avg_item.setBackground(QColor("#fff3cd"))
                avg_item.setForeground(QColor("#856404"))
            elif avg_r2 >= 0.70:
                avg_item.setBackground(QColor("#ffeaa7"))
                avg_item.setForeground(QColor("#6c5200"))
            else:
                avg_item.setBackground(QColor("#f8d7da"))
                avg_item.setForeground(QColor("#721c24"))
            
            if row == 0:
                font = avg_item.font()
                font.setBold(True)
                avg_item.setFont(font)
            
            avg_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, self.table.columnCount() - 1, avg_item)
        
        self.table.resizeColumnsToContents()
        print("[BestR2Tab] Table populated successfully")
    
    def _update_summary(self):
        """Update the summary bar."""
        if not self.average_results:
            self.total_files_label.setText("📁 0 Datasets Analyzed")
            self.best_model_label.setText("🏆 Best Model: None")
            self.avg_r2_label.setText("📈 Avg R²: N/A")
            self.quality_label.setText("🟢 Excellent: 0/0 (0%)")
            return
        
        n_files = len(self.best_results)
        best_model = self.average_results[0]["model"]
        best_avg_r2 = self.average_results[0]["avg_r2"]
        
        # Count quality distribution
        excellent = sum(1 for data in self.average_results if data["avg_r2"] >= 0.95)
        good = sum(1 for data in self.average_results if 0.85 <= data["avg_r2"] < 0.95)
        fair = sum(1 for data in self.average_results if 0.70 <= data["avg_r2"] < 0.85)
        poor = sum(1 for data in self.average_results if data["avg_r2"] < 0.70)
        
        # Update labels
        self.total_files_label.setText(f"📁 {n_files} Datasets Analyzed")
        self.best_model_label.setText(f"🏆 Best Model: {best_model} (Avg R² = {best_avg_r2:.4f})")
        self.avg_r2_label.setText(f"📈 Top Model Avg R²: {best_avg_r2:.4f}")
        
        quality_text = f"🟢 Excellent: {excellent}  🟡 Good: {good}  🟠 Fair: {fair}  🔴 Poor: {poor}"
        if excellent > 0:
            self.quality_label.setText(f"🟢 {quality_text}")
        elif good > 0:
            self.quality_label.setText(f"🟡 {quality_text}")
        elif fair > 0:
            self.quality_label.setText(f"🟠 {quality_text}")
        else:
            self.quality_label.setText(f"🔴 {quality_text}")
    
    def _generate_interpretation(self):
        """Generate the interpretation text."""
        if not self.average_results:
            self.interpretation_text.setText("No data available. Load multiple files and run analysis to see results here.")
            return
        
        lines = []
        lines.append("📊 BEST R² - MODEL PERFORMANCE INTERPRETATION")
        lines.append("=" * 60)
        lines.append("")
        
        # Best model
        best = self.average_results[0]
        lines.append(f"🏆 BEST OVERALL MODEL: {best['model']}")
        lines.append(f"   • Average R² = {best['avg_r2']:.4f} (Highest across all datasets)")
        lines.append(f"   • Standard Deviation = {best['std_r2']:.4f}")
        lines.append(f"   • Best model in {best['count_best']} out of {best['n_datasets']} datasets")
        
        if best['avg_r2'] >= 0.95:
            lines.append("   ✅ Excellent fit across all datasets")
        elif best['avg_r2'] >= 0.85:
            lines.append("   ✅ Good fit across all datasets")
        elif best['avg_r2'] >= 0.70:
            lines.append("   ⚠️ Fair fit - consider parameter optimization")
        else:
            lines.append("   ❌ Poor fit - model may not be suitable")
        
        lines.append("")
        lines.append("-" * 60)
        lines.append("")
        
        # Runner up
        if len(self.average_results) > 1:
            runner = self.average_results[1]
            lines.append(f"🥈 RUNNER UP: {runner['model']}")
            lines.append(f"   • Average R² = {runner['avg_r2']:.4f}")
            lines.append(f"   • Best in {runner['count_best']} out of {runner['n_datasets']} datasets")
            lines.append("")
        
        # Third place
        if len(self.average_results) > 2:
            third = self.average_results[2]
            lines.append(f"🥉 THIRD PLACE: {third['model']}")
            lines.append(f"   • Average R² = {third['avg_r2']:.4f}")
            lines.append(f"   • Best in {third['count_best']} out of {third['n_datasets']} datasets")
            lines.append("")
        
        lines.append("-" * 60)
        lines.append("")
        
        # Quality distribution
        excellent = sum(1 for data in self.average_results if data["avg_r2"] >= 0.95)
        good = sum(1 for data in self.average_results if 0.85 <= data["avg_r2"] < 0.95)
        fair = sum(1 for data in self.average_results if 0.70 <= data["avg_r2"] < 0.85)
        poor = sum(1 for data in self.average_results if data["avg_r2"] < 0.70)
        
        lines.append("📊 QUALITY DISTRIBUTION")
        lines.append(f"   🟢 Excellent (R² ≥ 0.95): {excellent} models")
        lines.append(f"   🟡 Good (R² ≥ 0.85): {good} models")
        lines.append(f"   🟠 Fair (R² ≥ 0.70): {fair} models")
        lines.append(f"   🔴 Poor (R² < 0.70): {poor} models")
        lines.append("")
        
        # Per dataset best model
        lines.append("-" * 60)
        lines.append("")
        lines.append("📁 BEST MODEL PER DATASET")
        for filename, data in self.best_results.items():
            if data["best_model"]:
                lines.append(f"   • {filename}: {data['best_model']} (R² = {data['best_r2']:.4f})")
            else:
                lines.append(f"   • {filename}: No successful fit")
        
        lines.append("")
        lines.append("-" * 60)
        lines.append("")
        
        # Recommendation
        lines.append("💡 RECOMMENDATION")
        if best['avg_r2'] >= 0.95:
            lines.append(f"   ✅ The {best['model']} model is HIGHLY RECOMMENDED for this data.")
            lines.append("   It consistently provides excellent fits across all datasets.")
        elif best['avg_r2'] >= 0.85:
            lines.append(f"   ✅ The {best['model']} model is RECOMMENDED for this data.")
            lines.append("   It provides good fits, but consider parameter optimization for better results.")
        elif best['avg_r2'] >= 0.70:
            lines.append(f"   ⚠️ The {best['model']} model may need improvement.")
            lines.append("   Consider adjusting parameters or collecting more data.")
        else:
            lines.append(f"   ❌ The {best['model']} model is NOT RECOMMENDED for this data.")
            lines.append("   Try different models or check data quality.")
        
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("Thermal Conductivity Modeling Suite v2.0")
        
        self.interpretation_text.setText("\n".join(lines))
    
    def export_results(self):
        """Export results to CSV."""
        if not self.average_results:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Best R² Results", "best_r2_results.csv",
            "CSV Files (*.csv)"
        )
        
        if not path:
            return
        
        try:
            # Build export data
            data = []
            filenames = list(self.best_results.keys())
            
            for row_data in self.average_results:
                row = {
                    "Rank": self.average_results.index(row_data) + 1,
                    "Model": row_data["model"],
                    "Avg R²": row_data["avg_r2"],
                    "Std Dev": row_data["std_r2"],
                    "Best In N Datasets": row_data["count_best"],
                    "Total Datasets": row_data["n_datasets"],
                }
                
                # Add individual dataset R² values
                for i, filename in enumerate(filenames):
                    if i < len(row_data["r2_values"]):
                        row[f"{filename}"] = row_data["r2_values"][i]
                    else:
                        row[f"{filename}"] = None
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
            
            QMessageBox.information(self, "Success", f"Results exported to:\n{path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")
    
    def export_excel(self):
        """Export results to Excel."""
        if not self.average_results:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Best R² Results", "best_r2_results.xlsx",
            "Excel Files (*.xlsx)"
        )
        
        if not path:
            return
        
        try:
            # Build export data
            data = []
            filenames = list(self.best_results.keys())
            
            for row_data in self.average_results:
                row = {
                    "Rank": self.average_results.index(row_data) + 1,
                    "Model": row_data["model"],
                    "Avg R²": row_data["avg_r2"],
                    "Std Dev": row_data["std_r2"],
                    "Best In N Datasets": row_data["count_best"],
                    "Total Datasets": row_data["n_datasets"],
                }
                
                for i, filename in enumerate(filenames):
                    if i < len(row_data["r2_values"]):
                        row[f"{filename}"] = row_data["r2_values"][i]
                    else:
                        row[f"{filename}"] = None
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_excel(path, index=False)
            
            QMessageBox.information(self, "Success", f"Results exported to:\n{path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")
    
    def show_bar_chart(self):
        """Show a bar chart of average R² values with improved readability."""
        if not self.average_results:
            QMessageBox.warning(self, "Warning", "No data to display.")
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            
            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Average R² Bar Chart")
            dialog.setMinimumSize(900, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Create figure with larger size
            fig = Figure(figsize=(10, 6), dpi=120)
            canvas = FigureCanvasQTAgg(fig)
            ax = fig.add_subplot(111)
            
            # Prepare data
            models = [data["model"] for data in self.average_results]
            avg_r2 = [data["avg_r2"] for data in self.average_results]
            
            # Colors based on R²
            colors = []
            for r2 in avg_r2:
                if r2 >= 0.95:
                    colors.append("#27ae60")
                elif r2 >= 0.85:
                    colors.append("#f39c12")
                elif r2 >= 0.70:
                    colors.append("#e67e22")
                else:
                    colors.append("#e74c3c")
            
            # Create bars
            bars = ax.bar(models, avg_r2, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on top of bars with larger font
            for bar, r2 in zip(bars, avg_r2):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{r2:.4f}', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold', rotation=0)
            
            # Add horizontal lines
            ax.axhline(y=0.95, color='#27ae60', linestyle='--', alpha=0.7, linewidth=2, label='Excellent (0.95)')
            ax.axhline(y=0.85, color='#f39c12', linestyle='--', alpha=0.7, linewidth=2, label='Good (0.85)')
            ax.axhline(y=0.70, color='#e67e22', linestyle='--', alpha=0.7, linewidth=2, label='Fair (0.70)')
            
            # Labels and title - Larger fonts
            ax.set_xlabel('Model', fontsize=14, fontweight='bold')
            ax.set_ylabel('Average R²', fontsize=14, fontweight='bold')
            ax.set_title('Model Performance Comparison (Average R² Across All Datasets)', 
                        fontsize=16, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.legend(loc='best', fontsize=11)
            
            # Rotate x labels if many models - with better readability
            if len(models) > 8:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
            else:
                plt.setp(ax.xaxis.get_majorticklabels(), fontsize=11)
            
            ax.grid(True, alpha=0.3, axis='y')
            fig.tight_layout()
            
            layout.addWidget(canvas)
            
            # Close button
            btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            btn_box.rejected.connect(dialog.reject)
            layout.addWidget(btn_box)
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display chart:\n{str(e)}")
    
    def show_details(self):
        """Show detailed results for each dataset."""
        if not self.best_results:
            QMessageBox.warning(self, "Warning", "No data to display.")
            return
        
        try:
            # Import model comparison widget - avoid circular import
            from .model_comparison_widget import ModelComparisonDialog
            
            # Get all results for the first file
            first_file = list(self.best_results.keys())[0]
            if first_file in self.best_results:
                file_results = self.best_results[first_file]["all_results"]
                
                # If no results, try to get from parent
                if not file_results or len(file_results) == 0:
                    if self.parent_window:
                        if hasattr(self.parent_window, 'multi_file_results') and self.parent_window.multi_file_results:
                            if first_file in self.parent_window.multi_file_results:
                                file_results = self.parent_window.multi_file_results[first_file]
                        elif hasattr(self.parent_window, 'fit_results'):
                            file_results = self.parent_window.fit_results
                
                if file_results and len(file_results) > 0:
                    dialog = ModelComparisonDialog(file_results, self.parent_window)
                    dialog.exec()
                else:
                    QMessageBox.warning(self, "Warning", "No results available for this file.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to show details:\n{str(e)}")