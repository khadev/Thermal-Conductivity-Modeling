"""
Fit Statistics Widget - Display fit statistics with visual indicators.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor


class FitStatsWidget(QWidget):
    """Widget for displaying fit statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        # Quality indicator
        self.quality_label = QLabel("No fit data")
        self.quality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quality_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.quality_label)
        
        # Statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.stats_table)
        
        # Progress bar for fit quality
        self.quality_bar = QProgressBar()
        self.quality_bar.setRange(0, 100)
        self.quality_bar.setFormat("Fit Quality: %p%")
        layout.addWidget(self.quality_bar)
    
    def update_stats(self, stats):
        """Update the statistics display."""
        if not stats:
            return
        
        # Quality indicator
        r2 = stats.get('r2', 0)
        if r2 >= 0.95:
            quality = "⭐ Excellent"
            color = "#27ae60"
            bar_value = 100
        elif r2 >= 0.85:
            quality = "✓ Good"
            color = "#f39c12"
            bar_value = 80
        elif r2 >= 0.70:
            quality = "⚠ Fair"
            color = "#e67e22"
            bar_value = 60
        else:
            quality = "✗ Poor"
            color = "#e74c3c"
            bar_value = 30
        
        self.quality_label.setText(f"Fit Quality: {quality} (R² = {r2:.4f})")
        self.quality_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold; padding: 10px;")
        
        # Update quality bar
        self.quality_bar.setValue(bar_value)
        
        # Update statistics table
        metrics = [
            ('R²', r2, '{:.4f}'),
            ('RMSE', stats.get('rmse', 0), '{:.4f}'),
            ('MAE', stats.get('mae', 0), '{:.4f}'),
            ('MAPE (%)', stats.get('mape', 0), '{:.2f}'),
            ('n', stats.get('n', 0), '{:d}'),
            ('DOF', stats.get('dof', 0), '{:d}')
        ]
        
        self.stats_table.setRowCount(len(metrics))
        for i, (name, value, fmt) in enumerate(metrics):
            self.stats_table.setItem(i, 0, QTableWidgetItem(name))
            try:
                self.stats_table.setItem(i, 1, QTableWidgetItem(fmt.format(value)))
            except:
                self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))
        
        self.stats_table.resizeColumnsToContents()
    
    def clear(self):
        """Clear all statistics."""
        self.quality_label.setText("No fit data")
        self.quality_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        self.quality_bar.setValue(0)
        self.stats_table.setRowCount(0)
