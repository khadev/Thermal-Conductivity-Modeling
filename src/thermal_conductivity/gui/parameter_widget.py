"""
Parameter Widget - Interactive parameter control for Smart Fit.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSlider, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class ParameterWidget(QWidget):
    """Widget for controlling model parameters."""
    
    parameter_changed = pyqtSignal(str, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.param_widgets = {}
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Parameter Controls")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Parameter table
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(4)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Lock", "Min/Max"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.param_table)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.auto_button = QPushButton("Auto-Initialize")
        self.auto_button.clicked.connect(self.auto_initialize)
        btn_layout.addWidget(self.auto_button)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        btn_layout.addWidget(self.reset_button)
        
        layout.addLayout(btn_layout)
    
    def set_parameters(self, param_names, default_values, bounds=None, param_errors=None):
        """Set the parameters to display."""
        self.param_widgets = {}
        
        self.param_table.setRowCount(len(param_names))
        
        for i, param_name in enumerate(param_names):
            # Parameter name
            self.param_table.setItem(i, 0, QTableWidgetItem(param_name))
            
            # Value with spinbox
            value_widget = QDoubleSpinBox()
            value_widget.setRange(-1e6, 1e6)
            value_widget.setDecimals(6)
            value_widget.setSingleStep(0.1)
            if default_values and i < len(default_values):
                value_widget.setValue(default_values[i])
            value_widget.valueChanged.connect(lambda val, p=param_name: self._on_value_changed(p, val))
            self.param_table.setCellWidget(i, 1, value_widget)
            self.param_widgets[param_name] = value_widget
            
            # Lock checkbox
            lock_widget = QCheckBox()
            lock_widget.setChecked(False)
            self.param_table.setCellWidget(i, 2, lock_widget)
            
            # Min/Max display
            if bounds and i < len(bounds):
                min_val, max_val = bounds[i]
                label = QLabel(f"{min_val:.2f} - {max_val:.2f}")
                self.param_table.setCellWidget(i, 3, label)
            else:
                self.param_table.setItem(i, 3, QTableWidgetItem("unbounded"))
        
        self.param_table.resizeColumnsToContents()
    
    def _on_value_changed(self, param_name, value):
        """Handle parameter value change."""
        self.parameter_changed.emit(param_name, value)
    
    def get_parameter_values(self):
        """Get current parameter values."""
        values = []
        for i in range(self.param_table.rowCount()):
            widget = self.param_table.cellWidget(i, 1)
            if widget:
                values.append(widget.value())
        return values
    
    def get_locked_parameters(self):
        """Get list of locked parameters."""
        locked = []
        for i in range(self.param_table.rowCount()):
            widget = self.param_table.cellWidget(i, 2)
            if widget and widget.isChecked():
                item = self.param_table.item(i, 0)
                if item:
                    locked.append(item.text())
        return locked
    
    def auto_initialize(self):
        """Auto-initialize parameters."""
        # This will be implemented by the parent
        pass
    
    def reset_to_defaults(self):
        """Reset parameters to defaults."""
        # This will be implemented by the parent
        pass
