"""Data loading dialog with mode-aware material selection."""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QComboBox, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QGroupBox, QWidget, QCheckBox, QCompleter
)
from PyQt6.QtCore import Qt, QStringListModel

from ..core.data_loader import load_data
from ..core.material_constants import (
    get_mode_info, get_materials_for_mode,
    get_default_materials, get_k_value,
    get_mode_style
)


class DataLoadDialog(QDialog):
    """Dialog for loading data with mode-aware material selection."""

    def __init__(self, mode="nanothermite", parent=None):
        super().__init__(parent)
        self.mode = mode
        
        mode_info = get_mode_info(mode)
        self.mode_label = mode_info.get("name", "Nanothermite")
        self.mode_icon = mode_info.get("icon", "🔬")
        
        materials = get_materials_for_mode(mode)
        self.matrix_list = materials.get("matrix_list", [])
        self.filler_list = materials.get("filler_list", [])
        self.matrix_label_text = mode_info.get("matrix_label", "Matrix")
        self.filler_label_text = mode_info.get("filler_label", "Filler")
        
        defaults = get_default_materials(mode)
        self.default_matrix = defaults.get("matrix", "Al")
        self.default_filler = defaults.get("filler", "CuO")
        
        self.setWindowTitle(f"Load Experimental Data - {self.mode_label}")
        self.setMinimumSize(600, 500)
        self.setModal(True)

        self.loaded_data = None
        self.file_path = None

        self._build_ui()
        self._update_mode_specific_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Mode indicator
        mode_indicator = QLabel(f"{self.mode_icon} Mode: {self.mode_label}")
        mode_indicator.setStyleSheet(f"""
            font-size: 14px; 
            font-weight: bold; 
            color: {get_mode_style(self.mode)['primary']};
            padding: 8px;
            background-color: {get_mode_style(self.mode)['background']};
            border-radius: 5px;
            border: 2px solid {get_mode_style(self.mode)['primary']};
        """)
        mode_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(mode_indicator)

        # File selection group
        file_group = QGroupBox("Data File")
        file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_layout = QVBoxLayout(file_group)

        file_select_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: #7f8c8d; border: 1px solid #bdc3c7; padding: 8px; border-radius: 4px;")
        self.file_label.setWordWrap(True)
        file_select_layout.addWidget(self.file_label, 1)

        btn_browse = QPushButton("Browse...")
        btn_browse.setStyleSheet(self._btn_style("#3498db"))
        btn_browse.clicked.connect(self.browse_file)
        file_select_layout.addWidget(btn_browse)

        file_layout.addLayout(file_select_layout)

        self.file_info = QLabel("Supported formats: .txt, .csv, .xlsx, .xls<br>"
                               "File must contain: <b>phi</b> (filler volume fraction) and <b>k_meas</b>")
        self.file_info.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        file_layout.addWidget(self.file_info)

        layout.addWidget(file_group)

        # Material selection group
        material_group = QGroupBox("Material Composition")
        material_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        material_layout = QFormLayout(material_group)

        # Matrix selection
        matrix_layout = QHBoxLayout()
        self.matrix_combo = QComboBox()
        self.matrix_combo.setEditable(True)
        self.matrix_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        matrix_items = ["Select or type matrix..."] + self.matrix_list
        self.matrix_combo.addItems(matrix_items)
        self.matrix_combo.setCurrentIndex(0)
        self.matrix_combo.setToolTip(f"Select the {self.matrix_label_text}")
        # Auto-select text when user starts typing
        self.matrix_combo.lineEdit().setPlaceholderText("Type to search or add new")
        matrix_layout.addWidget(self.matrix_combo, 1)

        btn_add_matrix = QPushButton("+")
        btn_add_matrix.setMaximumWidth(30)
        btn_add_matrix.setStyleSheet(self._btn_style("#27ae60"))
        btn_add_matrix.setToolTip("Add current text to matrix list")
        btn_add_matrix.clicked.connect(self.add_matrix)
        matrix_layout.addWidget(btn_add_matrix)

        self.matrix_label = QLabel(f"{self.matrix_label_text}:")
        material_layout.addRow(self.matrix_label, matrix_layout)

        # Filler selection
        filler_layout = QHBoxLayout()
        self.filler_combo = QComboBox()
        self.filler_combo.setEditable(True)
        self.filler_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        filler_items = ["Select or type filler..."] + self.filler_list
        self.filler_combo.addItems(filler_items)
        self.filler_combo.setCurrentIndex(0)
        self.filler_combo.setToolTip(f"Select the {self.filler_label_text}")
        self.filler_combo.lineEdit().setPlaceholderText("Type to search or add new")
        filler_layout.addWidget(self.filler_combo, 1)

        btn_add_filler = QPushButton("+")
        btn_add_filler.setMaximumWidth(30)
        btn_add_filler.setStyleSheet(self._btn_style("#27ae60"))
        btn_add_filler.setToolTip("Add current text to filler list")
        btn_add_filler.clicked.connect(self.add_filler)
        filler_layout.addWidget(btn_add_filler)

        self.filler_label = QLabel(f"{self.filler_label_text}:")
        material_layout.addRow(self.filler_label, filler_layout)

        # Material info
        self.material_info = QLabel("")
        self.material_info.setStyleSheet("color: #27ae60; font-weight: bold;")
        material_layout.addRow("", self.material_info)

        layout.addWidget(material_group)

        # Options group
        options_group = QGroupBox("Options")
        options_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        options_layout = QVBoxLayout(options_group)

        self.auto_plot_check = QCheckBox("Auto-plot after loading")
        self.auto_plot_check.setChecked(True)
        options_layout.addWidget(self.auto_plot_check)

        layout.addWidget(options_group)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_load = QPushButton("Load Data")
        btn_load.setStyleSheet(self._btn_style("#27ae60", bold=True))
        btn_load.setMinimumWidth(120)
        btn_load.clicked.connect(self.load_data)
        btn_layout.addWidget(btn_load)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet(self._btn_style("#e74c3c"))
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

    def _btn_style(self, color, bold=False):
        weight = "bold" if bold else "normal"
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: {weight};
            }}
            QPushButton:hover {{ background-color: {color}dd; }}
            QPushButton:pressed {{ background-color: {color}aa; }}
        """

    def _update_mode_specific_ui(self):
        mode_info = get_mode_info(self.mode)
        self.matrix_label.setText(f"{mode_info.get('matrix_label', 'Matrix')}:")
        self.filler_label.setText(f"{mode_info.get('filler_label', 'Filler')}:")
        self.matrix_label_text = mode_info.get('matrix_label', 'Matrix')
        self.filler_label_text = mode_info.get('filler_label', 'Filler')

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "",
            "Data Files (*.txt *.csv *.xlsx *.xls);;All Files (*.*)"
        )
        if path:
            filename = os.path.basename(path)
            skip_files = ['readme.md', 'readme.txt', 'requirements.txt', 'app.ico']
            if filename.lower() in skip_files:
                QMessageBox.warning(self, "Warning", f"'{filename}' is not a valid data file.")
                return
            self.file_path = path
            self.file_label.setText(path)
            self.file_label.setStyleSheet("color: #2c3e50; border: 1px solid #27ae60; padding: 8px; border-radius: 4px;")
            try:
                self.loaded_data = load_data(path, mode=self.mode)
                self.file_info.setText(f"Loaded: {len(self.loaded_data)} data points")
                self.file_info.setStyleSheet("color: #27ae60; font-size: 10px;")
            except Exception as e:
                self.file_info.setText(f"Error: {str(e)}")
                self.file_info.setStyleSheet("color: #e74c3c; font-size: 10px;")

    def add_matrix(self):
        text = self.matrix_combo.currentText().strip()
        if text and text not in self.matrix_list and text != "Select or type matrix...":
            self.matrix_list.append(text)
            current = self.matrix_combo.currentText()
            self.matrix_combo.clear()
            self.matrix_combo.addItems(["Select or type matrix..."] + sorted(self.matrix_list))
            index = self.matrix_combo.findText(current)
            if index >= 0:
                self.matrix_combo.setCurrentIndex(index)
            else:
                self.matrix_combo.setCurrentIndex(0)
            QMessageBox.information(self, "Added", f"Added '{text}' to matrix list")
        else:
            QMessageBox.information(self, "Info", "Text is empty, already in list, or is the placeholder.")

    def add_filler(self):
        text = self.filler_combo.currentText().strip()
        if text and text not in self.filler_list and text != "Select or type filler...":
            self.filler_list.append(text)
            current = self.filler_combo.currentText()
            self.filler_combo.clear()
            self.filler_combo.addItems(["Select or type filler..."] + sorted(self.filler_list))
            index = self.filler_combo.findText(current)
            if index >= 0:
                self.filler_combo.setCurrentIndex(index)
            else:
                self.filler_combo.setCurrentIndex(0)
            QMessageBox.information(self, "Added", f"Added '{text}' to filler list")
        else:
            QMessageBox.information(self, "Info", "Text is empty, already in list, or is the placeholder.")

    def get_material_info(self):
        matrix = self.matrix_combo.currentText().strip()
        filler = self.filler_combo.currentText().strip()
        if matrix and filler and matrix != "Select or type matrix..." and filler != "Select or type filler...":
            return f"{matrix}/{filler}"
        elif matrix and matrix != "Select or type matrix...":
            return matrix
        elif filler and filler != "Select or type filler...":
            return filler
        else:
            if self.mode == "nanothermite":
                return "Al/CuO"
            else:
                return "Epoxy/Graphene"

    def load_data(self):
        if not self.file_path:
            QMessageBox.warning(self, "Warning", "Please select a data file first.")
            return
        try:
            self.loaded_data = load_data(self.file_path, mode=self.mode)
            matrix = self.matrix_combo.currentText().strip()
            filler = self.filler_combo.currentText().strip()
            material = self.get_material_info()
            if matrix and matrix != "Select or type matrix...":
                k_matrix = get_k_value(self.mode, "matrix", matrix)
            else:
                defaults = get_default_materials(self.mode)
                matrix = defaults['matrix']
                k_matrix = get_k_value(self.mode, "matrix", matrix)
            if filler and filler != "Select or type filler...":
                k_filler = get_k_value(self.mode, "filler", filler)
            else:
                defaults = get_default_materials(self.mode)
                filler = defaults['filler']
                k_filler = get_k_value(self.mode, "filler", filler)
            self.loaded_data.attrs['mode'] = self.mode
            self.loaded_data.attrs['matrix'] = matrix
            self.loaded_data.attrs['filler'] = filler
            self.loaded_data.attrs['material'] = material
            self.loaded_data.attrs['k_matrix'] = k_matrix
            self.loaded_data.attrs['k_filler'] = k_filler
            QMessageBox.information(self, "Success",
                f"Data loaded successfully!\n\n"
                f"File: {self.file_path}\n"
                f"Data points: {len(self.loaded_data)}\n"
                f"Material: {material}\n"
                f"k_matrix = {k_matrix:.2f} W/m·K\n"
                f"k_filler = {k_filler:.2f} W/m·K\n\n"
                f"φ = Filler volume fraction, (1-φ) = Matrix volume fraction")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n\n{str(e)}")

    def get_data(self):
        if self.loaded_data is not None:
            return {
                'data': self.loaded_data,
                'mode': self.mode,
                'matrix': self.loaded_data.attrs.get('matrix', self.default_matrix),
                'filler': self.loaded_data.attrs.get('filler', self.default_filler),
                'material': self.loaded_data.attrs.get('material', self.get_material_info()),
                'file_path': self.file_path,
                'k_matrix': self.loaded_data.attrs.get('k_matrix', 0.2),
                'k_filler': self.loaded_data.attrs.get('k_filler', 300.0)
            }
        return None
