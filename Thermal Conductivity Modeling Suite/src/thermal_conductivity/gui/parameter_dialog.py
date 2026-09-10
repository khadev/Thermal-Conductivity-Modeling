"""Parameter input dialog for each thermal conductivity model."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QGroupBox, QScrollArea, QWidget,
    QDoubleSpinBox, QMessageBox, QTabWidget, QCheckBox
)
from PyQt6.QtCore import Qt

from ..core.models import MODEL_REGISTRY


class ModelParameterDialog(QDialog):
    """Dialog to configure parameters for all selected models."""

    def __init__(self, selected_models, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Parameters Configuration")
        self.setMinimumSize(600, 500)
        self.selected_models = selected_models
        self.param_values = {}

        self._build_ui()
        self._load_defaults()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("<h2>Configure Model Parameters</h2>"
                       "<p>Set initial guesses and bounds for fitted parameters. "
                       "Fixed models require no input.</p>")
        header.setWordWrap(True)
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)

        for model_name in self.selected_models:
            meta = MODEL_REGISTRY[model_name]
            group = QGroupBox(model_name)
            group_layout = QFormLayout(group)

            if meta["fixed"]:
                group_layout.addRow(QLabel("<i>Fixed model - no parameters to configure</i>"))
                self.param_values[model_name] = {"p0": None, "bounds": None}
            else:
                params = meta["params"]
                p0_defaults = meta.get("p0", [1.0] * len(params))
                bounds = meta.get("bounds", ([-float('inf')] * len(params), [float('inf')] * len(params)))
                bounds_low = bounds[0]
                bounds_high = bounds[1]

                p0_widgets = []
                low_widgets = []
                high_widgets = []

                for i, param in enumerate(params):
                    # Parameter name with tooltip
                    param_label = QLabel(f"{param}:")
                    param_label.setToolTip(f"Parameter: {param}")
                    group_layout.addRow(param_label)

                    # Initial guess
                    spin_p0 = QDoubleSpinBox()
                    spin_p0.setRange(-1e6, 1e6)
                    spin_p0.setDecimals(6)
                    spin_p0.setValue(float(p0_defaults[i]))
                    spin_p0.setSingleStep(0.1)
                    group_layout.addRow("Initial Guess:", spin_p0)
                    p0_widgets.append(spin_p0)

                    # Lower bound
                    spin_low = QDoubleSpinBox()
                    spin_low.setRange(-1e9, 1e9)
                    spin_low.setDecimals(6)
                    spin_low.setValue(float(bounds_low[i]) if bounds_low[i] != -float('inf') else -1e6)
                    spin_low.setSingleStep(0.1)
                    group_layout.addRow("Lower Bound:", spin_low)
                    low_widgets.append(spin_low)

                    # Upper bound
                    spin_high = QDoubleSpinBox()
                    spin_high.setRange(-1e9, 1e9)
                    spin_high.setDecimals(6)
                    spin_high.setValue(float(bounds_high[i]) if bounds_high[i] != float('inf') else 1e6)
                    spin_high.setSingleStep(0.1)
                    group_layout.addRow("Upper Bound:", spin_high)
                    high_widgets.append(spin_high)

                self.param_values[model_name] = {
                    "p0_widgets": p0_widgets,
                    "low_widgets": low_widgets,
                    "high_widgets": high_widgets,
                    "params": params
                }

            container_layout.addWidget(group)

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        btn_layout = QHBoxLayout()
        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.clicked.connect(self._load_defaults)
        btn_layout.addWidget(btn_reset)

        btn_layout.addStretch()

        btn_ok = QPushButton("Apply & Run")
        btn_ok.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px 20px;")
        btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(btn_ok)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

    def _load_defaults(self):
        for model_name, widgets in self.param_values.items():
            if widgets.get("p0_widgets") is None:
                continue
            meta = MODEL_REGISTRY[model_name]
            p0_defaults = meta.get("p0", [1.0] * len(widgets["params"]))
            bounds = meta.get("bounds", ([-float('inf')] * len(widgets["params"]), [float('inf')] * len(widgets["params"])))

            for i, spin in enumerate(widgets["p0_widgets"]):
                spin.setValue(float(p0_defaults[i]))
            for i, spin in enumerate(widgets["low_widgets"]):
                val = float(bounds[0][i]) if bounds[0][i] != -float('inf') else -1e6
                spin.setValue(val)
            for i, spin in enumerate(widgets["high_widgets"]):
                val = float(bounds[1][i]) if bounds[1][i] != float('inf') else 1e6
                spin.setValue(val)

    def get_custom_params(self):
        custom = {}
        for model_name, widgets in self.param_values.items():
            if widgets.get("p0_widgets") is None:
                continue

            p0 = [spin.value() for spin in widgets["p0_widgets"]]
            low = [spin.value() for spin in widgets["low_widgets"]]
            high = [spin.value() for spin in widgets["high_widgets"]]

            for i, (l, h) in enumerate(zip(low, high)):
                if l >= h:
                    QMessageBox.warning(self, "Invalid Bounds",
                                       f"{model_name}: Lower bound must be less than upper bound for parameter {widgets['params'][i]}.")
                    return None

            custom[model_name] = {"p0": p0, "bounds": (low, high)}

        return custom
