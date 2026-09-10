"""GUI module."""
from .main_window import MainWindow
from .data_dialog import DataLoadDialog
from .parameter_dialog import ModelParameterDialog
from .about_dialog import AboutDialog
from .splash_screen import SplashScreen
from .smart_fit_dialog import SmartFitDialog
from .model_comparison_widget import ModelComparisonDialog
from .model_discovery_dialog import ModelDiscoveryDialog
from .best_r2_tab import BestR2Tab
from .generate_data_dialog import GenerateDataDialog

__all__ = [
    'MainWindow',
    'DataLoadDialog',
    'ModelParameterDialog',
    'AboutDialog',
    'SplashScreen',
    'SmartFitDialog',
    'ModelComparisonDialog',
    'ModelDiscoveryDialog',
    'BestR2Tab',
    'GenerateDataDialog'
]