"""Application entry point with splash screen and error handling."""

import sys
import os
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon

# Ensure we can find the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import using absolute paths
from thermal_conductivity.gui.mode_selector import ModeSelectorDialog  # NEW
from thermal_conductivity.gui.main_window import MainWindow
from thermal_conductivity.gui.splash_screen import SplashScreen


def excepthook(exc_type, exc_value, exc_tb):
    """Global exception handler to catch and log uncaught exceptions."""
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"Uncaught exception: {tb}")
    
    try:
        with open("error_log.txt", "w", encoding='utf-8') as f:
            f.write(f"=== ERROR LOG ===\n")
            f.write(f"Exception Type: {exc_type}\n")
            f.write(f"Exception Value: {exc_value}\n")
            f.write(f"Traceback:\n{tb}")
        print(f"Error log written to error_log.txt")
    except Exception as e:
        print(f"Could not write error log: {e}")
    
    try:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Application Error")
        msg.setText(f"An unexpected error occurred:\n\n{str(exc_value)}")
        msg.setDetailedText(tb)
        msg.exec()
    except:
        pass
    
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def get_icon_path():
    """Get the icon path for the application."""
    if os.path.exists("app.ico"):
        return "app.ico"
    
    parent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "app.ico")
    if os.path.exists(parent_path):
        return parent_path
    
    exe_path = os.path.join(os.path.dirname(sys.executable), "app.ico")
    if os.path.exists(exe_path):
        return exe_path
    
    return None


def main():
    """Main application entry point."""
    sys.excepthook = excepthook
    
    app = QApplication(sys.argv)
    app.setApplicationName("Thermal Conductivity Modeling Suite")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("Oukil Khaled ibn El-walid Research")
    
    icon_path = get_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
        print(f"Icon loaded: {icon_path}")
    else:
        print("No icon found")

    # ===== NEW: Show Mode Selector First =====
    mode_selector = ModeSelectorDialog()
    
    def on_mode_selected(mode):
        """Handle mode selection from the mode selector dialog."""
        print(f"Mode selected: {mode}")
        
        # Create and show splash screen for loading
        splash = SplashScreen()
        splash.show()
        
        # Start loading sequence
        def load_step(step):
            mode_name = "Nanothermite" if mode == "nanothermite" else "Polymer Composite"
            messages = [
                f"Initializing {mode_name} components...",
                f"Loading thermal conductivity models for {mode_name}...",
                "Preparing visualization engine...",
                "Setting up export modules...",
                "Ready to launch!"
            ]
            progress = (step + 1) * 20
            splash.update_progress(progress, messages[step] if step < len(messages) else "")

            if step >= 4:
                try:
                    window = MainWindow(mode=mode)
                    window.show()
                    splash.finish_splash(window)
                except Exception as e:
                    splash.hide()
                    QMessageBox.critical(
                        None, 
                        "Fatal Error", 
                        f"Failed to start application:\n\n{str(e)}\n\n{traceback.format_exc()}"
                    )
                    sys.exit(1)
            else:
                QTimer.singleShot(400, lambda: load_step(step + 1))

        QTimer.singleShot(300, lambda: load_step(0))
    
    mode_selector.mode_selected.connect(on_mode_selected)
    mode_selector.exec()  # Show mode selector dialog

    try:
        sys.exit(app.exec())
    except Exception as e:
        print(f"Fatal application error: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()