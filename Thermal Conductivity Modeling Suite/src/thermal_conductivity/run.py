"""Entry point for PyInstaller executable."""
import sys
import os

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Now import using absolute paths
from thermal_conductivity.main import main

if __name__ == "__main__":
    main()
