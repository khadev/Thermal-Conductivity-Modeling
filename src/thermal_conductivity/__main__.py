"""Allow running as python -m thermal_conductivity."""

import sys
import traceback

# Add parent directory to path for PyInstaller compatibility
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {traceback.format_exc()}")
        sys.exit(1)
