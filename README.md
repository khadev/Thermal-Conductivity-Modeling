# 🔬 Thermal Conductivity Modeling Suite v2.0

Professional desktop application for experimental data analysis and modeling of thermal conductivity of composite materials.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

## 📊 Overview

The **Thermal Conductivity Modeling Suite** is a professional desktop application designed for researchers, scientists, and engineers working with composite materials. It provides advanced tools for:

- **Experimental data analysis** of thermal conductivity measurements
- **Model fitting** with 8+ thermal conductivity models
- **Publication-ready plots** with journal presets
- **Auto-discovery** of new mathematical models
- **Multi-file analysis** for batch processing
- **Professional PDF report generation**

## ✨ New in v2.0

- 🔍 **Auto-Discover New Models** - Automatically find mathematical models that fit your data
- 🔄 **Dual Mode Support** - Switch seamlessly between Nanothermite and Polymer Composite modes
- 🎨 **Publication-Ready Plots** - Journal presets (Thermochimica Acta, IJHMT, Nature, Colorblind)
- 📊 **Best R² Tab** - Compare model performance across multiple datasets
- 🧠 **Smart Fit** - Interactive model fitting with parameter optimization
- 💾 **Save Graph** - One-click export to PNG, PDF, SVG, TIFF (600 DPI LZW)

## 🚀 Features

### 📐 Thermal Conductivity Models
- **Basic Models**: Parallel, Series, Maxwell-Eucken, Bruggeman, Agari, Lewis-Nielsen, Percolation, Agari-Percolation
- **Classic Models**: Russell, Cheng-Vachon, Baschirow-Selenew, Hamilton-Crosser, EMT Equation, Geometric Mean, Böttcher, De Loor, Ce Wen Nan
- **Advanced Models**: Rayleigh, Halpin-Tsai, Hatta, Modified Hashin-Shtrikman, Tsao, Hamilton-Crosser Extended, Maxwell-Eucken Upper/Lower, GEM Equation
- **Auto-Discovered Models**: AI-driven model discovery from your experimental data

### 🔄 Dual Mode Support
| Feature | Nanothermite Mode 🔥 | Polymer Mode 🧪 |
|---------|---------------------|-----------------|
| Matrix | Reductant (Al, Mg, Ti, etc.) | Polymer Matrix (Epoxy, PE, PP, etc.) |
| Filler | Oxidizer (CuO, Fe₂O₃, etc.) | Conductive Filler (Graphene, CNT, etc.) |
| Trend | DECREASING with filler | INCREASING with filler |
| k_matrix | ~237 W/m·K (Al) | ~0.2 W/m·K (Epoxy) |
| k_filler | ~33 W/m·K (CuO) | ~300 W/m·K (Graphene) |

### 📊 Data Management
- Load data from TXT, CSV, XLSX files
- Multiple file support with batch analysis
- Manual data entry and editing
- Data validation and preprocessing

### 📈 Visualization
- Publication-ready plots with 8+ journal styles
- Interactive plot navigation (zoom, pan, save)
- Residual plots with statistical analysis
- Model comparison visualization

### 📝 Reporting
- Professional PDF report generation
- Export to Excel, CSV
- High-quality image export (PNG, PDF, SVG, TIFF)

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Option 1: From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/thermal-conductivity-modeling.git
cd thermal-conductivity-modeling

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
