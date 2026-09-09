"""
Fit Visualizer - Visualization functions for Smart Fit results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class FitVisualizer:
    """Visualization tools for Smart Fit results."""
    
    def __init__(self, canvas):
        self.canvas = canvas
        self.ax = canvas.axes
    
    def plot_fit(self, phi, k_meas, k_pred, k_std=None, model_name="Model", title=None):
        """Plot experimental data with fitted curve."""
        self.ax.clear()
        
        # Plot experimental data
        if k_std is not None and not np.all(np.isnan(k_std)):
            self.ax.errorbar(phi, k_meas, yerr=k_std, fmt='o', color='black',
                           markersize=6, capsize=4, label='Experimental')
        else:
            self.ax.scatter(phi, k_meas, color='black', s=50, label='Experimental')
        
        # Plot predicted curve
        phi_smooth = np.linspace(phi.min(), phi.max(), 200)
        # Need to interpolate k_pred for smooth curve
        if len(k_pred) == len(phi):
            from scipy.interpolate import interp1d
            try:
                interp_func = interp1d(phi, k_pred, kind='cubic', fill_value='extrapolate')
                k_smooth = interp_func(phi_smooth)
            except:
                k_smooth = np.interp(phi_smooth, phi, k_pred)
            self.ax.plot(phi_smooth, k_smooth, color='#1a5490', linewidth=2, label='Predicted')
        else:
            # If k_pred is already on a smooth grid
            self.ax.plot(phi_smooth, k_pred, color='#1a5490', linewidth=2, label='Predicted')
        
        self.ax.set_xlabel('Volume Fraction, $\\phi$')
        self.ax.set_ylabel('Thermal Conductivity, $k$ (W/m·K)')
        self.ax.set_title(title or f'{model_name} - Fit Result')
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def plot_residuals(self, phi, residuals, title="Residuals"):
        """Plot residuals."""
        self.ax.clear()
        
        self.ax.scatter(phi, residuals, color='#2c3e50', s=50, alpha=0.7)
        self.ax.axhline(0, color='black', linewidth=1, linestyle='--')
        self.ax.set_xlabel('Volume Fraction, $\\phi$')
        self.ax.set_ylabel('Residuals (W/m·K)')
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        
        # Add horizontal lines at ±std
        std_residuals = np.std(residuals)
        if std_residuals > 0:
            self.ax.axhline(std_residuals, color='red', linewidth=0.5, linestyle=':', alpha=0.5)
            self.ax.axhline(-std_residuals, color='red', linewidth=0.5, linestyle=':', alpha=0.5)
        
        self.canvas.draw()
    
    def plot_parameter_sensitivity(self, sensitivity):
        """Plot parameter sensitivity analysis."""
        self.ax.clear()
        
        if not sensitivity or 'message' in sensitivity:
            self.ax.text(0.5, 0.5, 'No sensitivity data available', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        # Check if it's a single parameter or multiple
        if len(sensitivity) == 1:
            # Single parameter - plot RMSE vs factor
            param_name, data = next(iter(sensitivity.items()))
            variations = data.get('variations', [])
            if variations:
                factors = [v['factor'] for v in variations]
                rmse_values = [v['rmse'] for v in variations]
                
                self.ax.plot(factors, rmse_values, marker='o', color='#1a5490', linewidth=2)
                self.ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='Optimal')
                self.ax.set_xlabel(f'{param_name} Variation Factor')
                self.ax.set_ylabel('RMSE (W/m·K)')
                self.ax.set_title(f'Sensitivity: {param_name}')
                self.ax.legend()
                self.ax.grid(True, alpha=0.3)
        else:
            # Multiple parameters - plot all
            colors = ['#1a5490', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad', '#1abc9c']
            for idx, (param_name, data) in enumerate(sensitivity.items()):
                variations = data.get('variations', [])
                if variations:
                    factors = [v['factor'] for v in variations]
                    rmse_values = [v['rmse'] for v in variations]
                    color = colors[idx % len(colors)]
                    self.ax.plot(factors, rmse_values, marker='o', color=color, 
                                linewidth=2, label=param_name)
            
            self.ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='Optimal')
            self.ax.set_xlabel('Parameter Variation Factor')
            self.ax.set_ylabel('RMSE (W/m·K)')
            self.ax.set_title('Parameter Sensitivity Analysis')
            self.ax.legend(loc='best')
            self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def plot_uncertainty(self, phi, k_meas, uncertainty, title="Uncertainty Analysis"):
        """Plot prediction with uncertainty bands."""
        self.ax.clear()
        
        # Plot experimental data
        self.ax.scatter(phi, k_meas, color='black', s=50, label='Experimental')
        
        # Plot mean prediction
        mean = uncertainty.get('mean', [])
        lower_68 = uncertainty.get('lower_68', [])
        upper_68 = uncertainty.get('upper_68', [])
        lower_95 = uncertainty.get('lower_95', [])
        upper_95 = uncertainty.get('upper_95', [])
        
        if len(mean) == len(phi):
            self.ax.plot(phi, mean, color='#1a5490', linewidth=2, label='Mean')
            self.ax.fill_between(phi, lower_68, upper_68, alpha=0.3, color='#1a5490', label='68% CI')
            self.ax.fill_between(phi, lower_95, upper_95, alpha=0.15, color='#1a5490', label='95% CI')
        
        self.ax.set_xlabel('Volume Fraction, $\\phi$')
        self.ax.set_ylabel('Thermal Conductivity, $k$ (W/m·K)')
        self.ax.set_title(title)
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def plot_comparison(self, all_results):
        """Plot comparison of multiple models."""
        self.ax.clear()
        
        colors = ['#1a5490', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad', '#1abc9c', '#2c3e50', '#e67e22']
        
        for idx, (model_name, result) in enumerate(all_results.items()):
            if not result.get('success', False):
                continue
            
            phi = result.get('phi', [])
            k_pred = result.get('y_pred', [])
            
            if len(phi) > 0 and len(k_pred) == len(phi):
                color = colors[idx % len(colors)]
                self.ax.plot(phi, k_pred, color=color, linewidth=2, 
                           label=f"{model_name} (R²={result.get('stats', {}).get('r2', 0):.3f})")
        
        self.ax.set_xlabel('Volume Fraction, $\\phi$')
        self.ax.set_ylabel('Thermal Conductivity, $k$ (W/m·K)')
        self.ax.set_title('Model Comparison')
        self.ax.legend(loc='best', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
