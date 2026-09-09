"""
Sensitivity Analyzer - Analyze parameter sensitivity and uncertainty.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


class SensitivityAnalyzer:
    """Analyze parameter sensitivity and uncertainty in model fitting."""
    
    def __init__(self, model_func, xdata, ydata, param_names, param_values, param_errors):
        self.model_func = model_func
        self.xdata = np.asarray(xdata)
        self.ydata = np.asarray(ydata)
        self.param_names = param_names
        self.param_values = param_values
        self.param_errors = param_errors
        self.n_params = len(param_names)
    
    def compute_sensitivity(self, variation_scale=0.1):
        """
        Compute parameter sensitivity using finite difference.
        
        Returns:
            dict: Sensitivity metrics for each parameter
        """
        sensitivity = {}
        base_y = self.model_func(self.xdata, *self.param_values)
        base_rmse = np.sqrt(np.mean((self.ydata - base_y)**2))
        
        for i, param_name in enumerate(self.param_names):
            variations = []
            for factor in [0.5, 0.7, 0.9, 0.95, 1.0, 1.05, 1.1, 1.3, 1.5, 2.0]:
                param_values_mod = self.param_values.copy()
                param_values_mod[i] = self.param_values[i] * factor
                
                try:
                    y_mod = self.model_func(self.xdata, *param_values_mod)
                    rmse = np.sqrt(np.mean((self.ydata - y_mod)**2))
                    variations.append({
                        'factor': factor,
                        'value': param_values_mod[i],
                        'rmse': rmse,
                        'change': (rmse - base_rmse) / base_rmse * 100 if base_rmse > 0 else 0
                    })
                except:
                    pass
            
            # Calculate sensitivity metrics
            if len(variations) > 1:
                factors = [v['factor'] for v in variations]
                rmse_values = [v['rmse'] for v in variations]
                
                # Calculate derivative at optimal point
                derivative = np.gradient(rmse_values, factors)[0] if len(factors) > 1 else 0
                
                sensitivity[param_name] = {
                    'base_value': self.param_values[i],
                    'base_rmse': base_rmse,
                    'variations': variations,
                    'derivative': derivative,
                    'sensitivity_score': abs(derivative) * self.param_values[i] / base_rmse if base_rmse > 0 else 0
                }
        
        return sensitivity
    
    def compute_uncertainty_propagation(self, n_samples=1000):
        """
        Compute uncertainty propagation using Monte Carlo simulation.
        
        Returns:
            dict: Uncertainty statistics for predictions
        """
        # Generate parameter samples assuming normal distribution
        samples = []
        for i in range(self.n_params):
            if self.param_errors[i] > 0:
                samples.append(np.random.normal(
                    self.param_values[i], 
                    self.param_errors[i], 
                    n_samples
                ))
            else:
                samples.append(np.full(n_samples, self.param_values[i]))
        
        # Generate predictions for each sample
        predictions = []
        for j in range(n_samples):
            param_sample = [samples[i][j] for i in range(self.n_params)]
            try:
                y_pred = self.model_func(self.xdata, *param_sample)
                predictions.append(y_pred)
            except:
                predictions.append(np.full_like(self.xdata, np.nan))
        
        predictions = np.array(predictions)
        
        # Calculate uncertainty statistics
        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'lower_68': np.percentile(predictions, 16, axis=0),
            'upper_68': np.percentile(predictions, 84, axis=0),
            'lower_95': np.percentile(predictions, 2.5, axis=0),
            'upper_95': np.percentile(predictions, 97.5, axis=0),
            'samples': samples
        }
    
    def compute_correlation_sensitivity(self, param_ranges=None, n_points=20):
        """
        Compute 2D sensitivity by varying two parameters simultaneously.
        
        Returns:
            dict: 2D sensitivity matrix
        """
        if param_ranges is None:
            param_ranges = {}
            for i, param_name in enumerate(self.param_names):
                param_ranges[param_name] = (
                    self.param_values[i] * 0.5,
                    self.param_values[i] * 1.5
                )
        
        # Compute 2D sensitivity for each pair of parameters
        results = {}
        for i in range(self.n_params):
            for j in range(i+1, self.n_params):
                param1 = self.param_names[i]
                param2 = self.param_names[j]
                
                # Create grid
                p1_values = np.linspace(param_ranges[param1][0], param_ranges[param1][1], n_points)
                p2_values = np.linspace(param_ranges[param2][0], param_ranges[param2][1], n_points)
                grid = np.zeros((n_points, n_points))
                
                for idx1, p1_val in enumerate(p1_values):
                    for idx2, p2_val in enumerate(p2_values):
                        param_vals = self.param_values.copy()
                        param_vals[i] = p1_val
                        param_vals[j] = p2_val
                        try:
                            y_pred = self.model_func(self.xdata, *param_vals)
                            rmse = np.sqrt(np.mean((self.ydata - y_pred)**2))
                            grid[idx1, idx2] = rmse
                        except:
                            grid[idx1, idx2] = np.nan
                
                results[f'{param1}_{param2}'] = {
                    'param1': param1,
                    'param2': param2,
                    'param1_values': p1_values,
                    'param2_values': p2_values,
                    'grid': grid
                }
        
        return results
    
    def print_sensitivity_summary(self, sensitivity):
        """Print a summary of sensitivity analysis."""
        print("\n" + "="*60)
        print("PARAMETER SENSITIVITY SUMMARY")
        print("="*60)
        print(f"{'Parameter':<20} {'Base Value':<12} {'Sensitivity Score':<15} {'Impact'}")
        print("-"*60)
        
        for param_name, data in sensitivity.items():
            score = data.get('sensitivity_score', 0)
            if score > 0.5:
                impact = "HIGH 🔴"
            elif score > 0.2:
                impact = "MEDIUM 🟡"
            else:
                impact = "LOW 🟢"
            
            print(f"{param_name:<20} {data['base_value']:<12.4f} {score:<15.4f} {impact}")
        print("="*60)
