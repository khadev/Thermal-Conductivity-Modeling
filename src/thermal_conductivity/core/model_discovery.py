"""
Model Discovery - Automatically find new models that fit experimental data.
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings
import re

from ..core.statistics import calculate_statistics


class ModelDiscovery:
    """Discover new models that fit experimental data."""
    
    def __init__(self, phi_data, k_data, k_red, k_oxy):
        """
        Initialize model discovery.
        
        Parameters:
        -----------
        phi_data : array
            Reductant volume fraction (φ)
        k_data : array
            Thermal conductivity data
        k_red : float
            Thermal conductivity of reductant (matrix)
        k_oxy : float
            Thermal conductivity of oxidizer (filler)
        """
        self.phi = np.asarray(phi_data)
        self.k = np.asarray(k_data)
        self.k_red = k_red
        self.k_oxy = k_oxy
        
        # Store discovered models
        self.discovered_models = []
        
    def discover_models(self, target_r2_min=0.7, target_r2_max=1.0, max_models=15):
        """
        Discover new models that fit the data well.
        
        Parameters:
        -----------
        target_r2_min : float
            Minimum R² for models to be included (default 0.7)
        target_r2_max : float
            Maximum R² for models to be included (default 1.0)
        max_models : int
            Maximum number of models to return
        """
        discovered = []
        model_families = self._generate_model_families()
        
        for family in model_families:
            try:
                result = self._test_model_family(family)
                if result and result['r2'] >= target_r2_min and result['r2'] <= target_r2_max:
                    discovered.append(result)
                    if len(discovered) >= max_models:
                        break
            except Exception as e:
                continue
        
        # Sort by R² descending
        discovered.sort(key=lambda x: x['r2'], reverse=True)
        self.discovered_models = discovered
        return discovered
    
    def _generate_model_families(self):
        """Generate candidate model families with safe exponential handling."""
        models = []
        
        # ===== Polynomial Models =====
        models.append({
            'name': 'Quadratic Polynomial',
            'expr': 'a + b*phi + c*phi**2',
            'params': ['a', 'b', 'c'],
            'func': lambda phi, a, b, c: a + b*phi + c*phi**2
        })
        
        models.append({
            'name': 'Cubic Polynomial',
            'expr': 'a + b*phi + c*phi**2 + d*phi**3',
            'params': ['a', 'b', 'c', 'd'],
            'func': lambda phi, a, b, c, d: a + b*phi + c*phi**2 + d*phi**3
        })
        
        # ===== Exponential Models =====
        models.append({
            'name': 'Exponential Decay',
            'expr': 'a*exp(-b*phi) + c',
            'params': ['a', 'b', 'c'],
            'func': lambda phi, a, b, c: a * np.exp(-b * phi) + c
        })
        
        models.append({
            'name': 'Double Exponential',
            'expr': 'a*exp(-b*phi) + c*exp(-d*phi)',
            'params': ['a', 'b', 'c', 'd'],
            'func': lambda phi, a, b, c, d: a * np.exp(-b * phi) + c * np.exp(-d * phi)
        })
        
        # ===== Power Law Models =====
        models.append({
            'name': 'Power Law',
            'expr': 'a*phi**b + c',
            'params': ['a', 'b', 'c'],
            'func': lambda phi, a, b, c: a * phi**b + c
        })
        
        # ===== Rational Models =====
        models.append({
            'name': 'Rational (linear/linear)',
            'expr': '(a + b*phi) / (1 + c*phi)',
            'params': ['a', 'b', 'c'],
            'func': lambda phi, a, b, c: (a + b * phi) / (1 + c * phi)
        })
        
        models.append({
            'name': 'Rational (quadratic/linear)',
            'expr': '(a + b*phi + c*phi**2) / (1 + d*phi)',
            'params': ['a', 'b', 'c', 'd'],
            'func': lambda phi, a, b, c, d: (a + b*phi + c*phi**2) / (1 + d*phi)
        })
        
        # ===== Asymptotic Models =====
        models.append({
            'name': 'Asymptotic Growth',
            'expr': 'a*(1 - exp(-b*phi)) + c',
            'params': ['a', 'b', 'c'],
            'func': lambda phi, a, b, c: a * (1 - np.exp(-b * phi)) + c
        })
        
        # ===== Hybrid Models =====
        models.append({
            'name': 'Hybrid Rational-Exponential',
            'expr': '(a + b*phi) / (c + d*phi) + e*exp(-f*phi)',
            'params': ['a', 'b', 'c', 'd', 'e', 'f'],
            'func': lambda phi, a, b, c, d, e, f: (a + b*phi) / (c + d*phi) + e * np.exp(-f * phi)
        })
        
        # ===== Physical Bounds Models =====
        models.append({
            'name': 'Modified Rule of Mixtures',
            'expr': 'k_red*(1-phi) + k_oxy*phi + a*phi*(1-phi)',
            'params': ['a'],
            'func': lambda phi, a: self.k_red*(1-phi) + self.k_oxy*phi + a*phi*(1-phi)
        })
        
        return models
    
    def _test_model_family(self, family):
        """Test a model family and return fit results with better error handling."""
        try:
            func = family['func']
            param_names = family['params']
            
            # Better initial guesses
            p0 = []
            bounds_low = []
            bounds_high = []
            
            for name in param_names:
                if name in ['a', 'c', 'e']:
                    p0.append(1.0)
                    bounds_low.append(-1000.0)
                    bounds_high.append(1000.0)
                elif name in ['b', 'd', 'f']:
                    p0.append(1.0)
                    bounds_low.append(0.0)
                    bounds_high.append(10.0)
                else:
                    p0.append(1.0)
                    bounds_low.append(-1000.0)
                    bounds_high.append(1000.0)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                
                try:
                    popt, pcov = curve_fit(func, self.phi, self.k, p0=p0, 
                                           bounds=(bounds_low, bounds_high),
                                           maxfev=10000)
                except:
                    popt, pcov = curve_fit(func, self.phi, self.k, p0=p0, maxfev=10000)
            
            if np.any(np.isnan(popt)) or np.any(np.isinf(popt)):
                return None
            
            y_pred = func(self.phi, *popt)
            
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return None
            
            stats = calculate_statistics(self.k, y_pred)
            
            eq_str = self._generate_equation_string(family['expr'], param_names, popt)
            
            return {
                'name': family['name'],
                'expr': family['expr'],
                'params': dict(zip(param_names, popt)),
                'param_names': param_names,
                'param_values': popt,
                'equation': eq_str,
                'function_string': self._generate_function_string(family['expr'], param_names),
                'r2': stats['r2'],
                'rmse': stats['rmse'],
                'mae': stats['mae'],
                'mape': stats['mape'],
                'y_pred': y_pred,
                'success': True,
                'message': 'Model discovered successfully'
            }
            
        except Exception as e:
            return None
    
    def _generate_equation_string(self, expr, param_names, param_values):
        """Generate a readable equation string with parameter values - FIXED."""
        try:
            eq = expr
            for name, value in zip(param_names, param_values):
                if abs(value) > 1000:
                    val_str = f"{value:.0f}"
                elif abs(value) > 100:
                    val_str = f"{value:.1f}"
                elif abs(value) > 1:
                    val_str = f"{value:.3f}"
                else:
                    val_str = f"{value:.4f}"
                eq = eq.replace(name, val_str)
            
            # Clean up
            eq = eq.replace('**', '^')
            eq = eq.replace('*', '·')
            eq = eq.replace('phi', 'φ')
            eq = eq.replace('exp', 'e')
            
            # Fix double operators
            eq = eq.replace('++', '+')
            eq = eq.replace('--', '+')
            eq = eq.replace('+-', '-')
            eq = eq.replace('-+', '-')
            
            # Remove zero coefficients
            eq = re.sub(r'0\.0+·\s*[^+\s]+', '', eq)
            eq = re.sub(r'0\.0+·\s*e\^\([^)]+\)', '', eq)
            
            # Clean up
            eq = eq.replace('  ', ' ')
            eq = eq.replace('· ', '·')
            eq = eq.replace(' ·', '·')
            
            # Remove trailing operators
            eq = re.sub(r'\s*\+\s*$', '', eq)
            eq = re.sub(r'\s*-\s*$', '', eq)
            eq = re.sub(r'\s*·\s*$', '', eq)
            
            return eq
        except Exception:
            return expr.replace('*', '·').replace('phi', 'φ')
    
    def _generate_function_string(self, expr, param_names):
        """Generate a Python function string for the discovered model."""
        param_str = ', '.join(param_names)
        return f"def new_model(phi, {param_str}):\n    return {expr}"
    
    def add_to_registry(self, model_name, model_data):
        """
        Create a model entry for the registry.
        
        Returns:
            dict: Model registry entry
        """
        expr = model_data['expr']
        param_names = model_data['param_names']
        param_values = model_data['param_values']
        
        # Create the function
        def model_func(phi_input, *params):
            phi_red = 1.0 - np.asarray(phi_input)
            
            local_vars = {'phi': phi_red, 'np': np, 'exp': np.exp}
            for name, value in zip(param_names, params):
                local_vars[name] = value
            try:
                result = eval(expr, {"__builtins__": {}}, local_vars)
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    return np.full_like(phi_input, np.nan)
                result = np.clip(result, 0, 1000.0)
                return np.asarray(result)
            except Exception as e:
                return np.full_like(phi_input, np.nan)
        
        bounds = self._get_bounds_for_params(param_names, param_values)
        
        return {
            'func': model_func,
            'params': param_names,
            'fixed': False,
            'color': '#e74c3c',
            'linestyle': '-',
            'linewidth': 2.0,
            'category': 'Discovered Models',
            'p0': param_values,
            'bounds': bounds,
            'description': f"Auto-discovered model: {model_data['name']} (R²={model_data['r2']:.4f})",
            'reference': "Auto-discovered by Thermal Conductivity Modeling Suite",
            'discovered': True,
            'original_name': model_data['name'],
            'equation': model_data['equation'],
            'expr': expr,
            'param_names': param_names,
            'param_values': param_values
        }
    
    def _get_bounds_for_params(self, param_names, param_values):
        """Get reasonable bounds for parameters."""
        bounds_low = []
        bounds_high = []
        
        for name in param_names:
            if name in ['b', 'd', 'f']:
                bounds_low.append(0.0)
                bounds_high.append(10.0)
            else:
                bounds_low.append(-1000.0)
                bounds_high.append(1000.0)
        
        return (bounds_low, bounds_high)
