
"""
Smart Fit Engine - Individual model fitting with advanced capabilities.
"""

import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import t
import pandas as pd

from ..core.models import MODEL_REGISTRY
from ..core.statistics import calculate_statistics


class SmartFitEngine:
    """Advanced fitting engine for individual models."""
    
    def __init__(self, phi_data, k_data, k_std=None):
        self.phi = np.asarray(phi_data)
        self.k = np.asarray(k_data)
        self.k_std = np.asarray(k_std) if k_std is not None else None
        self.results = {}
        
    def fit_model(self, model_name, p0=None, bounds=None, method='trf'):
        """
        Fit a single model with comprehensive analysis.
        
        Returns:
            dict: Complete fitting results including parameters, statistics, 
                  confidence intervals, and diagnostics.
        """
        if model_name not in MODEL_REGISTRY:
            return {'success': False, 'error': f'Model {model_name} not found'}
        
        meta = MODEL_REGISTRY[model_name]
        model_func = meta['func']
        
        # Get parameter info
        param_names = meta.get('params', [])
        is_fixed = meta.get('fixed', True)
        
        # For fixed models
        if is_fixed:
            try:
                y_pred = model_func(self.phi)
                stats = calculate_statistics(self.k, y_pred)
                return {
                    'success': True,
                    'model_name': model_name,
                    'params': {},
                    'y_pred': y_pred,
                    'stats': stats,
                    'is_fixed': True,
                    'message': 'Fixed model - no fitting required'
                }
            except Exception as e:
                return {
                    'success': False,
                    'model_name': model_name,
                    'error': str(e),
                    'message': f'Error in fixed model: {str(e)}'
                }
        
        # Get initial guesses and bounds
        if p0 is None:
            p0 = meta.get('p0', [1.0] * len(param_names))
        if bounds is None:
            bounds = meta.get('bounds', (-np.inf, np.inf))
        
        try:
            # Perform curve fitting
            if self.k_std is not None and not np.all(np.isnan(self.k_std)):
                popt, pcov = curve_fit(
                    model_func,
                    self.phi,
                    self.k,
                    p0=p0,
                    bounds=bounds,
                    maxfev=10000,
                    method=method,
                    sigma=self.k_std,
                    absolute_sigma=True
                )
            else:
                popt, pcov = curve_fit(
                    model_func,
                    self.phi,
                    self.k,
                    p0=p0,
                    bounds=bounds,
                    maxfev=10000,
                    method=method
                )
            
            # Calculate predictions
            y_pred = model_func(self.phi, *popt)
            
            # Calculate statistics
            stats = calculate_statistics(self.k, y_pred)
            
            # Calculate parameter errors
            param_errors = self._calculate_parameter_errors(popt, pcov)
            
            # Calculate confidence intervals
            param_intervals = self._calculate_confidence_intervals(popt, pcov)
            
            # Calculate correlations
            correlations = self._calculate_correlations(pcov)
            
            return {
                'success': True,
                'model_name': model_name,
                'params': dict(zip(param_names, popt)),
                'param_errors': dict(zip(param_names, param_errors)),
                'param_intervals': param_intervals,
                'correlations': correlations,
                'popt': popt,
                'pcov': pcov,
                'y_pred': y_pred,
                'stats': stats,
                'is_fixed': False,
                'n_parameters': len(popt),
                'message': 'Fitting completed successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'model_name': model_name,
                'error': str(e),
                'message': f'Fitting failed: {str(e)}'
            }
    
    def _calculate_parameter_errors(self, popt, pcov):
        """Calculate standard errors for parameters."""
        if pcov is not None:
            diag = np.diag(pcov)
            # Handle negative values (shouldn't happen, but just in case)
            diag = np.maximum(diag, 0)
            return np.sqrt(diag)
        return np.full_like(popt, np.nan)
    
    def _calculate_confidence_intervals(self, popt, pcov, alpha=0.05):
        """Calculate confidence intervals for parameters."""
        n = len(self.phi)
        p = len(popt)
        dof = n - p
        
        if pcov is not None and dof > 0:
            param_errors = self._calculate_parameter_errors(popt, pcov)
            t_val = t.ppf(1 - alpha/2, dof)
            intervals = {
                'lower': popt - t_val * param_errors,
                'upper': popt + t_val * param_errors,
                'confidence': 1 - alpha
            }
            return intervals
        return None
    
    def _calculate_correlations(self, pcov):
        """Calculate parameter correlation matrix."""
        if pcov is None:
            return None
        diag = np.sqrt(np.diag(pcov))
        if np.any(diag == 0):
            return None
        corr = pcov / np.outer(diag, diag)
        return corr
    
    def optimize_parameters(self, model_name, param_ranges=None, maxiter=1000):
        """
        Use global optimization to find optimal parameters.
        
        This is useful when local optimization (curve_fit) gets stuck.
        """
        if model_name not in MODEL_REGISTRY:
            return {'success': False, 'message': f'Model {model_name} not found'}
        
        meta = MODEL_REGISTRY[model_name]
        model_func = meta['func']
        param_names = meta.get('params', [])
        
        if not param_names:
            return {'success': False, 'message': 'Model has no parameters to optimize'}
        
        if meta.get('fixed', True):
            return {'success': False, 'message': 'Fixed model - no optimization needed'}
        
        # Define bounds for optimization
        if param_ranges is None:
            bounds = meta.get('bounds', (-np.inf, np.inf))
            if isinstance(bounds[0], (int, float)):
                bounds = (bounds, bounds)
            param_ranges = [(bounds[0][i], bounds[1][i]) for i in range(len(param_names))]
        
        # Replace inf with reasonable values
        for i, (low, high) in enumerate(param_ranges):
            if low == -np.inf:
                param_ranges[i] = (-1e6, high)
            if high == np.inf:
                param_ranges[i] = (low, 1e6)
        
        def objective(x):
            try:
                y_pred = model_func(self.phi, *x)
                # Use RMSE as objective
                rmse = np.sqrt(np.mean((self.k - y_pred)**2))
                # Add small penalty for unreasonable values
                if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                    return 1e10
                return rmse
            except:
                return 1e10
        
        # Run differential evolution
        try:
            result = differential_evolution(
                objective,
                param_ranges,
                maxiter=maxiter,
                popsize=15,
                tol=0.01,
                workers=1
            )
            
            if result.success:
                return {
                    'success': True,
                    'params': dict(zip(param_names, result.x)),
                    'objective': result.fun,
                    'iterations': result.nit,
                    'message': 'Optimization completed successfully'
                }
            else:
                return {
                    'success': False,
                    'message': result.message
                }
        except Exception as e:
            return {
                'success': False,
                'message': f'Optimization failed: {str(e)}'
            }
    
    def parameter_sensitivity(self, model_name, param_variation=0.1):
        """
        Analyze how parameter variations affect the fit.
        
        Returns:
            dict: Sensitivity analysis results
        """
        if model_name not in MODEL_REGISTRY:
            return None
        
        meta = MODEL_REGISTRY[model_name]
        if meta.get('fixed', True):
            return {'message': 'Fixed model - no sensitivity analysis'}
        
        # First, fit the model to get base parameters
        fit_result = self.fit_model(model_name)
        if not fit_result['success']:
            return None
        
        params = fit_result['params']
        param_names = list(params.keys())
        
        sensitivity = {}
        base_rmse = fit_result['stats']['rmse']
        
        for param_name in param_names:
            variations = []
            base_val = params[param_name]
            
            # Test different variation factors - wider range
            for factor in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
                test_params = params.copy()
                test_params[param_name] = base_val * factor
                
                try:
                    # Get parameter values in correct order
                    param_values = [test_params.get(name, 0) for name in param_names]
                    y_pred = meta['func'](self.phi, *param_values)
                    rmse = np.sqrt(np.mean((self.k - y_pred)**2))
                    variations.append({
                        'factor': factor,
                        'value': test_params[param_name],
                        'rmse': rmse,
                        'change': (rmse - base_rmse) / base_rmse * 100 if base_rmse > 0 else 0
                    })
                except:
                    pass
            
            sensitivity[param_name] = {
                'base_value': base_val,
                'base_rmse': base_rmse,
                'variations': variations
            }
        
        return sensitivity
