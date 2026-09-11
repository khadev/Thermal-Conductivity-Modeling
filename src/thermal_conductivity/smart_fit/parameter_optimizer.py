"""
Parameter Optimizer - Advanced parameter optimization with multiple methods.
"""

import numpy as np
from scipy.optimize import curve_fit, differential_evolution, basinhopping, minimize
import warnings


class ParameterOptimizer:
    """Advanced parameter optimization with multiple methods."""
    
    METHODS = ['curve_fit', 'differential_evolution', 'basinhopping', 'nelder_mead']
    
    def __init__(self, model_func, xdata, ydata, sigma=None):
        self.model_func = model_func
        self.xdata = np.asarray(xdata)
        self.ydata = np.asarray(ydata)
        self.sigma = np.asarray(sigma) if sigma is not None else None
    
    def optimize(self, p0, bounds=None, method='curve_fit', **kwargs):
        """Optimize parameters using specified method."""
        if method == 'curve_fit':
            return self._curve_fit_optimize(p0, bounds, **kwargs)
        elif method == 'differential_evolution':
            return self._de_optimize(p0, bounds, **kwargs)
        elif method == 'basinhopping':
            return self._basinhopping_optimize(p0, bounds, **kwargs)
        elif method == 'nelder_mead':
            return self._nelder_mead_optimize(p0, bounds, **kwargs)
        else:
            raise ValueError(f'Unknown method: {method}')
    
    def _curve_fit_optimize(self, p0, bounds=None, **kwargs):
        """Standard Levenberg-Marquardt optimization."""
        try:
            if self.sigma is not None:
                popt, pcov = curve_fit(
                    self.model_func, self.xdata, self.ydata,
                    p0=p0, bounds=bounds, sigma=self.sigma,
                    absolute_sigma=True, **kwargs
                )
            else:
                popt, pcov = curve_fit(
                    self.model_func, self.xdata, self.ydata,
                    p0=p0, bounds=bounds, **kwargs
                )
            return {
                'success': True,
                'popt': popt,
                'pcov': pcov,
                'method': 'curve_fit',
                'message': 'Optimization completed successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Curve fit failed: {str(e)}'
            }
    
    def _de_optimize(self, p0, bounds=None, maxiter=1000, popsize=15, **kwargs):
        """Differential evolution global optimization."""
        def objective(x):
            try:
                y_pred = self.model_func(self.xdata, *x)
                residuals = self.ydata - y_pred
                if self.sigma is not None:
                    residuals = residuals / self.sigma
                return np.sum(residuals**2)
            except:
                return 1e10
        
        # Create bounds from p0 and bounds
        if bounds is None or (isinstance(bounds[0], (int, float)) and bounds[0] == -np.inf):
            # If no bounds, use 100x around p0
            param_ranges = [(p0[i] * 0.01, p0[i] * 100) if p0[i] > 0 else (p0[i] - 100, p0[i] + 100) 
                           for i in range(len(p0))]
        else:
            if isinstance(bounds[0], (int, float)):
                bounds = (bounds, bounds)
            param_ranges = [(bounds[0][i], bounds[1][i]) for i in range(len(p0))]
        
        # Replace inf with reasonable values
        for i, (low, high) in enumerate(param_ranges):
            if low == -np.inf:
                param_ranges[i] = (-1e6, high)
            if high == np.inf:
                param_ranges[i] = (low, 1e6)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = differential_evolution(
                    objective,
                    param_ranges,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=0.01,
                    workers=1,
                    **kwargs
                )
            
            if result.success:
                return {
                    'success': True,
                    'popt': result.x,
                    'pcov': None,
                    'method': 'differential_evolution',
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
                'message': f'Differential evolution failed: {str(e)}'
            }
    
    def _basinhopping_optimize(self, p0, bounds=None, niter=100, **kwargs):
        """Basin-hopping global optimization."""
        def objective(x):
            try:
                y_pred = self.model_func(self.xdata, *x)
                residuals = self.ydata - y_pred
                if self.sigma is not None:
                    residuals = residuals / self.sigma
                return np.sum(residuals**2)
            except:
                return 1e10
        
        try:
            # First do a local optimization
            local_result = self._curve_fit_optimize(p0, bounds)
            if local_result['success']:
                x0 = local_result['popt']
            else:
                x0 = p0
            
            # Then basin-hopping
            result = basinhopping(objective, x0, niter=niter, **kwargs)
            
            if result.success:
                return {
                    'success': True,
                    'popt': result.x,
                    'pcov': None,
                    'method': 'basinhopping',
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
                'message': f'Basin-hopping failed: {str(e)}'
            }
    
    def _nelder_mead_optimize(self, p0, bounds=None, **kwargs):
        """Nelder-Mead simplex optimization."""
        def objective(x):
            try:
                y_pred = self.model_func(self.xdata, *x)
                residuals = self.ydata - y_pred
                if self.sigma is not None:
                    residuals = residuals / self.sigma
                return np.sum(residuals**2)
            except:
                return 1e10
        
        try:
            # Setup bounds for Nelder-Mead (uses constraints if bounds provided)
            if bounds is not None:
                from scipy.optimize import Bounds
                if isinstance(bounds[0], (int, float)):
                    bounds = (bounds, bounds)
                constraint = Bounds(bounds[0], bounds[1])
            else:
                constraint = None
            
            result = minimize(
                objective,
                p0,
                method='Nelder-Mead',
                bounds=constraint,
                options={'maxiter': 10000, 'maxfev': 20000, 'fatol': 1e-8},
                **kwargs
            )
            
            if result.success:
                return {
                    'success': True,
                    'popt': result.x,
                    'pcov': None,
                    'method': 'nelder_mead',
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
                'message': f'Nelder-Mead failed: {str(e)}'
            }
