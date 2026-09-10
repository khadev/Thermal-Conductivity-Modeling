
"""
Smart Fit Engine - Individual model fitting with advanced capabilities.
"""

import numpy as np
import inspect
import traceback
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import t
import pandas as pd

from ..core.models import MODEL_REGISTRY
from ..core.statistics import calculate_statistics


class SmartFitEngine:
    """Advanced fitting engine for individual models."""

    def __init__(self, phi_data, k_data, k_std=None, mode="nanothermite",
                 k_matrix=None, k_filler=None):
        self.phi = np.asarray(phi_data, dtype=float)
        self.k = np.asarray(k_data, dtype=float)
        self.k_std = np.asarray(k_std, dtype=float) if k_std is not None else None
        self.results = {}

        # Mode-aware material constants
        self.mode = mode
        if k_matrix is None:
            k_matrix = 237.0 if mode == "nanothermite" else 0.2
        if k_filler is None:
            k_filler = 33.0 if mode == "nanothermite" else 300.0
        self.k_matrix = float(k_matrix)
        self.k_filler = float(k_filler)

        # Sanitize data
        self._sanitize_data()

    # ========================================================================
    # DATA SANITIZATION
    # ========================================================================

    def _sanitize_data(self):
        """Remove NaN / Inf values from input data."""
        mask = np.isfinite(self.phi) & np.isfinite(self.k)
        if self.k_std is not None and len(self.k_std) == len(self.phi):
            mask &= np.isfinite(self.k_std) | np.isnan(self.k_std)
            self.k_std = self.k_std[mask]

        original_len = len(self.phi)
        self.phi = self.phi[mask]
        self.k = self.k[mask]

        if len(self.phi) < original_len:
            print(f"[SmartFitEngine] Removed {original_len - len(self.phi)} invalid data points")

        if len(self.phi) < 2:
            raise ValueError("Not enough valid data points (minimum 2).")

    # ========================================================================
    # MODEL WRAPPER (mode-aware)
    # ========================================================================

    def _make_wrapped_func(self, model_func):
        """Wrap a model function to inject mode + material constants."""
        sig = inspect.signature(model_func)
        has_mode = "mode" in sig.parameters
        has_k_matrix = "k_matrix" in sig.parameters
        has_k_filler = "k_filler" in sig.parameters

        mode = self.mode
        k_matrix = self.k_matrix
        k_filler = self.k_filler

        def wrapped(phi, *args):
            kwargs = {}
            if has_mode:
                kwargs["mode"] = mode
            if has_k_matrix:
                kwargs["k_matrix"] = k_matrix
            if has_k_filler:
                kwargs["k_filler"] = k_filler
            return model_func(phi, *args, **kwargs)

        return wrapped

    # ========================================================================
    # SINGLE MODEL FIT
    # ========================================================================

    def fit_model(self, model_name, p0=None, bounds=None, method='trf'):
        """
        Fit a single model with comprehensive analysis.
        """
        if model_name not in MODEL_REGISTRY:
            return {'success': False, 'error': f'Model {model_name} not found'}

        meta = MODEL_REGISTRY[model_name]
        model_func = meta['func']
        param_names = meta.get('params', [])
        is_fixed = meta.get('fixed', True)

        wrapped = self._make_wrapped_func(model_func)

        # ----------------------------------------------------------------
        # FIXED MODELS
        # ----------------------------------------------------------------
        if is_fixed:
            try:
                y_pred = wrapped(self.phi)
                if not np.all(np.isfinite(y_pred)):
                    return {
                        'success': False,
                        'model_name': model_name,
                        'message': 'Fixed model produced NaN or Inf values'
                    }
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

        # ----------------------------------------------------------------
        # GET p0 / BOUNDS
        # ----------------------------------------------------------------
        if p0 is None:
            p0 = meta.get('p0', [1.0] * len(param_names))
        if bounds is None:
            bounds = meta.get('bounds', (-np.inf, np.inf))

        # Sanitize p0
        p0 = [float(v) if np.isfinite(v) else 1.0 for v in p0]

        # Sanitize bounds
        try:
            low, high = bounds
            low = [float(x) if np.isfinite(x) else -1e6 for x in low]
            high = [float(x) if np.isfinite(x) else 1e6 for x in high]
            # Ensure low < high
            for i in range(len(low)):
                if low[i] >= high[i]:
                    low[i], high[i] = -1e6, 1e6
            bounds = (low, high)
        except Exception:
            bounds = (-np.inf, np.inf)

        # ----------------------------------------------------------------
        # SAFETY: ensure p0 gives finite residuals
        # ----------------------------------------------------------------
        try:
            test_pred = wrapped(self.phi, *p0)
            if not np.all(np.isfinite(test_pred)):
                print(f"[SmartFitEngine] p0 gives non-finite residuals for {model_name}, trying shifts...")
                rescued = False
                for shift in [0.1, 10.0, 0.01, 100.0, 0.001, 1000.0]:
                    p0_try = [v * shift if v != 0 else shift for v in p0]
                    try:
                        test_pred = wrapped(self.phi, *p0_try)
                        if np.all(np.isfinite(test_pred)):
                            p0 = p0_try
                            rescued = True
                            print(f"[SmartFitEngine] Rescued with shift {shift}")
                            break
                    except Exception:
                        continue
                if not rescued:
                    return {
                        'success': False,
                        'model_name': model_name,
                        'message': (
                            'Residuals are not finite at any initial guess. '
                            'Check material constants (k_matrix, k_filler) and data.'
                        )
                    }
        except Exception as e:
            return {
                'success': False,
                'model_name': model_name,
                'message': f'Initial guess evaluation failed: {e}'
            }

        # ----------------------------------------------------------------
        # ACTUAL FITTING
        # ----------------------------------------------------------------
        try:
            if self.k_std is not None and not np.all(np.isnan(self.k_std)):
                popt, pcov = curve_fit(
                    wrapped, self.phi, self.k,
                    p0=p0, bounds=bounds, maxfev=20000,
                    method=method,
                    sigma=self.k_std, absolute_sigma=True,
                )
            else:
                popt, pcov = curve_fit(
                    wrapped, self.phi, self.k,
                    p0=p0, bounds=bounds, maxfev=20000,
                    method=method,
                )

            y_pred = wrapped(self.phi, *popt)

            if not np.all(np.isfinite(y_pred)):
                return {
                    'success': False,
                    'model_name': model_name,
                    'message': 'Fitted model produced NaN or Inf values'
                }

            stats = calculate_statistics(self.k, y_pred)
            param_errors = self._calculate_parameter_errors(popt, pcov)
            param_intervals = self._calculate_confidence_intervals(popt, pcov)
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

    # ========================================================================
    # PARAMETER ANALYSIS
    # ========================================================================

    def _calculate_parameter_errors(self, popt, pcov):
        if pcov is not None:
            diag = np.diag(pcov)
            diag = np.maximum(diag, 0)
            return np.sqrt(diag)
        return np.full_like(popt, np.nan)

    def _calculate_confidence_intervals(self, popt, pcov, alpha=0.05):
        n = len(self.phi)
        p = len(popt)
        dof = n - p
        if pcov is not None and dof > 0:
            param_errors = self._calculate_parameter_errors(popt, pcov)
            t_val = t.ppf(1 - alpha / 2, dof)
            return {
                'lower': popt - t_val * param_errors,
                'upper': popt + t_val * param_errors,
                'confidence': 1 - alpha
            }
        return None

    def _calculate_correlations(self, pcov):
        if pcov is None:
            return None
        diag = np.sqrt(np.diag(pcov))
        if np.any(diag == 0):
            return None
        corr = pcov / np.outer(diag, diag)
        return corr

    # ========================================================================
    # GLOBAL OPTIMIZATION
    # ========================================================================

    def optimize_parameters(self, model_name, param_ranges=None, maxiter=1000):
        if model_name not in MODEL_REGISTRY:
            return {'success': False, 'message': f'Model {model_name} not found'}

        meta = MODEL_REGISTRY[model_name]
        model_func = meta['func']
        param_names = meta.get('params', [])

        if not param_names:
            return {'success': False, 'message': 'Model has no parameters'}
        if meta.get('fixed', True):
            return {'success': False, 'message': 'Fixed model - no optimization needed'}

        wrapped = self._make_wrapped_func(model_func)

        if param_ranges is None:
            bounds = meta.get('bounds', (-np.inf, np.inf))
            if isinstance(bounds[0], (int, float)):
                bounds = (bounds, bounds)
            param_ranges = [(bounds[0][i], bounds[1][i]) for i in range(len(param_names))]

        for i, (low, high) in enumerate(param_ranges):
            if not np.isfinite(low):
                low = -1e6
            if not np.isfinite(high):
                high = 1e6
            if low >= high:
                low, high = -1e6, 1e6
            param_ranges[i] = (low, high)

        def objective(x):
            try:
                y_pred = wrapped(self.phi, *x)
                if not np.all(np.isfinite(y_pred)):
                    return 1e10
                return np.sqrt(np.mean((self.k - y_pred) ** 2))
            except Exception:
                return 1e10

        try:
            result = differential_evolution(
                objective, param_ranges,
                maxiter=maxiter, popsize=15, tol=0.01, workers=1
            )
            if result.success:
                return {
                    'success': True,
                    'params': dict(zip(param_names, result.x)),
                    'objective': result.fun,
                    'iterations': result.nit,
                    'message': 'Optimization completed successfully'
                }
            return {'success': False, 'message': result.message}
        except Exception as e:
            return {'success': False, 'message': f'Optimization failed: {str(e)}'}

    # ========================================================================
    # SENSITIVITY
    # ========================================================================

    def parameter_sensitivity(self, model_name, param_variation=0.1):
        if model_name not in MODEL_REGISTRY:
            return None

        meta = MODEL_REGISTRY[model_name]
        if meta.get('fixed', True):
            return {'message': 'Fixed model - no sensitivity analysis'}

        fit_result = self.fit_model(model_name)
        if not fit_result['success']:
            return None

        wrapped = self._make_wrapped_func(meta['func'])
        params = fit_result['params']
        param_names = list(params.keys())
        base_rmse = fit_result['stats']['rmse']

        sensitivity = {}
        for param_name in param_names:
            variations = []
            base_val = params[param_name]
            for factor in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
                test_params = params.copy()
                test_params[param_name] = base_val * factor
                try:
                    param_values = [test_params.get(n, 0) for n in param_names]
                    y_pred = wrapped(self.phi, *param_values)
                    rmse = np.sqrt(np.mean((self.k - y_pred) ** 2))
                    variations.append({
                        'factor': factor,
                        'value': test_params[param_name],
                        'rmse': rmse,
                        'change': (rmse - base_rmse) / base_rmse * 100 if base_rmse > 0 else 0
                    })
                except Exception:
                    pass

            sensitivity[param_name] = {
                'base_value': base_val,
                'base_rmse': base_rmse,
                'variations': variations
            }

        return sensitivity
