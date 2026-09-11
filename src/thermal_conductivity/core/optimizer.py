"""Nonlinear least-squares fitting module for thermal conductivity models."""

import numpy as np
from scipy.optimize import curve_fit
import inspect

from .models import MODEL_REGISTRY, k_Al, k_CuO


def fit_model(model_name, phi_data, k_data, p0=None, bounds=None, mode="nanothermite",
              k_matrix=None, k_filler=None):
    """
    Fit a thermal conductivity model to experimental data.

    Parameters
    ----------
    model_name : str
        Name of the model from MODEL_REGISTRY
    phi_data : array_like
        Volume fraction array
    k_data : array_like
        Measured thermal conductivity array
    p0 : list, optional
        Initial parameter guesses. Uses registry defaults if None.
    bounds : tuple, optional
        Parameter bounds as (lower, upper). Uses registry defaults if None.
    mode : str, optional
        Mode for material constants: 'nanothermite' or 'polymer'
    k_matrix : float, optional
        Matrix thermal conductivity for the material actually selected in the
        GUI. If None, the mode's generic default (Al / Epoxy) is used.
    k_filler : float, optional
        Filler thermal conductivity for the material actually selected in the
        GUI. If None, the mode's generic default (CuO / Graphene) is used.

    Returns
    -------
    dict
        Dictionary containing fitted parameters, predictions, and statistics.
    """
    registry = MODEL_REGISTRY[model_name]
    model_func = registry["func"]

    # ===== WRAP THE FUNCTION TO PASS MODE AND MATERIAL CONSTANTS =====
    def wrapped_func(phi, *args):
        """Wrapper that passes mode and material constants to the model function."""
        sig = inspect.signature(model_func)
        kwargs = {}
        if 'mode' in sig.parameters:
            kwargs['mode'] = mode
        # FIXED: k_matrix/k_filler were never forwarded here, so if the user
        # picked a non-default material in the GUI (e.g. Mg instead of Al),
        # the fit silently used the generic mode defaults instead of the
        # material actually selected.
        if 'k_matrix' in sig.parameters and k_matrix is not None:
            kwargs['k_matrix'] = k_matrix
        if 'k_filler' in sig.parameters and k_filler is not None:
            kwargs['k_filler'] = k_filler
        return model_func(phi, *args, **kwargs)

    if registry["fixed"]:
        # No fitting needed for fixed models
        try:
            y_pred = wrapped_func(phi_data)
            # Check for NaN or Inf
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return {
                    "model_name": model_name,
                    "params": {},
                    "popt": None,
                    "pcov": None,
                    "y_pred": None,
                    "success": False,
                    "message": "Fixed model produced NaN or Inf values",
                }
            return {
                "model_name": model_name,
                "params": {},
                "popt": None,
                "pcov": None,
                "y_pred": y_pred,
                "success": True,
                "message": "Fixed model - no fitting required",
            }
        except Exception as e:
            return {
                "model_name": model_name,
                "params": {},
                "popt": None,
                "pcov": None,
                "y_pred": None,
                "success": False,
                "message": f"Error in fixed model: {str(e)}",
            }

    # Use registry defaults if not provided
    if p0 is None:
        p0 = registry.get("p0", [1.0] * len(registry["params"]))
    if bounds is None:
        bounds = registry.get("bounds", (-np.inf, np.inf))

    try:
        popt, pcov = curve_fit(
            wrapped_func, 
            phi_data, 
            k_data, 
            p0=p0, 
            bounds=bounds,
            maxfev=10000,
            method="trf",
        )

        y_pred = wrapped_func(phi_data, *popt)
        
        # Check for NaN or Inf in predictions
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return {
                "model_name": model_name,
                "params": {},
                "popt": None,
                "pcov": None,
                "y_pred": None,
                "success": False,
                "message": "Fitted model produced NaN or Inf values",
            }

        # Calculate parameter standard errors from covariance matrix
        if pcov is not None:
            perr = np.sqrt(np.diag(pcov))
        else:
            perr = [np.nan] * len(popt)

        return {
            "model_name": model_name,
            "params": dict(zip(registry["params"], popt)),
            "popt": popt,
            "pcov": pcov,
            "perr": perr,
            "y_pred": y_pred,
            "success": True,
            "message": "Fitting converged successfully",
        }
    except RuntimeError as e:
        return {
            "model_name": model_name,
            "params": {},
            "popt": None,
            "pcov": None,
            "perr": None,
            "y_pred": None,
            "success": False,
            "message": f"Fitting failed: {str(e)}",
        }
    except Exception as e:
        return {
            "model_name": model_name,
            "params": {},
            "popt": None,
            "pcov": None,
            "perr": None,
            "y_pred": None,
            "success": False,
            "message": f"Unexpected error: {str(e)}",
        }


def fit_all_models(phi_data, k_data, selected_models=None, custom_params=None, mode="nanothermite",
                    k_matrix=None, k_filler=None):
    """
    Fit all selected models to experimental data.

    Parameters
    ----------
    phi_data : array_like
        Volume fraction array
    k_data : array_like
        Measured thermal conductivity array
    selected_models : list, optional
        List of model names to fit. If None, fits all models.
    custom_params : dict, optional
        Custom initial parameters and bounds per model.
        Format: {"model_name": {"p0": [...], "bounds": ([...], [...])}}
    mode : str, optional
        Mode for material constants: 'nanothermite' or 'polymer'
    k_matrix : float, optional
        Matrix thermal conductivity for the material actually selected in the GUI.
    k_filler : float, optional
        Filler thermal conductivity for the material actually selected in the GUI.

    Returns
    -------
    dict
        Dictionary mapping model names to fit results.
    """
    if selected_models is None:
        selected_models = list(MODEL_REGISTRY.keys())

    if custom_params is None:
        custom_params = {}

    results = {}
    for model_name in selected_models:
        if model_name not in MODEL_REGISTRY:
            continue

        cp = custom_params.get(model_name, {})
        p0 = cp.get("p0")
        bounds = cp.get("bounds")

        # ===== PASS MODE AND MATERIAL CONSTANTS TO fit_model =====
        result = fit_model(model_name, phi_data, k_data, p0=p0, bounds=bounds, mode=mode,
                            k_matrix=k_matrix, k_filler=k_filler)
        results[model_name] = result

    return results