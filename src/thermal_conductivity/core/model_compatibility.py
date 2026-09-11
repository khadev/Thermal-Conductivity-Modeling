"""
Model Compatibility Checker
============================
Validates whether selected models are physically compatible with the
experimental data trend and magnitude.

Scientific basis:
    - Each model has a fixed physical shape (trend + curvature + magnitude).
    - If the model's prediction direction contradicts the data,
      R² becomes hugely negative and parameters become unphysical.

FIXED (see CHANGELOG): the previous version of this module hardcoded a single
global direction per model (MODEL_DIRECTION) and a static per-mode whitelist
(NANOTHERMITE_COMPATIBLE / POLYMER_COMPATIBLE / FILLER_GREATER_REQUIRED /
FILLER_LESS_REQUIRED). That assumed each model's direction was an intrinsic
property of the model - but for rule-of-mixtures-style models the actual
direction depends on which material (matrix or filler) has the higher
conductivity, which flips between modes. After fixing the phi-convention and
formula bugs in models.py, every model now correctly tracks that sign
automatically - so this module now determines each model's direction by
actually evaluating it at the mode's real k_matrix/k_filler, instead of
trusting a hardcoded table (which, notably, was itself wrong for several
models - e.g. it listed "Parallel" as nanothermite-incompatible even though
the model computes the correct decreasing trend there).
"""

import numpy as np
import inspect

from .models import get_model_registry


# ============================================================================
# DYNAMIC MODEL DIRECTION (replaces the old static MODEL_DIRECTION table)
# ============================================================================

def get_model_direction(model_name: str, mode: str, k_matrix: float, k_filler: float) -> str:
    """Evaluate the actual model function at the given mode's material
    constants to determine whether it predicts an increasing, decreasing,
    or flat trend - rather than assuming a fixed direction per model name.
    """
    registry = get_model_registry(mode)
    entry = registry.get(model_name)
    if entry is None:
        return "flexible"

    func = entry["func"]
    p0 = entry.get("p0", [])

    try:
        sig = inspect.signature(func)
        kwargs = {}
        if "mode" in sig.parameters:
            kwargs["mode"] = mode
        if "k_matrix" in sig.parameters:
            kwargs["k_matrix"] = k_matrix
        if "k_filler" in sig.parameters:
            kwargs["k_filler"] = k_filler

        phi_lo = np.array([0.05])
        phi_hi = np.array([0.95])
        v_lo = float(np.asarray(func(phi_lo, *p0, **kwargs)).ravel()[0])
        v_hi = float(np.asarray(func(phi_hi, *p0, **kwargs)).ravel()[0])

        if v_lo <= 0 or np.isnan(v_lo) or np.isnan(v_hi):
            return "flexible"
        if v_hi > v_lo * 1.02:
            return "increasing"
        elif v_hi < v_lo * 0.98:
            return "decreasing"
        else:
            return "flat"
    except Exception:
        return "flexible"


DILUTE_ONLY = {"Maxwell-Eucken", "Maxwell-Eucken Upper", "Russell"}
LAYERED_ONLY = {"Series", "Maxwell-Eucken Lower"}
PERCOLATION_BASED = {"Percolation", "GEM Equation", "Agari-Percolation"}


# ============================================================================
# DATA ANALYSIS
# ============================================================================

def analyze_data_trend(phi, k_meas):
    """Determine whether the data is increasing, decreasing, or flat."""
    phi = np.asarray(phi)
    k_meas = np.asarray(k_meas)

    slope = float(np.polyfit(phi, k_meas, 1)[0])
    k_start = float(k_meas[0])
    k_end = float(k_meas[-1])
    ratio = k_end / k_start if k_start > 0 else 1.0
    magnitude_range = float(k_meas.max() / max(k_meas.min(), 1e-12))

    if abs(k_end - k_start) / max(abs(k_start), 1e-12) < 0.05:
        trend = "flat"
    elif k_end > k_start:
        trend = "increasing"
    else:
        trend = "decreasing"

    return {
        "trend": trend,
        "slope": slope,
        "k_start": k_start,
        "k_end": k_end,
        "ratio": ratio,
        "magnitude_range": magnitude_range,
        "phi_start": float(phi.min()),
        "phi_end": float(phi.max()),
    }


# ============================================================================
# SINGLE MODEL CHECK
# ============================================================================

def check_model_compatibility(
    model_name: str,
    data_trend: dict,
    mode: str = "polymer",
    k_matrix: float = 0.2,
    k_filler: float = 300.0,
):
    reasons = []
    severity = "ok"
    compatible = True

    trend = data_trend["trend"]
    direction = get_model_direction(model_name, mode, k_matrix, k_filler)

    # ---------- RULE 1: Direction mismatch ----------
    # FIXED: direction is now computed from the model's actual behavior at
    # this mode's real material constants (see get_model_direction above),
    # not a static table - so this comparison is now mode-aware.
    if direction == "increasing" and trend == "decreasing":
        compatible = False
        severity = "error"
        reasons.append(
            f"<b>{model_name}</b> predicts that k<sub>eff</sub> "
            f"<b>increases</b> with filler fraction φ for the material "
            f"constants entered (k<sub>matrix</sub>={k_matrix:.2f}, "
            f"k<sub>filler</sub>={k_filler:.2f}), but your data "
            f"<b>decreases</b> from {data_trend['k_start']:.2f} to "
            f"{data_trend['k_end']:.2f} W/m·K."
        )

    elif direction == "decreasing" and trend == "increasing":
        compatible = False
        severity = "error"
        reasons.append(
            f"<b>{model_name}</b> predicts that k<sub>eff</sub> "
            f"<b>decreases</b> with φ for the material constants entered "
            f"(k<sub>matrix</sub>={k_matrix:.2f}, k<sub>filler</sub>="
            f"{k_filler:.2f}), but your data <b>increases</b> "
            f"from {data_trend['k_start']:.2f} to "
            f"{data_trend['k_end']:.2f} W/m·K."
        )

    # ---------- RULE 2: Material constant sanity ----------
    if mode == "polymer":
        if k_filler <= k_matrix:
            reasons.append(
                f"⚠ In polymer mode, k<sub>filler</sub> ({k_filler:.2f}) "
                f"should exceed k<sub>matrix</sub> ({k_matrix:.2f})."
            )
            if severity == "ok":
                severity = "warning"
    else:  # nanothermite
        if k_matrix <= k_filler:
            reasons.append(
                f"⚠ In nanothermite mode, k<sub>matrix</sub> "
                f"({k_matrix:.1f}) should exceed k<sub>filler</sub> "
                f"({k_filler:.1f})."
            )
            if severity == "ok":
                severity = "warning"

    # ---------- RULE 3: Dilute limit ----------
    if model_name in DILUTE_ONLY and data_trend["magnitude_range"] > 5:
        if severity == "ok":
            severity = "warning"
        reasons.append(
            f"<b>{model_name}</b> is valid mainly for dilute inclusions "
            f"(φ &lt; 0.10). Your data spans "
            f"{data_trend['magnitude_range']:.1f}× in k."
        )

    # ---------- RULE 4: Percolation models ----------
    if model_name in PERCOLATION_BASED and data_trend["magnitude_range"] < 2.0:
        if severity == "ok":
            severity = "warning"
        reasons.append(
            f"<b>{model_name}</b> assumes a percolation threshold. Your "
            f"data varies only {data_trend['magnitude_range']:.2f}×."
        )

    return {
        "model_name": model_name,
        "compatible": compatible,
        "severity": severity,
        "reason": " ".join(reasons) if reasons else "Compatible.",
        "details": reasons,
    }


# ============================================================================
# BATCH CHECK
# ============================================================================

def check_all_models(
    selected_models, phi, k_meas,
    mode="polymer", k_matrix=0.2, k_filler=300.0,
):
    data_trend = analyze_data_trend(phi, k_meas)
    results = {}
    incompatible = []
    warnings = []

    for name in selected_models:
        res = check_model_compatibility(
            name, data_trend, mode=mode,
            k_matrix=k_matrix, k_filler=k_filler,
        )
        results[name] = res
        if not res["compatible"]:
            incompatible.append(name)
        elif res["severity"] == "warning":
            warnings.append(name)

    return {
        "data_trend": data_trend,
        "results": results,
        "incompatible": incompatible,
        "warnings": warnings,
    }


# ============================================================================
# RECOMMENDED MODELS
# ============================================================================

def get_recommended_models(mode="polymer"):
    if mode == "polymer":
        return ["Agari", "Lewis-Nielsen", "Maxwell-Eucken",
                "Bruggeman", "Percolation", "GEM Equation"]
    return ["Maxwell-Eucken", "Cheng-Vachon", "Baschirow-Selenew",
            "Series", "Bruggeman", "Tsao"]
