"""
Model Compatibility Checker
============================
Validates whether selected models are physically compatible with the
experimental data trend and magnitude.

Scientific basis:
    - Each model has a fixed physical shape (trend + curvature + magnitude).
    - If the model's prediction direction contradicts the data,
      R² becomes hugely negative and parameters become unphysical.
"""

import numpy as np


# ============================================================================
# MODEL DIRECTION TABLE
# ============================================================================

MODEL_DIRECTION = {
    # ---- Polymer-friendly (k increases with φ) ----
    "Parallel": "increasing",
    "Maxwell-Eucken": "flexible",           # Mode-dependent
    "Maxwell-Eucken Upper": "increasing",
    "Bruggeman": "flexible",
    "Agari": "flexible",
    "Agari-Percolation": "flexible",
    "Lewis-Nielsen": "increasing",
    "Percolation": "increasing",
    "GEM Equation": "increasing",
    "Halpin-Tsai": "increasing",
    "Hatta": "increasing",
    "Modified Hashin-Shtrikman": "increasing",
    "Hamilton-Crosser": "increasing",
    "Hamilton-Crosser Extended": "increasing",
    "Rayleigh": "flexible",

    # ---- Nanothermite-friendly (k decreases with φ) ----
    "Series": "decreasing",
    "Maxwell-Eucken Lower": "decreasing",
    "Cheng-Vachon": "decreasing",
    "Baschirow-Selenew": "decreasing",
    "Tsao": "decreasing",

    # ---- Ambiguous / data-dependent ----
    "EMT Equation": "flexible",
    "Geometric Mean": "flexible",
    "Böttcher": "flexible",
    "De Loor": "flexible",
    "Ce Wen Nan": "flexible",
    "Russell": "flexible",
}


# ============================================================================
# MODEL CATEGORIES
# ============================================================================

POLYMER_COMPATIBLE = {
    "Agari", "Agari-Percolation", "Lewis-Nielsen", "Maxwell-Eucken",
    "Bruggeman", "Percolation", "GEM Equation", "Halpin-Tsai",
    "Hatta", "Modified Hashin-Shtrikman", "Hamilton-Crosser",
    "Hamilton-Crosser Extended", "Rayleigh", "Parallel",
}

NANOTHERMITE_COMPATIBLE = {
    "Agari", "Agari-Percolation",
    "Series", "Maxwell-Eucken", "Maxwell-Eucken Lower",
    "Cheng-Vachon", "Baschirow-Selenew", "Tsao",
    "Bruggeman", "Percolation", "GEM Equation",
    "Ce Wen Nan", "De Loor", "Böttcher",
    "Geometric Mean", "EMT Equation", "Russell",
}

# Models that REQUIRE k_filler > k_matrix (invalid for nanothermite)
FILLER_GREATER_REQUIRED = {
    "Parallel", "Maxwell-Eucken Upper",
    "Lewis-Nielsen", "Halpin-Tsai", "Hatta",
    "Hamilton-Crosser", "Hamilton-Crosser Extended",
    "Modified Hashin-Shtrikman", "Rayleigh",
}

# Models that REQUIRE k_filler < k_matrix (invalid for polymer)
FILLER_LESS_REQUIRED = {
    "Series", "Maxwell-Eucken Lower",
    "Cheng-Vachon", "Baschirow-Selenew",
}

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
    direction = MODEL_DIRECTION.get(model_name, "flexible")

    # ---------- RULE 1: Direction mismatch ----------
    if direction == "increasing" and trend == "decreasing":
        compatible = False
        severity = "error"
        reasons.append(
            f"<b>{model_name}</b> predicts that k<sub>eff</sub> "
            f"<b>increases</b> with filler fraction φ, but your data "
            f"<b>decreases</b> from {data_trend['k_start']:.1f} to "
            f"{data_trend['k_end']:.1f} W/m·K."
        )

    elif direction == "decreasing" and trend == "increasing":
        compatible = False
        severity = "error"
        reasons.append(
            f"<b>{model_name}</b> predicts that k<sub>eff</sub> "
            f"<b>decreases</b> with φ, but your data <b>increases</b> "
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

    # ---------- RULE 3: Models requiring k_filler > k_matrix ----------
    if model_name in FILLER_GREATER_REQUIRED and mode == "nanothermite":
        if k_filler < k_matrix:
            compatible = False
            severity = "error"
            reasons.append(
                f"<b>{model_name}</b> assumes k<sub>filler</sub> > "
                f"k<sub>matrix</sub>, which is <b>false</b> for "
                f"nanothermites (k<sub>matrix</sub> = {k_matrix:.1f}, "
                f"k<sub>filler</sub> = {k_filler:.1f} W/m·K)."
            )

    # ---------- RULE 4: Models requiring k_filler < k_matrix ----------
    if model_name in FILLER_LESS_REQUIRED and mode == "polymer":
        if k_filler > k_matrix:
            compatible = False
            severity = "error"
            reasons.append(
                f"<b>{model_name}</b> assumes k<sub>filler</sub> < "
                f"k<sub>matrix</sub>, which is <b>false</b> for polymer "
                f"composites (k<sub>matrix</sub> = {k_matrix:.2f}, "
                f"k<sub>filler</sub> = {k_filler:.1f} W/m·K)."
            )

    # ---------- RULE 5: Dilute limit ----------
    if model_name in DILUTE_ONLY and data_trend["magnitude_range"] > 5:
        if severity == "ok":
            severity = "warning"
        reasons.append(
            f"<b>{model_name}</b> is valid mainly for dilute inclusions "
            f"(φ &lt; 0.10). Your data spans "
            f"{data_trend['magnitude_range']:.1f}× in k."
        )

    # ---------- RULE 6: Percolation models ----------
    if model_name in PERCOLATION_BASED and data_trend["magnitude_range"] < 2.0:
        if severity == "ok":
            severity = "warning"
        reasons.append(
            f"<b>{model_name}</b> assumes a percolation threshold. Your "
            f"data varies only {data_trend['magnitude_range']:.2f}×."
        )

    # ---------- RULE 7: Mode compatibility list ----------
    if mode == "polymer" and model_name not in POLYMER_COMPATIBLE:
        compatible = False
        severity = "error"
        reasons.append(
            f"<b>{model_name}</b> is not in the recommended set for "
            f"polymer composites."
        )

    if mode == "nanothermite" and model_name not in NANOTHERMITE_COMPATIBLE:
        compatible = False
        severity = "error"
        reasons.append(
            f"<b>{model_name}</b> is not in the recommended set for "
            f"nanothermite composites."
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
