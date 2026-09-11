"""Thermal Conductivity Models for Composite Materials - Dual Mode Support."""

import numpy as np
from scipy.optimize import fsolve


# ============================================================================
# MATERIAL CONSTANTS (Defined locally to avoid circular import)
# ============================================================================

# Nanothermite Mode
NANO_k_Al = 237.0
NANO_k_CuO = 33.0
NANO_rho_Al = 2700.0
NANO_rho_CuO = 6315.0

# Polymer Mode
POLYMER_k_Epoxy = 0.2
POLYMER_k_Graphene = 300.0
POLYMER_rho_Epoxy = 1200.0
POLYMER_rho_Graphene = 2200.0

# Current mode (will be set by main window)
CURRENT_MODE = "nanothermite"


# ============================================================================
# HELPER FUNCTION TO GET CONSTANTS
# ============================================================================

def get_mode_constants(mode="nanothermite", k_matrix=None, k_filler=None):
    """Get the appropriate constants for the current mode."""
    if mode == "nanothermite":
        return {
            'k_matrix': k_matrix if k_matrix is not None else NANO_k_Al,
            'k_filler': k_filler if k_filler is not None else NANO_k_CuO,
            'rho_matrix': NANO_rho_Al,
            'rho_filler': NANO_rho_CuO,
            'mode': 'nanothermite',
            'matrix_name': 'Reductant',
            'filler_name': 'Oxidizer'
        }
    else:  # polymer
        return {
            'k_matrix': k_matrix if k_matrix is not None else POLYMER_k_Epoxy,
            'k_filler': k_filler if k_filler is not None else POLYMER_k_Graphene,
            'rho_matrix': POLYMER_rho_Epoxy,
            'rho_filler': POLYMER_rho_Graphene,
            'mode': 'polymer',
            'matrix_name': 'Polymer Matrix',
            'filler_name': 'Filler'
        }


def mass_to_volume_fraction(mass_fraction_filler, rho_matrix, rho_filler):
    """Convert mass fraction to volume fraction."""
    m_filler = mass_fraction_filler
    m_matrix = 1.0 - m_filler
    v_filler = m_filler / rho_filler
    v_matrix = m_matrix / rho_matrix
    return v_filler / (v_filler + v_matrix)


# ============================================================================
# BASIC MODELS (Universally applicable to both modes)
# ============================================================================

def parallel_model(phi, k_matrix=None, k_filler=None, mode=None):
    """
    Parallel Model (Upper Bound / Rule of Mixtures).
    k_eff = φ * k_filler + (1-φ) * k_matrix
    """
    phi = np.asarray(phi)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    return phi * k_filler + (1.0 - phi) * k_matrix


def series_model(phi, k_matrix=None, k_filler=None, mode=None):
    """
    Series Model (Lower Bound / Inverse Rule of Mixtures).
    1/k_eff = φ/k_filler + (1-φ)/k_matrix
    """
    phi = np.asarray(phi)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    k_matrix_safe = max(k_matrix, 1e-12)
    k_filler_safe = max(k_filler, 1e-12)
    return 1.0 / (phi / k_filler_safe + (1.0 - phi) / k_matrix_safe)


def maxwell_eucken(phi, k_matrix=None, k_filler=None, mode=None):
    """
    Maxwell-Eucken Model for dilute spherical inclusions.
    k_eff = k_matrix * [(k_filler + 2*k_matrix + 2*φ*(k_filler - k_matrix)) /
                        (k_filler + 2*k_matrix - φ*(k_filler - k_matrix))]
    """
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    numerator = k_filler + 2.0 * k_matrix + 2.0 * phi * (k_filler - k_matrix)
    denominator = k_filler + 2.0 * k_matrix - phi * (k_filler - k_matrix)
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    result = k_matrix * numerator / denominator
    return np.clip(result, 0, 5000.0)


def _bruggeman_scalar(phi, k_matrix, k_filler):
    """Bruggeman model for a single scalar value.

    FIXED: unconstrained fsolve from a linear-mixture starting guess routinely
    failed to converge (or converged to a spurious/negative root) once the
    filler/matrix conductivity ratio got large - e.g. polymer mode
    (300/0.2 = 1500x) returned NaN at every phi. Replaced with a bounded
    root-finder (brentq) searched over the physically valid k_eff range
    between k_matrix and k_filler.
    """
    from scipy.optimize import brentq

    def func(k_eff):
        return (1.0 - phi) - ((k_filler - k_eff) / (k_filler - k_matrix)) * ((k_matrix / k_eff) ** (1.0 / 3.0))
    
    lo = min(k_matrix, k_filler) * 0.999
    hi = max(k_matrix, k_filler) * 1.001
    try:
        f_lo, f_hi = func(lo), func(hi)
        if f_lo * f_hi > 0:
            # Bracket failed (shouldn't happen for phi in (0,1)) - fall back
            # to the linear mixture as a safe, non-NaN estimate.
            return k_matrix * (1.0 - phi) + k_filler * phi
        k_eff = brentq(func, lo, hi, maxiter=200)
        if k_eff <= 0 or np.isnan(k_eff) or np.isinf(k_eff):
            return k_matrix * (1.0 - phi) + k_filler * phi
        return k_eff
    except Exception:
        return k_matrix * (1.0 - phi) + k_filler * phi


def bruggeman_model(phi, k_matrix=None, k_filler=None, mode=None):
    """
    Bruggeman Effective Medium Model.
    (1-φ) = (k_filler - k_eff)/(k_filler - k_matrix) * (k_matrix/k_eff)^(1/3)
    """
    phi = np.asarray(phi)
    phi = np.clip(phi, 0.001, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    if phi.ndim == 0:
        return _bruggeman_scalar(float(phi), k_matrix, k_filler)
    return np.array([_bruggeman_scalar(p, k_matrix, k_filler) for p in phi])


def agari_model(phi, C1, C2, k_matrix=None, k_filler=None, mode=None):
    """
    Agari Empirical Model - CORRECTED.
    
    The CORRECT Agari equation is:
    log10(k_eff) = φ * C2 * log10(k_filler) + (1-φ) * log10(C1 * k_matrix)
    
    For Nanothermite: φ = oxidizer fraction, k_filler = oxidizer conductivity
    For Polymer: φ = filler fraction, k_filler = filler conductivity
    
    This means:
    - Nanothermite: k_filler (CuO=33) < k_matrix (Al=237) → DECREASING conductivity
    - Polymer: k_filler (Graphene=300) > k_matrix (Epoxy=0.2) → INCREASING conductivity
    """
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    k_matrix_safe = max(k_matrix, 1e-12)
    k_filler_safe = max(k_filler, 1e-12)
    C1_safe = max(C1, 1e-12)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # ===== CORRECTED: φ * C2 * log(k_filler) + (1-φ) * log(C1 * k_matrix) =====
        log_k = phi * C2 * np.log10(k_filler_safe) + (1.0 - phi) * np.log10(C1_safe * k_matrix_safe)
    
    log_k = np.where(np.isfinite(log_k), log_k, -10.0)
    result = 10.0 ** log_k
    return np.clip(result, 0, 5000.0)


def lewis_nielsen_model(phi, A, phi_m, k_matrix=None, k_filler=None, mode=None):
    """
    Lewis-Nielsen Model.
    k_eff = k_matrix * (1 + A*B*φ_filler) / (1 - B*ψ*φ_filler)
    where φ_filler = 1-φ
    """
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    A = max(A, 0.01)
    phi_m = max(phi_m, 0.01)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    # FIXED: Lewis-Nielsen is only physically valid for filler_frac < phi_m
    # (the maximum packing fraction) - beyond that, (1 - B*psi*filler_frac)
    # legitimately goes negative and the model has no physical meaning. The
    # original code let filler_frac run all the way to 0.99 regardless of
    # phi_m, which produced sign flips (results silently clipped near 0)
    # instead of a controlled breakdown. Clamp to the model's valid domain.
    filler_frac = np.clip(phi, 0, 0.98 * phi_m)  # phi IS the filler fraction
    B = (k_filler / k_matrix - 1.0) / (k_filler / k_matrix + A)
    psi = 1.0 + ((1.0 - phi_m) / (phi_m ** 2.0 + 1e-12)) * filler_frac
    result = k_matrix * (1.0 + A * B * filler_frac) / (1.0 - B * psi * filler_frac + 1e-12)
    return np.clip(result, 0, 5000.0)


def percolation_model(phi, k0, phi_c, t, k_matrix=None, mode=None):
    """
    Percolation Model.
    k_eff = k0 * (φ_filler - φ_c)^t for φ_filler > φ_c
    k_eff = k_matrix * (1 - (φ_c - φ_filler)/φ_c) for φ_filler <= φ_c
    where φ_filler = 1-φ
    """
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    phi_c = max(phi_c, 0.001)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, None)
    k_matrix = constants['k_matrix']
    
    filler_frac = phi  # FIXED: phi IS the filler fraction (was 1.0 - phi)
    result = np.empty_like(phi, dtype=float)
    mask_below = filler_frac <= phi_c
    mask_above = filler_frac > phi_c
    
    base = filler_frac[mask_above] - phi_c
    base = np.where(base <= 0, 1e-12, base)
    result[mask_above] = k0 * (base) ** t
    result[mask_below] = k_matrix * (1.0 - (filler_frac[mask_below] - phi_c) / (phi_c + 1e-12))
    return np.clip(result, 0, 5000.0)


def agari_percolation_model(phi, C1, C2, k0, phi_c, t, k_matrix=None, k_filler=None, mode=None):
    """
    Hybrid Agari-Percolation Model - CORRECTED.
    log10(k_eff) = φ * C2 * log10(k_filler) + (1-φ) * log10(C1 * k_matrix) + k0*(φ_filler - φ_c)^t
    """
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    k_matrix_safe = max(k_matrix, 1e-12)
    k_filler_safe = max(k_filler, 1e-12)
    C1_safe = max(C1, 1e-12)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # ===== CORRECTED =====
        agari_part_log = phi * C2 * np.log10(k_filler_safe) + (1.0 - phi) * np.log10(C1_safe * k_matrix_safe)
        agari_part = 10.0 ** np.clip(agari_part_log, -10, 10)
    
    percolation_part = percolation_model(phi, k0, phi_c, t, k_matrix, mode)
    result = agari_part + percolation_part
    return np.clip(result, 0, 5000.0)


# ============================================================================
# CLASSIC MODELS (unchanged - keep as is)
# ============================================================================

def emt_equation_model(phi, k_matrix=None, k_filler=None, mode=None):
    """EMT Equation Model - Effective Medium Theory."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0.001, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    def func(k_eff, phi_val):
        phi_filler = phi_val  # phi IS the filler fraction
        phi_matrix = 1.0 - phi_val
        term1 = phi_matrix * (k_matrix - k_eff) / (k_matrix + 2.0 * k_eff + 1e-12)
        term2 = phi_filler * (k_filler - k_eff) / (k_filler + 2.0 * k_eff + 1e-12)
        return term1 + term2
    
    # FIXED: the initial guess and every fallback below used
    # "k_matrix*p + k_filler*(1-p)" - i.e. still weighted by the OLD (matrix-
    # fraction) convention even after phi_filler was corrected to just "p".
    # That fed fsolve a systematically wrong starting point, which combined
    # with fsolve's lack of bounds, converged to spurious roots especially at
    # large k-ratios (e.g. polymer mode) - hence the wrong direction. Now
    # using the correct linear-mixture guess and a bounded solver.
    from scipy.optimize import brentq
    results = []
    for p in phi:
        linear_fallback = k_matrix * (1.0 - p) + k_filler * p
        try:
            lo, hi = min(k_matrix, k_filler) * 0.999, max(k_matrix, k_filler) * 1.001
            f_lo, f_hi = func(lo, p), func(hi, p)
            if f_lo * f_hi > 0:
                results.append(linear_fallback)
                continue
            k_eff = brentq(func, lo, hi, args=(p,), maxiter=200)
            k_eff = np.clip(k_eff, 0, 5000.0)
            if k_eff <= 0 or np.isnan(k_eff) or np.isinf(k_eff):
                results.append(linear_fallback)
            else:
                results.append(k_eff)
        except Exception:
            results.append(linear_fallback)
    return np.array(results)


def geometric_mean_model(phi, k_matrix=None, k_filler=None, mode=None):
    """
    Geometric Mean Model.
    k_eff = k_matrix^φ * k_filler^(1-φ)
    """
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    phi_filler = phi  # FIXED: phi IS the filler fraction
    phi_matrix = 1.0 - phi
    k_matrix_safe = np.maximum(k_matrix, 1e-12)
    k_filler_safe = np.maximum(k_filler, 1e-12)
    result = (k_matrix_safe ** phi_matrix) * (k_filler_safe ** phi_filler)
    return np.clip(result, 0, 5000.0)


def _bottcher_scalar(phi_filler, k_matrix, k_filler):
    """Böttcher self-consistent equation for a single scalar phi.

    (k_eff - k_matrix) / (3*k_eff) = phi_filler * (k_filler - k_matrix) / (k_filler + 2*k_eff)

    FIXED: the previous implementation never used k_filler at all (result was
    just k_matrix / phi), so it could not represent the actual physics of a
    two-component mixture. Solved with a bounded root-finder instead of the
    unconstrained solver used elsewhere, since this equation is quadratic in
    k_eff and an unconstrained guess can converge to a spurious root.
    """
    from scipy.optimize import brentq

    def func(k_eff):
        return (k_eff - k_matrix) / (3.0 * k_eff) - phi_filler * (k_filler - k_matrix) / (k_filler + 2.0 * k_eff)

    lo = max(min(k_matrix, k_filler) * 0.01, 1e-6)
    hi = max(k_matrix, k_filler) * 3.0
    try:
        f_lo, f_hi = func(lo), func(hi)
        if f_lo * f_hi > 0:
            return k_matrix * (1.0 - phi_filler) + k_filler * phi_filler
        return brentq(func, lo, hi, maxiter=200)
    except Exception:
        return k_matrix * (1.0 - phi_filler) + k_filler * phi_filler


def bottcher_model(phi, k_matrix=None, k_filler=None, mode=None):
    """Böttcher Model (self-consistent two-phase mixing equation, 1945/1952)."""
    phi = np.asarray(phi)
    phi_filler = np.clip(phi, 0.001, 0.99)  # phi IS the filler fraction
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    if phi_filler.ndim == 0:
        result = _bottcher_scalar(float(phi_filler), k_matrix, k_filler)
    else:
        result = np.array([_bottcher_scalar(p, k_matrix, k_filler) for p in phi_filler])
    return np.clip(result, 0, 5000.0)


def de_loor_model(phi, k_matrix=None, k_filler=None, mode=None):
    """De Loor Model (1956) - spherical-inclusion mixing formula.

    k_eff = k_matrix * [1 + 3*phi_filler*(k_filler-k_matrix) /
                        (k_filler + 2*k_matrix - phi_filler*(k_filler-k_matrix))]

    FIXED: the previous implementation fetched k_filler but never used it in
    the formula, and its denominator (1 - 2*phi_filler) went negative for any
    phi_filler > 0.5, so every result was clipped to 0.
    """
    phi = np.asarray(phi)
    phi_filler = np.clip(phi, 0.001, 0.99)  # phi IS the filler fraction
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    diff_k = k_filler - k_matrix
    denominator = k_filler + 2.0 * k_matrix - phi_filler * diff_k
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    result = k_matrix * (1.0 + 3.0 * phi_filler * diff_k / denominator)
    return np.clip(result, 0, 5000.0)


def ce_wen_nan_model(phi, k_matrix=None, k_filler=None, mode=None):
    """Nan et al. (1997/2004) Model - spherical particles, ideal interface.

    FIXED: the original formula did not satisfy the boundary conditions
    (k_eff should equal k_matrix at phi=0 and k_filler at phi=1); at phi=1 it
    evaluated to 3*k_matrix + k_filler instead of k_filler, so it always grew
    without bound regardless of mode. In the ideal-interface, spherical-
    particle limit the Nan et al. effective-medium result reduces to the
    classic Maxwell form below, which does satisfy both boundaries.
    """
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    phi_filler = phi  # phi IS the filler fraction
    numerator = k_filler + 2.0 * k_matrix + 2.0 * phi_filler * (k_filler - k_matrix)
    denominator = k_filler + 2.0 * k_matrix - phi_filler * (k_filler - k_matrix)
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    result = k_matrix * numerator / denominator
    return np.clip(result, 0, 5000.0)


def hamilton_crosser_model(phi, n, k_matrix=None, k_filler=None, mode=None):
    """Hamilton-Crosser Model."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    n = max(n, 0.1)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    phi_filler = phi  # FIXED: phi IS the filler fraction
    numerator = k_filler + (n - 1.0) * k_matrix + (n - 1.0) * phi_filler * (k_filler - k_matrix)
    denominator = k_filler + (n - 1.0) * k_matrix - phi_filler * (k_filler - k_matrix)
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    result = k_matrix * numerator / denominator
    return np.clip(result, 0, 5000.0)


def russell_model(phi, k_matrix=None, k_filler=None, mode=None):
    """Russell Model."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    phi_filler = phi  # FIXED: phi IS the filler fraction
    phi_filler_23 = phi_filler ** (2.0/3.0)
    k_ratio = k_matrix / (k_filler + 1e-12)
    numerator = phi_filler_23 + k_ratio * (1.0 - phi_filler_23)
    # FIXED: denominator was missing "+ phi_filler" inside the k_ratio term,
    # which broke the boundary condition (k_eff should equal k_filler at
    # phi_filler=1) and made the model increase regardless of mode.
    denominator = phi_filler_23 - phi_filler + k_ratio * (1.0 - phi_filler_23 + phi_filler)
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    result = k_matrix * numerator / denominator
    return np.clip(result, 0, 5000.0)


def cheng_vachon_model(phi, k_matrix=None, k_filler=None, mode=None):
    """Cheng-Vachon Model."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0.001, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    results = []
    for p in phi:
        phi_filler = p  # FIXED: phi IS the filler fraction (was 1.0 - p)
        try:
            B = np.sqrt(3.0 * phi_filler / 2.0)
            C = np.sqrt(2.0 * phi_filler / 3.0)
            diff_k = k_filler - k_matrix
            if np.abs(diff_k) < 1e-12:
                results.append(k_matrix)
                continue
            
            sqrt_arg1 = C * diff_k * (k_matrix + B * diff_k)
            if sqrt_arg1 <= 0:
                results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
                continue
                
            sqrt_arg2 = C * diff_k
            if sqrt_arg2 <= 0:
                results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
                continue
                
            term1 = 1.0 / np.sqrt(sqrt_arg1 + 1e-12)
            term2 = np.sqrt(k_matrix + B * diff_k)
            term3 = (B / 2.0) * np.sqrt(sqrt_arg2 + 1e-12)
            
            log_arg = (term2 + term3) / (term2 - term3 + 1e-12)
            if log_arg <= 0:
                results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
                continue
            log_term = np.log(log_arg)
            
            k_eff = 1.0 / ((1.0 - B) / (k_matrix + 1e-12) + term1 * log_term)
            if k_eff <= 0 or np.isnan(k_eff) or np.isinf(k_eff):
                results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
            else:
                results.append(np.clip(k_eff, 0, 5000.0))
        except Exception:
            results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
    return np.array(results)


def baschirow_selenew_model(phi, k_matrix=None, k_filler=None, mode=None):
    """Baschirow-Selenew Model."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0.001, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    results = []
    for p in phi:
        phi_filler = p  # FIXED: phi IS the filler fraction (was 1.0 - p)
        try:
            a = (6.0 * phi_filler / np.pi) ** (1.0/3.0)
            diff_k = k_matrix - k_filler
            if np.abs(diff_k) < 1e-12:
                results.append(k_matrix)
                continue
            P = k_filler / diff_k
            term1 = np.pi * a**2 / 4.0
            term2 = a * np.pi * P / 2.0
            log_arg = 1.0 + a / (P + 1e-12)
            if log_arg <= 0:
                results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
                continue
            term3 = 1.0 - (P / (a + 1e-12)) * np.log(log_arg)
            k_eff = (1.0 - term1 + term2 * term3) * k_matrix
            if k_eff <= 0 or np.isnan(k_eff) or np.isinf(k_eff):
                results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
            else:
                results.append(np.clip(k_eff, 0, 5000.0))
        except Exception:
            results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
    return np.array(results)


# ============================================================================
# ADVANCED MODELS (unchanged - keep as is)
# ============================================================================

def gem_equation_model(phi, t, phi_c, k_matrix=None, k_filler=None, mode=None):
    """GEM (Generalized Effective Medium) Equation Model."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0.001, 0.99)
    t = max(t, 0.1)
    phi_c = np.clip(phi_c, 0.01, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    A = (1.0 - phi_c) / (phi_c + 1e-12)
    A = np.clip(A, 0.01, 100.0)
    
    k_matrix_pow = k_matrix ** (1.0 / t) if k_matrix > 0 else 1e-12
    k_filler_pow = k_filler ** (1.0 / t) if k_filler > 0 else 1e-12
    
    results = []
    for p in phi:
        phi_filler = p  # FIXED: phi IS the filler fraction (was phi_matrix = p)
        phi_matrix = 1.0 - p
        try:
            def gem_func(k_eff):
                k_eff = max(k_eff, 1e-12)
                k_eff_pow = k_eff ** (1.0 / t)
                denom1 = k_matrix_pow + A * k_eff_pow + 1e-12
                denom2 = k_filler_pow + A * k_eff_pow + 1e-12
                term1 = phi_matrix * (k_matrix_pow - k_eff_pow) / denom1
                term2 = phi_filler * (k_filler_pow - k_eff_pow) / denom2
                return term1 + term2
            
            lower = min(k_matrix, k_filler) * 0.1
            upper = max(k_matrix, k_filler) * 2.0
            lower = max(lower, 0.1)
            upper = max(upper, lower + 1.0)
            
            f_lower = gem_func(lower)
            f_upper = gem_func(upper)
            
            if f_lower * f_upper > 0:
                k_eff = k_matrix * phi_matrix + k_filler * phi_filler
            else:
                for _ in range(100):
                    mid = (lower + upper) / 2
                    f_mid = gem_func(mid)
                    if abs(f_mid) < 1e-10:
                        lower = upper = mid
                        break
                    if f_lower * f_mid < 0:
                        upper = mid
                        f_upper = f_mid
                    else:
                        lower = mid
                        f_lower = f_mid
                k_eff = (lower + upper) / 2
            
            k_eff = np.clip(k_eff, 0.1, 5000.0)
            if np.isnan(k_eff) or np.isinf(k_eff) or k_eff <= 0:
                results.append(k_matrix * phi_matrix + k_filler * phi_filler)
            else:
                results.append(k_eff)
        except Exception:
            results.append(k_matrix * phi_matrix + k_filler * phi_filler)
    return np.array(results)


def rayleigh_model(phi, C1, C2, k_matrix=None, k_filler=None, mode=None):
    """Rayleigh Model."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    # FIXED: Rayleigh's classical formula is derived for low-conductivity
    # inclusions embedded in a higher-conductivity continuous host (that's
    # required for gamma=(ratio+1)/(ratio-1) to stay positive/well-behaved).
    # The original code always used k_matrix as the host, which is correct
    # for nanothermite (k_matrix > k_filler) but means ratio < 1 in polymer
    # mode (k_filler > k_matrix), making gamma negative and the whole
    # expression unstable. Using whichever phase actually has the higher
    # conductivity as the host fixes this in both modes.
    phi_filler = phi  # phi IS the filler fraction
    phi_matrix = 1.0 - phi
    if k_matrix >= k_filler:
        k_host, k_incl, phi_incl = k_matrix, k_filler, phi_filler
    else:
        k_host, k_incl, phi_incl = k_filler, k_matrix, phi_matrix

    ratio = k_host / (k_incl + 1e-12)
    if np.abs(ratio - 1.0) < 1e-12:
        gamma = 1e6
    else:
        gamma = (ratio + 1.0) / (ratio - 1.0)
    
    denominator = gamma + phi_incl - (C1 / (gamma + 1e-12)) * phi_incl**2 - (C2**2 / (gamma + 1e-12)) * phi_incl**2
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    result = k_host * (1.0 - 2.0 * phi_incl / denominator)
    return np.clip(result, 0, 5000.0)


def halpin_tsai_model(phi, eta, aspect_ratio, k_matrix=None, k_filler=None, mode=None):
    """Halpin-Tsai Model."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']  # FIXED: was never extracted, so the model
                                       # ignored the filler material entirely and
                                       # could only ever predict an increase.
    
    phi_filler = phi  # FIXED: phi IS the filler fraction (was 1.0 - phi)
    aspect_ratio = max(aspect_ratio, 1.0)
    eta = np.clip(eta, 0, 0.99)
    
    if aspect_ratio > 1.0:
        factor = np.sqrt(3.0) * np.log(aspect_ratio)
    else:
        factor = 2.0
    
    # FIXED: the classic Halpin-Tsai reinforcement factor is derived from the
    # filler/matrix conductivity ratio, so its sign flips automatically with
    # mode (negative when k_filler < k_matrix). The fitted `eta` now scales
    # the magnitude/efficiency of that material-driven effect (0-1), rather
    # than being the only source of a term that could never go negative.
    ratio = k_filler / (k_matrix + 1e-12)
    eta_material = (ratio - 1.0) / (ratio + factor)
    eta_effective = eta * eta_material
    
    numerator = 1.0 + factor * eta_effective * phi_filler
    denominator = 1.0 - eta_effective * phi_filler
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    result = k_matrix * numerator / denominator
    return np.clip(result, 0, 5000.0)


def hatta_model(phi, S, k_matrix=None, k_filler=None, mode=None):
    """Hatta Model."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    phi_filler = phi  # FIXED: phi IS the filler fraction (was 1.0 - phi)
    S = max(S, 0.01)
    
    diff_k = k_filler - k_matrix
    if np.abs(diff_k) < 1e-12:
        denominator = S * phi + 1e6
    else:
        denominator = S * phi + k_matrix / diff_k
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    result = k_matrix * (1.0 + phi_filler / denominator)
    return np.clip(result, 0, 5000.0)


def modified_hashin_shtrikman_model(phi, lamda, k_matrix=None, k_filler=None, mode=None):
    """Modified Hashin-Shtrikman Model."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    phi_filler = phi  # FIXED: phi IS the filler fraction (was phi_matrix = phi)
    phi_matrix = 1.0 - phi
    lamda = np.clip(lamda, 0, 1.0)
    
    # FIXED: normalize by k_matrix before applying the Clausius-Mossotti-style
    # factor (k-1)/(k+2). That form assumes conductivities of order unity; with
    # raw W/m*K values (tens to hundreds) both terms saturated near +1 and the
    # model was nearly insensitive to phi and to which material was which.
    k_matrix_n = 1.0
    k_filler_n = k_filler / (k_matrix + 1e-12)
    M = ((k_matrix_n - 1.0) / (k_matrix_n + 2.0)) * phi_matrix + ((k_filler_n - 1.0) / (k_filler_n + 2.0)) * phi_filler
    numerator = 1.0 + 2.0 * lamda * M
    denominator = 1.0 - lamda * M
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    result = k_matrix * numerator / denominator
    return np.clip(result, 0, 5000.0)


def tsao_model(phi, k_matrix=None, k_filler=None, mode=None):
    """Tsao Model."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0.001, 0.99)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    results = []
    for p in phi:
        phi_filler = p  # FIXED: phi IS the filler fraction (was 1.0 - p)
        try:
            B = np.sqrt(3.0 * phi_filler / 2.0)
            C = 4.0 * np.sqrt(2.0 / (3.0 * phi_filler + 1e-12))
            diff_k = k_filler - k_matrix
            if np.abs(diff_k) < 1e-12:
                results.append(k_matrix)
                continue
            
            sqrt_arg1 = C * diff_k * (k_matrix + B * diff_k)
            if sqrt_arg1 <= 0:
                results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
                continue
                
            sqrt_arg2 = C * diff_k
            if sqrt_arg2 <= 0:
                results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
                continue
                
            term1 = 1.0 / np.sqrt(sqrt_arg1 + 1e-12)
            term2 = np.sqrt(k_matrix + B * diff_k)
            term3 = (B / 2.0) * np.sqrt(sqrt_arg2 + 1e-12)
            
            log_arg = (term2 + term3) / (term2 - term3 + 1e-12)
            if log_arg <= 0:
                results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
                continue
            log_term = np.log(log_arg)
            
            k_eff = 1.0 / ((1.0 - B) / (k_matrix + 1e-12) + term1 * log_term)
            if k_eff <= 0 or np.isnan(k_eff) or np.isinf(k_eff):
                results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
            else:
                results.append(np.clip(k_eff, 0, 5000.0))
        except Exception:
            results.append(k_matrix * (1.0 - phi_filler) + k_filler * phi_filler)
    return np.array(results)


def hamilton_crosser_extended_model(phi, n, phi_i, k_matrix=None, k_filler=None, mode=None):
    """Hamilton-Crosser Extended Model."""
    phi = np.asarray(phi)
    phi = np.clip(phi, 0, 0.99)
    n = max(n, 0.1)
    phi_i = np.clip(phi_i, 0.01, 1.0)
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    phi_filler = phi  # phi IS the filler fraction
    phi_effective = phi_filler * phi_i
    # FIXED: signs here were flipped relative to the base Hamilton-Crosser
    # formula (hamilton_crosser_model above), which breaks the k_eff -> k_filler
    # boundary condition as phi_effective -> 1 and made the direction wrong
    # in both modes. Mirrored to match the verified-correct base formula.
    numerator = k_filler + (n - 1.0) * k_matrix + (n - 1.0) * phi_effective * (k_filler - k_matrix)
    denominator = k_filler + (n - 1.0) * k_matrix - phi_effective * (k_filler - k_matrix)
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    result = k_matrix * numerator / denominator
    return np.clip(result, 0, 5000.0)


def _hashin_shtrikman_bound(phi_filler, k_matrix, k_filler, upper):
    """Rigorous Hashin-Shtrikman bound, keyed on which PHASE conducts better
    (not on which phase is geometrically 'matrix' vs 'filler').

    FIXED: the previous Upper/Lower implementations always used k_matrix as
    the 'host' phase for Upper and k_filler as host for Lower, regardless of
    which material actually had the higher conductivity. That's correct for
    nanothermite (k_matrix > k_filler) but backwards for polymer mode
    (k_filler > k_matrix) - which is exactly why Upper had the wrong
    direction in polymer mode and could even produce a negative (clipped to
    0) result for large conductivity ratios. Using k_hi/k_lo (rather than
    k_matrix/k_filler) makes both bounds correct in either mode automatically.
    """
    k_hi, k_lo = max(k_matrix, k_filler), min(k_matrix, k_filler)
    # phi_hi = volume fraction of whichever phase has the higher conductivity
    phi_hi = phi_filler if k_filler >= k_matrix else (1.0 - phi_filler)
    phi_lo = 1.0 - phi_hi

    if upper:
        host, other, phi_other = k_hi, k_lo, phi_lo
    else:
        host, other, phi_other = k_lo, k_hi, phi_hi

    phi_host = 1.0 - phi_other
    denom = 1.0 / (other - host + 1e-12) + phi_host / (3.0 * host + 1e-12)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return host + phi_other / denom


def maxwell_eucken_upper(phi, k_matrix=None, k_filler=None, mode=None):
    """Maxwell-Eucken / Hashin-Shtrikman Upper Bound (HS+)."""
    phi = np.asarray(phi)
    phi_filler = np.clip(phi, 0, 0.99)  # phi IS the filler fraction
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    result = _hashin_shtrikman_bound(phi_filler, k_matrix, k_filler, upper=True)
    return np.clip(result, 0, 5000.0)


def maxwell_eucken_lower(phi, k_matrix=None, k_filler=None, mode=None):
    """Maxwell-Eucken / Hashin-Shtrikman Lower Bound (HS-)."""
    phi = np.asarray(phi)
    phi_filler = np.clip(phi, 0, 0.99)  # phi IS the filler fraction
    
    if mode is None:
        mode = "nanothermite"
    
    constants = get_mode_constants(mode, k_matrix, k_filler)
    k_matrix = constants['k_matrix']
    k_filler = constants['k_filler']
    
    result = _hashin_shtrikman_bound(phi_filler, k_matrix, k_filler, upper=False)
    return np.clip(result, 0, 5000.0)


# ============================================================================
# MODEL CATEGORIES AND REGISTRY
# ============================================================================

def get_model_category_names(mode="nanothermite"):
    """Get category names based on mode."""
    if mode == "nanothermite":
        return {
            "Basic Models": ["Parallel", "Series", "Maxwell-Eucken", "Bruggeman", 
                           "Agari", "Lewis-Nielsen", "Percolation", "Agari-Percolation"],
            "Classic Models": ["Russell", "Cheng-Vachon", "Baschirow-Selenew", 
                             "Hamilton-Crosser", "EMT Equation", "Geometric Mean",
                             "Böttcher", "De Loor", "Ce Wen Nan"],
            "Advanced Models": ["Rayleigh", "Halpin-Tsai", "Hatta", 
                              "Modified Hashin-Shtrikman", "Tsao", 
                              "Hamilton-Crosser Extended",
                              "Maxwell-Eucken Upper", "Maxwell-Eucken Lower",
                              "GEM Equation"],
            "Discovered Models": []
        }
    else:  # polymer
        return {
            "Basic Models": ["Parallel", "Series", "Maxwell-Eucken", "Bruggeman", 
                           "Agari", "Lewis-Nielsen", "Percolation", "Agari-Percolation"],
            "Classic Models": ["Russell", "Cheng-Vachon", "Baschirow-Selenew", 
                             "Hamilton-Crosser", "EMT Equation", "Geometric Mean",
                             "Böttcher", "De Loor", "Ce Wen Nan"],
            "Advanced Models": ["Rayleigh", "Halpin-Tsai", "Hatta", 
                              "Modified Hashin-Shtrikman", "Tsao", 
                              "Hamilton-Crosser Extended",
                              "Maxwell-Eucken Upper", "Maxwell-Eucken Lower",
                              "GEM Equation"],
            "Discovered Models": []
        }


def get_model_registry(mode="nanothermite"):
    """Get the model registry with appropriate descriptions for the mode."""
    
    if mode == "nanothermite":
        matrix_desc = "reductant"
        filler_desc = "oxidizer"
        mode_name = "Nanothermite"
    else:
        matrix_desc = "polymer matrix"
        filler_desc = "filler"
        mode_name = "Polymer Composite"
    
    registry = {
        # ===== BASIC MODELS =====
        "Parallel": {
            "func": parallel_model,
            "params": [],
            "fixed": True,
            "color": "#1f77b4",
            "linestyle": "-",
            "linewidth": 2.0,
            "category": "Basic Models",
            "description": f"Upper Hashin-Shtrikman bound (Rule of Mixtures) - φ = {filler_desc} fraction",
            "reference": "Hashin & Shtrikman (1962)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Series": {
            "func": series_model,
            "params": [],
            "fixed": True,
            "color": "#ff7f0e",
            "linestyle": "--",
            "linewidth": 2.0,
            "category": "Basic Models",
            "description": f"Lower Hashin-Shtrikman bound (Inverse Rule of Mixtures) - φ = {filler_desc} fraction",
            "reference": "Hashin & Shtrikman (1962)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Maxwell-Eucken": {
            "func": maxwell_eucken,
            "params": [],
            "fixed": True,
            "color": "#2ca02c",
            "linestyle": "-.",
            "linewidth": 2.0,
            "category": "Basic Models",
            "description": f"Dilute spherical inclusions in continuous matrix - φ = {filler_desc} fraction",
            "reference": "Maxwell (1873), Eucken (1932)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Bruggeman": {
            "func": bruggeman_model,
            "params": [],
            "fixed": True,
            "color": "#d62728",
            "linestyle": ":",
            "linewidth": 2.5,
            "category": "Basic Models",
            "description": f"Effective medium with particle interactions - φ = {filler_desc} fraction",
            "reference": "Bruggeman (1935)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Agari": {
            "func": agari_model,
            "params": ["C1", "C2"],
            "fixed": False,
            "color": "#9467bd",
            "linestyle": "-",
            "linewidth": 2.0,
            "category": "Basic Models",
            "p0": [1.0, 1.0],
            "bounds": ([0.0, -10.0], [10.0, 10.0]),
            "description": f"Empirical model with conductive pathways - φ = {filler_desc} fraction",
            "reference": "Agari et al. (1993)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Lewis-Nielsen": {
            "func": lewis_nielsen_model,
            "params": ["A", "phi_m"],
            "fixed": False,
            "color": "#8c564b",
            "linestyle": "-",
            "linewidth": 2.0,
            "category": "Basic Models",
            "p0": [2.0, 0.6],
            "bounds": ([0.1, 0.1], [100.0, 0.9]),
            "description": f"Particle geometry and maximum packing fraction - φ = {filler_desc} fraction",
            "reference": "Lewis & Nielsen (1970)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Percolation": {
            "func": percolation_model,
            "params": ["k0", "phi_c", "t"],
            "fixed": False,
            "color": "#e377c2",
            "linestyle": "-",
            "linewidth": 2.0,
            "category": "Basic Models",
            "p0": [50.0, 0.3, 2.0],
            "bounds": ([1.0, 0.01, 0.1], [1000.0, 0.9, 10.0]),
            "description": f"Thermal percolation threshold model - φ = {filler_desc} fraction",
            "reference": "Kirkpatrick (1973)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Agari-Percolation": {
            "func": agari_percolation_model,
            "params": ["C1", "C2", "k0", "phi_c", "t"],
            "fixed": False,
            "color": "#7f7f7f",
            "linestyle": "-",
            "linewidth": 2.5,
            "category": "Basic Models",
            "p0": [1.0, 1.0, 50.0, 0.3, 2.0],
            "bounds": ([0.0, -10.0, 1.0, 0.01, 0.1], [10.0, 10.0, 1000.0, 0.9, 10.0]),
            "description": f"Hybrid Agari-Percolation model - φ = {filler_desc} fraction",
            "reference": "Agari (1993) + Percolation",
            "mode": mode,
            "mode_name": mode_name
        },
        # ===== CLASSIC MODELS =====
        "EMT Equation": {
            "func": emt_equation_model,
            "params": [],
            "fixed": True,
            "color": "#2ecc71",
            "linestyle": "-.",
            "linewidth": 1.8,
            "category": "Classic Models",
            "description": f"Effective Medium Theory - Two-phase symmetric model - φ = {filler_desc} fraction",
            "reference": "Håkansson & Ross (1990)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Geometric Mean": {
            "func": geometric_mean_model,
            "params": [],
            "fixed": True,
            "color": "#3498db",
            "linestyle": "--",
            "linewidth": 1.8,
            "category": "Classic Models",
            "description": f"Logarithmic mixing rule - φ = {filler_desc} fraction",
            "reference": "Bruggeman (1935), Böttcher (1952)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Böttcher": {
            "func": bottcher_model,
            "params": [],
            "fixed": True,
            "color": "#9b59b6",
            "linestyle": ":",
            "linewidth": 1.8,
            "category": "Classic Models",
            "description": f"Simple rational function model - φ = {filler_desc} fraction",
            "reference": "Böttcher (1952)",
            "mode": mode,
            "mode_name": mode_name
        },
        "De Loor": {
            "func": de_loor_model,
            "params": [],
            "fixed": True,
            "color": "#1abc9c",
            "linestyle": "-.",
            "linewidth": 1.8,
            "category": "Classic Models",
            "description": f"Rational function with volume fraction - φ = {filler_desc} fraction",
            "reference": "DeLoor (1956)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Ce Wen Nan": {
            "func": ce_wen_nan_model,
            "params": [],
            "fixed": True,
            "color": "#e67e22",
            "linestyle": "--",
            "linewidth": 1.8,
            "category": "Classic Models",
            "description": f"Modified mixing rule with shape factor - φ = {filler_desc} fraction",
            "reference": "Nan et al. (2004)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Hamilton-Crosser": {
            "func": hamilton_crosser_model,
            "params": ["n"],
            "fixed": False,
            "color": "#e74c3c",
            "linestyle": "-",
            "linewidth": 1.8,
            "category": "Classic Models",
            "p0": [3.0],
            "bounds": ([0.1], [100.0]),
            "description": f"Considers filler shape (sphericity parameter) - φ = {filler_desc} fraction",
            "reference": "Hamilton & Crosser (1962)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Cheng-Vachon": {
            "func": cheng_vachon_model,
            "params": [],
            "fixed": True,
            "color": "#16a085",
            "linestyle": ":",
            "linewidth": 1.8,
            "category": "Classic Models",
            "description": f"Parabolic distribution of discontinuous phase - φ = {filler_desc} fraction",
            "reference": "Cheng & Vachon (1969)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Russell": {
            "func": russell_model,
            "params": [],
            "fixed": True,
            "color": "#c0392b",
            "linestyle": "-.",
            "linewidth": 1.8,
            "category": "Classic Models",
            "description": f"Cubical dispersed phase assumption - φ = {filler_desc} fraction",
            "reference": "Russell (1935)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Baschirow-Selenew": {
            "func": baschirow_selenew_model,
            "params": [],
            "fixed": True,
            "color": "#8e44ad",
            "linestyle": "--",
            "linewidth": 1.8,
            "category": "Classic Models",
            "description": f"Spherical fillers with isotropic phases - φ = {filler_desc} fraction",
            "reference": "Leung et al. (2013)",
            "mode": mode,
            "mode_name": mode_name
        },
        # ===== ADVANCED MODELS =====
        "GEM Equation": {
            "func": gem_equation_model,
            "params": ["t", "phi_c"],
            "fixed": False,
            "color": "#f1c40f",
            "linestyle": "-",
            "linewidth": 2.0,
            "category": "Advanced Models",
            "p0": [2.0, 0.3],
            "bounds": ([1.0, 0.01], [5.0, 0.9]),
            "description": f"Generalized Effective Medium (GEM) - Combines percolation and EMT - φ = {filler_desc} fraction",
            "reference": "McLachlan et al. (1990)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Rayleigh": {
            "func": rayleigh_model,
            "params": ["C1", "C2"],
            "fixed": False,
            "color": "#d35400",
            "linestyle": "-",
            "linewidth": 1.8,
            "category": "Advanced Models",
            "p0": [0.3058, 0.0134],
            "bounds": ([0.0, 0.0], [1.0, 1.0]),
            "description": f"Thermal barrier effects perpendicular to fibers - φ = {filler_desc} fraction",
            "reference": "Rayleigh (1911)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Halpin-Tsai": {
            "func": halpin_tsai_model,
            "params": ["eta", "aspect_ratio"],
            "fixed": False,
            "color": "#27ae60",
            "linestyle": "-",
            "linewidth": 1.8,
            "category": "Advanced Models",
            "p0": [0.5, 10.0],
            "bounds": ([0.0, 1.0], [1.0, 1000.0]),
            "description": f"Plane field equation for plate-shaped composites - φ = {filler_desc} fraction",
            "reference": "Affdl & Kardos (1976)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Hatta": {
            "func": hatta_model,
            "params": ["S"],
            "fixed": False,
            "color": "#2980b9",
            "linestyle": "-",
            "linewidth": 1.8,
            "category": "Advanced Models",
            "p0": [0.5],
            "bounds": ([0.01], [10.0]),
            "description": f"Laminar thermally conductive fillers - φ = {filler_desc} fraction",
            "reference": "Hatta et al. (1992)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Modified Hashin-Shtrikman": {
            "func": modified_hashin_shtrikman_model,
            "params": ["lamda"],
            "fixed": False,
            "color": "#2c3e50",
            "linestyle": "-",
            "linewidth": 1.8,
            "category": "Advanced Models",
            "p0": [0.5],
            "bounds": ([0.0], [2.0]),
            "description": f"Corrects for particle distribution, size, phonon scattering - φ = {filler_desc} fraction",
            "reference": "Ngo et al. (2017)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Tsao": {
            "func": tsao_model,
            "params": [],
            "fixed": True,
            "color": "#f39c12",
            "linestyle": ":",
            "linewidth": 1.8,
            "category": "Advanced Models",
            "description": f"Parabolic distribution without additional parameters - φ = {filler_desc} fraction",
            "reference": "Tsao (1961)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Hamilton-Crosser Extended": {
            "func": hamilton_crosser_extended_model,
            "params": ["n", "phi_i"],
            "fixed": False,
            "color": "#8e44ad",
            "linestyle": "-",
            "linewidth": 1.8,
            "category": "Advanced Models",
            "p0": [3.0, 0.5],
            "bounds": ([0.1, 0.01], [100.0, 1.0]),
            "description": f"Considers filler geometry, particle size, packing sequence - φ = {filler_desc} fraction",
            "reference": "Agarwal et al. (2008)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Maxwell-Eucken Upper": {
            "func": maxwell_eucken_upper,
            "params": [],
            "fixed": True,
            "color": "#1abc9c",
            "linestyle": "--",
            "linewidth": 1.8,
            "category": "Advanced Models",
            "description": f"Maxwell-Eucken Upper Bound (HS+) - φ = {filler_desc} fraction",
            "reference": "Hashin & Shtrikman (1962)",
            "mode": mode,
            "mode_name": mode_name
        },
        "Maxwell-Eucken Lower": {
            "func": maxwell_eucken_lower,
            "params": [],
            "fixed": True,
            "color": "#e74c3c",
            "linestyle": "-.",
            "linewidth": 1.8,
            "category": "Advanced Models",
            "description": f"Maxwell-Eucken Lower Bound (HS-) - φ = {filler_desc} fraction",
            "reference": "Hashin & Shtrikman (1962)",
            "mode": mode,
            "mode_name": mode_name
        },
    }
    
    return registry


# ============================================================================
# HELPER FUNCTION TO ADD DISCOVERED MODELS
# ============================================================================

def add_discovered_model(model_name, model_data, mode="nanothermite"):
    """Add a discovered model to the registry."""
    registry = get_model_registry(mode)
    
    required_fields = ['func', 'params', 'p0', 'bounds', 'description']
    for field in required_fields:
        if field not in model_data:
            raise ValueError(f"Missing required field: {field}")
    
    registry[model_name] = model_data
    
    # === FIX: Update the global registry too ===
    global MODEL_REGISTRY
    MODEL_REGISTRY[model_name] = model_data
    
    # Also update categories
    global MODEL_CATEGORIES
    if "Discovered Models" not in MODEL_CATEGORIES:
        MODEL_CATEGORIES["Discovered Models"] = []
    if model_name not in MODEL_CATEGORIES["Discovered Models"]:
        MODEL_CATEGORIES["Discovered Models"].append(model_name)
    
    return registry


# ============================================================================
# FUNCTION TO UPDATE GLOBAL REGISTRY
# ============================================================================

def set_global_registry(mode="nanothermite"):
    """Update the global MODEL_REGISTRY and MODEL_CATEGORIES for a given mode."""
    global MODEL_REGISTRY, MODEL_CATEGORIES
    MODEL_REGISTRY = get_model_registry(mode)
    MODEL_CATEGORIES = get_model_category_names(mode)
    
    # Ensure Discovered Models category exists
    if "Discovered Models" not in MODEL_CATEGORIES:
        MODEL_CATEGORIES["Discovered Models"] = []
    
    # Find all discovered models and add them to the category
    discovered_models = [name for name, meta in MODEL_REGISTRY.items() 
                        if meta.get("discovered", False)]
    MODEL_CATEGORIES["Discovered Models"] = discovered_models
    
    return MODEL_REGISTRY, MODEL_CATEGORIES


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

# Legacy constants - kept for backward compatibility
k_Al = NANO_k_Al
k_CuO = NANO_k_CuO
rho_Al = NANO_rho_Al
rho_CuO = NANO_rho_CuO

# Legacy function
def mass_to_volume_fraction_legacy(mass_fraction_cuo):
    """Legacy function for CuO mass fraction conversion."""
    return mass_to_volume_fraction(mass_fraction_cuo, rho_Al, rho_CuO)

# Legacy model categories (will be updated by main window)
MODEL_CATEGORIES = get_model_category_names("nanothermite")
MODEL_REGISTRY = get_model_registry("nanothermite")