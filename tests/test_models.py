import numpy as np
import pytest
from thermal_conductivity.core.models import (
    parallel_model, series_model, maxwell_eucken,
    bruggeman_model, agari_model, lewis_nielsen_model,
    percolation_model, agari_percolation_model,
    mass_to_volume_fraction, k_Al, k_CuO
)


def test_mass_to_volume_fraction():
    assert mass_to_volume_fraction(0.0) == 0.0
    assert mass_to_volume_fraction(1.0) == 1.0


def test_parallel_model_bounds():
    phi = np.linspace(0, 1, 11)
    k = parallel_model(phi)
    assert np.isclose(k[0], k_Al)
    assert np.isclose(k[-1], k_CuO)


def test_series_model_bounds():
    phi = np.linspace(0, 1, 11)
    k = series_model(phi)
    assert np.isclose(k[0], k_Al)
    assert np.isclose(k[-1], k_CuO)


def test_maxwell_eucken_at_zero():
    assert np.isclose(maxwell_eucken(0.0), k_Al)


def test_bruggeman_at_bounds():
    assert np.isclose(bruggeman_model(0.0), k_Al, rtol=0.01)
    assert np.isclose(bruggeman_model(1.0), k_CuO, rtol=0.01)


def test_agari_with_unity_params():
    k = agari_model(0.5, 1.0, 1.0)
    assert np.isfinite(k)
    assert k > 0


def test_percolation_below_threshold():
    k = percolation_model(0.1, 50, 0.3, 2.0)
    assert np.isfinite(k)
