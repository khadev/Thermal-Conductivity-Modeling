"""
generate_polymer_data_all_models.py
Generate polymer data that fits multiple models well.
"""

import numpy as np
import pandas as pd

def generate_polymer_data_all_models():
    """Generate polymer data that fits multiple models well."""
    
    phi = np.linspace(0, 0.3, 15)
    k_matrix = 0.2
    k_filler = 300.0
    
    # Use Maxwell-Eucken model
    numerator = k_filler + 2*k_matrix + 2*phi*(k_filler - k_matrix)
    denominator = k_filler + 2*k_matrix - phi*(k_filler - k_matrix)
    k_meas = k_matrix * numerator / denominator
    
    # Add small noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, len(phi))
    k_meas_noisy = k_meas + noise
    k_meas_noisy = np.maximum(k_meas_noisy, 0.01)
    
    df = pd.DataFrame({
        'phi': phi,
        'k_meas': k_meas_noisy,
        'k_std_dev': np.full(len(phi), 0.02)
    })
    
    return df

if __name__ == "__main__":
    df = generate_polymer_data_all_models()
    df.to_csv('polymer_maxwell_data.txt', sep='\t', index=False)
    print("Generated polymer_maxwell_data.txt")
    print(f"  φ range: {df['phi'].min():.3f} - {df['phi'].max():.3f}")
    print(f"  k at φ=0: {df['k_meas'].iloc[0]:.4f} W/m·K")
    print(f"  k at φ=0.3: {df['k_meas'].iloc[-1]:.4f} W/m·K")
    print(f"  Trend: {'INCREASING ✓' if df['k_meas'].iloc[-1] > df['k_meas'].iloc[0] else 'DECREASING ✗'}")