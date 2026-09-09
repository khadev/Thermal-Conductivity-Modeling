"""
Convert phi_red to Red_volume_fraction (φ)
and Oxy_volume_fraction (1-φ)
"""

import pandas as pd
import numpy as np

# Densities (kg/m³)
rho_Al = 2700.0
rho_CuO = 6315.0

def convert_mass_to_volume(df):
    """
    Convert phi_red to Red_volume_fraction (φ) and Oxy_volume_fraction (1-φ).
    
    Parameters:
    -----------
    df : DataFrame with column 'phi_red'
    
    Returns:
    --------
    DataFrame with added columns:
        - Red_volume_fraction (φ)  = Al volume fraction (matrix)
        - Oxy_volume_fraction (1-φ) = CuO volume fraction (filler)
    """
    # Mass fractions
    m_cuo = df["phi_red"].values
    m_al = 1.0 - m_cuo
    
    # Volume fractions
    v_cuo = m_cuo / rho_CuO
    v_al = m_al / rho_Al
    total_v = v_cuo + v_al
    
    # Reductant volume fraction (φ) = Al fraction
    df["Red_volume_fraction"] = v_al / total_v
    
    # Oxidizer volume fraction (1-φ) = CuO fraction
    df["Oxy_volume_fraction"] = v_cuo / total_v
    
    # Verify: Red + Oxy = 1
    # df["Check"] = df["Red_volume_fraction"] + df["Oxy_volume_fraction"]
    
    return df


# ============================================================================
# Example Usage
# ============================================================================

# Create sample data with phi_red
data = {
    "phi_red": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 
                          0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 
                          0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
    "k_meas": [237.0, 180.5, 142.3, 110.2, 85.4, 65.1, 50.8, 
               42.3, 38.1, 36.2, 35.5, 35.1, 34.8, 34.2, 
               33.8, 33.5, 33.2, 33.1, 33.0, 33.0, 33.0],
    "k_std_dev": [5.0, 4.2, 3.8, 3.1, 2.5, 2.0, 1.8, 
                  1.5, 1.4, 1.3, 1.2, 1.2, 1.1, 1.0, 
                  1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.5]
}

df = pd.DataFrame(data)

# Convert
df = convert_mass_to_volume(df)

# ============================================================================
# Display Results
# ============================================================================

print("=" * 80)
print("CONVERSION RESULTS")
print("=" * 80)
print(f"{'CuO Mass':<12} {'Red (φ)':<14} {'Oxy (1-φ)':<14} {'k_meas':<10}")
print("-" * 80)

for i, row in df.iterrows():
    print(f"{row['phi_red']:<12.4f} "
          f"{row['Red_volume_fraction']:<14.6f} "
          f"{row['Oxy_volume_fraction']:<14.6f} "
          f"{row['k_meas']:<10.1f}")

print("-" * 80)
print(f"Total data points: {len(df)}")
print(f"φ (Red) range: {df['Red_volume_fraction'].min():.4f} to {df['Red_volume_fraction'].max():.4f}")
print(f"φ (Red) + (1-φ) (Oxy) = {df['Red_volume_fraction'].iloc[0] + df['Oxy_volume_fraction'].iloc[0]:.2f} (should be 1.00)")

# ============================================================================
# Save to CSV
# ============================================================================

df.to_csv("converted_data.csv", index=False)
print("\nData saved to: converted_data.csv")

# ============================================================================
# Verify the New DataTable Format
# ============================================================================

print("\n" + "=" * 80)
print("NEW DATA TABLE FORMAT (φ = Reductant)")
print("=" * 80)
print(f"{'Phi Red (φ)':<14} {'k_meas (W/m·K)':<16} {'k_std_dev':<12} {'Oxy (1-φ)':<14}")
print("-" * 80)

for i, row in df.iterrows():
    print(f"{row['Red_volume_fraction']:<14.6f} "
          f"{row['k_meas']:<16.2f} "
          f"{row['k_std_dev']:<12.2f} "
          f"{row['Oxy_volume_fraction']:<14.6f}")
