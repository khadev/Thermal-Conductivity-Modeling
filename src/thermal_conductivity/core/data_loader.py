"""Data loading and validation – φ = FILLER volume fraction for both modes."""

import pandas as pd
import numpy as np
import os

from .material_constants import (
    get_mode_info, get_materials_for_mode,
    get_default_materials, get_k_value,
    get_rho_value
)


def load_data(filepath, mode="nanothermite"):
    """
    Load thermal conductivity data from txt, csv, or Excel.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    mode : str
        'nanothermite' or 'polymer'
    
    For BOTH modes: φ = FILLER volume fraction
    - Nanothermite: φ = Oxidizer (filler) fraction, (1-φ) = Reductant (matrix)
    - Polymer: φ = Filler fraction, (1-φ) = Polymer Matrix fraction
    """
    filename = os.path.basename(filepath)
    skip_files = ['readme.md', 'readme.txt', 'requirements.txt', 'app.ico', 'run.py', 'setup.py']
    if filename.lower() in skip_files:
        raise ValueError(f"'{filename}' is not a valid data file.")
    
    ext = str(filepath).split(".")[-1].lower()

    if ext not in ["txt", "csv", "tsv", "xlsx", "xls"]:
        raise ValueError(f"Unsupported format: {ext}. Please use .txt, .csv, .xlsx, or .xls files.")

    df = None
    try:
        if ext in ["txt", "csv", "tsv"]:
            delimiters = ['\t', ',', ';', '|', ' ']
            df = None
            last_error = None
            
            for delim in delimiters:
                try:
                    df = pd.read_csv(filepath, sep=delim, engine='python', 
                                   encoding='utf-8-sig',
                                   skipinitialspace=True)
                    if df.shape[1] >= 2:
                        break
                    else:
                        df = None
                except Exception as e:
                    last_error = str(e)
                    continue
            
            if df is None or df.shape[1] < 2:
                try:
                    df = pd.read_csv(filepath, sep=None, engine='python', 
                                   encoding='utf-8-sig',
                                   skipinitialspace=True)
                except Exception as e:
                    raise ValueError(f"Could not parse file. Last error: {last_error}")
                    
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported format: {ext}")
            
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")

    if df is None or df.empty:
        raise ValueError("File is empty.")

    df.columns = df.columns.str.strip()
    
    # Try to find phi column (filler fraction)
    phi_col = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['phi', 'φ', 'filler_volume_fraction', 'filler fraction', 
                         'volume_fraction', 'vol_fraction', 'phi_filler']:
            phi_col = col
            break
        if 'phi' in col_lower or 'filler' in col_lower:
            phi_col = col
            break
    
    # Try to find k_meas column
    k_col = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['k_meas', 'k', 'thermal_conductivity', 'conductivity', 
                         'k_meas (w/m·k)', 'k (w/m·k)']:
            k_col = col
            break
        if 'k_' in col_lower or 'conduct' in col_lower:
            k_col = col
            break
    
    if phi_col is None and len(df.columns) >= 2:
        phi_col = df.columns[0]
        print(f"[DEBUG] Assuming first column '{phi_col}' is phi (filler fraction)")
    
    if k_col is None and len(df.columns) >= 2:
        if phi_col == df.columns[0]:
            k_col = df.columns[1]
        else:
            k_col = df.columns[1]
        print(f"[DEBUG] Assuming column '{k_col}' is k_meas")
    
    if phi_col is None:
        raise ValueError(f"Missing column: 'phi'. Found: {', '.join(df.columns)}")
    if k_col is None:
        raise ValueError(f"Missing column: 'k_meas'. Found: {', '.join(df.columns)}")
    
    try:
        df['phi'] = pd.to_numeric(df[phi_col], errors='coerce')
        df['k_meas'] = pd.to_numeric(df[k_col], errors='coerce')
    except Exception as e:
        raise ValueError(f"Failed to convert data to numeric: {str(e)}")
    
    # Check for k_std_dev
    if "k_std_dev" not in df.columns:
        std_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'std' in col_lower or 'dev' in col_lower:
                std_col = col
                break
        
        if std_col:
            try:
                df['k_std_dev'] = pd.to_numeric(df[std_col], errors='coerce')
            except:
                df["k_std_dev"] = np.nan
        else:
            df["k_std_dev"] = np.nan
    else:
        df["k_std_dev"] = pd.to_numeric(df["k_std_dev"], errors='coerce')
    
    # Drop rows with NaN values
    initial_count = len(df)
    df = df.dropna(subset=['phi', 'k_meas'])
    df = df[(df['phi'] >= 0) & (df['phi'] <= 1)]
    
    if len(df) < initial_count:
        print(f"[DEBUG] Dropped {initial_count - len(df)} rows with invalid values")
    
    if len(df) < 2:
        raise ValueError(f"Need at least 2 valid data points. Found {len(df)}.")
    
    df = df.sort_values('phi').reset_index(drop=True)

    # ===== IMPORTANT: φ = FILLER fraction for BOTH modes =====
    # For nanothermite: φ = Oxidizer (filler), (1-φ) = Reductant (matrix)
    # For polymer: φ = Filler, (1-φ) = Polymer Matrix
    
    df["Filler_volume_fraction"] = df["phi"].values
    df["Matrix_volume_fraction"] = 1.0 - df["phi"].values
    
    # Legacy compatibility columns
    if mode == "nanothermite":
        df["Oxy_volume_fraction"] = df["phi"].values
        df["Red_volume_fraction"] = 1.0 - df["phi"].values
        df["phi_red"] = 1.0 - df["phi"].values  # Reductant fraction
        df["phi_oxy"] = df["phi"].values        # Oxidizer fraction
    else:  # polymer
        df["Red_volume_fraction"] = df["phi"].values
        df["Oxy_volume_fraction"] = 1.0 - df["phi"].values
        df["phi_red"] = df["phi"].values

    df["k_std_dev"] = df["k_std_dev"].fillna(0)

    print(f"[DEBUG] Successfully loaded {len(df)} data points")
    print(f"[DEBUG] φ (filler) range: {df['phi'].min():.3f} - {df['phi'].max():.3f}")
    print(f"[DEBUG] k range: {df['k_meas'].min():.2f} - {df['k_meas'].max():.2f}")
    print(f"[DEBUG] Mode: {mode}")

    return df


def get_mode_lists(mode="nanothermite"):
    """Get material lists based on mode."""
    mode_info = get_mode_info(mode)
    materials = get_materials_for_mode(mode)
    
    return {
        'matrix_list': materials.get('matrix_list', []),
        'filler_list': materials.get('filler_list', []),
        'matrix_label': mode_info.get('matrix_label', 'Matrix'),
        'filler_label': mode_info.get('filler_label', 'Filler'),
        'mode_label': mode_info.get('name', 'Unknown'),
        'mode_icon': mode_info.get('icon', '🔬')
    }


def get_material_constants(mode="nanothermite", matrix=None, filler=None):
    """Get material constants based on mode and specific materials."""
    if matrix is None:
        defaults = get_default_materials(mode)
        matrix = defaults['matrix']
    if filler is None:
        defaults = get_default_materials(mode)
        filler = defaults['filler']
    
    return {
        'k_matrix': get_k_value(mode, 'matrix', matrix),
        'k_filler': get_k_value(mode, 'filler', filler),
        'rho_matrix': get_rho_value(mode, 'matrix', matrix),
        'rho_filler': get_rho_value(mode, 'filler', filler),
        'mode': mode,
        'matrix': matrix,
        'filler': filler
    }


def get_default_materials(mode="nanothermite"):
    """Get default matrix and filler names for a mode."""
    from .material_constants import get_default_materials as get_defaults
    return get_defaults(mode)
