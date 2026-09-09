"""Central repository for all material data - Thermal Conductivity Suite."""

# ============================================================================
# MODE DEFINITIONS
# ============================================================================

MODES = {
    "nanothermite": {
        "id": "nanothermite",
        "name": "Nanothermite",
        "icon": "🔥",
        "description": "Energetic Material Composites",
        "default_matrix": "Al",
        "default_filler": "CuO",
        "matrix_label": "Reductant (Matrix)",
        "filler_label": "Oxidizer (Filler)",
        "phi_label": "Filler Volume Fraction, φ",
        "one_minus_phi_label": "Matrix Volume Fraction, (1-φ)",
        "color_primary": "#2c3e50",
        "color_secondary": "#2980b9",
        "color_accent": "#3498db",
        "background": "#f0f2f5",
        "surface": "#ffffff",
        "text": "#2c3e50",
        "text_secondary": "#7f8c8d",
        "gradient_start": "#2c3e50",
        "gradient_end": "#34495e"
    },
    "polymer": {
        "id": "polymer",
        "name": "Polymer Composite",
        "icon": "🧪",
        "description": "Polymer Matrix Composites",
        "default_matrix": "Epoxy",
        "default_filler": "Graphene",
        "matrix_label": "Polymer Matrix",
        "filler_label": "Filler",
        "phi_label": "Filler Volume Fraction, φ",
        "one_minus_phi_label": "Matrix Volume Fraction, (1-φ)",
        "color_primary": "#2c3e50",
        "color_secondary": "#27ae60",
        "color_accent": "#2ecc71",
        "background": "#f0f2f5",
        "surface": "#ffffff",
        "text": "#2c3e50",
        "text_secondary": "#7f8c8d",
        "gradient_start": "#2c3e50",
        "gradient_end": "#27ae60"
    }
}


# ============================================================================
# NANOTHERMITE MATERIALS
# ============================================================================

NANOTHERMITE_OXIDIZERS = [
    "CuO", "Fe2O3", "MnO2", "MoO3", "WO3", "Bi2O3",
    "SnO2", "PbO2", "Cr2O3", "Co3O4", "NiO", "V2O5",
    "KClO4", "NH4ClO4", "KNO3", "NaNO3", "NH4NO3",
    "Teflon", "PTFE", "PFOA"
]

NANOTHERMITE_REDUCTANTS = [
    "Al", "Mg", "Ti", "Zr", "B", "Si", "Fe", "Ni", "Zn", "Sn", "Cu",
    "AlH3", "MgH2", "LiAlH4", "NaBH4", "Carbon", "Graphite", "Graphene", "CNT"
]

NANOTHERMITE_K_MATRIX = {
    "Al": 237.0, "Mg": 156.0, "Ti": 21.9, "Zr": 22.6,
    "B": 27.4, "Si": 148.0, "Cu": 401.0, "Fe": 80.2,
    "Ni": 90.7, "Zn": 116.0, "Sn": 66.6,
    "AlH3": 20.0, "MgH2": 15.0, "LiAlH4": 10.0, "NaBH4": 12.0,
    "Carbon": 5.0, "Graphite": 150.0, "Graphene": 300.0, "CNT": 200.0
}

NANOTHERMITE_K_FILLER = {
    "CuO": 33.0, "Fe2O3": 4.0, "MnO2": 5.0, "MoO3": 14.0,
    "WO3": 14.0, "Bi2O3": 2.5, "SnO2": 4.5, "PbO2": 2.0,
    "Cr2O3": 3.0, "Co3O4": 6.0, "NiO": 12.0, "V2O5": 8.0,
    "KClO4": 2.0, "NH4ClO4": 2.5, "KNO3": 3.0, "NaNO3": 2.8, "NH4NO3": 2.2,
    "Teflon": 0.25, "PTFE": 0.25, "PFOA": 0.20
}

NANOTHERMITE_RHO_MATRIX = {
    "Al": 2700.0, "Mg": 1740.0, "Ti": 4500.0, "Zr": 6500.0,
    "B": 2460.0, "Si": 2330.0, "Cu": 8960.0, "Fe": 7870.0,
    "Ni": 8900.0, "Zn": 7140.0, "Sn": 7310.0,
    "AlH3": 1480.0, "MgH2": 1450.0, "LiAlH4": 917.0, "NaBH4": 1070.0,
    "Carbon": 2260.0, "Graphite": 2260.0, "Graphene": 2200.0, "CNT": 2100.0
}

NANOTHERMITE_RHO_FILLER = {
    "CuO": 6315.0, "Fe2O3": 5240.0, "MnO2": 5200.0, "MoO3": 4690.0,
    "WO3": 7160.0, "Bi2O3": 8900.0, "SnO2": 6950.0, "PbO2": 9100.0,
    "Cr2O3": 5200.0, "Co3O4": 6000.0, "NiO": 6670.0, "V2O5": 3360.0,
    "KClO4": 2520.0, "NH4ClO4": 1950.0, "KNO3": 2110.0, "NaNO3": 2260.0, "NH4NO3": 1720.0,
    "Teflon": 2200.0, "PTFE": 2200.0, "PFOA": 1800.0
}


# ============================================================================
# POLYMER COMPOSITE MATERIALS
# ============================================================================

THERMOPLASTICS = [
    "Polyethylene (PE)", "Polypropylene (PP)", "Polystyrene (PS)",
    "Polyvinyl Chloride (PVC)", "Polyethylene Terephthalate (PET)",
    "Polyamide (Nylon)", "Polycarbonate (PC)", "Poly(methyl methacrylate) (PMMA)",
    "Acrylonitrile Butadiene Styrene (ABS)", "Polyether Ether Ketone (PEEK)",
    "Polytetrafluoroethylene (PTFE)", "Polyimide (PI)",
    "Polyurethane (PU)", "Polylactic Acid (PLA)"
]

THERMOSETS = [
    "Epoxy", "Phenolic", "Polyester", "Vinyl Ester",
    "Polyurethane Thermoset", "Bismaleimide (BMI)",
    "Cyanate Ester", "Silicone", "Melamine"
]

ELASTOMERS = [
    "Natural Rubber (NR)", "Styrene-Butadiene Rubber (SBR)",
    "Nitrile Rubber (NBR)", "Neoprene (CR)",
    "Silicone Rubber", "EPDM Rubber",
    "Fluoroelastomer (FKM)", "Butyl Rubber (IIR)"
]

CERAMIC_FILLERS = [
    "Al2O3 (Alumina)", "SiO2 (Silica)", "BN (Boron Nitride)",
    "SiC (Silicon Carbide)", "AlN (Aluminum Nitride)",
    "BeO (Beryllium Oxide)", "ZrO2 (Zirconia)",
    "TiO2 (Titania)", "ZnO (Zinc Oxide)", "MgO (Magnesia)",
    "Si3N4 (Silicon Nitride)", "Diamond"
]

CARBON_FILLERS = [
    "Graphene", "Carbon Nanotube (CNT)", "Graphite",
    "Carbon Black", "Carbon Fiber", "Activated Carbon",
    "Fullerene (C60)"
]

METALLIC_FILLERS = [
    "Al (Aluminum)", "Cu (Copper)", "Ag (Silver)",
    "Au (Gold)", "Fe (Iron)", "Ni (Nickel)",
    "Zn (Zinc)", "Sn (Tin)", "Stainless Steel"
]

FIBER_FILLERS = [
    "Glass Fiber", "Carbon Fiber", "Aramid Fiber (Kevlar)",
    "Basalt Fiber", "Natural Fiber (Hemp, Flax)",
    "Cellulose Fiber", "Polymer Fiber"
]

POLYMER_MATRICES = THERMOPLASTICS + THERMOSETS + ELASTOMERS
POLYMER_FILLERS = CERAMIC_FILLERS + CARBON_FILLERS + METALLIC_FILLERS + FIBER_FILLERS

POLYMER_K_MATRIX = {
    "Polyethylene (PE)": 0.40, "Polypropylene (PP)": 0.20,
    "Polystyrene (PS)": 0.15, "Polyvinyl Chloride (PVC)": 0.16,
    "Polyethylene Terephthalate (PET)": 0.30, "Polyamide (Nylon)": 0.25,
    "Polycarbonate (PC)": 0.20, "Poly(methyl methacrylate) (PMMA)": 0.20,
    "Acrylonitrile Butadiene Styrene (ABS)": 0.20, "Polyether Ether Ketone (PEEK)": 0.25,
    "Polytetrafluoroethylene (PTFE)": 0.25, "Polyimide (PI)": 0.30,
    "Polyurethane (PU)": 0.20, "Polylactic Acid (PLA)": 0.15,
    "Epoxy": 0.20, "Phenolic": 0.20, "Polyester": 0.20,
    "Vinyl Ester": 0.20, "Polyurethane Thermoset": 0.20,
    "Bismaleimide (BMI)": 0.25, "Cyanate Ester": 0.20,
    "Silicone": 0.20, "Melamine": 0.30,
    "Natural Rubber (NR)": 0.15, "Styrene-Butadiene Rubber (SBR)": 0.20,
    "Nitrile Rubber (NBR)": 0.25, "Neoprene (CR)": 0.20,
    "Silicone Rubber": 0.20, "EPDM Rubber": 0.20,
    "Fluoroelastomer (FKM)": 0.30, "Butyl Rubber (IIR)": 0.20
}

POLYMER_K_FILLER = {
    "Al2O3 (Alumina)": 30.0, "SiO2 (Silica)": 1.4,
    "BN (Boron Nitride)": 60.0, "SiC (Silicon Carbide)": 100.0,
    "AlN (Aluminum Nitride)": 180.0, "BeO (Beryllium Oxide)": 250.0,
    "ZrO2 (Zirconia)": 3.0, "TiO2 (Titania)": 8.0,
    "ZnO (Zinc Oxide)": 30.0, "MgO (Magnesia)": 45.0,
    "Si3N4 (Silicon Nitride)": 30.0, "Diamond": 1000.0,
    "Graphene": 300.0, "Carbon Nanotube (CNT)": 200.0,
    "Graphite": 150.0, "Carbon Black": 6.0,
    "Carbon Fiber": 50.0, "Activated Carbon": 5.0,
    "Fullerene (C60)": 0.4,
    "Al (Aluminum)": 237.0, "Cu (Copper)": 401.0,
    "Ag (Silver)": 430.0, "Au (Gold)": 317.0,
    "Fe (Iron)": 80.0, "Ni (Nickel)": 90.0,
    "Zn (Zinc)": 116.0, "Sn (Tin)": 67.0,
    "Stainless Steel": 16.0,
    "Glass Fiber": 1.0, "Carbon Fiber": 50.0,
    "Aramid Fiber (Kevlar)": 0.5, "Basalt Fiber": 1.5,
    "Natural Fiber (Hemp, Flax)": 0.1, "Cellulose Fiber": 0.1,
    "Polymer Fiber": 0.2
}

POLYMER_RHO_MATRIX = {
    "Polyethylene (PE)": 950, "Polypropylene (PP)": 910,
    "Polystyrene (PS)": 1050, "Polyvinyl Chloride (PVC)": 1400,
    "Polyethylene Terephthalate (PET)": 1380, "Polyamide (Nylon)": 1140,
    "Polycarbonate (PC)": 1200, "Poly(methyl methacrylate) (PMMA)": 1190,
    "Acrylonitrile Butadiene Styrene (ABS)": 1050, "Polyether Ether Ketone (PEEK)": 1320,
    "Polytetrafluoroethylene (PTFE)": 2200, "Polyimide (PI)": 1420,
    "Polyurethane (PU)": 1200, "Polylactic Acid (PLA)": 1240,
    "Epoxy": 1200, "Phenolic": 1300, "Polyester": 1200,
    "Vinyl Ester": 1200, "Polyurethane Thermoset": 1200,
    "Bismaleimide (BMI)": 1300, "Cyanate Ester": 1200,
    "Silicone": 1100, "Melamine": 1500,
    "Natural Rubber (NR)": 920, "Styrene-Butadiene Rubber (SBR)": 940,
    "Nitrile Rubber (NBR)": 1000, "Neoprene (CR)": 1230,
    "Silicone Rubber": 1100, "EPDM Rubber": 850,
    "Fluoroelastomer (FKM)": 1800, "Butyl Rubber (IIR)": 920
}

POLYMER_RHO_FILLER = {
    "Al2O3 (Alumina)": 3900, "SiO2 (Silica)": 2650,
    "BN (Boron Nitride)": 2190, "SiC (Silicon Carbide)": 3210,
    "AlN (Aluminum Nitride)": 3260, "BeO (Beryllium Oxide)": 3010,
    "ZrO2 (Zirconia)": 5600, "TiO2 (Titania)": 4230,
    "ZnO (Zinc Oxide)": 5610, "MgO (Magnesia)": 3580,
    "Si3N4 (Silicon Nitride)": 3210, "Diamond": 3510,
    "Graphene": 2200, "Carbon Nanotube (CNT)": 2100,
    "Graphite": 2260, "Carbon Black": 1800,
    "Carbon Fiber": 1800, "Activated Carbon": 2000,
    "Fullerene (C60)": 1700,
    "Al (Aluminum)": 2700, "Cu (Copper)": 8960,
    "Ag (Silver)": 10500, "Au (Gold)": 19300,
    "Fe (Iron)": 7870, "Ni (Nickel)": 8900,
    "Zn (Zinc)": 7140, "Sn (Tin)": 7310,
    "Stainless Steel": 8000,
    "Glass Fiber": 2600, "Carbon Fiber": 1800,
    "Aramid Fiber (Kevlar)": 1440, "Basalt Fiber": 2700,
    "Natural Fiber (Hemp, Flax)": 1500, "Cellulose Fiber": 1500,
    "Polymer Fiber": 1300
}


# ============================================================================
# GETTER FUNCTIONS
# ============================================================================

def get_mode_info(mode_id):
    return MODES.get(mode_id, MODES["nanothermite"])

def get_available_modes():
    return list(MODES.keys())

def get_materials_for_mode(mode_id):
    if mode_id == "nanothermite":
        return {
            "matrix_list": NANOTHERMITE_REDUCTANTS.copy(),
            "filler_list": NANOTHERMITE_OXIDIZERS.copy()
        }
    elif mode_id == "polymer":
        return {
            "matrix_list": POLYMER_MATRICES.copy(),
            "filler_list": POLYMER_FILLERS.copy()
        }
    return {"matrix_list": [], "filler_list": []}

def get_k_value(mode_id, material_type, material_name):
    if mode_id == "nanothermite":
        if material_type == "matrix":
            return NANOTHERMITE_K_MATRIX.get(material_name, 237.0)
        elif material_type == "filler":
            return NANOTHERMITE_K_FILLER.get(material_name, 33.0)
    elif mode_id == "polymer":
        if material_type == "matrix":
            return POLYMER_K_MATRIX.get(material_name, 0.2)
        elif material_type == "filler":
            return POLYMER_K_FILLER.get(material_name, 300.0)
    return None

def get_rho_value(mode_id, material_type, material_name):
    if mode_id == "nanothermite":
        if material_type == "matrix":
            return NANOTHERMITE_RHO_MATRIX.get(material_name, 2700.0)
        elif material_type == "filler":
            return NANOTHERMITE_RHO_FILLER.get(material_name, 6315.0)
    elif mode_id == "polymer":
        if material_type == "matrix":
            return POLYMER_RHO_MATRIX.get(material_name, 1200.0)
        elif material_type == "filler":
            return POLYMER_RHO_FILLER.get(material_name, 2200.0)
    return None

def get_default_materials(mode_id):
    mode_info = get_mode_info(mode_id)
    return {
        "matrix": mode_info.get("default_matrix", "Al"),
        "filler": mode_info.get("default_filler", "CuO")
    }

def get_mode_style(mode_id):
    mode_info = get_mode_info(mode_id)
    return {
        "primary": mode_info.get("color_primary", "#2c3e50"),
        "secondary": mode_info.get("color_secondary", "#2980b9"),
        "accent": mode_info.get("color_accent", "#3498db"),
        "background": mode_info.get("background", "#f0f2f5"),
        "surface": mode_info.get("surface", "#ffffff"),
        "text": mode_info.get("text", "#2c3e50"),
        "text_secondary": mode_info.get("text_secondary", "#7f8c8d"),
        "gradient_start": mode_info.get("gradient_start", "#2c3e50"),
        "gradient_end": mode_info.get("gradient_end", "#34495e"),
        "mode_icon": mode_info.get("icon", "🔬"),
        "mode_label": mode_info.get("name", "Unknown"),
        "mode_subtitle": mode_info.get("description", "")
    }


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

OXIDIZERS = NANOTHERMITE_OXIDIZERS
REDUCTANTS = NANOTHERMITE_REDUCTANTS
k_Al = NANOTHERMITE_K_MATRIX.get("Al", 237.0)
k_CuO = NANOTHERMITE_K_FILLER.get("CuO", 33.0)
rho_Al = NANOTHERMITE_RHO_MATRIX.get("Al", 2700.0)
rho_CuO = NANOTHERMITE_RHO_FILLER.get("CuO", 6315.0)


# ============================================================================
# DEBUG - Verify lists are loaded
# ============================================================================

print(f"[material_constants] Nanothermite reductants: {len(NANOTHERMITE_REDUCTANTS)} items")
print(f"[material_constants] Nanothermite oxidizers: {len(NANOTHERMITE_OXIDIZERS)} items")
print(f"[material_constants] Polymer matrices: {len(POLYMER_MATRICES)} items")
print(f"[material_constants] Polymer fillers: {len(POLYMER_FILLERS)} items")
