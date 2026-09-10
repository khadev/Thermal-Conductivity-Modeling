"""Generate professional PDF reports with full model interpretations - Mode-aware."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import tempfile
import os
import numpy as np
from datetime import datetime


def generate_pdf_report(filepath, df, fit_results, metadata=None, figure_paths=None, mode="nanothermite"):
    """
    Generate a publication-quality PDF report with full model interpretations.
    
    Parameters:
    -----------
    filepath : str
        Path to save the PDF
    df : pandas.DataFrame
        Experimental data
    fit_results : dict
        Dictionary of fit results
    metadata : dict, optional
        Additional metadata for the report
    figure_paths : list, optional
        List of figure file paths to include
    mode : str
        'nanothermite' or 'polymer'
    """
    
    # Get mode-specific text
    if mode == "nanothermite":
        mode_name = "Nanothermite Composites"
        material_type = "energetic material"
        convention_text = "φ = Reductant (matrix) volume fraction, (1-φ) = Oxidizer (filler) volume fraction"
        summary_template = "This report presents the results of thermal conductivity modeling for {material} composites."
        best_model_text = "The fitted parameters can be used to predict thermal conductivity for other compositions."
    else:
        mode_name = "Polymer Composites"
        material_type = "polymer matrix"
        convention_text = "φ = Polymer matrix volume fraction, (1-φ) = Filler volume fraction"
        summary_template = "This report presents the results of thermal conductivity modeling for {material} composites."
        best_model_text = "The fitted parameters can be used to predict thermal conductivity for other polymer composite formulations."

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
        title="Thermal Conductivity Modeling Report"
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Heading1"],
        fontSize=20, spaceAfter=20, alignment=TA_CENTER,
        textColor=colors.HexColor("#1a5490")
    )

    heading1_style = ParagraphStyle(
        "CustomHeading1", parent=styles["Heading1"],
        fontSize=16, spaceAfter=12, textColor=colors.HexColor("#1a5490")
    )

    heading2_style = ParagraphStyle(
        "CustomHeading2", parent=styles["Heading2"],
        fontSize=14, spaceAfter=10, textColor=colors.HexColor("#2c3e50")
    )

    body_style = ParagraphStyle(
        "CustomBody", parent=styles["Normal"],
        fontSize=10, spaceAfter=8, alignment=TA_JUSTIFY, leading=14
    )

    interpretation_style = ParagraphStyle(
        "Interpretation", parent=styles["Normal"],
        fontSize=10, spaceAfter=10, alignment=TA_JUSTIFY,
        leading=14, leftIndent=10, rightIndent=10,
        backColor=colors.HexColor("#f8f9fa")
    )

    model_title_style = ParagraphStyle(
        "ModelTitle", parent=styles["Heading3"],
        fontSize=12, spaceAfter=6, textColor=colors.HexColor("#2c3e50")
    )

    story = []

    # ===== COVER PAGE =====
    story.append(Paragraph(f"Thermal Conductivity Modeling Report - {mode_name}", title_style))
    story.append(Spacer(1, 12))

    # Metadata
    if metadata:
        for key, val in metadata.items():
            if key not in ["Developer", "Software", "Mode"]:
                story.append(Paragraph(f"<b>{key}:</b> {val}", styles["Normal"]))
        story.append(Spacer(1, 6))
        if "Developer" in metadata:
            story.append(Paragraph(f"<b>Developer:</b> {metadata['Developer']}", styles["Normal"]))
        if "Software" in metadata:
            story.append(Paragraph(f"<b>Software:</b> {metadata['Software']}", styles["Normal"]))
        if "Mode" in metadata:
            story.append(Paragraph(f"<b>Mode:</b> {metadata['Mode']}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles["Normal"]))
    story.append(Spacer(1, 24))

    # ===== CONVENTION NOTE =====
    story.append(Paragraph("Volume Fraction Convention", heading2_style))
    story.append(Paragraph(convention_text, body_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#bdc3c7")))
    story.append(Spacer(1, 12))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading1_style))

    # Calculate summary statistics
    total_models = len(fit_results)
    successful = sum(1 for r in fit_results.values() if r.get("success", False))

    # Get material info from metadata
    material_info = "Unknown"
    if metadata:
        material_info = metadata.get('Material', metadata.get('material', 'Unknown'))
        if 'matrix' in metadata and 'filler' in metadata:
            material_info = f"{metadata.get('matrix', '')}/{metadata.get('filler', '')}"

    if successful > 0:
        best_model = max(
            fit_results.items(),
            key=lambda x: x[1].get("r2", -1) if x[1].get("success", False) else -1
        )
        best_r2 = best_model[1].get("r2", 0)
        best_rmse = best_model[1].get("rmse", 0)
        best_name = best_model[0]
        best_params = best_model[1].get("params", {})
        param_str = ", ".join([f"{k}={v:.3f}" for k, v in best_params.items()]) if best_params else "N/A"

        summary_text = summary_template.format(material=material_info)
        summary_text += f"""
        A total of <b>{total_models}</b> models were evaluated, with <b>{successful}</b> models fitting successfully.
        The <b>{best_name}</b> model showed the best performance with an R² value of <b>{best_r2:.4f}</b>
        and RMSE of <b>{best_rmse:.2f} W/m·K</b>. The optimized parameters for this model were:
        <b>{param_str}</b>.
        """
        story.append(Paragraph(summary_text, body_style))
    else:
        story.append(Paragraph("No models were successfully fitted to the experimental data.", body_style))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#bdc3c7")))
    story.append(Spacer(1, 12))

    # ===== MODELS REFERENCE =====
    story.append(Paragraph("1. Models Reference", heading1_style))
    story.append(Spacer(1, 6))

    # Import model registry for descriptions
    try:
        from ..core.models import get_model_registry
        MODEL_REGISTRY = get_model_registry(mode)
    except:
        MODEL_REGISTRY = {}

    # Get model categories based on mode
    if mode == "nanothermite":
        categories = [
            ("Basic Models", ["Parallel", "Series", "Maxwell-Eucken", "Bruggeman", 
                            "Agari", "Lewis-Nielsen", "Percolation", "Agari-Percolation"]),
            ("Classic Models", ["Russell", "Cheng-Vachon", "Baschirow-Selenew", 
                              "Hamilton-Crosser", "EMT Equation", "Geometric Mean",
                              "Böttcher", "De Loor", "Ce Wen Nan"]),
            ("Advanced Models", ["Rayleigh", "Halpin-Tsai", "Hatta", 
                               "Modified Hashin-Shtrikman", "Tsao", 
                               "Hamilton-Crosser Extended",
                               "Maxwell-Eucken Upper", "Maxwell-Eucken Lower",
                               "GEM Equation"])
        ]
    else:
        categories = [
            ("Basic Models", ["Parallel", "Series", "Maxwell-Eucken", "Bruggeman", 
                            "Agari", "Lewis-Nielsen", "Percolation", "Agari-Percolation"]),
            ("Classic Models", ["Russell", "Cheng-Vachon", "Baschirow-Selenew", 
                              "Hamilton-Crosser", "EMT Equation", "Geometric Mean",
                              "Böttcher", "De Loor", "Ce Wen Nan"]),
            ("Advanced Models", ["Rayleigh", "Halpin-Tsai", "Hatta", 
                               "Modified Hashin-Shtrikman", "Tsao", 
                               "Hamilton-Crosser Extended",
                               "Maxwell-Eucken Upper", "Maxwell-Eucken Lower",
                               "GEM Equation"])
        ]

    for category_name, model_names in categories:
        story.append(Paragraph(category_name, heading2_style))
        for model_name in model_names:
            meta = MODEL_REGISTRY.get(model_name, {})
            color = meta.get("color", "#2c3e50")
            description = meta.get("description", "No description available")
            params = meta.get("params", [])

            story.append(Paragraph(
                f'<font color="{color}"><b>{model_name}</b></font>',
                model_title_style
            ))
            story.append(Paragraph(
                f"<i>Description:</i> {description}",
                body_style
            ))
            if params:
                story.append(Paragraph(
                    f"<i>Fitted Parameters:</i> {', '.join(params)}",
                    body_style
                ))
            if meta.get("reference"):
                story.append(Paragraph(
                    f"<i>Reference:</i> {meta['reference']}",
                    body_style
                ))
            story.append(Spacer(1, 6))

    # Discovered models
    discovered_models = [name for name, meta in MODEL_REGISTRY.items() 
                        if meta.get('discovered', False)]
    if discovered_models:
        story.append(Paragraph("Discovered Models", heading2_style))
        for model_name in discovered_models:
            meta = MODEL_REGISTRY.get(model_name, {})
            story.append(Paragraph(
                f'<font color="#e74c3c"><b>🔍 {model_name}</b></font>',
                model_title_style
            ))
            story.append(Paragraph(
                f"<i>Description:</i> {meta.get('description', 'Auto-discovered model')}",
                body_style
            ))
            if meta.get("params"):
                story.append(Paragraph(
                    f"<i>Fitted Parameters:</i> {', '.join(meta['params'])}",
                    body_style
                ))
            story.append(Spacer(1, 6))

    story.append(PageBreak())

    # ===== EXPERIMENTAL DATA =====
    story.append(Paragraph("2. Experimental Data Summary", heading1_style))
    story.append(Spacer(1, 6))

    if df is not None:
        # Get appropriate column names based on mode
        if 'phi' in df.columns:
            phi_col = 'phi'
        elif 'Red_volume_fraction' in df.columns:
            phi_col = 'Red_volume_fraction'
        else:
            phi_col = df.columns[0]

        phi_vals = df[phi_col].values

        # Get the appropriate fraction column
        if mode == "nanothermite":
            if 'Oxy_volume_fraction' in df.columns:
                fraction_col = 'Oxy_volume_fraction'
            else:
                fraction_col = df.columns[1] if len(df.columns) > 1 else phi_col
        else:
            if 'Filler_volume_fraction' in df.columns:
                fraction_col = 'Filler_volume_fraction'
            elif 'Oxy_volume_fraction' in df.columns:
                fraction_col = 'Oxy_volume_fraction'
            else:
                fraction_col = df.columns[1] if len(df.columns) > 1 else phi_col

        fraction_vals = df[fraction_col].values if fraction_col in df.columns else 1.0 - phi_vals
        k_meas = df["k_meas"].values

        story.append(Paragraph(f"<b>Number of Data Points:</b> {len(df)}", body_style))
        story.append(Paragraph(f"<b>φ Range:</b> {phi_vals.min():.3f} - {phi_vals.max():.3f}", body_style))
        story.append(Paragraph(f"<b>(1-φ) Range:</b> {fraction_vals.min():.3f} - {fraction_vals.max():.3f}", body_style))
        story.append(Paragraph(f"<b>Thermal Conductivity Range:</b> {k_meas.min():.2f} - {k_meas.max():.2f} W/m·K", body_style))
        story.append(Paragraph(f"<b>Mean Thermal Conductivity:</b> {k_meas.mean():.2f} W/m·K", body_style))
        story.append(Paragraph(f"<b>Standard Deviation:</b> {k_meas.std():.2f} W/m·K", body_style))

        # Data table
        story.append(Spacer(1, 10))
        story.append(Paragraph("Experimental Data Table", heading2_style))

        # Determine column names for table
        if mode == "nanothermite":
            col1_name = "φ (Reductant)"
            col2_name = "(1-φ) (Oxidizer)"
        else:
            col1_name = "φ (Polymer)"
            col2_name = "(1-φ) (Filler)"

        table_data = [["#", col1_name, col2_name, "k_meas (W/m·K)"]]
        for i, row in df.iterrows():
            if i < 20:
                phi_val = row.get(phi_col, 0)
                frac_val = row.get(fraction_col, 1.0 - phi_val)
                table_data.append([
                    str(i+1),
                    f"{phi_val:.4f}",
                    f"{frac_val:.4f}",
                    f"{row['k_meas']:.2f}"
                ])

        if len(df) > 20:
            table_data.append(["", "...", "...", "..."])

        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5490")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
        ]))
        story.append(t)

    story.append(PageBreak())

    # ===== MODEL RESULTS WITH INTERPRETATIONS =====
    story.append(Paragraph("3. Model Results and Interpretations", heading1_style))
    story.append(Spacer(1, 6))

    # Sort models by R²
    sorted_models = sorted(
        fit_results.items(),
        key=lambda x: x[1].get("r2", -1) if x[1].get("success", False) else -1,
        reverse=True
    )

    for rank, (model_name, result) in enumerate(sorted_models, 1):
        success = result.get("success", False)

        if success:
            if rank == 1:
                badge = "🏆 "
                color = "#27ae60"
            elif rank == 2:
                badge = "🥈 "
                color = "#f39c12"
            elif rank == 3:
                badge = "🥉 "
                color = "#e67e22"
            else:
                badge = f"#{rank} "
                color = "#2c3e50"

            story.append(Paragraph(
                f'<font color="{color}"><b>{badge}{model_name}</b></font>',
                heading2_style
            ))
        else:
            story.append(Paragraph(
                f'<b>❌ {model_name} (Failed)</b>',
                heading2_style
            ))

        if not success:
            story.append(Paragraph(
                f"<i>Status:</i> This model failed to fit the experimental data.",
                body_style
            ))
            story.append(Paragraph(
                f"<i>Error:</i> {result.get('message', 'Unknown error')}",
                body_style
            ))
            story.append(Spacer(1, 10))
            continue

        meta = MODEL_REGISTRY.get(model_name, {})
        params = result.get("params", {})
        stats = result.get("stats", {})

        # Statistics table
        r2_val = stats.get('r2', 0)
        rmse_val = stats.get('rmse', 0)
        mape_val = stats.get('mape', 0)

        stats_data = [
            ["Metric", "Value", "Interpretation"],
            ["R²", f"{r2_val:.4f}", _get_r2_interpretation(r2_val)],
            ["RMSE", f"{rmse_val:.2f} W/m·K", _get_rmse_interpretation(rmse_val, df)],
            ["MAE", f"{stats.get('mae', 0):.2f} W/m·K", "Mean absolute error"],
            ["MAPE", f"{mape_val:.2f}%", _get_mape_interpretation(mape_val)],
        ]

        t = Table(stats_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5490")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("SPAN", (2, 1), (2, -1)),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

        if params:
            story.append(Paragraph("<b>Optimized Parameters:</b>", body_style))
            param_str = ", ".join([f"{k} = {v:.4f}" for k, v in params.items()])
            story.append(Paragraph(param_str, body_style))

        story.append(Spacer(1, 4))

        story.append(Paragraph("<b>Interpretation:</b>", body_style))
        interpretation = _get_model_interpretation(
            model_name,
            r2_val,
            rmse_val,
            params,
            meta.get('description', ''),
            mode
        )
        story.append(Paragraph(interpretation, interpretation_style))

        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="80%", thickness=0.5, color=colors.HexColor("#bdc3c7")))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # ===== MODEL COMPARISON =====
    story.append(Paragraph("4. Model Comparison", heading1_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Statistical Comparison of All Models", heading2_style))

    comp_data = [["Rank", "Model", "R²", "RMSE", "MAE", "MAPE (%)", "Status"]]
    for rank, (model_name, result) in enumerate(sorted_models, 1):
        if result.get("success", False):
            stats = result.get("stats", {})
            if rank == 1:
                rank_str = f"🏆 {rank}"
            elif rank == 2:
                rank_str = f"🥈 {rank}"
            elif rank == 3:
                rank_str = f"🥉 {rank}"
            else:
                rank_str = str(rank)
            comp_data.append([
                rank_str,
                model_name,
                f"{stats.get('r2', 0):.4f}",
                f"{stats.get('rmse', 0):.2f}",
                f"{stats.get('mae', 0):.2f}",
                f"{stats.get('mape', 0):.2f}",
                "✅"
            ])
        else:
            comp_data.append([
                str(rank),
                model_name,
                "-",
                "-",
                "-",
                "-",
                "❌"
            ])

    t = Table(comp_data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5490")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # ===== COMPARISON INTERPRETATION =====
    story.append(Paragraph("Comparison Interpretation", heading2_style))

    successful_models = [(n, r) for n, r in sorted_models if r.get("success", False)]

    if successful_models:
        best = successful_models[0]
        best_r2 = best[1].get("stats", {}).get("r2", 0)
        best_rmse = best[1].get("stats", {}).get("rmse", 0)

        comparison_text = f"""
        The <b>{best[0]}</b> model provides the best fit to the experimental data with an R² of
        <b>{best_r2:.4f}</b> and RMSE of <b>{best_rmse:.2f} W/m·K</b>.
        """

        if len(successful_models) > 1:
            second = successful_models[1]
            second_r2 = second[1].get("stats", {}).get("r2", 0)
            comparison_text += f"""
            The second-best model is <b>{second[0]}</b> with an R² of <b>{second_r2:.4f}</b>.
            """

        if best_r2 >= 0.95:
            recommendation = "⭐ The model shows excellent agreement with experimental data."
        elif best_r2 >= 0.85:
            recommendation = "✓ The model shows good agreement with experimental data."
        elif best_r2 >= 0.70:
            recommendation = "⚠ The model shows moderate agreement with experimental data."
        else:
            recommendation = "✗ The model shows poor agreement with experimental data."

        comparison_text += f"\n\n<b>Recommendation:</b> {recommendation}"

        story.append(Paragraph(comparison_text, body_style))
    else:
        story.append(Paragraph(
            "No models were successfully fitted to the experimental data.",
            body_style
        ))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#bdc3c7")))

    # ===== RECOMMENDATIONS =====
    story.append(Paragraph("5. Recommendations", heading1_style))
    story.append(Spacer(1, 6))

    recommendations = _get_recommendations(fit_results, sorted_models, mode)
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", body_style))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#bdc3c7")))

    # ===== FIGURES =====
    if figure_paths:
        story.append(Paragraph("6. Figures", heading1_style))
        story.append(Spacer(1, 6))

        for fig_path in figure_paths:
            if os.path.exists(fig_path):
                try:
                    img = Image(fig_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    story.append(Paragraph(f"Figure could not be loaded: {fig_path}", body_style))

    # ===== FOOTER =====
    story.append(Spacer(1, 24))
    story.append(Paragraph(
        f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        styles["Normal"]
    ))
    story.append(Paragraph(
        f"Thermal Conductivity Modeling Suite v2.0 - {mode_name} Mode | Oukil Khaled Research",
        styles["Normal"]
    ))

    doc.build(story)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_r2_interpretation(r2):
    if r2 >= 0.95:
        return "⭐ Excellent - Model explains >95% of variance"
    elif r2 >= 0.85:
        return "✓ Good - Model explains 85-95% of variance"
    elif r2 >= 0.70:
        return "⚠ Fair - Model explains 70-85% of variance"
    elif r2 >= 0.50:
        return "△ Poor - Model explains 50-70% of variance"
    else:
        return "✗ Very Poor - Model explains <50% of variance"


def _get_rmse_interpretation(rmse, df):
    if df is not None and len(df) > 0:
        k_mean = df["k_meas"].mean()
        if k_mean > 0:
            relative_rmse = (rmse / k_mean) * 100
            if relative_rmse < 5:
                return f"Excellent (RMSE < 5% of mean k = {k_mean:.2f} W/m·K)"
            elif relative_rmse < 10:
                return f"Good (RMSE < 10% of mean k)"
            elif relative_rmse < 20:
                return f"Moderate (RMSE < 20% of mean k)"
            else:
                return f"Poor (RMSE > 20% of mean k)"
    return f"RMSE: {rmse:.2f} W/m·K"


def _get_mape_interpretation(mape):
    if mape < 5:
        return "⭐ Excellent - Very accurate predictions"
    elif mape < 10:
        return "✓ Good - Reasonably accurate predictions"
    elif mape < 20:
        return "⚠ Moderate - Acceptable predictions"
    else:
        return "✗ Poor - Inaccurate predictions"


def _get_model_interpretation(model_name, r2, rmse, params, description, mode="nanothermite"):
    """Get mode-specific model interpretation."""
    
    # Base interpretations - mode independent
    base_interpretations = {
        "Parallel": """
        The Parallel model represents the upper Hashin-Shtrikman bound, assuming perfect thermal
        contact between phases. This model typically overestimates thermal conductivity as it
        assumes heat flows in parallel through both phases.
        """,
        "Series": """
        The Series model represents the lower Hashin-Shtrikman bound, assuming heat flows
        perpendicular to the layers. This model typically underestimates thermal conductivity.
        """,
        "Maxwell-Eucken": """
        The Maxwell-Eucken model assumes dilute spherical inclusions in a continuous matrix
        with no particle interaction. This model works best at low filler loadings (<10%).
        """,
        "Bruggeman": """
        The Bruggeman model accounts for particle interactions and is valid for higher filler
        loadings. It treats both phases symmetrically.
        """,
        "Agari": """
        The Agari model is an empirical model that accounts for filler geometry, polymer
        crystallinity, and conductive pathways. The parameters C₁ and C₂ represent the
        matrix's ability to form conductive chains and the filler's contribution to
        conductivity, respectively.
        """,
        "Lewis-Nielsen": """
        The Lewis-Nielsen model incorporates particle geometry (parameter A) and maximum
        packing fraction (φₘ). This makes it suitable for a wide range of filler shapes
        and loadings.
        """,
        "Percolation": """
        The Percolation model describes the formation of a conductive network above a
        critical threshold (φc). The power-law behavior with exponent t captures the
        transition from insulating to conducting behavior.
        """,
        "GEM Equation": """
        The Generalized Effective Medium (GEM) equation combines percolation and effective
        medium theories. The parameter t represents the critical exponent (typically 1.6-2.0
        for thermal conductivity), and φc is the percolation threshold.
        """
    }

    # Get mode-specific material names
    if mode == "nanothermite":
        matrix_name = "reductant"
        filler_name = "oxidizer"
        material_context = "energetic material"
    else:
        matrix_name = "polymer matrix"
        filler_name = "filler"
        material_context = "polymer composite"

    # Get base interpretation or create generic one
    interpretation = base_interpretations.get(model_name, "")
    
    if not interpretation:
        interpretation = f"""
        The {model_name} model was fitted to the experimental data with R² = {r2:.4f} and
        RMSE = {rmse:.2f} W/m·K for this {material_context} composite.
        """

    # Add mode-specific context
    if r2 >= 0.95:
        assessment = "⭐ <b>Assessment:</b> This model shows excellent agreement with experimental data."
    elif r2 >= 0.85:
        assessment = "✓ <b>Assessment:</b> This model shows good agreement with experimental data."
    elif r2 >= 0.70:
        assessment = "⚠ <b>Assessment:</b> This model shows moderate agreement with experimental data."
    else:
        assessment = "✗ <b>Assessment:</b> This model shows poor agreement with experimental data."

    interpretation += f"\n\n{assessment}"

    return interpretation


def _get_recommendations(fit_results, sorted_models, mode="nanothermite"):
    """Get mode-specific recommendations."""
    recommendations = []

    successful = [(n, r) for n, r in sorted_models if r.get("success", False)]

    if not successful:
        recommendations.append("No models were successfully fitted. Check your data quality and model parameters.")
        return recommendations

    best = successful[0]
    best_r2 = best[1].get("stats", {}).get("r2", 0)
    best_model = best[0]

    recommendations.append(f"For this dataset, the <b>{best_model}</b> model provides the best fit (R² = {best_r2:.4f}).")

    if best_r2 >= 0.95:
        recommendations.append("⭐ The model shows excellent agreement with experimental data.")
        recommendations.append("The fitted parameters can be used to predict thermal conductivity for other compositions.")
    elif best_r2 >= 0.85:
        recommendations.append("✓ The model shows good agreement with experimental data.")
        recommendations.append("Consider additional experimental data points to improve the fit quality.")
    elif best_r2 >= 0.70:
        recommendations.append("⚠ The model shows moderate agreement with experimental data.")
        if mode == "nanothermite":
            recommendations.append("Consider using a more sophisticated model or increasing the number of data points.")
        else:
            recommendations.append("Consider using a more sophisticated model or checking filler dispersion quality.")
    else:
        recommendations.append("✗ The model shows poor agreement with experimental data.")
        recommendations.append("Consider checking data quality, trying different models, or using custom parameters.")

    # Add mode-specific recommendations
    if mode == "nanothermite":
        recommendations.append("For nanothermite composites, consider the Agari-Percolation model for percolation effects.")
    else:
        recommendations.append("For polymer composites, consider the Halpin-Tsai model for anisotropic fillers.")

    return recommendations