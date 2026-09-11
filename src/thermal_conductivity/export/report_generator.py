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
import re
import numpy as np
from datetime import datetime


# ============================================================================
# HELPER: SANITIZE TEXT (remove LaTeX leaks, fix typos)
# ============================================================================

def _clean_text(text):
    """Remove LaTeX-style markers and normalize text for PDF rendering."""
    if text is None:
        return ""
    text = str(text)

    # Replace LaTeX-style R² with Unicode R²
    text = text.replace("\\(R^2\\)", "R²")
    text = text.replace("\\(\\mathrm{R}^2\\)", "R²")
    text = text.replace("\\(\\mathbb{R}^2\\)", "R²")
    text = text.replace("R^2", "R²")

    # Replace \\(\\phi\\) with φ
    text = text.replace("\\(\\phi\\)", "φ")
    text = text.replace("\\phi", "φ")

    # Fix "Agar" → "Agari" (in case it slips through)
    text = re.sub(r"\bAgar\b", "Agari", text)
    text = re.sub(r"\bAgaric", "Agari", text)

    # Fix "Bottcher" → "Böttcher"
    text = re.sub(r"\bBottcher\b", "Böttcher", text)

    return text


def _get_stat(result, key, default=0.0):
    """
    Get a statistic from a fit result dict, handling both layouts:
      - Nested: result["stats"]["r2"]
      - Flat:   result["r2"]
    """
    if not isinstance(result, dict):
        return default

    stats = result.get("stats")
    if isinstance(stats, dict) and key in stats:
        val = stats[key]
        if val is not None and np.isfinite(val):
            return float(val)

    val = result.get(key, default)
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return default
    return val


def generate_pdf_report(filepath, df, fit_results, metadata=None,
                        figure_paths=None, mode="nanothermite"):
    """Generate a publication-quality PDF report."""

    # ========================================================================
    # MODE-SPECIFIC TEXT
    # ========================================================================
    if mode == "nanothermite":
        mode_name = "Nanothermite Composites"
        material_type = "energetic material"
        convention_text = (
            "φ = Filler (oxidizer) volume fraction, "
            "(1-φ) = Matrix (reductant) volume fraction"
        )
        summary_template = (
            "This report presents the results of thermal conductivity modeling "
            "for {material} composites."
        )
    else:
        mode_name = "Polymer Composites"
        material_type = "polymer matrix"
        convention_text = (
            "φ = Filler volume fraction, "
            "(1-φ) = Polymer matrix volume fraction"
        )
        summary_template = (
            "This report presents the results of thermal conductivity modeling "
            "for {material} composites."
        )

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
        title="Thermal Conductivity Modeling Report",
        author="Oukil Khaled ibn El-walid",
        subject=f"Thermal conductivity analysis - {mode_name}"
    )

    styles = getSampleStyleSheet()

    # ========================================================================
    # CUSTOM STYLES
    # ========================================================================
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
        fontSize=12, spaceAfter=4, textColor=colors.HexColor("#2c3e50")
    )

    desc_style = ParagraphStyle(
        "DescriptionStyle", parent=styles["Normal"],
        fontSize=9, spaceAfter=4, leading=12,
        textColor=colors.HexColor("#555555")
    )

    meta_label_style = ParagraphStyle(
        "MetaLabel", parent=styles["Normal"],
        fontSize=10, leading=13, textColor=colors.HexColor("#2c3e50")
    )

    meta_value_style = ParagraphStyle(
        "MetaValue", parent=styles["Normal"],
        fontSize=10, leading=13, textColor=colors.HexColor("#2c3e50")
    )

    story = []

    # ========================================================================
    # COVER PAGE
    # ========================================================================
    story.append(Paragraph(
        f"Thermal Conductivity Modeling Report - {mode_name}", title_style
    ))
    story.append(Spacer(1, 12))

    # ---- Metadata table (each cell wrapped in Paragraph → HTML renders) ----
    if metadata:
        meta_rows = []
        display_order = [
            ("Material", "Material"),
            ("Matrix", "Matrix"),
            ("Filler", "Filler"),
            ("k_matrix", "k_matrix"),
            ("k_filler", "k_filler"),
            ("Data Points", "Data Points"),
            ("Developer", "Developer"),
            ("Software", "Software"),
            ("Mode", "Mode"),
        ]
        for key, label in display_order:
            if key in metadata:
                val = metadata[key]
                val_str = f"{val}" if not isinstance(val, str) else val
                meta_rows.append([
                    Paragraph(f"<b>{label}:</b>", meta_label_style),
                    Paragraph(_clean_text(val_str), meta_value_style)
                ])

        if meta_rows:
            t = Table(meta_rows, colWidths=[1.4*inch, 5.0*inch])
            t.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t)

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"<b>Report Date:</b> "
        f"{datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        meta_value_style
    ))
    story.append(Spacer(1, 24))

    # ========================================================================
    # CONVENTION NOTE
    # ========================================================================
    story.append(Paragraph("Volume Fraction Convention", heading2_style))
    story.append(Paragraph(_clean_text(convention_text), body_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#bdc3c7")
    ))
    story.append(Spacer(1, 12))

    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    story.append(Paragraph("Executive Summary", heading1_style))

    total_models = len(fit_results)
    successful = sum(
        1 for r in fit_results.values() if r.get("success", False)
    )

    material_info = "Unknown"
    if metadata:
        material_info = metadata.get(
            'Material', metadata.get('material', 'Unknown')
        )
        if 'matrix' in metadata and 'filler' in metadata:
            material_info = (
                f"{metadata.get('matrix', '')}/{metadata.get('filler', '')}"
            )

    if successful > 0:
        best_model = max(
            fit_results.items(),
            key=lambda x: _get_stat(x[1], 'r2', -1)
                          if x[1].get("success", False) else -1
        )
        best_r2 = _get_stat(best_model[1], 'r2', 0)
        best_rmse = _get_stat(best_model[1], 'rmse', 0)
        best_name = _clean_text(best_model[0])
        best_params = best_model[1].get("params", {})
        param_str = (
            ", ".join([f"{k}={v:.3f}" for k, v in best_params.items()])
            if best_params else "N/A"
        )

        summary_text = summary_template.format(material=material_info)
        summary_text += (
            f" A total of <b>{total_models}</b> models were evaluated, "
            f"with <b>{successful}</b> models fitting successfully. "
            f"The <b>{best_name}</b> model showed the best performance "
            f"with an R² value of <b>{best_r2:.4f}</b> and RMSE of "
            f"<b>{best_rmse:.2f} W/m·K</b>. "
            f"The optimized parameters for this model were: "
            f"<b>{param_str}</b>."
        )
        story.append(Paragraph(summary_text, body_style))
    else:
        story.append(Paragraph(
            "No models were successfully fitted to the experimental data.",
            body_style
        ))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#bdc3c7")
    ))
    story.append(Spacer(1, 12))

    # ========================================================================
    # MODELS REFERENCE
    # ========================================================================
    story.append(Paragraph("1. Models Reference", heading1_style))
    story.append(Spacer(1, 6))

    try:
        from ..core.models import get_model_registry
        MODEL_REGISTRY = get_model_registry(mode)
    except Exception:
        MODEL_REGISTRY = {}

    categories = [
        ("Basic Models", ["Parallel", "Series", "Maxwell-Eucken",
                          "Bruggeman", "Agari", "Lewis-Nielsen",
                          "Percolation", "Agari-Percolation"]),
        ("Classic Models", ["Russell", "Cheng-Vachon",
                            "Baschirow-Selenew", "Hamilton-Crosser",
                            "EMT Equation", "Geometric Mean",
                            "Böttcher", "De Loor", "Ce Wen Nan"]),
        ("Advanced Models", ["Rayleigh", "Halpin-Tsai", "Hatta",
                             "Modified Hashin-Shtrikman", "Tsao",
                             "Hamilton-Crosser Extended",
                             "Maxwell-Eucken Upper",
                             "Maxwell-Eucken Lower", "GEM Equation"])
    ]

    for category_name, model_names in categories:
        story.append(Paragraph(category_name, heading2_style))
        for model_name in model_names:
            meta = MODEL_REGISTRY.get(model_name, {})
            color = meta.get("color", "#2c3e50")
            description = _clean_text(
                meta.get("description", "No description available")
            )
            params = meta.get("params", [])

            # Model name (with spacing after)
            story.append(Paragraph(
                f'<font color="{color}"><b>{model_name}</b></font>',
                model_title_style
            ))
            story.append(Spacer(1, 3))

            # Description
            story.append(Paragraph(
                f"<i>Description:</i> {description}", desc_style
            ))

            if params:
                story.append(Paragraph(
                    f"<i>Fitted Parameters:</i> {', '.join(params)}",
                    desc_style
                ))
            if meta.get("reference"):
                story.append(Paragraph(
                    f"<i>Reference:</i> {_clean_text(meta['reference'])}",
                    desc_style
                ))

            story.append(Spacer(1, 8))

    # Discovered models
    discovered_models = [
        name for name, meta in MODEL_REGISTRY.items()
        if meta.get('discovered', False)
    ]
    if discovered_models:
        story.append(Paragraph("Discovered Models", heading2_style))
        for model_name in discovered_models:
            meta = MODEL_REGISTRY.get(model_name, {})
            story.append(Paragraph(
                f'<font color="#e74c3c"><b>{model_name}</b></font>',
                model_title_style
            ))
            story.append(Spacer(1, 3))
            story.append(Paragraph(
                f"<i>Description:</i> "
                f"{_clean_text(meta.get('description', 'Auto-discovered model'))}",
                desc_style
            ))
            if meta.get("params"):
                story.append(Paragraph(
                    f"<i>Fitted Parameters:</i> {', '.join(meta['params'])}",
                    desc_style
                ))
            story.append(Spacer(1, 8))

    story.append(PageBreak())

    # ========================================================================
    # EXPERIMENTAL DATA
    # ========================================================================
    story.append(Paragraph("2. Experimental Data Summary", heading1_style))
    story.append(Spacer(1, 6))

    if df is not None:
        if 'Filler_volume_fraction' in df.columns:
            phi_col = 'Filler_volume_fraction'
        elif 'phi' in df.columns:
            phi_col = 'phi'
        elif 'Red_volume_fraction' in df.columns:
            phi_col = 'Red_volume_fraction'
        else:
            phi_col = df.columns[0]

        phi_vals = df[phi_col].values

        if 'Matrix_volume_fraction' in df.columns:
            fraction_col = 'Matrix_volume_fraction'
        else:
            fraction_col = None
        fraction_vals = (
            df[fraction_col].values if fraction_col
            else 1.0 - phi_vals
        )

        k_meas = df["k_meas"].values

        story.append(Paragraph(
            f"<b>Number of Data Points:</b> {len(df)}", body_style
        ))
        story.append(Paragraph(
            f"<b>φ Range:</b> {phi_vals.min():.3f} - "
            f"{phi_vals.max():.3f}", body_style
        ))
        story.append(Paragraph(
            f"<b>(1-φ) Range:</b> {fraction_vals.min():.3f} - "
            f"{fraction_vals.max():.3f}", body_style
        ))
        story.append(Paragraph(
            f"<b>Thermal Conductivity Range:</b> "
            f"{k_meas.min():.2f} - {k_meas.max():.2f} W/m·K", body_style
        ))
        story.append(Paragraph(
            f"<b>Mean Thermal Conductivity:</b> "
            f"{k_meas.mean():.2f} W/m·K", body_style
        ))
        story.append(Paragraph(
            f"<b>Standard Deviation:</b> "
            f"{k_meas.std():.2f} W/m·K", body_style
        ))

        # Data table
        story.append(Spacer(1, 10))
        story.append(Paragraph("Experimental Data Table", heading2_style))

        table_data = [["#", "φ (Filler)", "(1-φ) (Matrix)", "k_meas (W/m·K)"]]
        for i, row in df.iterrows():
            if i < 20:
                phi_val = row.get(phi_col, 0)
                frac_val = (
                    row.get(fraction_col, 1.0 - phi_val)
                    if fraction_col else (1.0 - phi_val)
                )
                table_data.append([
                    str(i + 1),
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
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f0f4f8")]),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
        ]))
        story.append(t)

    story.append(PageBreak())

    # ========================================================================
    # MODEL RESULTS WITH INTERPRETATIONS
    # ========================================================================
    story.append(Paragraph(
        "3. Model Results and Interpretations", heading1_style
    ))
    story.append(Spacer(1, 6))

    sorted_models = sorted(
        fit_results.items(),
        key=lambda x: _get_stat(x[1], 'r2', -1)
                      if x[1].get("success", False) else -1,
        reverse=True
    )

    for rank, (model_name, result) in enumerate(sorted_models, 1):
        success = result.get("success", False)
        clean_name = _clean_text(model_name)

        if success:
            if rank == 1:
                badge = "#1 [Best] "
                color = "#27ae60"
            elif rank == 2:
                badge = "#2 "
                color = "#f39c12"
            elif rank == 3:
                badge = "#3 "
                color = "#e67e22"
            else:
                badge = f"#{rank} "
                color = "#2c3e50"

            story.append(Paragraph(
                f'<font color="{color}"><b>{badge}{clean_name}</b></font>',
                heading2_style
            ))
        else:
            story.append(Paragraph(
                f'<b>[Failed] {clean_name}</b>', heading2_style
            ))

        if not success:
            story.append(Paragraph(
                "<i>Status:</i> This model failed to fit the experimental data.",
                body_style
            ))
            story.append(Paragraph(
                f"<i>Error:</i> {_clean_text(result.get('message', 'Unknown error'))}",
                body_style
            ))
            story.append(Spacer(1, 10))
            continue

        meta = MODEL_REGISTRY.get(model_name, {})
        params = result.get("params", {})

        r2_val = _get_stat(result, 'r2', 0)
        rmse_val = _get_stat(result, 'rmse', 0)
        mae_val = _get_stat(result, 'mae', 0)
        mape_val = _get_stat(result, 'mape', 0)

        # Clean metric table (no SPAN)
        stats_data = [
            ["Metric", "Value", "Interpretation"],
            ["R²", f"{r2_val:.4f}", _get_r2_interpretation(r2_val)],
            ["RMSE", f"{rmse_val:.2f} W/m·K",
             _get_rmse_interpretation(rmse_val, df)],
            ["MAE", f"{mae_val:.2f} W/m·K", "Mean absolute error"],
            ["MAPE", f"{mape_val:.2f}%", _get_mape_interpretation(mape_val)],
        ]

        t = Table(stats_data, repeatRows=1, colWidths=[1.0*inch, 1.5*inch, 3.5*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5490")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (1, -1), "CENTER"),
            ("ALIGN", (2, 1), (2, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f0f4f8")]),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

        if params:
            story.append(Paragraph(
                "<b>Optimized Parameters:</b>", body_style
            ))
            param_str = ", ".join(
                [f"{k} = {v:.4f}" for k, v in params.items()]
            )
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
        story.append(Paragraph(
            _clean_text(interpretation), interpretation_style
        ))

        story.append(Spacer(1, 10))
        story.append(HRFlowable(
            width="80%", thickness=0.5, color=colors.HexColor("#bdc3c7")
        ))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # ========================================================================
    # MODEL COMPARISON
    # ========================================================================
    story.append(Paragraph("4. Model Comparison", heading1_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "Statistical Comparison of All Models", heading2_style
    ))

    comp_data = [["Rank", "Model", "R²", "RMSE", "MAE", "MAPE (%)", "Status"]]
    for rank, (model_name, result) in enumerate(sorted_models, 1):
        clean_name = _clean_text(model_name)
        if result.get("success", False):
            r2 = _get_stat(result, 'r2', 0)
            rmse = _get_stat(result, 'rmse', 0)
            mae = _get_stat(result, 'mae', 0)
            mape = _get_stat(result, 'mape', 0)

            if rank == 1:
                rank_str = "#1"
            elif rank == 2:
                rank_str = "#2"
            elif rank == 3:
                rank_str = "#3"
            else:
                rank_str = str(rank)

            comp_data.append([
                rank_str,
                clean_name,
                f"{r2:.4f}",
                f"{rmse:.2f}",
                f"{mae:.2f}",
                f"{mape:.2f}",
                "OK"
            ])
        else:
            comp_data.append([
                str(rank), clean_name, "-", "-", "-", "-", "FAIL"
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
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f4f8")]),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # ---- Comparison interpretation ----
    story.append(Paragraph("Comparison Interpretation", heading2_style))

    successful_models = [
        (n, r) for n, r in sorted_models if r.get("success", False)
    ]

    if successful_models:
        best = successful_models[0]
        best_r2 = _get_stat(best[1], 'r2', 0)
        best_rmse = _get_stat(best[1], 'rmse', 0)
        best_name = _clean_text(best[0])

        comparison_text = (
            f"The <b>{best_name}</b> model provides the best fit to the "
            f"experimental data with an R² of <b>{best_r2:.4f}</b> and "
            f"RMSE of <b>{best_rmse:.2f} W/m·K</b>."
        )

        if len(successful_models) > 1:
            second = successful_models[1]
            second_r2 = _get_stat(second[1], 'r2', 0)
            second_name = _clean_text(second[0])
            comparison_text += (
                f" The second-best model is <b>{second_name}</b> with an "
                f"R² of <b>{second_r2:.4f}</b>."
            )

        if best_r2 >= 0.95:
            recommendation = (
                "The model shows excellent agreement with experimental data."
            )
        elif best_r2 >= 0.85:
            recommendation = (
                "The model shows good agreement with experimental data."
            )
        elif best_r2 >= 0.70:
            recommendation = (
                "The model shows moderate agreement with experimental data."
            )
        else:
            recommendation = (
                "The model shows poor agreement with experimental data."
            )

        comparison_text += f"\n\n<b>Recommendation:</b> {recommendation}"

        story.append(Paragraph(comparison_text, body_style))
    else:
        story.append(Paragraph(
            "No models were successfully fitted to the experimental data.",
            body_style
        ))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#bdc3c7")
    ))

    # ========================================================================
    # RECOMMENDATIONS (data-driven)
    # ========================================================================
    story.append(Paragraph("5. Recommendations", heading1_style))
    story.append(Spacer(1, 6))

    recommendations = _get_recommendations(fit_results, sorted_models, mode)
    for rec in recommendations:
        story.append(Paragraph(f"• {_clean_text(rec)}", body_style))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#bdc3c7")
    ))

    # ========================================================================
    # FIGURES
    # ========================================================================
    if figure_paths:
        story.append(Paragraph("6. Figures", heading1_style))
        story.append(Spacer(1, 6))

        for fig_path in figure_paths:
            if os.path.exists(fig_path):
                try:
                    img = Image(fig_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception:
                    story.append(Paragraph(
                        f"Figure could not be loaded: {fig_path}", body_style
                    ))

    # ========================================================================
    # FOOTER
    # ========================================================================
    story.append(Spacer(1, 24))
    story.append(Paragraph(
        f"Report generated on "
        f"{datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        styles["Normal"]
    ))
    story.append(Paragraph(
        f"Thermal Conductivity Modeling Suite v3.0 - {mode_name} Mode | "
        f"Oukil Khaled Research",
        styles["Normal"]
    ))

    doc.build(story)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_r2_interpretation(r2):
    if r2 >= 0.95:
        return "Excellent - Model explains >95% of variance"
    elif r2 >= 0.85:
        return "Good - Model explains 85-95% of variance"
    elif r2 >= 0.70:
        return "Fair - Model explains 70-85% of variance"
    elif r2 >= 0.50:
        return "Poor - Model explains 50-70% of variance"
    else:
        return "Very Poor - Model explains <50% of variance"


def _get_rmse_interpretation(rmse, df):
    if df is not None and len(df) > 0:
        k_mean = df["k_meas"].mean()
        if k_mean > 0:
            relative_rmse = (rmse / k_mean) * 100
            if relative_rmse < 5:
                return f"Excellent (RMSE < 5% of mean k = {k_mean:.2f} W/m·K)"
            elif relative_rmse < 10:
                return "Good (RMSE < 10% of mean k)"
            elif relative_rmse < 20:
                return "Moderate (RMSE < 20% of mean k)"
            else:
                return "Poor (RMSE > 20% of mean k)"
    return f"RMSE: {rmse:.2f} W/m·K"


def _get_mape_interpretation(mape):
    if mape < 5:
        return "Excellent - Very accurate predictions"
    elif mape < 10:
        return "Good - Reasonably accurate predictions"
    elif mape < 20:
        return "Moderate - Acceptable predictions"
    else:
        return "Poor - Inaccurate predictions"


def _get_model_interpretation(model_name, r2, rmse, params, description,
                              mode="nanothermite"):
    """Get mode-specific model interpretation."""

    base_interpretations = {
        "Parallel": (
            "The Parallel model represents the upper Hashin-Shtrikman bound, "
            "assuming perfect thermal contact between phases. This model "
            "typically overestimates thermal conductivity as it assumes heat "
            "flows in parallel through both phases."
        ),
        "Series": (
            "The Series model represents the lower Hashin-Shtrikman bound, "
            "assuming heat flows perpendicular to the layers. This model "
            "typically underestimates thermal conductivity."
        ),
        "Maxwell-Eucken": (
            "The Maxwell-Eucken model assumes dilute spherical inclusions in "
            "a continuous matrix with no particle interaction. This model "
            "works best at low filler loadings (<10%)."
        ),
        "Bruggeman": (
            "The Bruggeman model accounts for particle interactions and is "
            "valid for higher filler loadings. It treats both phases "
            "symmetrically."
        ),
        "Agari": (
            "The Agari model is an empirical model that accounts for filler "
            "geometry, polymer crystallinity, and conductive pathways. The "
            "parameters C1 and C2 represent the matrix's ability to form "
            "conductive chains and the filler's contribution to conductivity, "
            "respectively."
        ),
        "Lewis-Nielsen": (
            "The Lewis-Nielsen model incorporates particle geometry "
            "(parameter A) and maximum packing fraction (phi_m). This makes "
            "it suitable for a wide range of filler shapes and loadings."
        ),
        "Percolation": (
            "The Percolation model describes the formation of a conductive "
            "network above a critical threshold (phi_c). The power-law "
            "behavior with exponent t captures the transition from insulating "
            "to conducting behavior."
        ),
        "GEM Equation": (
            "The Generalized Effective Medium (GEM) equation combines "
            "percolation and effective medium theories. The parameter t "
            "represents the critical exponent (typically 1.6-2.0 for thermal "
            "conductivity), and phi_c is the percolation threshold."
        ),
    }

    if mode == "nanothermite":
        material_context = "energetic material"
    else:
        material_context = "polymer composite"

    interpretation = base_interpretations.get(model_name, "")

    if not interpretation:
        interpretation = (
            f"The {model_name} model was fitted to the experimental data with "
            f"R² = {r2:.4f} and RMSE = {rmse:.2f} W/m·K for this "
            f"{material_context} composite."
        )

    if r2 >= 0.95:
        assessment = (
            "<b>Assessment:</b> This model shows excellent agreement with "
            "experimental data."
        )
    elif r2 >= 0.85:
        assessment = (
            "<b>Assessment:</b> This model shows good agreement with "
            "experimental data."
        )
    elif r2 >= 0.70:
        assessment = (
            "<b>Assessment:</b> This model shows moderate agreement with "
            "experimental data."
        )
    else:
        assessment = (
            "<b>Assessment:</b> This model shows poor agreement with "
            "experimental data."
        )

    interpretation += f"\n\n{assessment}"

    return interpretation


def _get_recommendations(fit_results, sorted_models, mode="nanothermite"):
    """Data-driven recommendations based on actual fit results."""
    recommendations = []

    successful = [
        (n, r) for n, r in sorted_models if r.get("success", False)
    ]

    if not successful:
        recommendations.append(
            "No models were successfully fitted. Check your data quality "
            "and model parameters."
        )
        return recommendations

    # ---- Best model ----
    best = successful[0]
    best_r2 = _get_stat(best[1], 'r2', 0)
    best_name = _clean_text(best[0])

    recommendations.append(
        f"For this dataset, the <b>{best_name}</b> model provides the "
        f"best fit (R² = {best_r2:.4f})."
    )

    # ---- Data-driven top-3 recommendation ----
    top_3 = successful[:3]
    top_names = [_clean_text(n) for n, _ in top_3]
    if len(top_names) > 1:
        recommendations.append(
            f"Based on this dataset, the top-performing models were: "
            f"<b>{', '.join(top_names)}</b>."
        )

    # ---- Quality assessment ----
    if best_r2 >= 0.95:
        recommendations.append(
            "The model shows excellent agreement with experimental data."
        )
        recommendations.append(
            "The fitted parameters can be used to predict thermal "
            "conductivity for other compositions."
        )
    elif best_r2 >= 0.85:
        recommendations.append(
            "The model shows good agreement with experimental data."
        )
        recommendations.append(
            "Consider additional experimental data points to improve the "
            "fit quality."
        )
    elif best_r2 >= 0.70:
        recommendations.append(
            "The model shows moderate agreement with experimental data."
        )
        if mode == "nanothermite":
            recommendations.append(
                "Consider using a more sophisticated model or increasing "
                "the number of data points."
            )
        else:
            recommendations.append(
                "Consider using a more sophisticated model or checking "
                "filler dispersion quality."
            )
    else:
        recommendations.append(
            "The model shows poor agreement with experimental data."
        )
        recommendations.append(
            "Consider checking data quality, trying different models, or "
            "using custom parameters."
        )

    return recommendations