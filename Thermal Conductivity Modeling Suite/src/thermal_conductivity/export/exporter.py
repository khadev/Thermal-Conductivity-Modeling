"""Export results to images, Excel, and CSV - High DPI export."""

import os
import pandas as pd
import numpy as np


def export_plot(canvas, filepath, dpi=300):
    """Save matplotlib figure to file with high DPI."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".tiff":
        canvas.fig.savefig(filepath, dpi=600, format="tiff", 
                          pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
    elif ext == ".png":
        canvas.fig.savefig(filepath, dpi=600, format="png", bbox_inches="tight")
    elif ext in [".svg", ".pdf"]:
        canvas.fig.savefig(filepath, format=ext[1:], bbox_inches="tight")
    else:
        canvas.fig.savefig(filepath, dpi=300, bbox_inches="tight")


def export_excel(filepath, df, fit_results):
    """Export data and results to Excel workbook."""
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # Experimental Data
        export_df = df.copy()
        # Use Filler and Matrix columns
        if "Filler_volume_fraction" in export_df.columns and "Matrix_volume_fraction" in export_df.columns:
            export_df = export_df.rename(columns={
                "Filler_volume_fraction": "Filler_volume_fraction (φ)",
                "Matrix_volume_fraction": "Matrix_volume_fraction (1-φ)"
            })
        # Handle legacy columns if present
        elif "Red_volume_fraction" in export_df.columns and "Oxy_volume_fraction" in export_df.columns:
            export_df = export_df.rename(columns={
                "Red_volume_fraction": "Filler_volume_fraction (φ)",
                "Oxy_volume_fraction": "Matrix_volume_fraction (1-φ)"
            })
        export_df.to_excel(writer, sheet_name="Experimental Data", index=False)

        # Statistics
        stats_rows = []
        for name, res in fit_results.items():
            if res["success"]:
                stats_rows.append({
                    "Model": name,
                    "R2": res.get("r2", np.nan),
                    "RMSE": res.get("rmse", np.nan),
                    "MAE": res.get("mae", np.nan),
                    "MAPE_%": res.get("mape", np.nan),
                    "Status": "Success"
                })
            else:
                stats_rows.append({
                    "Model": name,
                    "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MAPE_%": np.nan,
                    "Status": res.get("message", "Failed")
                })
        pd.DataFrame(stats_rows).to_excel(writer, sheet_name="Statistics", index=False)

        # Fitted Parameters
        param_rows = []
        for name, res in fit_results.items():
            if res.get("params"):
                row = {"Model": name}
                row.update({k: v for k, v in res["params"].items()})
                param_rows.append(row)
        if param_rows:
            pd.DataFrame(param_rows).to_excel(writer, sheet_name="Fitted Parameters", index=False)

        # Predictions
        pred_dict = {}
        # Use Filler fraction as φ
        if "Filler_volume_fraction" in df.columns:
            pred_dict["Filler_volume_fraction (φ)"] = df["Filler_volume_fraction"].values
        else:
            pred_dict["φ"] = df["phi"].values
        pred_dict["k_meas"] = df["k_meas"].values
        for name, res in fit_results.items():
            if res["success"] and res.get("y_pred") is not None:
                pred_dict[name] = res["y_pred"]
        pd.DataFrame(pred_dict).to_excel(writer, sheet_name="Predictions", index=False)


def export_csv(filepath, df, fit_results):
    """Export predictions to CSV."""
    pred_dict = {}
    if "Filler_volume_fraction" in df.columns:
        pred_dict["Filler_volume_fraction (φ)"] = df["Filler_volume_fraction"].values
    else:
        pred_dict["φ"] = df["phi"].values
    pred_dict["k_meas"] = df["k_meas"].values
    for name, res in fit_results.items():
        if res["success"] and res.get("y_pred") is not None:
            pred_dict[name] = res["y_pred"]
    pd.DataFrame(pred_dict).to_csv(filepath, index=False)
