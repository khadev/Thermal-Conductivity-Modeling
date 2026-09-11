"""Publication-ready matplotlib plots – φ = FILLER fraction with compact legend inside."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..core.models import get_model_registry


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=9, height=7, dpi=150):
        self.fig = Figure(figsize=(width, height), dpi=150)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 13
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 9


def create_main_plot(canvas, df, fit_results, style_name="Thermochimica Acta",
                     log_scale=True, show_legend=True, legend_format="compact",
                     material="Al/CuO", matrix="Al", filler="CuO",
                     x_col="Filler_volume_fraction", x_label=None,
                     phi_col="Filler_volume_fraction",
                     x_min=0.0, x_max=1.0, max_models_to_show=8,
                     mode="nanothermite"):
    from .styles import PLOT_STYLES
    style = PLOT_STYLES.get(style_name, PLOT_STYLES["Thermochimica Acta"])
    style['axes.labelsize'] = 13
    style['axes.titlesize'] = 14
    style['legend.fontsize'] = 9
    style['xtick.labelsize'] = 12
    style['ytick.labelsize'] = 12

    with plt.rc_context(style):
        ax = canvas.axes
        ax.clear()

        if x_col in df.columns:
            xdata = df[x_col].values
        else:
            xdata = df["Filler_volume_fraction"].values
                
        k_meas = df["k_meas"].values
        k_err = df["k_std_dev"].values if "k_std_dev" in df.columns else np.full(len(df), np.nan)

        mask = (xdata >= x_min) & (xdata <= x_max)
        xdata_filtered = xdata[mask]
        k_meas_filtered = k_meas[mask]
        k_err_filtered = k_err[mask] if len(k_err) == len(xdata) else np.full(len(xdata_filtered), np.nan)

        # Experimental data
        has_err = not np.all(np.isnan(k_err_filtered))
        if has_err:
            ax.errorbar(xdata_filtered, k_meas_filtered, yerr=k_err_filtered, fmt="o", color="black",
                       markersize=8, capsize=4, label="Experimental", zorder=5, linewidth=1.5)
        else:
            ax.scatter(xdata_filtered, k_meas_filtered, color="black", s=80, zorder=5, label="Experimental")

        MODEL_REGISTRY = get_model_registry(mode)

        x_smooth = np.linspace(x_min, x_max, 200)
        phi_filler_smooth = np.clip(x_smooth, 0, 1)

        sorted_results = sorted(
            fit_results.items(),
            key=lambda x: x[1].get("r2", -1) if x[1].get("success", False) else -1,
            reverse=True
        )
        
        discovered = []
        standard = []
        for name, result in sorted_results:
            if not result.get("success", False):
                continue
            r2 = result.get("r2", -1)
            if not np.isfinite(r2):
                continue
            if MODEL_REGISTRY.get(name, {}).get("discovered", False):
                discovered.append((name, result))
            else:
                standard.append((name, result))
        
        standard_to_show = standard[:5]
        discovered_to_show = discovered[:3]
        models_to_show = standard_to_show + discovered_to_show
        
        if len(models_to_show) > max_models_to_show:
            models_to_show = models_to_show[:max_models_to_show]

        for name, result in models_to_show:
            if not result.get("success", False):
                continue
            y_pred = result.get("y_pred")
            if y_pred is None:
                continue
            meta = MODEL_REGISTRY.get(name, {})
            try:
                if result.get("popt") is not None:
                    y_smooth = meta["func"](phi_filler_smooth, *result["popt"])
                else:
                    y_smooth = meta["func"](phi_filler_smooth)
                if np.any(np.isnan(y_smooth)) or np.any(np.isinf(y_smooth)):
                    continue
                r2 = result.get("r2", 0)
                label = f"{name} (R²={r2:.3f})"
                linestyle = meta.get("linestyle", "-")
                if meta.get("discovered", False):
                    linestyle = "--"
                if name == models_to_show[0][0]:
                    color = "#27ae60"
                    linewidth = 2.5
                    alpha = 1.0
                else:
                    color = meta.get("color", "#2c3e50")
                    linewidth = meta.get("linewidth", 1.8)
                    alpha = 0.85
                ax.plot(x_smooth, y_smooth, color=color,
                       linestyle=linestyle, linewidth=linewidth,
                       label=label, alpha=alpha, zorder=3)
            except Exception as e:
                print(f"Error plotting {name}: {e}")
                continue

        if x_label is None:
            x_label = f"{filler} Volume Fraction, φ"
        
        ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
        ax.set_ylabel("Thermal Conductivity, $k$ (W/m·K)", fontsize=13, fontweight='bold')
        ax.set_title(f"{material}", fontsize=14, fontweight='bold')

        if log_scale:
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")

        # === FORCE PLAIN NUMBERS ===
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False, useMathText=False))
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

        ax.set_xlim(x_min, x_max)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='best', framealpha=0.9, fontsize=8,
                         handlelength=1.2, handletextpad=0.5,
                         borderpad=0.3, labelspacing=0.2)

        canvas.fig.tight_layout()
        canvas.draw()


def create_residuals_plot(canvas, df, fit_results, style_name="Thermochimica Acta",
                          show_legend=True, matrix="Al", filler="CuO",
                          x_col="Filler_volume_fraction", x_label=None,
                          x_min=0.0, x_max=1.0, max_models_to_show=8,
                          mode="nanothermite"):
    from .styles import PLOT_STYLES
    style = PLOT_STYLES.get(style_name, PLOT_STYLES["Thermochimica Acta"])
    style['axes.labelsize'] = 13
    style['axes.titlesize'] = 14
    style['legend.fontsize'] = 9
    style['xtick.labelsize'] = 12
    style['ytick.labelsize'] = 12

    with plt.rc_context(style):
        ax = canvas.axes
        ax.clear()

        if x_col in df.columns:
            xdata = df[x_col].values
        else:
            xdata = df["Filler_volume_fraction"].values

        mask = (xdata >= x_min) & (xdata <= x_max)
        xdata_filtered = xdata[mask]

        MODEL_REGISTRY = get_model_registry(mode)

        sorted_results = sorted(
            fit_results.items(),
            key=lambda x: x[1].get("r2", -1) if x[1].get("success", False) else -1,
            reverse=True
        )
        
        discovered = []
        standard = []
        for name, result in sorted_results:
            if not result.get("success", False):
                continue
            r2 = result.get("r2", -1)
            if not np.isfinite(r2):
                continue
            if MODEL_REGISTRY.get(name, {}).get("discovered", False):
                discovered.append((name, result))
            else:
                standard.append((name, result))
        
        standard_to_show = standard[:5]
        discovered_to_show = discovered[:3]
        models_to_show = standard_to_show + discovered_to_show
        
        if len(models_to_show) > max_models_to_show:
            models_to_show = models_to_show[:max_models_to_show]

        all_residuals = []
        best_model_name = models_to_show[0][0] if models_to_show else None

        for name, result in models_to_show:
            if not result.get("success", False) or result.get("residuals") is None:
                continue
            residuals = result["residuals"]
            if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
                continue
            meta = MODEL_REGISTRY.get(name, {})
            residuals_filtered = residuals[mask]
            all_residuals.extend(residuals_filtered)
            marker = "s" if meta.get("discovered", False) else "o"
            if name == best_model_name:
                color = "#27ae60"
                size = 70
                alpha = 1.0
                zorder = 5
            else:
                color = meta.get("color", "#2c3e50")
                size = 50
                alpha = 0.7
                zorder = 3
            ax.scatter(xdata_filtered, residuals_filtered, color=color,
                      marker=marker, s=size, alpha=alpha, label=name, zorder=zorder)

        ax.axhline(0, color="black", linewidth=1.5, linestyle="-", zorder=2)

        if all_residuals:
            mean_resid = np.mean(all_residuals)
            std_resid = np.std(all_residuals)
            stats_text = f"Mean: {mean_resid:.3f} W/m·K\nStd: {std_resid:.3f} W/m·K"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
                   fontsize=10, family='monospace')
            max_resid = np.max(np.abs(all_residuals))
            if max_resid > 0:
                y_max = max_resid * 1.3
                ax.set_ylim(-y_max, y_max)

        if x_label is None:
            x_label = f"{filler} Volume Fraction, φ"

        ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
        ax.set_ylabel("Residuals (W/m·K)", fontsize=13, fontweight='bold')
        ax.set_title("Model Residuals", fontsize=14, fontweight='bold')

        ax.set_xlim(x_min, x_max)

        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False, useMathText=False))
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='best', framealpha=0.9, fontsize=8,
                         handlelength=1.2, handletextpad=0.5,
                         borderpad=0.3, labelspacing=0.2)

        canvas.fig.tight_layout()
        canvas.draw()
