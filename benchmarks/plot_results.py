"""Plot benchmark results from benchmarks/results/ CSV files.

Layout: 3 rows × 2 columns
  rows    : expression-structure counts | build/compile/lambdify times | eval times
  columns : case (a) plain pull_production vs N | case (b) limit steps vs K
  y-axis  : shared across each row (same scale for both cases)
  x-axis  : shared within each column (counts share x with timings)

Usage:
    python benchmarks/plot_results.py                       # most recent run
    python benchmarks/plot_results.py --ts 20260430_063850  # specific timestamp
    python benchmarks/plot_results.py --out plot.png        # save to file
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── colour / style scheme ─────────────────────────────────────────────────────

PHASE_COLOR = {
    "build":          "tab:orange",
    "compile":        "tab:blue",
    "lambdify_np":    "tab:red",
    "lambdify_math":  "tab:green",
    "eval_np":        "tab:pink",
    "eval_math":      "tab:olive",
    "matrix_build":   "tab:gray",
    "matrix_solve":   "tab:purple",
    "bw_db_setup":    "tab:brown",
    "bw_lci_solve":   "sienna",
}

STRUCT_COLOR = {
    "raw_ops_total":    "tab:blue",
    "raw_ops_max_slot": "tab:red",
    "n_intermediates":  "tab:orange",
    "n_free_symbols":   "tab:green",
}

PHASE_LABEL = {
    "build":          "build",
    "compile":        "compile",
    "lambdify_np":    "lambdify (numpy)",
    "lambdify_math":  "lambdify (math)",
    "eval_np":        "eval (numpy)",
    "eval_math":      "eval (math)",
    "matrix_build":   "scipy matrix build",
    "matrix_solve":   "scipy matrix solve",
    "bw_db_setup":    "BW db setup",
    "bw_lci_solve":   "BW lci solve",
}

# Solid = chain/primary; dashed = fan/secondary variant
STYLE_PRIMARY   = dict(linestyle="-",  marker="o", markersize=5, lw=1.5)
STYLE_SECONDARY = dict(linestyle="--", marker="s", markersize=5, lw=1.5)
STYLE_SCIPY     = dict(linestyle=":",  marker="^", markersize=5, lw=1.5)
STYLE_BW        = dict(linestyle="-.", marker="D", markersize=5, lw=1.5)

# ── CSV loading ───────────────────────────────────────────────────────────────

def _load(prefix: str, ts: str | None) -> pd.DataFrame | None:
    if ts:
        path = os.path.join(RESULTS_DIR, f"{prefix}_{ts}.csv")
        return pd.read_csv(path) if os.path.exists(path) else None
    files = glob.glob(os.path.join(RESULTS_DIR, f"{prefix}_*.csv"))
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    print(f"  {prefix}: {os.path.basename(latest)}")
    return pd.read_csv(latest)


# ── drawing helpers ───────────────────────────────────────────────────────────

def _loglog_axis(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _line(ax, df: pd.DataFrame, xcol: str, ycol: str, color: str,
          label: str, **style):
    """Plot one line, dropping NaN/blank rows."""
    if df is None or ycol not in df.columns:
        return
    sub = df[[xcol, ycol]].dropna()
    if sub.empty:
        return
    ax.plot(sub[xcol], sub[ycol], color=color, label=label, **style)


def _variant_legend(ax, primary_label="chain", secondary_label="fan"):
    """Add a small grey legend distinguishing primary (solid) from secondary (dashed)."""
    ax.add_artist(
        ax.legend(
            handles=[
                Line2D([0], [0], color="0.4", lw=1.5, linestyle="-",
                       marker="o", markersize=4, label=primary_label),
                Line2D([0], [0], color="0.4", lw=1.5, linestyle="--",
                       marker="s", markersize=4, label=secondary_label),
            ],
            fontsize=7, loc="lower right", framealpha=0.8,
            title="variant", title_fontsize=7,
        )
    )


def _reference_line(ax, slope: float = 1.0, label: str = "linear"):
    """Add a subtle power-law reference line anchored at the centre of the view.

    For a log-log plot the line appears straight with the given slope.
    Must be called after data is plotted so auto-scaling has run.
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if any(v <= 0 for v in (xmin, xmax, ymin, ymax)):
        return
    # Anchor at geometric centre of current view
    xc = 10 ** ((np.log10(xmin) + np.log10(xmax)) / 2)
    yc = 10 ** ((np.log10(ymin) + np.log10(ymax)) / 2)
    c = yc / xc ** slope
    x = np.array([xmin, xmax])
    y = np.clip(c * x ** slope, ymin * 0.1, ymax * 10)
    ax.plot(x, y, color="0.75", lw=1.0, linestyle="--",
            zorder=0, label=f"~{label}")


# ── panel functions ───────────────────────────────────────────────────────────

def _panel_counts(ax, primary: pd.DataFrame | None, secondary: pd.DataFrame | None,
                  xcol: str, title: str):
    """Expression-structure counts panel (top row)."""
    _loglog_axis(ax, "", "count (ops / symbols)", title)

    # One legend entry per metric colour; primary/secondary distinguished by style
    for col, color in STRUCT_COLOR.items():
        _line(ax, primary,   xcol, col, color, col, **STYLE_PRIMARY)
        _line(ax, secondary, xcol, col, color, "_nolegend_", **STYLE_SECONDARY)

    leg = ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    ax.add_artist(leg)
    if secondary is not None:
        _variant_legend(ax)


def _panel_build(ax, primary: pd.DataFrame | None, secondary: pd.DataFrame | None,
                 xcol: str, title: str,
                 scipy_df: pd.DataFrame | None = None,
                 bw_df: pd.DataFrame | None = None,
                 phases=("build", "compile", "lambdify_np", "lambdify_math")):
    """Build / compile / lambdify timing panel (middle row)."""
    _loglog_axis(ax, "", "time (s)", title)

    for phase in phases:
        color = PHASE_COLOR.get(phase, "black")
        label = PHASE_LABEL.get(phase, phase)
        _line(ax, primary,   xcol, phase, color, label, **STYLE_PRIMARY)
        _line(ax, secondary, xcol, phase, color, "_nolegend_", **STYLE_SECONDARY)

    if scipy_df is not None:
        _line(ax, scipy_df, xcol, "matrix_build", PHASE_COLOR["matrix_build"],
              PHASE_LABEL["matrix_build"], **STYLE_SCIPY)
    if bw_df is not None:
        _line(ax, bw_df, xcol, "bw_db_setup", PHASE_COLOR["bw_db_setup"],
              PHASE_LABEL["bw_db_setup"], **STYLE_BW)

    leg = ax.legend(fontsize=7, loc="upper left", ncol=1, framealpha=0.8)
    ax.add_artist(leg)
    if secondary is not None:
        _variant_legend(ax)


def _panel_eval(ax, primary: pd.DataFrame | None, secondary: pd.DataFrame | None,
                xcol: str, xlabel: str, title: str,
                scipy_df: pd.DataFrame | None = None,
                bw_df: pd.DataFrame | None = None,
                phases=("eval_np", "eval_math")):
    """Eval timing panel (bottom row)."""
    _loglog_axis(ax, xlabel, "time (s)", title)

    for phase in phases:
        color = PHASE_COLOR.get(phase, "black")
        label = PHASE_LABEL.get(phase, phase)
        _line(ax, primary,   xcol, phase, color, label, **STYLE_PRIMARY)
        _line(ax, secondary, xcol, phase, color, "_nolegend_", **STYLE_SECONDARY)

    if scipy_df is not None:
        _line(ax, scipy_df, xcol, "matrix_solve", PHASE_COLOR["matrix_solve"],
              PHASE_LABEL["matrix_solve"], **STYLE_SCIPY)
    if bw_df is not None:
        _line(ax, bw_df, xcol, "bw_lci_solve", PHASE_COLOR["bw_lci_solve"],
              PHASE_LABEL["bw_lci_solve"], **STYLE_BW)

    leg = ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    ax.add_artist(leg)
    if secondary is not None:
        _variant_legend(ax)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot flowprog benchmark results")
    parser.add_argument("--ts", help="Timestamp suffix of run to plot (default: most recent)")
    parser.add_argument("--out", default=None, help="Output file path (default: show window)")
    args = parser.parse_args()

    print("Loading CSVs:")
    chain    = _load("case_a_chain", args.ts)
    fan      = _load("case_a_fan",   args.ts)
    scipy_df = _load("case_a_scipy", args.ts)
    bw_df    = _load("case_a_bw",    args.ts)
    b_df     = _load("case_b",       args.ts)

    fig, axes = plt.subplots(
        3, 2,
        figsize=(13, 12),
        constrained_layout=True,
    )

    # ── share x within each column ────────────────────────────────────────────
    for row in range(1, 3):
        axes[row, 0].sharex(axes[0, 0])
        axes[row, 1].sharex(axes[0, 1])

    # ── share y across each row ───────────────────────────────────────────────
    for col in range(2):
        axes[0, col].sharey(axes[0, 1 - col])
        axes[1, col].sharey(axes[1, 1 - col])
        axes[2, col].sharey(axes[2, 1 - col])

    # ── fill panels ───────────────────────────────────────────────────────────
    _panel_counts(axes[0, 0], chain, fan,   xcol="n",
                  title="Case (a) — expression structure")
    _panel_counts(axes[0, 1], b_df,  None,  xcol="k",
                  title="Case (b) — expression structure")

    _panel_build(axes[1, 0], chain, fan, xcol="n",
                 title="Case (a) — build / lambdify",
                 scipy_df=scipy_df, bw_df=bw_df)
    _panel_build(axes[1, 1], b_df, None, xcol="k",
                 title="Case (b) — build / lambdify")

    _panel_eval(axes[2, 0], chain, fan, xcol="n", xlabel="N (processes)",
                title="Case (a) — eval",
                scipy_df=scipy_df, bw_df=bw_df)
    _panel_eval(axes[2, 1], b_df, None, xcol="k", xlabel="K (limit steps)",
                title="Case (b) — eval",
                phases=("eval_math",))

    # ── hide x-tick labels on top two rows ───────────────────────────────────
    for row in range(2):
        for col in range(2):
            plt.setp(axes[row, col].get_xticklabels(), visible=False)
            axes[row, col].set_xlabel("")

    # ── add linear reference lines after auto-scaling ─────────────────────────
    # Force auto-scale so get_xlim/get_ylim return data-driven limits
    fig.canvas.draw()
    for ax in axes.flat:
        if ax.lines:  # only panels that have data
            _reference_line(ax, slope=1.0, label="linear")

    fig.suptitle("Flowprog benchmark results", fontsize=12, fontweight="bold")

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
