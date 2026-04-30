"""Plot benchmark results from benchmarks/results/ CSV files.

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
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── colour / style scheme ─────────────────────────────────────────────────────

# One colour per phase, shared across subplots
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

# Linestyle / marker per model variant
VARIANT_STYLE = {
    "chain":  dict(linestyle="-",  marker="o", markersize=5),
    "fan":    dict(linestyle="--", marker="s", markersize=5),
    "scipy":  dict(linestyle=":",  marker="^", markersize=5),
    "bw":     dict(linestyle="-.", marker="D", markersize=5),
    "single": dict(linestyle="-",  marker="o", markersize=5),
}

# Human-readable labels
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


# ── axis formatting ───────────────────────────────────────────────────────────

def _loglog_axis(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=8)
    # Integer ticks on x-axis (N or K are always integers)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _add_line(ax, df: pd.DataFrame, xcol: str, ycol: str, color: str,
              label: str, **style):
    sub = df[[xcol, ycol]].dropna()
    if sub.empty:
        return
    ax.plot(sub[xcol], sub[ycol], color=color, label=label, **style)


# ── per-panel plot functions ──────────────────────────────────────────────────

def _plot_a_counts(ax, chain: pd.DataFrame | None, fan: pd.DataFrame | None):
    _loglog_axis(ax, "", "count (ops / symbols)", "Case (a) — expression structure")
    for col, color in STRUCT_COLOR.items():
        if chain is not None and col in chain.columns:
            _add_line(ax, chain, "n", col, color,
                      f"{col} chain", **VARIANT_STYLE["chain"])
        if fan is not None and col in fan.columns:
            _add_line(ax, fan, "n", col, color,
                      f"{col} fan", **VARIANT_STYLE["fan"])
    ax.legend(fontsize=7, ncol=2)


def _plot_a_timings(ax, chain: pd.DataFrame | None, fan: pd.DataFrame | None,
                    scipy_df: pd.DataFrame | None, bw_df: pd.DataFrame | None):
    _loglog_axis(ax, "N (processes)", "time (s)", "Case (a) — timings")

    fp_phases = ["build", "compile", "lambdify_np", "lambdify_math",
                 "eval_np", "eval_math"]
    for phase in fp_phases:
        color = PHASE_COLOR[phase]
        label = PHASE_LABEL[phase]
        if chain is not None and phase in chain.columns:
            # Only the chain variant gets a legend entry; fan uses the same
            # colour with dashed style so the reader can see any divergence
            # without doubling the legend size.
            _add_line(ax, chain, "n", phase, color,
                      label, **VARIANT_STYLE["chain"])
        if fan is not None and phase in fan.columns:
            _add_line(ax, fan, "n", phase, color,
                      "_nolegend_", **VARIANT_STYLE["fan"])

    if scipy_df is not None:
        for col in ["matrix_build", "matrix_solve"]:
            if col in scipy_df.columns:
                _add_line(ax, scipy_df, "n", col,
                          PHASE_COLOR[col], PHASE_LABEL[col],
                          **VARIANT_STYLE["scipy"])

    if bw_df is not None:
        for col in ["bw_db_setup", "bw_lci_solve"]:
            if col in bw_df.columns:
                _add_line(ax, bw_df, "n", col,
                          PHASE_COLOR[col], PHASE_LABEL[col],
                          **VARIANT_STYLE["bw"])

    from matplotlib.lines import Line2D
    # Primary legend: one entry per phase (colours)
    leg1 = ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.add_artist(leg1)
    # Secondary legend: linestyle key (chain vs fan)
    ax.legend(
        handles=[
            Line2D([0], [0], color="gray", linestyle="-",  marker="o",
                   markersize=4, label="chain"),
            Line2D([0], [0], color="gray", linestyle="--", marker="s",
                   markersize=4, label="fan (dashed)"),
        ],
        fontsize=7, loc="lower right",
        framealpha=0.8,
    )


def _plot_b_counts(ax, b_df: pd.DataFrame | None):
    _loglog_axis(ax, "", "count (ops / symbols)", "Case (b) — expression structure")
    if b_df is None:
        return
    for col, color in STRUCT_COLOR.items():
        if col in b_df.columns:
            _add_line(ax, b_df, "k", col, color, col, **VARIANT_STYLE["single"])
    ax.legend(fontsize=7)


def _plot_b_timings(ax, b_df: pd.DataFrame | None):
    _loglog_axis(ax, "K (limit steps)", "time (s)", "Case (b) — timings")
    if b_df is None:
        return
    phases = ["build", "compile", "lambdify_np", "lambdify_math", "eval_math"]
    for phase in phases:
        if phase in b_df.columns:
            color = PHASE_COLOR.get(phase, "black")
            label = PHASE_LABEL.get(phase, phase)
            _add_line(ax, b_df, "k", phase, color, label,
                      **VARIANT_STYLE["single"])
    ax.legend(fontsize=7)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot flowprog benchmark results")
    parser.add_argument("--ts", help="Timestamp of run to plot (default: most recent)")
    parser.add_argument("--out", default=None, help="Output file path (default: show)")
    args = parser.parse_args()

    print("Loading CSVs:")
    chain  = _load("case_a_chain", args.ts)
    fan    = _load("case_a_fan",   args.ts)
    scipy_df = _load("case_a_scipy", args.ts)
    bw_df  = _load("case_a_bw",    args.ts)
    b_df   = _load("case_b",       args.ts)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        constrained_layout=True,
    )
    # Share x-axis within each column (counts top, timings bottom)
    axes[0, 0].sharex(axes[1, 0])
    axes[0, 1].sharex(axes[1, 1])

    _plot_a_counts( axes[0, 0], chain, fan)
    _plot_a_timings(axes[1, 0], chain, fan, scipy_df, bw_df)
    _plot_b_counts( axes[0, 1], b_df)
    _plot_b_timings(axes[1, 1], b_df)

    # Hide redundant x-axis tick labels on the top row
    for ax in axes[0, :]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")

    fig.suptitle("Flowprog benchmark results", fontsize=12, fontweight="bold")

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
