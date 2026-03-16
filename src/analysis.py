"""
Analysis and Visualization Module
===================================
Takes SnapshotResult objects and produces:
- Statistical comparisons across conditions
- P&L curve plots
- Heatmaps
- Ranking tables
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from src.snapshot_engine import SnapshotResult
from src.config import OUTPUT_DIR, MIN_OCCURRENCES, P_VALUE_THRESHOLD


# ─── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "text.color": "#ccc",
    "xtick.color": "#999",
    "ytick.color": "#999",
    "grid.color": "#222",
    "grid.alpha": 0.6,
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#333",
    "font.family": "monospace",
    "font.size": 10,
})

COLORS = ["#00d4aa", "#ff6b6b", "#4ecdc4", "#ffd93d", "#6bcb77",
          "#e8a87c", "#95e1d3", "#f38181", "#aa96da", "#fcbad3"]


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ─── Ranking Table ───────────────────────────────────────────────────────────

def build_ranking_table(results: list[SnapshotResult],
                        min_occurrences: int = MIN_OCCURRENCES) -> pd.DataFrame:
    """
    Build a ranked comparison table from multiple SnapshotResults.
    Filters by minimum occurrences and sorts by mean P&L at peak bar.
    """
    rows = []
    for r in results:
        if r.n_occurrences < min_occurrences:
            continue
        stats = r.summary_stats()
        if stats:
            rows.append(stats)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("mean_pnl_at_peak", ascending=False)
    return df.reset_index(drop=True)


# ─── P&L Curve Plots ────────────────────────────────────────────────────────

def plot_pnl_curve(result: SnapshotResult,
                   filename: str = None,
                   show_bands: bool = True) -> str:
    """
    Plot the forward P&L curve for a single condition.
    Shows mean, median, and percentile bands.
    """
    if result.n_occurrences == 0:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))
    n_bars = len(result.mean_curve)
    x = np.arange(1, n_bars + 1)

    # Percentile bands
    if show_bands:
        ax.fill_between(x, result.pct_10 * 100, result.pct_90 * 100,
                        alpha=0.1, color=COLORS[0], label="10-90th pct")
        ax.fill_between(x, result.pct_25 * 100, result.pct_75 * 100,
                        alpha=0.2, color=COLORS[0], label="25-75th pct")

    # Mean and median
    ax.plot(x, result.mean_curve * 100, color=COLORS[0], linewidth=2,
            label=f"Mean (n={result.n_occurrences:,})")
    ax.plot(x, result.median_curve * 100, color=COLORS[1], linewidth=1.5,
            linestyle="--", label="Median")

    # Mark peak
    peak = result.peak_bar_mean
    if peak < n_bars:
        ax.axvline(x=peak + 1, color="#ffd93d", linewidth=0.8, linestyle=":",
                   label=f"Peak bar (Bar+{peak+1})")

    ax.axhline(0, color="#666", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Bars After Entry")
    ax.set_ylabel("P&L (%)")
    ax.set_title(f"{result.condition_name}  |  {result.ticker} {result.timeframe}\n"
                 f"n={result.n_occurrences:,}  |  Peak P&L: {result.mean_curve[peak]*100:.3f}% at Bar+{peak+1}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    if filename is None:
        safe_name = result.condition_name.replace(" ", "_").replace("|", "_")
        filename = f"pnl_curve_{safe_name}_{result.ticker}_{result.timeframe}.png"

    return _save(fig, filename)


def plot_pnl_curves_comparison(results: list[SnapshotResult],
                               title: str = "P&L Curve Comparison",
                               filename: str = "pnl_comparison.png") -> str:
    """Plot multiple P&L curves on the same chart for comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, r in enumerate(results):
        if r.n_occurrences == 0:
            continue
        n_bars = len(r.mean_curve)
        x = np.arange(1, n_bars + 1)
        label = f"{r.condition_name} (n={r.n_occurrences:,})"
        ax.plot(x, r.mean_curve * 100, color=COLORS[i % len(COLORS)],
                linewidth=1.8, label=label, alpha=0.9)

    ax.axhline(0, color="#666", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Bars After Entry")
    ax.set_ylabel("Mean P&L (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    return _save(fig, filename)


def plot_htf_comparison(result: SnapshotResult,
                        factor_name: str,
                        values: list,
                        filename: str = None) -> str:
    """
    Compare P&L curves for the same condition sliced by different HTF context values.
    e.g., "breakout_above_prior_day_high" sliced by above_ema_20 = 1 vs 0.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, val in enumerate(values):
        try:
            sliced = result.slice_by_htf(factor_name, val)
        except ValueError:
            continue
        if sliced.n_occurrences < 30:
            continue
        n_bars = len(sliced.mean_curve)
        x = np.arange(1, n_bars + 1)
        label = f"{factor_name}={val} (n={sliced.n_occurrences:,})"
        ax.plot(x, sliced.mean_curve * 100, color=COLORS[i % len(COLORS)],
                linewidth=1.8, label=label)

    ax.axhline(0, color="#666", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Bars After Entry")
    ax.set_ylabel("Mean P&L (%)")
    ax.set_title(f"{result.condition_name} — HTF Split by {factor_name}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    if filename is None:
        filename = f"htf_split_{factor_name}_{result.condition_name.replace(' ', '_')}.png"

    return _save(fig, filename)


def plot_peak_bar_distribution(result: SnapshotResult,
                               filename: str = None) -> str:
    """Histogram of which bar each trade hits its max P&L."""
    if result.n_occurrences == 0:
        return ""

    fig, ax = plt.subplots(figsize=(14, 4))
    peaks_raw = result.peak_bar_distribution
    if len(peaks_raw) == 0 or np.all(np.isnan(peaks_raw)):
        plt.close(fig)
        return ""
    peaks = peaks_raw[~np.isnan(peaks_raw)] + 1  # convert to 1-indexed
    if len(peaks) == 0:
        plt.close(fig)
        return ""
    ax.hist(peaks, bins=min(50, max(1, len(set(peaks)))), color=COLORS[0],
            alpha=0.7, edgecolor="#333")
    ax.axvline(x=np.median(peaks), color=COLORS[1], linewidth=2,
               linestyle="--", label=f"Median: Bar+{int(np.median(peaks))}")
    ax.set_xlabel("Bar of Maximum P&L")
    ax.set_ylabel("Count")
    ax.set_title(f"{result.condition_name} — Peak P&L Timing Distribution",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if filename is None:
        filename = f"peak_dist_{result.condition_name.replace(' ', '_')}.png"

    return _save(fig, filename)


def plot_mfe_mae(result: SnapshotResult, filename: str = None) -> str:
    """Scatter plot of MFE vs MAE for each occurrence."""
    if result.n_occurrences == 0:
        return ""

    fig, ax = plt.subplots(figsize=(8, 8))
    mfe_raw = result.mfe_distribution
    mae_raw = result.mae_distribution
    if len(mfe_raw) == 0 or len(mae_raw) == 0:
        plt.close(fig)
        return ""
    mfe = mfe_raw * 100
    mae = mae_raw * 100

    ax.scatter(mae, mfe, alpha=0.3, s=10, color=COLORS[0])
    ax.axhline(0, color="#666", linewidth=0.5)
    ax.axvline(0, color="#666", linewidth=0.5)
    ax.set_xlabel("Max Adverse Excursion (%)")
    ax.set_ylabel("Max Favorable Excursion (%)")
    ax.set_title(f"{result.condition_name} — MFE vs MAE\n"
                 f"Avg MFE: {np.mean(mfe):.3f}% | Avg MAE: {np.mean(mae):.3f}%",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if filename is None:
        filename = f"mfe_mae_{result.condition_name.replace(' ', '_')}.png"

    return _save(fig, filename)


def plot_win_rate_evolution(result: SnapshotResult, filename: str = None) -> str:
    """Plot win rate at each bar after entry."""
    if result.n_occurrences == 0:
        return ""

    fig, ax = plt.subplots(figsize=(14, 4))
    wr = result.win_rate_curve * 100
    x = np.arange(1, len(wr) + 1)

    ax.plot(x, wr, color=COLORS[0], linewidth=2)
    ax.axhline(50, color="#666", linewidth=0.8, linestyle="--", label="50%")
    ax.set_xlabel("Bars After Entry")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title(f"{result.condition_name} — Win Rate Evolution",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    if filename is None:
        filename = f"win_rate_{result.condition_name.replace(' ', '_')}.png"

    return _save(fig, filename)
