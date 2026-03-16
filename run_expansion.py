#!/usr/bin/env python3
"""
Asset Expansion Test
====================
Downloads data for additional ETFs and runs the validated strategy rules
on each one to test whether the edge generalizes beyond SPY/QQQ.

Usage:
    python run_expansion.py              # download + backtest
    python run_expansion.py --skip-dl    # skip download, just backtest
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import time as dt_time
from dataclasses import dataclass

from src.config import (
    INTRADAY_TIMEFRAMES, HTF_TIMEFRAMES,
    INTRADAY_START, DAILY_START, DATA_END, DATA_DIR, OUTPUT_DIR
)
from src.downloader import download_all, load_data
from src.htf_context import compute_all_htf_factors
from src.conditions import INTRADAY_CONDITIONS

# ── ETFs to test ──
# Major index ETFs + sector ETFs + bonds/commodities
EXPANSION_ETFS = [
    # Index ETFs (similar to SPY/QQQ)
    "IWM",   # Russell 2000
    "DIA",   # Dow Jones
    # Sector ETFs
    "XLK",   # Tech
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLI",   # Industrials
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    # Cross-asset
    "GLD",   # Gold
    "TLT",   # Long-term Treasuries
]

# ── Strategy rules to test on each ETF ──
# We test the 4 slow-drift rules that worked on SPY/QQQ
# (Gap rule is too specific to large-cap; skip it)
@dataclass
class ExpansionRule:
    name: str
    condition_name: str
    htf_factor: str
    max_hold_bars: int

RULES_TO_TEST = [
    ExpansionRule("HighVolWideRange_First30min", "first_30_min", "prior_day_high_vol_wide_range", 55),
    ExpansionRule("PriorDayWeak_First30min", "first_30_min", "prior_day_close_near_low", 55),
    ExpansionRule("PriorDayWeak_New50High", "new_50bar_high", "prior_day_close_near_low", 50),
]

# ── Backtest Parameters ──
INITIAL_CAPITAL = 100_000
POSITION_FRACTION = 0.15
SLIPPAGE_PER_SIDE = 0.0001
COMMISSION_PER_SHARE = 0.005


def download_expansion_data():
    """Download 5m, daily, weekly data for all expansion ETFs."""
    print("=" * 80)
    print("DOWNLOADING EXPANSION ETF DATA")
    print(f"  ETFs: {', '.join(EXPANSION_ETFS)}")
    print("=" * 80)
    download_all(tickers=EXPANSION_ETFS)
    print("\nDownload complete!")


def run_single_backtest(ticker, rule, intraday, htf):
    """Run a single rule on a single ticker. Returns list of trade dicts."""
    # Get condition function
    if rule.condition_name not in INTRADAY_CONDITIONS:
        return []

    cond_fn = INTRADAY_CONDITIONS[rule.condition_name]

    try:
        mask = cond_fn(intraday)
        if isinstance(mask, pd.Series):
            mask = mask.fillna(False).astype(bool)
    except Exception:
        return []

    # HTF factor lookup
    htf_factor_series = htf.get(rule.htf_factor)
    if htf_factor_series is None:
        return []

    close = intraday["close"].values
    high = intraday["high"].values
    low = intraday["low"].values
    dates = intraday.index.date
    times = intraday.index

    trades = []
    in_trade = False
    current_entry_price = 0.0
    current_entry_time = None
    current_shares = 0
    bars_since_entry = 0
    traded_dates = set()
    capital = INITIAL_CAPITAL

    for i in range(len(intraday)):
        bar_date = dates[i]
        bar_time = times[i]

        # ── Exit logic ──
        if in_trade:
            bars_since_entry += 1
            exit_price = None
            exit_reason = None

            # Time exit
            if bars_since_entry >= rule.max_hold_bars:
                exit_price = close[i]
                exit_reason = "time_exit"

            # EOD exit
            if exit_reason is None:
                if i + 1 < len(intraday) and dates[i + 1] != bar_date:
                    exit_price = close[i]
                    exit_reason = "eod_exit"
                elif i + 1 >= len(intraday):
                    exit_price = close[i]
                    exit_reason = "eod_exit"

            if exit_price is not None:
                exit_adj = exit_price * (1 - SLIPPAGE_PER_SIDE)
                pnl_per_share = exit_adj - current_entry_price
                pnl = pnl_per_share * current_shares
                commission = COMMISSION_PER_SHARE * current_shares * 2
                pnl -= commission
                capital += pnl

                trades.append({
                    "rule": f"{rule.name}_{ticker}",
                    "ticker": ticker,
                    "entry_time": current_entry_time,
                    "exit_time": bar_time,
                    "entry_price": current_entry_price,
                    "exit_price": exit_adj,
                    "shares": current_shares,
                    "exit_reason": exit_reason,
                    "pnl": pnl,
                    "pnl_pct": pnl / (current_entry_price * current_shares) if current_shares > 0 else 0,
                    "bars_held": bars_since_entry,
                })
                in_trade = False

        # ── Entry logic ──
        if not in_trade and (mask.iloc[i] if isinstance(mask, pd.Series) else mask[i]):
            # Check HTF factor
            bar_date_ts = pd.Timestamp(bar_date)
            if bar_date_ts in htf_factor_series.index:
                htf_val = htf_factor_series.loc[bar_date_ts]
            else:
                prior = htf_factor_series.index[htf_factor_series.index <= bar_date_ts]
                htf_val = htf_factor_series.loc[prior[-1]] if len(prior) > 0 else 0

            if htf_val != 1:
                continue

            # Only first per session
            if bar_date in traded_dates:
                continue

            entry_price = close[i] * (1 + SLIPPAGE_PER_SIDE)
            shares = int((capital * POSITION_FRACTION) / entry_price)
            if shares <= 0:
                continue

            position_value = shares * entry_price
            if position_value > capital * 0.95:
                shares = int(capital * 0.95 / entry_price)
                if shares <= 0:
                    continue

            current_entry_price = entry_price
            current_entry_time = bar_time
            current_shares = shares
            in_trade = True
            bars_since_entry = 0
            traded_dates.add(bar_date)

    return trades


def run_expansion_backtest():
    """Run all rules on all expansion ETFs + SPY/QQQ for comparison."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []  # (ticker, rule_name, n_trades, win_rate, total_pnl, avg_pnl, sharpe)

    all_tickers = ["SPY", "QQQ"] + EXPANSION_ETFS

    for ticker in all_tickers:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")

        # Load data
        try:
            intraday = load_data(ticker, "5m")
            daily = load_data(ticker, "1D")
            weekly = load_data(ticker, "1W")
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            continue

        # Strip tz for HTF
        daily_clean = daily.copy()
        if daily_clean.index.tz is not None:
            daily_clean.index = daily_clean.index.tz_localize(None)
        weekly_clean = weekly.copy()
        if weekly_clean.index.tz is not None:
            weekly_clean.index = weekly_clean.index.tz_localize(None)

        # Filter regular hours
        if hasattr(intraday.index, 'time'):
            time_mask = [(dt_time(9, 30) <= t <= dt_time(16, 0))
                         for t in intraday.index.time]
            intraday = intraday[time_mask]

        if len(intraday) == 0:
            print("  No intraday data")
            continue

        # HTF context
        htf = compute_all_htf_factors(daily_clean, weekly_clean)

        for rule in RULES_TO_TEST:
            trades = run_single_backtest(ticker, rule, intraday, htf)

            if len(trades) == 0:
                all_results.append({
                    "ticker": ticker, "rule": rule.name,
                    "n_trades": 0, "win_rate": 0, "total_pnl": 0,
                    "avg_pnl": 0, "avg_pnl_pct": 0, "sharpe": 0,
                })
                print(f"  {rule.name}: 0 trades")
                continue

            pnls = [t["pnl"] for t in trades]
            pnl_pcts = [t["pnl_pct"] for t in trades]
            n = len(trades)
            wr = sum(1 for p in pnls if p > 0) / n
            total = sum(pnls)
            avg = np.mean(pnls)
            avg_pct = np.mean(pnl_pcts)

            # Daily P&L for Sharpe
            daily_pnl = {}
            for t in trades:
                d = t["entry_time"].date() if hasattr(t["entry_time"], 'date') else pd.Timestamp(t["entry_time"]).date()
                daily_pnl[d] = daily_pnl.get(d, 0) + t["pnl"]
            daily_vals = list(daily_pnl.values())
            sharpe = (np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252)
                      if len(daily_vals) > 1 and np.std(daily_vals) > 0 else 0)

            all_results.append({
                "ticker": ticker, "rule": rule.name,
                "n_trades": n, "win_rate": wr, "total_pnl": total,
                "avg_pnl": avg, "avg_pnl_pct": avg_pct, "sharpe": sharpe,
            })

            color = "\033[92m" if total > 0 else "\033[91m"
            reset = "\033[0m"
            print(f"  {rule.name}: {n:>4} trades, WR {wr:.1%}, "
                  f"PnL {color}${total:>+8,.0f}{reset}, "
                  f"avg {avg_pct*100:+.4f}%, Sharpe {sharpe:+.2f}")

    # ── Save results ──
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTPUT_DIR, "expansion_results.csv"), index=False)

    # ── Summary table ──
    print(f"\n\n{'='*80}")
    print("EXPANSION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Ticker':<8} {'Rule':<35} {'Trades':>6} {'WR':>6} {'Total PnL':>10} {'Avg %':>8} {'Sharpe':>7}")
    print("-" * 80)

    for _, row in df.sort_values(["rule", "ticker"]).iterrows():
        if row["n_trades"] == 0:
            print(f"{row['ticker']:<8} {row['rule']:<35} {'—':>6}")
            continue
        color = "\033[92m" if row["total_pnl"] > 0 else "\033[91m"
        reset = "\033[0m"
        print(f"{row['ticker']:<8} {row['rule']:<35} {row['n_trades']:>6} "
              f"{row['win_rate']:>5.1%} {color}${row['total_pnl']:>+9,.0f}{reset} "
              f"{row['avg_pnl_pct']*100:>+7.4f}% {row['sharpe']:>+6.2f}")

    # ── Aggregate by ticker ──
    print(f"\n\n{'='*80}")
    print("AGGREGATE BY TICKER (all rules combined)")
    print(f"{'='*80}")
    agg = df.groupby("ticker").agg(
        total_trades=("n_trades", "sum"),
        total_pnl=("total_pnl", "sum"),
        avg_sharpe=("sharpe", "mean"),
    ).sort_values("total_pnl", ascending=False)

    for ticker, row in agg.iterrows():
        if row["total_trades"] == 0:
            continue
        color = "\033[92m" if row["total_pnl"] > 0 else "\033[91m"
        reset = "\033[0m"
        print(f"  {ticker:<8} {int(row['total_trades']):>5} trades  "
              f"{color}PnL ${row['total_pnl']:>+10,.0f}{reset}  "
              f"avg Sharpe {row['avg_sharpe']:+.2f}")

    # ── Heatmap: ticker × rule ──
    pivot = df.pivot_table(index="ticker", columns="rule", values="total_pnl", aggfunc="sum")
    pivot = pivot.reindex(index=["SPY", "QQQ"] + EXPANSION_ETFS)

    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.6)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    data = pivot.values
    vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 1
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in pivot.columns], color="white", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color="white", fontsize=11)

    for yi in range(len(pivot.index)):
        for xi in range(len(pivot.columns)):
            val = data[yi, xi]
            if not np.isnan(val):
                color_t = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(xi, yi, f"${val:,.0f}", ha="center", va="center",
                        fontsize=8, color=color_t, fontweight="bold")

    ax.set_title("Expansion Test: Total P&L by Ticker × Rule",
                 fontsize=14, fontweight="bold", color="white", pad=15)
    plt.colorbar(im, ax=ax, label="Total P&L ($)")
    fig.savefig(os.path.join(OUTPUT_DIR, "expansion_heatmap.png"),
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)

    # ── Bar chart: aggregate by ticker ──
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    tickers_sorted = agg.index.tolist()
    pnls = agg["total_pnl"].values
    colors = ["#00ff88" if p > 0 else "#ff4444" for p in pnls]

    bars = ax.bar(range(len(tickers_sorted)), pnls, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(tickers_sorted)))
    ax.set_xticklabels(tickers_sorted, color="white", fontsize=11)
    ax.set_ylabel("Total P&L ($)", color="white", fontsize=12)
    ax.set_title("Strategy P&L by ETF (All Rules Combined, $100k Capital, 15% Position Size)",
                 fontsize=13, fontweight="bold", color="white")
    ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, axis="y")

    # Add value labels
    for bar, val in zip(bars, pnls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (50 if val >= 0 else -300),
                f"${val:,.0f}", ha="center", va="bottom" if val >= 0 else "top",
                color="white", fontsize=9, fontweight="bold")

    fig.savefig(os.path.join(OUTPUT_DIR, "expansion_bar_chart.png"),
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)

    print(f"\n  Results saved → {OUTPUT_DIR}/")
    print(f"  • expansion_results.csv")
    print(f"  • expansion_heatmap.png")
    print(f"  • expansion_bar_chart.png")


if __name__ == "__main__":
    if "--skip-dl" not in sys.argv:
        download_expansion_data()
    run_expansion_backtest()
