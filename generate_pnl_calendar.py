#!/usr/bin/env python3
"""
Daily PnL Calendar Generator
=============================
Reads v5 trades, rescales to $1k risk per trade,
and generates monthly calendar heatmaps showing:
  - Daily PnL (color-coded green/red)
  - Number of trades taken that day
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import calendar

from src.config import OUTPUT_DIR

# ── Load trades ──
trades = pd.read_csv(os.path.join(OUTPUT_DIR, "phase5v5_trades.csv"), parse_dates=["entry_time", "exit_time"])

# ── Rescale to $1k risk per trade ──
# Each trade used ~15% of $100k = ~$15k notional
# pnl_pct is the return on that position
# For $1k risk per trade, we use pnl_pct * $1000 as the dollar PnL
# (treating $1k as the capital allocated per trade)
RISK_PER_TRADE = 1000
trades["pnl_scaled"] = trades["pnl_pct"] * RISK_PER_TRADE

# ── Aggregate by day ──
trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
trades["trade_date"] = trades["entry_time"].dt.date
daily = trades.groupby("trade_date").agg(
    pnl=("pnl_scaled", "sum"),
    n_trades=("pnl_scaled", "count"),
).reset_index()
daily["trade_date"] = pd.to_datetime(daily["trade_date"])

# Create a complete date range
all_dates = pd.date_range(daily["trade_date"].min(), daily["trade_date"].max(), freq="B")
daily_full = pd.DataFrame({"trade_date": all_dates})
daily_full = daily_full.merge(daily, on="trade_date", how="left")
daily_full["pnl"] = daily_full["pnl"].fillna(0)
daily_full["n_trades"] = daily_full["n_trades"].fillna(0).astype(int)

# ── Summary stats ──
trading_days = daily[daily["n_trades"] > 0]
print(f"Total trading days: {len(trading_days)}")
print(f"Total trades: {int(trading_days['n_trades'].sum())}")
print(f"Total PnL (scaled to $1k/trade): ${trading_days['pnl'].sum():,.2f}")
print(f"Avg daily PnL: ${trading_days['pnl'].mean():,.2f}")
print(f"Best day: ${trading_days['pnl'].max():,.2f}")
print(f"Worst day: ${trading_days['pnl'].min():,.2f}")
print(f"Win days: {(trading_days['pnl'] > 0).sum()} / {len(trading_days)} = {(trading_days['pnl'] > 0).mean():.1%}")
print(f"Avg trades per active day: {trading_days['n_trades'].mean():.1f}")

# ── Generate calendar pages ──
# Group by year-month
daily_full["year"] = daily_full["trade_date"].dt.year
daily_full["month"] = daily_full["trade_date"].dt.month
daily_full["day"] = daily_full["trade_date"].dt.day

year_months = sorted(daily_full[["year", "month"]].drop_duplicates().values.tolist())

# Color scale
max_abs_pnl = max(abs(trading_days["pnl"].max()), abs(trading_days["pnl"].min()))

# Create figure with all months
n_months = len(year_months)
cols = 3
rows = (n_months + cols - 1) // cols

fig = plt.figure(figsize=(24, rows * 5.5))
fig.patch.set_facecolor("#1a1a2e")

gs = GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.15)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

for idx, (yr, mo) in enumerate(year_months):
    ax = fig.add_subplot(gs[idx // cols, idx % cols])
    ax.set_facecolor("#1a1a2e")

    month_data = daily_full[(daily_full["year"] == yr) & (daily_full["month"] == mo)]

    # Calendar grid
    cal = calendar.Calendar(firstweekday=0)  # Monday start
    month_days = cal.monthdayscalendar(yr, mo)

    n_weeks = len(month_days)

    # Draw grid
    for week_idx, week in enumerate(month_days):
        for dow, day in enumerate(week):
            if dow >= 5:  # Skip weekends
                continue
            if day == 0:
                continue

            x = dow
            y = n_weeks - 1 - week_idx

            # Find this day's data
            day_row = month_data[month_data["day"] == day]

            if len(day_row) > 0 and day_row.iloc[0]["n_trades"] > 0:
                pnl = day_row.iloc[0]["pnl"]
                n_tr = int(day_row.iloc[0]["n_trades"])

                # Color intensity based on PnL magnitude
                intensity = min(abs(pnl) / max_abs_pnl, 1.0) * 0.85 + 0.15
                if pnl > 0:
                    color = (0, intensity * 0.9, intensity * 0.4, 0.85)
                elif pnl < 0:
                    color = (intensity * 0.9, 0, 0, 0.85)
                else:
                    color = (0.3, 0.3, 0.3, 0.5)

                rect = mpatches.FancyBboxPatch(
                    (x - 0.45, y - 0.42), 0.9, 0.84,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor="#444", linewidth=0.5
                )
                ax.add_patch(rect)

                # PnL text
                pnl_str = f"${pnl:+,.0f}" if abs(pnl) >= 1 else "$0"
                ax.text(x, y + 0.1, pnl_str, ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")
                # Trade count
                ax.text(x, y - 0.2, f"{n_tr}t", ha="center", va="center",
                        fontsize=6, color="#cccccc")
            else:
                # No trades - grey cell
                rect = mpatches.FancyBboxPatch(
                    (x - 0.45, y - 0.42), 0.9, 0.84,
                    boxstyle="round,pad=0.05",
                    facecolor="#222244", edgecolor="#333", linewidth=0.3
                )
                ax.add_patch(rect)

            # Day number
            ax.text(x - 0.35, y + 0.32, str(day), ha="left", va="top",
                    fontsize=5.5, color="#888888")

    # Month total
    month_pnl = month_data["pnl"].sum()
    month_trades = int(month_data["n_trades"].sum())
    color_m = "#00ff88" if month_pnl >= 0 else "#ff4444"

    ax.set_title(f"{calendar.month_name[mo]} {yr}   |   ${month_pnl:+,.0f}  ({month_trades} trades)",
                 fontsize=11, fontweight="bold", color=color_m, pad=8)

    # Weekday headers
    for i, wd in enumerate(WEEKDAYS[:5]):
        ax.text(i, n_weeks - 0.05, wd, ha="center", va="bottom",
                fontsize=7, color="#aaaaaa", fontweight="bold")

    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-0.6, n_weeks + 0.3)
    ax.axis("off")

fig.suptitle("Daily P&L Calendar  |  All v5 Strategies  |  $1,000 Risk Per Trade",
             fontsize=18, fontweight="bold", color="white", y=0.995)

out_path = os.path.join(OUTPUT_DIR, "pnl_calendar.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close(fig)
print(f"\nSaved → {out_path}")

# ── Also generate yearly summary ──
yearly = trading_days.copy()
yearly["year"] = yearly["trade_date"].dt.year
yearly_agg = yearly.groupby("year").agg(
    total_pnl=("pnl", "sum"),
    trading_days=("pnl", "count"),
    win_days=("pnl", lambda x: (x > 0).sum()),
    total_trades=("n_trades", "sum"),
    best_day=("pnl", "max"),
    worst_day=("pnl", "min"),
).reset_index()
yearly_agg["win_rate"] = yearly_agg["win_days"] / yearly_agg["trading_days"]

print(f"\n{'='*70}")
print("YEARLY SUMMARY ($1k risk per trade)")
print(f"{'='*70}")
print(f"{'Year':>6} {'PnL':>10} {'Days':>6} {'WR':>6} {'Trades':>7} {'Best':>8} {'Worst':>8}")
print("-" * 70)
for _, row in yearly_agg.iterrows():
    print(f"{int(row['year']):>6} ${row['total_pnl']:>+8,.0f} {int(row['trading_days']):>6} "
          f"{row['win_rate']:>5.1%} {int(row['total_trades']):>7} "
          f"${row['best_day']:>+7,.0f} ${row['worst_day']:>+7,.0f}")
print("-" * 70)
print(f"{'TOTAL':>6} ${yearly_agg['total_pnl'].sum():>+8,.0f} {int(yearly_agg['trading_days'].sum()):>6} "
      f"{(trading_days['pnl'] > 0).mean():>5.1%} {int(yearly_agg['total_trades'].sum()):>7} "
      f"${yearly_agg['best_day'].max():>+7,.0f} ${yearly_agg['worst_day'].min():>+7,.0f}")
