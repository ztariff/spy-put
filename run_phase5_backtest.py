#!/usr/bin/env python3
"""
Phase 5: Full Strategy Backtest
=================================
Takes validated signals from Phase 4 and runs a realistic backtest with:
- Defined entry/exit rules derived from P&L snapshot analysis
- Stop losses based on MAE distributions
- Profit targets based on MFE distributions
- Time-based exits derived from peak bar analysis
- Position sizing (fixed fractional)
- Slippage and commission modeling

Usage:
    python run_phase5_backtest.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import time as dt_time
from dataclasses import dataclass, field

from src.config import ACTIVE_TICKERS, OUTPUT_DIR
from src.downloader import load_data
from src.htf_context import compute_all_htf_factors
from src.conditions import INTRADAY_CONDITIONS, HTF_BRIDGE_CONDITIONS


# ── Strategy Definitions ──
# Each strategy defines entry conditions, exit rules, and parameters
# Parameters are derived from Phase 3 P&L snapshot analysis

@dataclass
class StrategyRule:
    """A single strategy rule with entry/exit logic."""
    name: str
    condition_name: str        # intraday condition to trigger entry
    htf_factor: str            # HTF factor that must be = 1
    ticker: str
    timeframe: str
    direction: str = "long"    # "long" or "short"

    # Exit parameters (derived from P&L snapshot peak bars and MFE/MAE)
    max_hold_bars: int = 30       # Time-based exit (bars after entry)
    stop_loss_pct: float = 0.004  # Stop loss (% of entry price)
    profit_target_pct: float = 0.008  # Profit target (% of entry price)

    # Entry timing
    entry_after_bar: int = 0      # Wait N bars after condition before entering
    only_first_per_session: bool = True  # Only take first signal per day


# ═══════════════════════════════════════════════════════════════════
# v4: NO STOPS on slow-drift signals — time exits only
# Key insight from v1-v3: stops destroy the edge on slow-drift signals.
#   Time exits are profitable EVERY year (even 2022 bear market).
#   Stop losses trigger on 27% of trades from normal intraday noise.
#   Solution: remove stops entirely, let the drift play out.
# ═══════════════════════════════════════════════════════════════════

# Strategy A: Gap Continuation (fast scalp — keeps tight stop + target)
STRATEGY_A_RULES = [
    # ★★★ STRONG PASS — OOS t=3.65, peak bar 8-9
    StrategyRule(
        name="GapLarge_First30min_SPY",
        condition_name="first_30_min",
        htf_factor="gap_up_large",
        ticker="SPY", timeframe="5m",
        max_hold_bars=12,          # Peak bar ~8, cut by 12
        stop_loss_pct=0.003,       # Tight 0.3% stop (fast signal = small adverse)
        profit_target_pct=0.002,   # 0.2% scalp target — high hit rate
    ),
]

# Strategy B: Slow drift signals — NO stops, time exit + EOD exit only
STRATEGY_B_RULES = [
    # ★★★ STRONG PASS — best rule, OOS t=3.54, peak bar 57
    StrategyRule(
        name="HighVolWideRange_First30min_SPY",
        condition_name="first_30_min",
        htf_factor="prior_day_high_vol_wide_range",
        ticker="SPY", timeframe="5m",
        max_hold_bars=55,
        stop_loss_pct=0.0,         # NO STOP — let the drift play out
        profit_target_pct=0.0,
    ),
    # ★★ PASS — OOS t=1.63
    StrategyRule(
        name="PriorDayWeak_First30min_SPY",
        condition_name="first_30_min",
        htf_factor="prior_day_close_near_low",
        ticker="SPY", timeframe="5m",
        max_hold_bars=55,
        stop_loss_pct=0.0,         # NO STOP
        profit_target_pct=0.0,
    ),
    # ★★ PASS — OOS t=1.62
    StrategyRule(
        name="PriorDayWeak_First30min_QQQ",
        condition_name="first_30_min",
        htf_factor="prior_day_close_near_low",
        ticker="QQQ", timeframe="5m",
        max_hold_bars=55,
        stop_loss_pct=0.0,         # NO STOP
        profit_target_pct=0.0,
    ),
    # ★★★ STRONG PASS — OOS t=3.12
    StrategyRule(
        name="PriorDayWeak_New50High_SPY",
        condition_name="new_50bar_high",
        htf_factor="prior_day_close_near_low",
        ticker="SPY", timeframe="5m",
        max_hold_bars=50,
        stop_loss_pct=0.0,         # NO STOP
        profit_target_pct=0.0,
    ),
    # NR7_New50High_SPY removed — only rule with negative P&L in no-stop backtest
]

ALL_RULES = STRATEGY_A_RULES + STRATEGY_B_RULES

# ── Backtest Parameters ──
INITIAL_CAPITAL = 100_000
RISK_PER_TRADE = 0.005       # Risk 0.5% of capital per trade (for rules WITH stops)
POSITION_FRACTION = 0.15     # 15% of capital per trade (for rules WITHOUT stops)
SLIPPAGE_PER_SIDE = 0.0001   # 1 bps slippage per side
COMMISSION_PER_SHARE = 0.005  # $0.005 per share

@dataclass
class Trade:
    """Record of a single trade."""
    rule_name: str
    ticker: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp = None
    exit_price: float = 0.0
    shares: int = 0
    direction: str = "long"
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0


def run_backtest(rules=None, use_oos_only=False):
    """
    Run event-driven backtest across all rules.

    Parameters
    ----------
    rules : list of StrategyRule, or None for ALL_RULES
    use_oos_only : bool
        If True, only backtest on the last 20% of data (out-of-sample)
    """
    if rules is None:
        rules = ALL_RULES

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_trades = []
    data_cache = {}

    print("=" * 80)
    print("PHASE 5: FULL STRATEGY BACKTEST")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}  |  Risk/trade: {RISK_PER_TRADE*100:.1f}%  |  "
          f"Slippage: {SLIPPAGE_PER_SIDE*10000:.1f} bps/side")
    if use_oos_only:
        print("  MODE: Out-of-sample only (last 20% of data)")
    print("=" * 80)

    for rule in rules:
        print(f"\n{'─'*60}")
        print(f"  Rule: {rule.name}")
        print(f"  {rule.condition_name} × {rule.htf_factor} ({rule.ticker}/{rule.timeframe})")
        stop_str = f"{rule.stop_loss_pct*100:.2f}%" if rule.stop_loss_pct > 0 else "NONE"
        sizing_str = f"risk-based {RISK_PER_TRADE*100:.1f}%" if rule.stop_loss_pct > 0 else f"fixed {POSITION_FRACTION*100:.0f}%"
        print(f"  Stop: {stop_str}  Target: {rule.profit_target_pct*100:.2f}%  "
              f"MaxHold: {rule.max_hold_bars} bars  Sizing: {sizing_str}")
        print(f"{'─'*60}")

        # Load data
        cache_key = (rule.ticker, rule.timeframe)
        if cache_key not in data_cache:
            try:
                intraday = load_data(rule.ticker, rule.timeframe)
                daily = load_data(rule.ticker, "1D")
                weekly = load_data(rule.ticker, "1W")
                data_cache[cache_key] = (intraday, daily, weekly)
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue

        intraday, daily, weekly = data_cache[cache_key]

        # Strip tz
        daily_clean = daily.copy()
        if daily_clean.index.tz is not None:
            daily_clean.index = daily_clean.index.tz_localize(None)
        weekly_clean = weekly.copy()
        if weekly_clean.index.tz is not None:
            weekly_clean.index = weekly_clean.index.tz_localize(None)

        # OOS filter
        if use_oos_only:
            unique_dates = sorted(set(intraday.index.date))
            oos_start = unique_dates[int(len(unique_dates) * 0.8)]
            intraday = intraday[intraday.index.date >= oos_start]
            daily_for_htf = daily_clean  # need full history for HTF computation
            weekly_for_htf = weekly_clean
        else:
            daily_for_htf = daily_clean
            weekly_for_htf = weekly_clean

        # HTF context
        htf = compute_all_htf_factors(daily_for_htf, weekly_for_htf)

        # Filter regular hours
        if hasattr(intraday.index, 'time'):
            time_mask = [(dt_time(9, 30) <= t <= dt_time(16, 0))
                         for t in intraday.index.time]
            intraday = intraday[time_mask]

        if len(intraday) == 0:
            print("  No data in period")
            continue

        # Get condition function
        is_bridge = False
        if rule.condition_name in INTRADAY_CONDITIONS:
            cond_fn = INTRADAY_CONDITIONS[rule.condition_name]
        elif rule.condition_name in HTF_BRIDGE_CONDITIONS:
            cond_fn = HTF_BRIDGE_CONDITIONS[rule.condition_name]
            is_bridge = True
        else:
            print(f"  Condition not found: {rule.condition_name}")
            continue

        # Evaluate condition
        try:
            if is_bridge:
                mask = cond_fn(intraday, daily_clean)
                if isinstance(mask, pd.Series):
                    mask = mask.reindex(intraday.index).fillna(False).astype(bool)
            else:
                mask = cond_fn(intraday)
                if isinstance(mask, pd.Series):
                    mask = mask.fillna(False).astype(bool)
        except Exception as e:
            print(f"  Condition error: {e}")
            continue

        # Build HTF factor lookup (date → value)
        htf_factor_series = htf.get(rule.htf_factor)
        if htf_factor_series is None:
            print(f"  HTF factor not found: {rule.htf_factor}")
            continue

        # ── Event-driven backtest ──
        close = intraday["close"].values
        high = intraday["high"].values
        low = intraday["low"].values
        dates = intraday.index.date
        times = intraday.index

        capital = INITIAL_CAPITAL
        rule_trades = []
        in_trade = False
        current_trade = None
        bars_since_entry = 0
        traded_dates = set()

        for i in range(len(intraday)):
            bar_date = dates[i]
            bar_time = times[i]

            # ── Exit logic (check before entry) ──
            if in_trade:
                bars_since_entry += 1
                current_price = close[i]
                bar_high = high[i]
                bar_low = low[i]

                exit_price = None
                exit_reason = None

                if rule.direction == "long":
                    # Stop loss (intra-bar check using low) — skip if no stop
                    if rule.stop_loss_pct > 0:
                        stop_price = current_trade.entry_price * (1 - rule.stop_loss_pct)
                        if bar_low <= stop_price:
                            exit_price = stop_price
                            exit_reason = "stop_loss"

                    # Profit target (intra-bar check using high) — skip if 0
                    if rule.profit_target_pct > 0 and exit_reason is None:
                        target_price = current_trade.entry_price * (1 + rule.profit_target_pct)
                        if bar_high >= target_price:
                            exit_price = target_price
                            exit_reason = "profit_target"

                    # Time exit
                    if bars_since_entry >= rule.max_hold_bars and exit_reason is None:
                        exit_price = current_price
                        exit_reason = "time_exit"

                    # End of session exit
                    if i + 1 < len(intraday) and dates[i + 1] != bar_date and exit_reason is None:
                        exit_price = current_price
                        exit_reason = "eod_exit"
                    elif i + 1 >= len(intraday) and exit_reason is None:
                        exit_price = current_price
                        exit_reason = "eod_exit"

                if exit_price is not None:
                    # Apply slippage
                    exit_price_adj = exit_price * (1 - SLIPPAGE_PER_SIDE)

                    pnl_per_share = exit_price_adj - current_trade.entry_price
                    pnl = pnl_per_share * current_trade.shares
                    commission = COMMISSION_PER_SHARE * current_trade.shares * 2  # round trip
                    pnl -= commission

                    current_trade.exit_time = bar_time
                    current_trade.exit_price = exit_price_adj
                    current_trade.pnl = pnl
                    current_trade.pnl_pct = pnl / (current_trade.entry_price * current_trade.shares)
                    current_trade.bars_held = bars_since_entry
                    current_trade.exit_reason = exit_reason

                    capital += pnl
                    rule_trades.append(current_trade)
                    in_trade = False

            # ── Entry logic ──
            if not in_trade and mask.iloc[i] if isinstance(mask, pd.Series) else mask[i]:
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
                if rule.only_first_per_session and bar_date in traded_dates:
                    continue

                # Calculate position size
                entry_price = close[i] * (1 + SLIPPAGE_PER_SIDE)  # slippage on entry
                if rule.stop_loss_pct > 0:
                    # Risk-based sizing: risk X% of capital per trade
                    risk_amount = capital * RISK_PER_TRADE
                    stop_distance = entry_price * rule.stop_loss_pct
                    shares = int(risk_amount / stop_distance)
                else:
                    # No stop → fixed fraction sizing (invest ~5% of capital per trade)
                    position_budget = capital * POSITION_FRACTION
                    shares = int(position_budget / entry_price)
                if shares <= 0:
                    continue

                # Don't exceed capital
                position_value = shares * entry_price
                if position_value > capital * 0.95:
                    shares = int(capital * 0.95 / entry_price)
                    if shares <= 0:
                        continue

                current_trade = Trade(
                    rule_name=rule.name,
                    ticker=rule.ticker,
                    entry_time=bar_time,
                    entry_price=entry_price,
                    shares=shares,
                    direction=rule.direction,
                )
                in_trade = True
                bars_since_entry = 0
                traded_dates.add(bar_date)

        # ── Rule Summary ──
        all_trades.extend(rule_trades)

        if rule_trades:
            pnls = [t.pnl for t in rule_trades]
            pnl_pcts = [t.pnl_pct for t in rule_trades]
            wins = [t for t in rule_trades if t.pnl > 0]
            exits = {}
            for t in rule_trades:
                exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

            print(f"  Trades: {len(rule_trades):,}")
            print(f"  Win rate: {len(wins)/len(rule_trades)*100:.1f}%")
            print(f"  Total P&L: ${sum(pnls):,.2f}")
            print(f"  Avg P&L/trade: ${np.mean(pnls):.2f} ({np.mean(pnl_pcts)*100:.4f}%)")
            print(f"  Max win: ${max(pnls):.2f}  Max loss: ${min(pnls):.2f}")
            print(f"  Avg bars held: {np.mean([t.bars_held for t in rule_trades]):.1f}")
            print(f"  Exits: {exits}")
        else:
            print("  No trades generated")

    # ── Portfolio Summary ──
    if not all_trades:
        print("\nNo trades across all rules.")
        return

    print(f"\n\n{'='*80}")
    print("PORTFOLIO BACKTEST SUMMARY")
    print(f"{'='*80}")

    # Sort all trades by entry time
    all_trades.sort(key=lambda t: t.entry_time)

    # Build equity curve
    equity = [INITIAL_CAPITAL]
    equity_dates = [all_trades[0].entry_time]
    running_capital = INITIAL_CAPITAL

    for trade in all_trades:
        running_capital += trade.pnl
        equity.append(running_capital)
        equity_dates.append(trade.exit_time if trade.exit_time else trade.entry_time)

    equity = np.array(equity)
    peak_equity = np.maximum.accumulate(equity)
    drawdown = (equity - peak_equity) / peak_equity

    total_pnl = equity[-1] - INITIAL_CAPITAL
    total_return = total_pnl / INITIAL_CAPITAL
    n_trades = len(all_trades)
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')

    max_dd = abs(drawdown.min())

    # Approximate Sharpe (daily P&L)
    daily_pnl = {}
    for t in all_trades:
        d = t.entry_time.date()
        daily_pnl[d] = daily_pnl.get(d, 0) + t.pnl
    if daily_pnl:
        daily_returns = list(daily_pnl.values())
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                  if np.std(daily_returns) > 0 else 0)
    else:
        sharpe = 0

    # Calmar ratio
    calmar = total_return / max_dd if max_dd > 0 else 0

    # Per-rule breakdown
    rule_stats = {}
    for t in all_trades:
        if t.rule_name not in rule_stats:
            rule_stats[t.rule_name] = {"trades": 0, "wins": 0, "pnl": 0}
        rule_stats[t.rule_name]["trades"] += 1
        if t.pnl > 0:
            rule_stats[t.rule_name]["wins"] += 1
        rule_stats[t.rule_name]["pnl"] += t.pnl

    print(f"\n  Initial Capital:   ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Final Capital:     ${equity[-1]:>12,.2f}")
    print(f"  Total P&L:         ${total_pnl:>12,.2f}  ({total_return*100:.2f}%)")
    print(f"  Total Trades:      {n_trades:>8,}")
    print(f"  Win Rate:          {win_rate*100:>8.1f}%")
    print(f"  Avg Win:           ${avg_win:>12,.2f}")
    print(f"  Avg Loss:          ${avg_loss:>12,.2f}")
    print(f"  Profit Factor:     {profit_factor:>8.2f}")
    print(f"  Max Drawdown:      {max_dd*100:>8.2f}%")
    print(f"  Sharpe Ratio:      {sharpe:>8.2f}")
    print(f"  Calmar Ratio:      {calmar:>8.2f}")

    print(f"\n  Per-Rule Breakdown:")
    for name, stats in sorted(rule_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        print(f"    {name:40s}  trades={stats['trades']:>4}  "
              f"WR={wr:.1f}%  P&L=${stats['pnl']:>10,.2f}")

    # ── Save results ──
    trade_records = [{
        "rule": t.rule_name,
        "ticker": t.ticker,
        "entry_time": t.entry_time,
        "entry_price": t.entry_price,
        "exit_time": t.exit_time,
        "exit_price": t.exit_price,
        "shares": t.shares,
        "direction": t.direction,
        "exit_reason": t.exit_reason,
        "pnl": t.pnl,
        "pnl_pct": t.pnl_pct,
        "bars_held": t.bars_held,
    } for t in all_trades]

    pd.DataFrame(trade_records).to_csv(
        os.path.join(OUTPUT_DIR, "phase5_trades.csv"), index=False
    )

    # ── Equity Curve Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    # Equity curve
    ax1 = axes[0]
    ax1.set_facecolor("#1a1a2e")
    ax1.plot(equity_dates, equity, color="#00d2ff", linewidth=1.5, label="Equity")
    ax1.fill_between(equity_dates, INITIAL_CAPITAL, equity,
                     where=(equity >= INITIAL_CAPITAL), alpha=0.15, color="#00ff88")
    ax1.fill_between(equity_dates, INITIAL_CAPITAL, equity,
                     where=(equity < INITIAL_CAPITAL), alpha=0.15, color="#ff4444")
    ax1.axhline(INITIAL_CAPITAL, color="#666", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Portfolio Value ($)", color="white")
    ax1.set_title(f"Momentum Strategy Backtest — Sharpe {sharpe:.2f} | "
                  f"Return {total_return*100:.1f}% | MaxDD {max_dd*100:.1f}%",
                  fontsize=14, fontweight="bold", color="white")
    ax1.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    ax1.tick_params(colors="white")
    ax1.grid(True, alpha=0.2)

    # Drawdown
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    ax2.fill_between(equity_dates, drawdown * 100, 0, alpha=0.4, color="#ff4444")
    ax2.set_ylabel("Drawdown (%)", color="white")
    ax2.set_xlabel("Date", color="white")
    ax2.tick_params(colors="white")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "phase5_equity_curve.png"),
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)

    # ── Monthly returns heatmap ──
    monthly_pnl = {}
    for t in all_trades:
        key = (t.entry_time.year, t.entry_time.month)
        monthly_pnl[key] = monthly_pnl.get(key, 0) + t.pnl

    if monthly_pnl:
        years = sorted(set(k[0] for k in monthly_pnl.keys()))
        months = range(1, 13)

        fig, ax = plt.subplots(figsize=(14, max(3, len(years))))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        data = np.full((len(years), 12), np.nan)
        for (y, m), pnl in monthly_pnl.items():
            yi = years.index(y)
            data[yi, m - 1] = pnl

        # Normalize for coloring
        vmax = np.nanmax(np.abs(data))
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                       vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                           color="white")
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years, color="white")
        ax.set_title("Monthly P&L ($)", fontsize=14, fontweight="bold", color="white")

        # Add text annotations
        for yi in range(len(years)):
            for mi in range(12):
                val = data[yi, mi]
                if not np.isnan(val):
                    color = "black" if abs(val) < vmax * 0.5 else "white"
                    ax.text(mi, yi, f"${val:,.0f}", ha="center", va="center",
                            fontsize=7, color=color)

        plt.colorbar(im, ax=ax, label="P&L ($)")
        fig.savefig(os.path.join(OUTPUT_DIR, "phase5_monthly_returns.png"),
                    dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)

    print(f"\n  Results saved → {OUTPUT_DIR}/")
    print(f"  • phase5_trades.csv — all trade records")
    print(f"  • phase5_equity_curve.png — equity curve with drawdown")
    print(f"  • phase5_monthly_returns.png — monthly P&L heatmap")
    print("=" * 80)


if __name__ == "__main__":
    # First run on full data, then optionally run OOS-only
    import sys
    if "--oos" in sys.argv:
        print("Running out-of-sample backtest only...")
        run_backtest(use_oos_only=True)
    else:
        run_backtest(use_oos_only=False)
