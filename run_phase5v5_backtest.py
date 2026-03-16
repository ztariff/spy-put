#!/usr/bin/env python3
"""
Phase 5 v5: Filtered Backtest with Long + Short Rules
======================================================
Adds intraday confirmation filters from Phase 6 findings:
  - LONG filters: Wide OR, RVOL > 1.5, Lower Half Prior Range
  - SHORT rules: inverse HTF conditions + anti-filters (Above OR Mid, Upper Half Range)

Usage:
    python run_phase5v5_backtest.py
    python run_phase5v5_backtest.py --no-filters   # baseline comparison
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import time as dt_time
from dataclasses import dataclass, field
from typing import List, Callable, Optional

from src.config import OUTPUT_DIR
from src.downloader import load_data
from src.htf_context import compute_all_htf_factors
from src.conditions import INTRADAY_CONDITIONS


# ── Intraday Feature Computer ──
def compute_entry_features(intraday: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday features at each bar for entry filtering."""
    feat = pd.DataFrame(index=intraday.index)
    dates = intraday.index.date
    c = intraday["close"]
    h = intraday["high"]
    l = intraday["low"]
    o = intraday["open"]
    v = intraday["volume"]

    # RVOL
    avg_vol_20 = v.rolling(20).mean()
    feat["bar_rvol"] = v / avg_vol_20.replace(0, np.nan)
    feat["bar_rvol_gt_1_5"] = (feat["bar_rvol"] > 1.5).astype(int)

    # Opening range (first 6 bars = 30 min on 5m)
    or_high = h.groupby(dates).transform(lambda x: x.iloc[:6].max() if len(x) >= 6 else np.nan)
    or_low = l.groupby(dates).transform(lambda x: x.iloc[:6].min() if len(x) >= 6 else np.nan)
    or_mid = (or_high + or_low) / 2
    session_open = o.groupby(dates).transform("first")

    feat["above_or_mid"] = (c > or_mid).astype(int)
    feat["below_or_mid"] = (c < or_mid).astype(int)
    feat["above_or_high"] = (c > or_high).astype(int)
    feat["below_or_low"] = (c < or_low).astype(int)

    # OR width vs average
    or_width_pct = (or_high - or_low) / session_open
    daily_or_width = or_width_pct.groupby(dates).first()
    avg_or_width = daily_or_width.rolling(20).mean()
    or_width_ratio = daily_or_width / avg_or_width.replace(0, np.nan)
    or_ratio_mapped = or_width_ratio.reindex(pd.DatetimeIndex(dates), method="ffill")
    or_ratio_mapped.index = intraday.index
    feat["or_wide"] = (or_ratio_mapped > 1.5).astype(int)
    feat["or_narrow"] = (or_ratio_mapped < 0.7).astype(int)

    # Prior day range position
    daily_clean = daily.copy()
    if daily_clean.index.tz is not None:
        daily_clean.index = daily_clean.index.tz_localize(None)
    daily_clean.index = daily_clean.index.normalize()

    prev_high = daily_clean["high"].shift(1)
    prev_low = daily_clean["low"].shift(1)
    prev_close = daily_clean["close"].shift(1)
    bar_dates_idx = pd.DatetimeIndex(dates)
    ph = prev_high.reindex(bar_dates_idx, method="ffill"); ph.index = intraday.index
    pl = prev_low.reindex(bar_dates_idx, method="ffill"); pl.index = intraday.index
    pc = prev_close.reindex(bar_dates_idx, method="ffill"); pc.index = intraday.index

    prior_range = (ph - pl).replace(0, np.nan)
    feat["pos_in_prior_range"] = (c - pl) / prior_range
    feat["in_upper_half_prior_range"] = (feat["pos_in_prior_range"] > 0.5).astype(int)
    feat["in_lower_half_prior_range"] = (feat["pos_in_prior_range"] < 0.5).astype(int)
    feat["above_prior_day_high"] = (c > ph).astype(int)
    feat["below_prior_day_low"] = (c < pl).astype(int)

    # Prior week range
    pw_high = daily_clean["high"].rolling(5).max().shift(1)
    pw_low = daily_clean["low"].rolling(5).min().shift(1)
    pwh = pw_high.reindex(bar_dates_idx, method="ffill"); pwh.index = intraday.index
    pwl = pw_low.reindex(bar_dates_idx, method="ffill"); pwl.index = intraday.index
    feat["below_prior_week_low"] = (c < pwl).astype(int)
    feat["above_prior_week_high"] = (c > pwh).astype(int)

    # VWAP
    cum_vol_price = (c * v).groupby(dates).cumsum()
    cum_vol = v.groupby(dates).cumsum()
    vwap = cum_vol_price / cum_vol.replace(0, np.nan)
    feat["above_vwap"] = (c > vwap).astype(int)
    feat["below_vwap"] = (c < vwap).astype(int)

    # Session return
    session_return = (c - session_open) / session_open
    feat["session_return_positive"] = (session_return > 0).astype(int)
    feat["session_return_negative"] = (session_return < 0).astype(int)

    # First bar direction
    first_close = c.groupby(dates).transform("first")
    first_open = o.groupby(dates).transform("first")
    feat["first_bar_green"] = (first_close > first_open).astype(int)
    feat["first_bar_red"] = (first_close < first_open).astype(int)

    return feat


# ── Strategy Rule with Filters ──
@dataclass
class StrategyRule:
    name: str
    condition_name: str
    htf_factor: str
    ticker: str
    timeframe: str = "5m"
    direction: str = "long"
    max_hold_bars: int = 55
    stop_loss_pct: float = 0.0
    profit_target_pct: float = 0.0
    only_first_per_session: bool = True
    # Intraday filters: list of (feature_col, required_value) tuples
    # ALL must be satisfied for entry (AND logic)
    intraday_filters: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# v5 RULES: Filtered longs + short-side rules
# ═══════════════════════════════════════════════════════════════════

# ── LONG RULES (with Phase 6 confirmation filters) ──
LONG_RULES = [
    # Gap scalp — fast signal, keeps stop + target (no intraday filter needed)
    StrategyRule(
        name="GapLarge_First30min_SPY",
        condition_name="first_30_min",
        htf_factor="gap_up_large",
        ticker="SPY",
        direction="long",
        max_hold_bars=12,
        stop_loss_pct=0.003,
        profit_target_pct=0.002,
        intraday_filters=[],  # fast scalp, don't filter
    ),
    # HighVol WideRange — filter: Wide OR day + RVOL confirmation
    StrategyRule(
        name="HighVolWR_30min_SPY_filtered",
        condition_name="first_30_min",
        htf_factor="prior_day_high_vol_wide_range",
        ticker="SPY",
        direction="long",
        max_hold_bars=55,
        intraday_filters=[("or_wide", 1)],  # +0.22% improvement
    ),
    # PriorDayWeak SPY — filter: Wide OR or RVOL > 1.5
    StrategyRule(
        name="PriorDayWeak_30min_SPY_filtered",
        condition_name="first_30_min",
        htf_factor="prior_day_close_near_low",
        ticker="SPY",
        direction="long",
        max_hold_bars=55,
        intraday_filters=[("bar_rvol_gt_1_5", 1)],  # +0.017%, keeps sample size
    ),
    # PriorDayWeak QQQ — unfiltered (filters didn't help much on QQQ)
    StrategyRule(
        name="PriorDayWeak_30min_QQQ",
        condition_name="first_30_min",
        htf_factor="prior_day_close_near_low",
        ticker="QQQ",
        direction="long",
        max_hold_bars=55,
        intraday_filters=[],
    ),
    # PriorDayWeak New50High — filter: RVOL > 2.0 (biggest improver +0.31%)
    StrategyRule(
        name="PriorDayWeak_50Hi_SPY_filtered",
        condition_name="new_50bar_high",
        htf_factor="prior_day_close_near_low",
        ticker="SPY",
        direction="long",
        max_hold_bars=50,
        intraday_filters=[("bar_rvol_gt_1_5", 1)],  # +0.037%, good balance
    ),
]

# ── SHORT RULES ──
# Inverse of the long thesis: prior day closed strong + price already
# extended above OR midpoint = fade the strength
SHORT_RULES = [
    # Prior day closed near high (strong close) + first 30 min + above OR mid
    # The "Above OR Midpoint" was the WORST long filter (-0.23%) → potential short signal
    StrategyRule(
        name="PriorDayStrong_AboveOR_SPY_short",
        condition_name="first_30_min",
        htf_factor="prior_day_close_near_high",
        ticker="SPY",
        direction="short",
        max_hold_bars=55,
        intraday_filters=[("above_or_mid", 1)],  # extended above OR = fade
    ),
    StrategyRule(
        name="PriorDayStrong_AboveOR_QQQ_short",
        condition_name="first_30_min",
        htf_factor="prior_day_close_near_high",
        ticker="QQQ",
        direction="short",
        max_hold_bars=55,
        intraday_filters=[("above_or_mid", 1)],
    ),
    # HighVolWR_UpperHalf_SPY_short REMOVED — 33% WR, -$1,098
    # NR7_BelowOR_SPY_short REMOVED — 40.6% WR, -$2,098
]

USE_FILTERS = "--no-filters" not in sys.argv
ALL_RULES = LONG_RULES + (SHORT_RULES if USE_FILTERS else [])

# ── Backtest Parameters ──
INITIAL_CAPITAL = 100_000
RISK_PER_TRADE = 0.005
POSITION_FRACTION = 0.15
SLIPPAGE_PER_SIDE = 0.0001
COMMISSION_PER_SHARE = 0.005


@dataclass
class Trade:
    rule_name: str
    ticker: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp = None
    exit_price: float = 0.0
    shares: int = 0
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0


def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_trades = []
    data_cache = {}

    mode = "FILTERED + SHORTS" if USE_FILTERS else "NO FILTERS (baseline)"
    print("=" * 80)
    print(f"PHASE 5 v5: {mode}")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}  |  Rules: {len(ALL_RULES)}")
    print(f"  Long rules: {sum(1 for r in ALL_RULES if r.direction=='long')}  "
          f"Short rules: {sum(1 for r in ALL_RULES if r.direction=='short')}")
    print("=" * 80)

    for rule in ALL_RULES:
        print(f"\n{'─'*60}")
        print(f"  Rule: {rule.name} [{rule.direction.upper()}]")
        print(f"  {rule.condition_name} × {rule.htf_factor} ({rule.ticker}/{rule.timeframe})")
        filters_str = ", ".join(f"{f[0]}={f[1]}" for f in rule.intraday_filters) if rule.intraday_filters else "NONE"
        print(f"  Filters: {filters_str}")
        print(f"{'─'*60}")

        # Load data
        cache_key = rule.ticker
        if cache_key not in data_cache:
            try:
                intraday = load_data(rule.ticker, "5m")
                daily = load_data(rule.ticker, "1D")
                weekly = load_data(rule.ticker, "1W")
            except Exception as e:
                print(f"  Skip: {e}")
                continue

            daily_clean = daily.copy()
            if daily_clean.index.tz is not None:
                daily_clean.index = daily_clean.index.tz_localize(None)
            weekly_clean = weekly.copy()
            if weekly_clean.index.tz is not None:
                weekly_clean.index = weekly_clean.index.tz_localize(None)

            # Filter regular hours
            if hasattr(intraday.index, 'time'):
                time_mask = [(dt_time(9, 30) <= t <= dt_time(16, 0)) for t in intraday.index.time]
                intraday = intraday[time_mask]

            htf = compute_all_htf_factors(daily_clean, weekly_clean)
            features = compute_entry_features(intraday, daily)
            data_cache[cache_key] = (intraday, daily_clean, htf, features)

        intraday, daily_clean, htf, features = data_cache[cache_key]

        # Get condition
        if rule.condition_name in INTRADAY_CONDITIONS:
            cond_fn = INTRADAY_CONDITIONS[rule.condition_name]
        else:
            print(f"  Condition not found: {rule.condition_name}")
            continue

        try:
            mask = cond_fn(intraday)
            if isinstance(mask, pd.Series):
                mask = mask.fillna(False).astype(bool)
        except Exception as e:
            print(f"  Condition error: {e}")
            continue

        # HTF factor
        htf_series = htf.get(rule.htf_factor)
        if htf_series is None:
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

            # ── EXIT ──
            if in_trade:
                bars_since_entry += 1
                bar_high = high[i]
                bar_low = low[i]
                current_price = close[i]
                exit_price = None
                exit_reason = None

                if current_trade.direction == "long":
                    # Stop loss
                    if rule.stop_loss_pct > 0:
                        stop_px = current_trade.entry_price * (1 - rule.stop_loss_pct)
                        if bar_low <= stop_px:
                            exit_price = stop_px
                            exit_reason = "stop_loss"
                    # Profit target
                    if rule.profit_target_pct > 0 and exit_reason is None:
                        tgt_px = current_trade.entry_price * (1 + rule.profit_target_pct)
                        if bar_high >= tgt_px:
                            exit_price = tgt_px
                            exit_reason = "profit_target"
                else:  # short
                    # Stop loss (price goes UP)
                    if rule.stop_loss_pct > 0:
                        stop_px = current_trade.entry_price * (1 + rule.stop_loss_pct)
                        if bar_high >= stop_px:
                            exit_price = stop_px
                            exit_reason = "stop_loss"
                    # Profit target (price goes DOWN)
                    if rule.profit_target_pct > 0 and exit_reason is None:
                        tgt_px = current_trade.entry_price * (1 - rule.profit_target_pct)
                        if bar_low <= tgt_px:
                            exit_price = tgt_px
                            exit_reason = "profit_target"

                # Time exit
                if bars_since_entry >= rule.max_hold_bars and exit_reason is None:
                    exit_price = current_price
                    exit_reason = "time_exit"

                # EOD exit
                if exit_reason is None:
                    if i + 1 < len(intraday) and dates[i + 1] != bar_date:
                        exit_price = current_price
                        exit_reason = "eod_exit"
                    elif i + 1 >= len(intraday):
                        exit_price = current_price
                        exit_reason = "eod_exit"

                if exit_price is not None:
                    # Slippage direction depends on trade direction
                    if current_trade.direction == "long":
                        exit_adj = exit_price * (1 - SLIPPAGE_PER_SIDE)
                        pnl_per_share = exit_adj - current_trade.entry_price
                    else:
                        exit_adj = exit_price * (1 + SLIPPAGE_PER_SIDE)
                        pnl_per_share = current_trade.entry_price - exit_adj

                    pnl = pnl_per_share * current_trade.shares
                    commission = COMMISSION_PER_SHARE * current_trade.shares * 2
                    pnl -= commission

                    current_trade.exit_time = bar_time
                    current_trade.exit_price = exit_adj
                    current_trade.pnl = pnl
                    current_trade.pnl_pct = pnl / (current_trade.entry_price * current_trade.shares) if current_trade.shares > 0 else 0
                    current_trade.bars_held = bars_since_entry
                    current_trade.exit_reason = exit_reason

                    capital += pnl
                    rule_trades.append(current_trade)
                    in_trade = False

            # ── ENTRY ──
            if not in_trade and (mask.iloc[i] if isinstance(mask, pd.Series) else mask[i]):
                # HTF check
                bar_date_ts = pd.Timestamp(bar_date)
                if bar_date_ts in htf_series.index:
                    htf_val = htf_series.loc[bar_date_ts]
                else:
                    prior = htf_series.index[htf_series.index <= bar_date_ts]
                    htf_val = htf_series.loc[prior[-1]] if len(prior) > 0 else 0

                if htf_val != 1:
                    continue

                # Only first per session
                if rule.only_first_per_session and bar_date in traded_dates:
                    continue

                # ── INTRADAY FILTERS ──
                if USE_FILTERS and rule.intraday_filters:
                    filters_pass = True
                    for feat_col, required_val in rule.intraday_filters:
                        if feat_col in features.columns:
                            feat_val = features[feat_col].iloc[i]
                            if pd.isna(feat_val) or int(feat_val) != required_val:
                                filters_pass = False
                                break
                        else:
                            filters_pass = False
                            break
                    if not filters_pass:
                        continue

                # Position sizing
                entry_price = close[i] * (1 + SLIPPAGE_PER_SIDE) if rule.direction == "long" else close[i] * (1 - SLIPPAGE_PER_SIDE)
                if rule.stop_loss_pct > 0:
                    risk_amount = capital * RISK_PER_TRADE
                    stop_distance = entry_price * rule.stop_loss_pct
                    shares = int(risk_amount / stop_distance)
                else:
                    shares = int((capital * POSITION_FRACTION) / entry_price)
                if shares <= 0:
                    continue

                position_value = shares * entry_price
                if position_value > capital * 0.95:
                    shares = int(capital * 0.95 / entry_price)
                    if shares <= 0:
                        continue

                current_trade = Trade(
                    rule_name=rule.name,
                    ticker=rule.ticker,
                    direction=rule.direction,
                    entry_time=bar_time,
                    entry_price=entry_price,
                    shares=shares,
                )
                in_trade = True
                bars_since_entry = 0
                traded_dates.add(bar_date)

        all_trades.extend(rule_trades)

        if rule_trades:
            pnls = [t.pnl for t in rule_trades]
            wins = sum(1 for p in pnls if p > 0)
            exits = {}
            for t in rule_trades:
                exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
            print(f"  Trades: {len(rule_trades)}, WR: {wins/len(rule_trades)*100:.1f}%, "
                  f"PnL: ${sum(pnls):,.0f}, Avg: ${np.mean(pnls):.2f}")
            print(f"  Exits: {exits}")
        else:
            print("  No trades")

    if not all_trades:
        print("\nNo trades.")
        return

    # ── Portfolio Summary ──
    all_trades.sort(key=lambda t: t.entry_time)

    equity = [INITIAL_CAPITAL]
    equity_dates = [all_trades[0].entry_time]
    running = INITIAL_CAPITAL
    for t in all_trades:
        running += t.pnl
        equity.append(running)
        equity_dates.append(t.exit_time or t.entry_time)

    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak

    total_pnl = equity[-1] - INITIAL_CAPITAL
    total_ret = total_pnl / INITIAL_CAPITAL
    n_trades = len(all_trades)
    win_rate = sum(1 for t in all_trades if t.pnl > 0) / n_trades
    max_dd = abs(dd.min())

    daily_pnl = {}
    for t in all_trades:
        d = t.entry_time.date()
        daily_pnl[d] = daily_pnl.get(d, 0) + t.pnl
    daily_vals = list(daily_pnl.values())
    sharpe = np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252) if np.std(daily_vals) > 0 else 0

    # Long vs Short breakdown
    long_trades = [t for t in all_trades if t.direction == "long"]
    short_trades = [t for t in all_trades if t.direction == "short"]
    long_pnl = sum(t.pnl for t in long_trades)
    short_pnl = sum(t.pnl for t in short_trades)

    print(f"\n\n{'='*80}")
    print(f"PORTFOLIO SUMMARY — {mode}")
    print(f"{'='*80}")
    print(f"  Final Capital:  ${equity[-1]:>12,.0f}")
    print(f"  Total P&L:      ${total_pnl:>12,.0f}  ({total_ret*100:.2f}%)")
    print(f"  Total Trades:   {n_trades:>8}")
    print(f"  Win Rate:       {win_rate*100:>8.1f}%")
    print(f"  Max Drawdown:   {max_dd*100:>8.2f}%")
    print(f"  Sharpe Ratio:   {sharpe:>8.2f}")
    print(f"\n  LONG:  {len(long_trades)} trades, PnL ${long_pnl:>+10,.0f}")
    print(f"  SHORT: {len(short_trades)} trades, PnL ${short_pnl:>+10,.0f}")

    # Per-rule
    print(f"\n  Per-Rule Breakdown:")
    rule_stats = {}
    for t in all_trades:
        if t.rule_name not in rule_stats:
            rule_stats[t.rule_name] = {"trades": 0, "wins": 0, "pnl": 0, "dir": t.direction}
        rule_stats[t.rule_name]["trades"] += 1
        if t.pnl > 0: rule_stats[t.rule_name]["wins"] += 1
        rule_stats[t.rule_name]["pnl"] += t.pnl

    for name, s in sorted(rule_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr = s["wins"]/s["trades"]*100 if s["trades"] > 0 else 0
        dir_tag = "L" if s["dir"] == "long" else "S"
        print(f"    [{dir_tag}] {name:45s}  trades={s['trades']:>4}  WR={wr:.1f}%  PnL=${s['pnl']:>+10,.0f}")

    # ── Save trades ──
    records = [{
        "rule": t.rule_name, "ticker": t.ticker, "direction": t.direction,
        "entry_time": t.entry_time, "entry_price": t.entry_price,
        "exit_time": t.exit_time, "exit_price": t.exit_price,
        "shares": t.shares, "exit_reason": t.exit_reason,
        "pnl": t.pnl, "pnl_pct": t.pnl_pct, "bars_held": t.bars_held,
    } for t in all_trades]
    pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR, "phase5v5_trades.csv"), index=False)

    # ── Equity Curve ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    ax1 = axes[0]
    ax1.set_facecolor("#1a1a2e")
    ax1.plot(equity_dates, equity, color="#00d2ff", linewidth=1.5, label="Equity")
    ax1.fill_between(equity_dates, INITIAL_CAPITAL, equity,
                     where=(equity >= INITIAL_CAPITAL), alpha=0.15, color="#00ff88")
    ax1.fill_between(equity_dates, INITIAL_CAPITAL, equity,
                     where=(equity < INITIAL_CAPITAL), alpha=0.15, color="#ff4444")
    ax1.axhline(INITIAL_CAPITAL, color="#666", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Portfolio Value ($)", color="white")
    ax1.set_title(f"v5 Backtest ({mode}) — Sharpe {sharpe:.2f} | Return {total_ret*100:.1f}% | "
                  f"MaxDD {max_dd*100:.1f}% | Long ${long_pnl:+,.0f} Short ${short_pnl:+,.0f}",
                  fontsize=12, fontweight="bold", color="white")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    ax1.tick_params(colors="white")
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    ax2.fill_between(equity_dates, dd * 100, 0, alpha=0.4, color="#ff4444")
    ax2.set_ylabel("Drawdown (%)", color="white")
    ax2.set_xlabel("Date", color="white")
    ax2.tick_params(colors="white")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "phase5v5_equity_curve.png"),
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)

    # ── Monthly Returns ──
    monthly_pnl = {}
    for t in all_trades:
        key = (t.entry_time.year, t.entry_time.month)
        monthly_pnl[key] = monthly_pnl.get(key, 0) + t.pnl

    if monthly_pnl:
        years = sorted(set(k[0] for k in monthly_pnl))
        fig, ax = plt.subplots(figsize=(14, max(3, len(years))))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        data = np.full((len(years), 12), np.nan)
        for (y, m), pnl in monthly_pnl.items():
            data[years.index(y), m-1] = pnl

        vmax = np.nanmax(np.abs(data))
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], color="white")
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years, color="white")
        ax.set_title(f"Monthly P&L ($) — v5 {mode}", fontsize=14, fontweight="bold", color="white")
        for yi in range(len(years)):
            for mi in range(12):
                val = data[yi, mi]
                if not np.isnan(val):
                    clr = "black" if abs(val) < vmax*0.5 else "white"
                    ax.text(mi, yi, f"${val:,.0f}", ha="center", va="center", fontsize=7, color=clr)
        plt.colorbar(im, ax=ax, label="P&L ($)")
        fig.savefig(os.path.join(OUTPUT_DIR, "phase5v5_monthly.png"),
                    dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)

    print(f"\n  Saved → {OUTPUT_DIR}/phase5v5_*.csv/.png")
    print("=" * 80)


if __name__ == "__main__":
    run_backtest()
