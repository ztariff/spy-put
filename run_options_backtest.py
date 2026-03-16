#!/usr/bin/env python3
"""
Options-Based Backtest for v5 Strategies
=========================================
Replaces stock positions with 0DTE options (calls for longs, puts for shorts)
using real Polygon.io options data. Tests multiple delta targets per strategy.

Usage:
    python run_options_backtest.py                  # Full run, all deltas
    python run_options_backtest.py --delta 0.50     # Single delta
    python run_options_backtest.py --smoke          # Smoke test (5 trades)
    python run_options_backtest.py --rule PriorDayWeak_30min_QQQ  # Single rule
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

from src.config import OUTPUT_DIR, DATA_DIR
from src.options_client import OptionsClient
from src.options_data import (
    select_expiry, estimate_realized_vol, time_to_expiry_years,
    select_strike, get_strikes_for_type,
)
from src.black_scholes import call_price, put_price, call_delta, put_delta


# ── Configuration ────────────────────────────────────────────────────────────

PREMIUM_BUDGET = 10_000       # $ allocated per trade (this IS the max risk)
SLIPPAGE_PER_SIDE = 0.02      # $ per share (options), ~2 cents
RISK_FREE_RATE = 0.045        # Annualized
DELTA_TARGETS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
ENTRY_DELAY_MINUTES = 1       # Wait 1 min after signal to enter (avoid opening auction noise)

# Optimal delta per rule (from full backtest ROI analysis)
OPTIMAL_DELTAS = {
    "GapLarge_First30min_SPY":            0.50,   # ATM
    "HighVolWR_30min_SPY_filtered":       0.10,   # Deep OTM — high WR leveraged
    "PriorDayStrong_AboveOR_QQQ_short":   0.70,   # ITM puts
    "PriorDayStrong_AboveOR_SPY_short":   0.70,   # ITM puts
    "PriorDayWeak_30min_QQQ":             0.50,   # ATM
    "PriorDayWeak_30min_SPY_filtered":    0.50,   # ATM
    "PriorDayWeak_50Hi_SPY_filtered":     0.50,   # ATM
}

# Premium budget per delta — scale up with delta for more directional exposure
PREMIUM_BY_DELTA = {
    0.10: 10_000,
    0.20: 20_000,
    0.30: 30_000,
    0.40: 40_000,
    0.50: 50_000,
    0.60: 60_000,
    0.70: 75_000,
    0.80: 80_000,
    0.90: 90_000,
}


@dataclass
class OptionsTrade:
    """One completed options trade."""
    rule: str
    ticker: str
    direction: str
    entry_time: str
    exit_time: str

    # Underlying
    underlying_entry_price: float
    underlying_exit_price: float
    equity_pnl_pct: float         # Original stock P&L % for comparison

    # Option details
    option_ticker: str
    strike: float
    expiry_date: str
    option_type: str              # CALL or PUT
    target_delta: float
    actual_delta: float

    # Pricing
    option_entry_price: float     # Premium per share at entry
    option_exit_price: float      # Premium per share at exit
    num_contracts: int
    premium_paid: float           # Total premium = entry_price * contracts * 100

    # P&L
    pnl: float                    # Dollar P&L on option position
    pnl_pct: float                # Return on premium
    max_loss: float               # = premium_paid (capped risk)

    # Context
    iv_estimate: float
    tte_at_entry: float           # Time to expiry in years at entry
    bars_held: int
    exit_reason: str
    status: str                   # 'ok', 'no_data', 'no_contracts', etc.


# ── Daily Data Loader ────────────────────────────────────────────────────────

def load_daily_bars(ticker: str) -> pd.DataFrame:
    """Load cached daily bars for IV estimation."""
    path = os.path.join(DATA_DIR, "1D", f"{ticker}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    path2 = os.path.join(DATA_DIR, f"{ticker}_1D.parquet")
    if os.path.exists(path2):
        return pd.read_parquet(path2)
    print(f"  Warning: No daily bars found for {ticker}")
    return pd.DataFrame()


# ── Core Backtest Logic ──────────────────────────────────────────────────────

def simulate_option_trade(
    trade_row: pd.Series,
    target_delta: float,
    client: OptionsClient,
    daily_bars: dict,
) -> OptionsTrade:
    """
    Simulate replacing a single equity trade with an options trade.

    Parameters
    ----------
    trade_row : Series from phase5v5_trades.csv
    target_delta : float (0.30, 0.50, etc.)
    client : OptionsClient
    daily_bars : dict of {ticker: DataFrame} for IV estimation

    Returns
    -------
    OptionsTrade
    """
    rule = trade_row["rule"]
    ticker = trade_row["ticker"]
    direction = trade_row["direction"]
    raw_entry_time = pd.to_datetime(trade_row["entry_time"], utc=True).tz_convert("America/New_York")
    exit_time = pd.to_datetime(trade_row["exit_time"], utc=True).tz_convert("America/New_York")
    # Delay entry by 1 minute to avoid opening auction noise on options
    entry_time = raw_entry_time + pd.Timedelta(minutes=ENTRY_DELAY_MINUTES)
    entry_price = float(trade_row["entry_price"])  # Underlying price at original signal
    exit_price = float(trade_row["exit_price"])
    equity_pnl_pct = float(trade_row["pnl_pct"])
    bars_held = int(trade_row["bars_held"])
    exit_reason = trade_row["exit_reason"]

    trade_date = entry_time.date()
    is_call = (direction == "long")
    option_type = "CALL" if is_call else "PUT"
    cp_char = "C" if is_call else "P"

    # Base result for error cases
    def error_trade(status):
        return OptionsTrade(
            rule=rule, ticker=ticker, direction=direction,
            entry_time=str(entry_time), exit_time=str(exit_time),
            underlying_entry_price=entry_price, underlying_exit_price=exit_price,
            equity_pnl_pct=equity_pnl_pct,
            option_ticker="", strike=0, expiry_date="", option_type=option_type,
            target_delta=target_delta, actual_delta=0,
            option_entry_price=0, option_exit_price=0, num_contracts=0, premium_paid=0,
            pnl=0, pnl_pct=0, max_loss=0,
            iv_estimate=0, tte_at_entry=0, bars_held=bars_held,
            exit_reason=exit_reason, status=status,
        )

    # 1. Find available expiries
    available_expiries = client.get_available_expiries(ticker, trade_date)
    if not available_expiries:
        return error_trade("no_expiries")

    expiry = select_expiry(available_expiries, trade_date)
    if expiry is None:
        return error_trade("no_expiry_selected")

    # 2. Get contracts for this expiry
    contracts = client.get_option_contracts(ticker, expiry, contract_type=option_type.lower())
    if not contracts:
        # Try without type filter
        contracts = client.get_option_contracts(ticker, expiry)
        contracts = [c for c in contracts if c["type"].upper() == option_type]

    if not contracts:
        return error_trade("no_contracts")

    strikes = get_strikes_for_type(contracts, option_type)
    if not strikes:
        return error_trade("no_strikes")

    # 3. Estimate IV from recent realized vol
    db = daily_bars.get(ticker, pd.DataFrame())
    iv = estimate_realized_vol(db, trade_date) if not db.empty else 0.20

    # 4. Compute time to expiry
    tte = time_to_expiry_years(entry_time, expiry)

    # 5. Select strike by delta
    strike, actual_delta = select_strike(
        underlying_price=entry_price,
        available_strikes=strikes,
        target_delta=target_delta,
        T=tte,
        r=RISK_FREE_RATE,
        sigma=iv,
        is_call=is_call,
    )
    if strike is None:
        return error_trade("no_strike_match")

    # 6. Construct option ticker and fetch bars
    option_ticker = client.construct_ticker(ticker, expiry, strike, cp_char)

    # Get option price at entry
    opt_entry_price = client.get_option_price_at_time(option_ticker, entry_time, trade_date)
    if np.isnan(opt_entry_price) or opt_entry_price <= 0:
        return error_trade("no_entry_price")

    # Get option price at exit
    opt_exit_price = client.get_option_price_at_time(option_ticker, exit_time, trade_date)
    if np.isnan(opt_exit_price):
        # If exit bar not found, use theoretical price
        tte_exit = time_to_expiry_years(exit_time, expiry)
        if is_call:
            opt_exit_price = call_price(exit_price, strike, tte_exit, RISK_FREE_RATE, iv)
        else:
            opt_exit_price = put_price(exit_price, strike, tte_exit, RISK_FREE_RATE, iv)
        if opt_exit_price <= 0:
            opt_exit_price = 0.01

    # 7. Skip penny options — too illiquid and spread-dominated
    if opt_entry_price < 0.05:
        return error_trade("penny_option")

    # 8. Position sizing: delta-scaled premium budget, cap at 500 contracts
    budget = PREMIUM_BY_DELTA.get(target_delta, PREMIUM_BUDGET)
    cost_per_contract = opt_entry_price * 100  # Each contract = 100 shares
    if cost_per_contract <= 0:
        return error_trade("zero_premium")

    num_contracts = max(1, min(500, int(budget / cost_per_contract)))
    premium_paid = num_contracts * cost_per_contract

    # 9. P&L calculation (percentage-based slippage: ~1% of option price per side)
    slippage_pct = 0.01  # 1% of option price per side
    entry_adj = opt_entry_price * (1 + slippage_pct)
    exit_adj = opt_exit_price * (1 - slippage_pct)
    exit_adj = max(exit_adj, 0)
    pnl_per_share = exit_adj - entry_adj
    pnl = pnl_per_share * num_contracts * 100
    # Cap loss at premium paid (long options can't lose more than premium)
    pnl = max(pnl, -premium_paid)
    pnl_pct = pnl / premium_paid if premium_paid > 0 else 0

    return OptionsTrade(
        rule=rule, ticker=ticker, direction=direction,
        entry_time=str(entry_time), exit_time=str(exit_time),
        underlying_entry_price=entry_price, underlying_exit_price=exit_price,
        equity_pnl_pct=equity_pnl_pct,
        option_ticker=option_ticker, strike=strike,
        expiry_date=str(expiry), option_type=option_type,
        target_delta=target_delta, actual_delta=actual_delta,
        option_entry_price=opt_entry_price, option_exit_price=opt_exit_price,
        num_contracts=num_contracts, premium_paid=premium_paid,
        pnl=pnl, pnl_pct=pnl_pct, max_loss=premium_paid,
        iv_estimate=iv, tte_at_entry=tte, bars_held=bars_held,
        exit_reason=exit_reason, status="ok",
    )


# ── Main Runner ──────────────────────────────────────────────────────────────

def run_options_backtest(
    equity_trades: pd.DataFrame,
    delta_targets: list = None,
    smoke_test: bool = False,
    rule_filter: str = None,
    optimal_mode: bool = False,
) -> pd.DataFrame:
    """
    Run the full options backtest.

    Parameters
    ----------
    equity_trades : DataFrame of v5 equity trades
    delta_targets : list of floats (delta values to test)
    smoke_test : bool - if True, only process 5 trades
    rule_filter : str - if set, only process trades for this rule
    optimal_mode : bool - if True, use per-rule optimal deltas from OPTIMAL_DELTAS

    Returns
    -------
    DataFrame of all OptionsTrade results
    """
    if delta_targets is None:
        delta_targets = DELTA_TARGETS

    client = OptionsClient()

    # Load daily bars for IV estimation
    print("Loading daily bars for IV estimation...")
    daily_bars = {}
    for ticker in equity_trades["ticker"].unique():
        db = load_daily_bars(ticker)
        if not db.empty:
            daily_bars[ticker] = db
            print(f"  {ticker}: {len(db)} daily bars loaded")

    # Filter trades
    trades_to_process = equity_trades.copy()
    if rule_filter:
        trades_to_process = trades_to_process[trades_to_process["rule"] == rule_filter]
        print(f"Filtered to rule: {rule_filter} ({len(trades_to_process)} trades)")
    if smoke_test:
        trades_to_process = trades_to_process.head(5)
        print(f"Smoke test: processing {len(trades_to_process)} trades")

    # In optimal mode, each trade uses its rule's best delta
    if optimal_mode:
        print(f"\n*** OPTIMAL MODE: Per-rule delta assignment ***")
        for rule, delta in sorted(OPTIMAL_DELTAS.items()):
            cnt = len(trades_to_process[trades_to_process["rule"] == rule])
            print(f"  {rule:<45} → Δ {delta:.2f}  ({cnt} trades)")
        total = len(trades_to_process)
        print(f"\nProcessing {total} trades (1 optimal delta each)")
    else:
        total = len(trades_to_process) * len(delta_targets)
        print(f"\nProcessing {len(trades_to_process)} trades × {len(delta_targets)} deltas = {total} option trades")
        print(f"Delta targets: {delta_targets}")

    print(f"Premium budget: ${PREMIUM_BUDGET:,}")
    print()

    all_results = []
    completed = 0
    errors_by_status = {}

    if optimal_mode:
        # One pass: each trade gets its rule's optimal delta
        print(f"── Optimal Delta Run ─────────────────────────────────")
        for idx, (_, trade_row) in enumerate(trades_to_process.iterrows()):
            rule = trade_row["rule"]
            delta = OPTIMAL_DELTAS.get(rule, 0.50)  # fallback ATM
            result = simulate_option_trade(trade_row, delta, client, daily_bars)
            all_results.append(asdict(result))

            completed += 1
            if result.status != "ok":
                errors_by_status[result.status] = errors_by_status.get(result.status, 0) + 1

            if completed % 50 == 0 or completed == total:
                ok_count = sum(1 for r in all_results if r["status"] == "ok")
                print(f"  [{completed}/{total}] {ok_count} ok, errors: {errors_by_status}")
    else:
        for delta in delta_targets:
            print(f"── Delta {delta:.2f} ─────────────────────────────────")
            for idx, (_, trade_row) in enumerate(trades_to_process.iterrows()):
                result = simulate_option_trade(trade_row, delta, client, daily_bars)
                all_results.append(asdict(result))

                completed += 1
                if result.status != "ok":
                    errors_by_status[result.status] = errors_by_status.get(result.status, 0) + 1

                if completed % 50 == 0 or completed == total:
                    ok_count = sum(1 for r in all_results if r["status"] == "ok")
                    print(f"  [{completed}/{total}] {ok_count} ok, errors: {errors_by_status}")

    results_df = pd.DataFrame(all_results)

    # Summary
    ok = results_df[results_df["status"] == "ok"]
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total attempted: {len(results_df)}")
    print(f"Successful: {len(ok)} ({len(ok)/len(results_df)*100:.1f}%)")
    print(f"Errors: {dict(errors_by_status)}")

    if len(ok) > 0:
        if optimal_mode:
            # Group by rule for optimal mode
            total_pnl = ok["pnl"].sum()
            total_premium = ok["premium_paid"].sum()
            roi = total_pnl / total_premium * 100 if total_premium > 0 else 0
            wr = (ok["pnl"] > 0).mean()
            print(f"\n  OPTIMAL PORTFOLIO: {len(ok)} trades")
            print(f"    Total P&L:   ${total_pnl:>+12,.0f}")
            print(f"    Avg P&L:     ${ok['pnl'].mean():>+8,.0f}")
            print(f"    Win Rate:    {wr:.1%}")
            print(f"    Total Prem:  ${total_premium:>12,.0f}")
            print(f"    ROI:         {roi:>+.1f}%")
            print(f"\n  Per Rule:")
            for rule in sorted(ok["rule"].unique()):
                r_ok = ok[ok["rule"] == rule]
                r_pnl = r_ok["pnl"].sum()
                r_prem = r_ok["premium_paid"].sum()
                r_roi = r_pnl / r_prem * 100 if r_prem > 0 else 0
                r_wr = (r_ok["pnl"] > 0).mean()
                delta = r_ok["target_delta"].iloc[0]
                print(f"    {rule:<45} Δ{delta:.1f}  ${r_pnl:>+10,.0f}  {r_wr:.1%} WR  {r_roi:>+.1f}% ROI")
        else:
            for delta in delta_targets:
                delta_ok = ok[ok["target_delta"] == delta]
                if len(delta_ok) == 0:
                    continue
                total_pnl = delta_ok["pnl"].sum()
                avg_pnl = delta_ok["pnl"].mean()
                wr = (delta_ok["pnl"] > 0).mean()
                total_premium = delta_ok["premium_paid"].sum()
                roi = total_pnl / total_premium * 100 if total_premium > 0 else 0
                print(f"\n  Delta {delta:.2f}: {len(delta_ok)} trades")
                print(f"    Total P&L:   ${total_pnl:>+12,.0f}")
                print(f"    Avg P&L:     ${avg_pnl:>+8,.0f}")
                print(f"    Win Rate:    {wr:.1%}")
                print(f"    Total Prem:  ${total_premium:>12,.0f}")
                print(f"    ROI:         {roi:>+.1f}%")

    return results_df


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options backtest for v5 strategies")
    parser.add_argument("--smoke", action="store_true", help="Run on 5 trades only")
    parser.add_argument("--delta", type=float, help="Single delta target")
    parser.add_argument("--rule", type=str, help="Filter to single rule")
    parser.add_argument("--optimal", action="store_true",
                        help="Use per-rule optimal deltas (production mode)")
    args = parser.parse_args()

    deltas = [args.delta] if args.delta else DELTA_TARGETS

    # Load equity trades
    trades_path = os.path.join(OUTPUT_DIR, "phase5v5_trades.csv")
    print(f"Loading equity trades from {trades_path}")
    equity_trades = pd.read_csv(trades_path)
    print(f"Loaded {len(equity_trades)} trades across {equity_trades['rule'].nunique()} rules")

    results = run_options_backtest(
        equity_trades=equity_trades,
        delta_targets=deltas,
        smoke_test=args.smoke,
        rule_filter=args.rule,
        optimal_mode=args.optimal,
    )

    # Save results
    suffix = "_optimal" if args.optimal else ""
    out_path = os.path.join(OUTPUT_DIR, f"options_backtest_trades{suffix}.csv")
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
