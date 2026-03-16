#!/usr/bin/env python3
"""
Delta Sweep Optimization — 1-min pricing
=========================================
Tests deltas 0.30 → 0.80 (step 0.05) for QQQ Short, SPY Short, 50Hi Weak.
Uses 1-min bar CLOSEs at actual signal times (same methodology as run_reprice_1min.py).
Strike selection via Black-Scholes (same as original backtest).
All contract metadata already cached locally — only fetches 1-min bars via API.

Input:  output/options_931filter_pt50.csv  (signal list, unpriced)
Output: output/delta_sweep_results.csv
Cache:  output/delta_sweep_cache.json

Run:
    python run_delta_sweep.py
"""

import os, time, json
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.config import OUTPUT_DIR, DATA_DIR
from src.options_data import (
    estimate_realized_vol, time_to_expiry_years,
    select_strike, get_strikes_for_type,
)
from src.options_client import OptionsClient

INPUT_CSV   = os.path.join(OUTPUT_DIR, "options_931filter_pt50.csv")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "delta_sweep_results.csv")
CACHE_FILE  = os.path.join(OUTPUT_DIR, "delta_sweep_cache.json")
CACHE_DIR   = os.path.join(DATA_DIR, "options_cache")

API_KEY  = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
BASE_URL = "https://api.polygon.io"
SLIPPAGE = 0.01
SLEEP    = 0.2

# Strategies and delta ranges to test
SWEEP_CONFIG = {
    'PriorDayStrong_AboveOR_QQQ_short': {
        'label':   'QQQ Short',
        'deltas':  [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        'budget':  104000,
        'is_call': False,   # Puts
    },
    'PriorDayStrong_AboveOR_SPY_short': {
        'label':   'SPY Short',
        'deltas':  [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        'budget':  99000,
        'is_call': False,   # Puts
    },
    'PriorDayWeak_50Hi_SPY_filtered': {
        'label':   '50Hi Weak',
        'deltas':  [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        'budget':  48000,
        'is_call': True,    # Calls
    },
}

RISK_FREE_RATE = 0.05
ET = ZoneInfo("America/New_York")


def ts_to_ms(ts_str: str) -> int:
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    return int(ts.timestamp() * 1000)


def fetch_1min_close(option_ticker: str, trade_date: str,
                     signal_time_str: str, session: requests.Session) -> float:
    """1-min bar CLOSE at actual signal time."""
    url = (f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/minute"
           f"/{trade_date}/{trade_date}"
           f"?adjusted=false&sort=asc&limit=1000&apiKey={API_KEY}")
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as e:
        print(f"    [ERR] {option_ticker}: {e}")
        return float("nan")

    if not results:
        return float("nan")

    signal_ms = ts_to_ms(signal_time_str)
    valid = [r for r in results if r["t"] <= signal_ms]
    if not valid:
        valid = results[:1]
    return float(valid[-1]["c"])


def load_daily_bars() -> dict:
    """Load 1D underlying bars for IV estimation."""
    daily = {}
    d1_dir = os.path.join(DATA_DIR, "1D")
    if not os.path.exists(d1_dir):
        return daily
    for f in os.listdir(d1_dir):
        if not f.endswith(".parquet"):
            continue
        ticker = f.replace(".parquet", "").split("_")[0]
        try:
            import pyarrow  # noqa
            df = pd.read_parquet(os.path.join(d1_dir, f))
            daily[ticker] = df
        except Exception:
            pass
    return daily


def load_contracts(ticker: str, expiry: str, cp: str) -> list:
    """Load cached contract list."""
    fname = os.path.join(CACHE_DIR, f"contracts_{ticker}_{expiry}_{cp.lower()}.json")
    if not os.path.exists(fname):
        return []
    with open(fname) as f:
        return json.load(f)


def main():
    print(f"Loading {INPUT_CSV}...")
    df_all = pd.read_csv(INPUT_CSV)
    df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])

    # Load daily bars for IV estimation (best effort)
    print("Loading daily bars for IV estimation...")
    daily_bars = load_daily_bars()
    print(f"  {len(daily_bars)} tickers loaded" if daily_bars else "  (pyarrow not available — using fixed 20% IV)")

    # Load price cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached prices\n")

    # Also load 1-min reprice cache to reuse existing prices
    reprice_cache_file = os.path.join(OUTPUT_DIR, "reprice_1min_cache.json")
    reprice_cache = {}
    if os.path.exists(reprice_cache_file):
        with open(reprice_cache_file) as f:
            reprice_cache = json.load(f)
        print(f"  Loaded {len(reprice_cache)} prices from 1-min reprice cache\n")

    session = requests.Session()
    session.headers["User-Agent"] = "0DTE-DeltaSweep/1.0"

    client = OptionsClient(API_KEY)

    all_rows = []
    fetched  = 0
    errors   = 0

    for rule, cfg in SWEEP_CONFIG.items():
        df = df_all[df_all['rule'] == rule].copy()
        label   = cfg['label']
        deltas  = cfg['deltas']
        budget  = cfg['budget']
        is_call  = cfg['is_call']
        cp_char  = 'C' if is_call else 'P'
        cp_word  = 'call' if is_call else 'put'
        cp_full  = 'CALL' if is_call else 'PUT'   # for get_strikes_for_type

        print(f"\n{'='*60}")
        print(f"{label}  ({len(df)} trades, testing {len(deltas)} deltas)")
        print(f"{'='*60}")

        for target_delta in deltas:
            pnls       = []
            skipped    = 0

            for _, row in df.iterrows():
                trade_date  = row['trade_date'].date()
                entry_str   = str(row['entry_time'])
                exit_str    = str(row['exit_time'])
                entry_hm    = entry_str[11:16]
                exit_hm     = exit_str[11:16]
                date_str    = str(trade_date)
                ticker      = row['ticker']

                # Underlying price at entry (from original row)
                underlying  = row['option_entry_price']  # we'll use expiry + entry info
                # Actually use the row's underlying price embedded in option_ticker
                # Reconstruct: use entry time and estimate underlying from option data
                # Better: use the underlying_entry_price if available, else skip
                # We stored it originally — let's use the strike as a proxy anchor
                # and the existing expiry/option_type from the row
                expiry_str  = str(row['expiry_date'])[:10]
                expiry_date = date.fromisoformat(expiry_str)

                # Load available strikes from cache
                contracts   = load_contracts(ticker, expiry_str, cp_word)
                if not contracts:
                    skipped += 1
                    continue
                strikes     = get_strikes_for_type(contracts, cp_full)
                if not strikes:
                    skipped += 1
                    continue

                # Estimate underlying price from original option ticker
                # Use original entry price + original strike to back out underlying
                orig_strike = float(row['strike'])

                # IV estimation
                db = daily_bars.get(ticker, pd.DataFrame())
                iv = estimate_realized_vol(db, trade_date) if not db.empty else 0.20

                # Time to expiry
                entry_ts = pd.Timestamp(entry_str)
                if entry_ts.tzinfo is None:
                    entry_ts = entry_ts.tz_localize("America/New_York")
                tte = time_to_expiry_years(entry_ts, expiry_date)
                if tte <= 0:
                    tte = 1 / (252 * 6.5)  # minimum ~1 bar

                # We need underlying price — reconstruct from the original delta/strike
                # Use the original option ticker's strike and delta to back-solve S
                # Simpler: use orig_strike as a proxy (underlying ≈ strike / (1 ± small offset))
                # Best available: read underlying price from the original trade's option ticker
                # Parse from option_ticker: O:SPY210104P00376000 → strike 376
                # We'll approximate underlying using orig_strike and the original delta
                # For puts: S ≈ strike * exp(some offset) -- but we don't have S directly
                # Use the underlying price stored if we can find it in a 1D bar
                if not db.empty:
                    try:
                        trade_dt = pd.Timestamp(trade_date)
                        if trade_dt.tzinfo is not None:
                            trade_dt = trade_dt.tz_localize(None)
                        # Find closest date
                        idx = db.index
                        if hasattr(idx, 'tz') and idx.tz is not None:
                            idx = idx.tz_localize(None)
                        loc = idx.searchsorted(trade_dt)
                        if loc >= len(db): loc = len(db) - 1
                        underlying_price = float(db.iloc[loc]['close'])
                    except Exception:
                        underlying_price = orig_strike  # fallback
                else:
                    # Fallback: use original strike as rough proxy for underlying
                    # (will slightly misestimate delta but same strike selected most of the time)
                    underlying_price = orig_strike

                # Select strike for this delta
                new_strike, actual_delta = select_strike(
                    underlying_price=underlying_price,
                    available_strikes=strikes,
                    target_delta=target_delta,
                    T=tte,
                    r=RISK_FREE_RATE,
                    sigma=iv,
                    is_call=is_call,
                )
                if new_strike is None:
                    skipped += 1
                    continue

                # Construct option ticker
                new_ticker = client.construct_ticker(ticker, expiry_date, new_strike, cp_char)

                # Entry price from 1-min bar
                entry_key = f"{new_ticker}_{date_str}_entry_{entry_hm}"
                if entry_key in cache:
                    ep = cache[entry_key]
                elif entry_key in reprice_cache:
                    ep = reprice_cache[entry_key]
                else:
                    ep = fetch_1min_close(new_ticker, date_str, entry_str, session)
                    if np.isnan(ep) or ep <= 0:
                        ep = float("nan")
                        errors += 1
                    cache[entry_key] = ep
                    fetched += 1
                    time.sleep(SLEEP)

                if np.isnan(ep) or ep <= 0 or ep < 0.05:
                    skipped += 1
                    continue

                # Exit price from 1-min bar
                exit_key = f"{new_ticker}_{date_str}_exit_{exit_hm}"
                if exit_key in cache:
                    xp = cache[exit_key]
                elif exit_key in reprice_cache:
                    xp = reprice_cache[exit_key]
                else:
                    xp = fetch_1min_close(new_ticker, date_str, exit_str, session)
                    if np.isnan(xp) or xp <= 0:
                        xp = 0.01
                        errors += 1
                    cache[exit_key] = xp
                    fetched += 1
                    time.sleep(SLEEP)

                # Save cache periodically
                if fetched > 0 and fetched % 100 == 0:
                    with open(CACHE_FILE, "w") as f:
                        json.dump(cache, f)
                    print(f"  [{label} d={target_delta}] {fetched} fetched, {errors} errors")

                # P&L
                contracts_n = max(1, int(budget / (ep * 100)))
                premium     = contracts_n * ep * 100
                entry_adj   = ep * (1 + SLIPPAGE)
                exit_adj    = max(xp * (1 - SLIPPAGE), 0)
                pnl         = max((exit_adj - entry_adj) * contracts_n * 100, -premium)
                pnl_pct     = pnl / premium if premium > 0 else 0

                pnls.append({
                    'rule':        rule,
                    'label':       label,
                    'trade_date':  date_str,
                    'target_delta':target_delta,
                    'actual_delta':round(actual_delta, 3) if actual_delta else target_delta,
                    'strike':      new_strike,
                    'option_ticker': new_ticker,
                    'entry_price': ep,
                    'exit_price':  xp,
                    'contracts':   contracts_n,
                    'premium':     round(premium, 2),
                    'pnl':         round(pnl, 2),
                    'pnl_pct':     round(pnl_pct, 4),
                    'win':         1 if pnl > 0 else 0,
                })

            if pnls:
                all_rows.extend(pnls)
                total  = sum(r['pnl'] for r in pnls)
                wr     = sum(r['win'] for r in pnls) / len(pnls) * 100
                n_ok   = len(pnls)
                print(f"  delta={target_delta:.2f}  trades={n_ok:>3} (skip={skipped})  "
                      f"WR={wr:>5.1f}%  P&L=${total:>12,.0f}  avg=${total/n_ok:>8,.0f}")

    # Final cache save
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    if not all_rows:
        print("\nNo results generated.")
        return

    results = pd.DataFrame(all_rows)
    results.to_csv(OUTPUT_CSV, index=False)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("DELTA SWEEP SUMMARY")
    print(f"{'='*75}")

    for rule, cfg in SWEEP_CONFIG.items():
        label = cfg['label']
        sub   = results[results['label'] == label]
        if sub.empty:
            continue
        print(f"\n{label}  (budget=${cfg['budget']:,})")
        print(f"  {'Delta':>7}  {'Trades':>7}  {'WR%':>7}  {'Total P&L':>13}  {'Avg/Trade':>10}  {'Sharpe':>7}")
        print(f"  {'-'*62}")

        all_dates = pd.bdate_range(results['trade_date'].min(), results['trade_date'].max())

        # Find best delta by total P&L for star marker
        best_pnl = max(
            (sub[sub['target_delta']==td]['pnl'].sum()
             for td in cfg['deltas']
             if not sub[sub['target_delta']==td].empty),
            default=float('-inf')
        )

        for td in cfg['deltas']:
            g = sub[sub['target_delta'] == td]
            if g.empty:
                continue
            total = g['pnl'].sum()
            wr    = g['win'].mean() * 100
            avg   = g['pnl'].mean()
            daily = g.groupby('trade_date')['pnl'].sum()
            daily_full = daily.reindex(all_dates, fill_value=0)
            mu, sig = daily_full.mean(), daily_full.std()
            sharpe = (mu / sig * np.sqrt(252)) if sig > 0 else 0
            star = '  ◄ best' if abs(total - best_pnl) < 1 else ''
            print(f"  {td:>7.2f}  {len(g):>7}  {wr:>6.1f}%  ${total:>12,.0f}  ${avg:>9,.0f}  {sharpe:>7.2f}{star}")

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Total API calls: {fetched}  |  errors: {errors}")


if __name__ == "__main__":
    main()
