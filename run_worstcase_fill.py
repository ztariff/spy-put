#!/usr/bin/env python3
"""
Worst-Case Fill — Real 1-Second Polygon Data
=============================================
For each trade in the 9:31-filter CSV, fetches actual 1-second option bars
from Polygon for a 10-second window starting at the trade's ACTUAL entry time
and uses the HIGH of that window as the worst-case fill price.

Recomputes:
  - option_entry_price  (HIGH of 10-sec window)
  - num_contracts       (budget / (worst_price * 100))
  - premium_paid
  - pnl
  - pnl_pct

Input:  output/options_931filter_pt50.csv
Output: output/options_931filter_worstcase.csv

Run on your machine:
    python run_worstcase_fill.py
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import OUTPUT_DIR

INPUT_CSV   = os.path.join(OUTPUT_DIR, "options_931filter_pt50.csv")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "options_931filter_worstcase.csv")
CACHE_FILE  = os.path.join(OUTPUT_DIR, "worstcase_price_cache.json")

API_KEY  = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
BASE_URL = "https://api.polygon.io"
SLIPPAGE = 0.01   # 1% per side

# Polygon rate limit — adjust if you're on a higher tier
# Starter: 5 req/min → sleep 12s between calls
# Options Advanced: unlimited → set to 0
SLEEP_BETWEEN_CALLS = 0.2   # seconds


ET = ZoneInfo("America/New_York")


def entry_window_utc(entry_time_str: str):
    """
    Given an entry_time string (e.g. '2021-01-04 09:31:00-05:00'),
    return (from_ms, to_ms) in Unix milliseconds for a 10-second
    window starting at the ACTUAL entry time.
    """
    try:
        ts = pd.Timestamp(entry_time_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("America/New_York")
        else:
            ts = ts.tz_convert("America/New_York")
    except Exception:
        return None, None

    from_ms = int(ts.timestamp() * 1000)
    to_ms   = int((ts + pd.Timedelta(seconds=10)).timestamp() * 1000)
    return from_ms, to_ms


def fetch_1sec_high(option_ticker: str, from_ms: int, to_ms: int,
                    session: requests.Session) -> float:
    """
    Fetch 1-second bars for option_ticker between from_ms and to_ms.
    Returns the HIGH of that window. Returns NaN if no data.
    """
    url = (f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/second"
           f"/{from_ms}/{to_ms}"
           f"?adjusted=false&sort=asc&limit=50&apiKey={API_KEY}")
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [ERR] {option_ticker}: {e}")
        return float("nan")

    results = data.get("results", [])
    if not results:
        return float("nan")

    high = max(r["h"] for r in results)
    return float(high)


def fetch_1min_high(option_ticker: str, trade_date: str,
                    entry_time_str: str,
                    session: requests.Session) -> float:
    """
    Fallback: fetch 1-minute bars and return HIGH of the bar at entry time.
    """
    url = (f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/minute"
           f"/{trade_date}/{trade_date}"
           f"?adjusted=false&sort=asc&limit=1000&apiKey={API_KEY}")
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return float("nan")

    results = data.get("results", [])
    if not results:
        return float("nan")

    # Find bar at actual entry time
    try:
        ts = pd.Timestamp(entry_time_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("America/New_York")
        else:
            ts = ts.tz_convert("America/New_York")
        target_ms = int(ts.timestamp() * 1000)
    except Exception:
        return float("nan")

    # Allow bars within ±60s of entry time
    window = [r for r in results if abs(r["t"] - target_ms) <= 60_000]
    if not window:
        # Fall back to nearest bar
        window = sorted(results, key=lambda r: abs(r["t"] - target_ms))[:1]
    if not window:
        return float("nan")

    return float(max(r["h"] for r in window))


def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    n = len(df)
    print(f"  {n} trades")

    # Show entry time breakdown by strategy
    name_map = {
        'GapLarge_First30min_SPY': 'GapLarge',
        'HighVolWR_30min_SPY_filtered': 'HighVolWR',
        'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
        'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
        'PriorDayWeak_30min_QQQ': 'QQQ Weak',
        'PriorDayWeak_30min_SPY_filtered': 'SPY Weak',
        'PriorDayWeak_50Hi_SPY_filtered': '50Hi Weak',
    }
    df['_strat'] = df['rule'].map(name_map)
    # Extract HH:MM from entry_time
    df['_entry_hhmm'] = df['entry_time'].astype(str).str[11:16]
    print("\nEntry time distribution by strategy:")
    for s in ['GapLarge','HighVolWR','QQQ Short','SPY Short','QQQ Weak','SPY Weak','50Hi Weak']:
        g = df[df['_strat']==s]
        if len(g)==0: continue
        times = g['_entry_hhmm'].value_counts().head(3).to_dict()
        print(f"  {s:<14}: {times}")
    print()

    # Load cache (avoid re-fetching already-done tickers)
    # Cache key includes entry_hhmm so different entry times for same ticker/date don't collide
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached prices")

    session = requests.Session()
    session.headers.update({"User-Agent": "0DTE-Backtest/1.0"})

    worst_prices = []
    fetched = 0
    fallbacks = 0
    errors = 0

    print(f"Fetching 1-second Polygon data for {n} trades...")
    print(f"(Sleep between calls: {SLEEP_BETWEEN_CALLS}s)\n")

    for i, row in df.iterrows():
        ticker       = row["option_ticker"]
        date         = str(row["trade_date"])[:10]
        entry_hhmm   = str(row["entry_time"])[11:16]
        # Include entry time in cache key so same ticker on same day but different times don't collide
        cache_key    = f"{ticker}_{date}_{entry_hhmm}"

        # Use cache if available
        if cache_key in cache:
            # Enforce floor: worst-case must be >= original entry price
            worst_prices.append(max(cache[cache_key], row["option_entry_price"]))
            continue

        from_ms, to_ms = entry_window_utc(row["entry_time"])
        if from_ms is None:
            worst_prices.append(row["option_entry_price"])
            errors += 1
            continue

        # Try 1-second bars first
        high = fetch_1sec_high(ticker, from_ms, to_ms, session)

        if np.isnan(high) or high <= 0:
            # Fall back to 1-minute bar at actual entry time
            high = fetch_1min_high(ticker, date, row["entry_time"], session)
            fallbacks += 1

        if np.isnan(high) or high <= 0:
            # Last resort: use original entry price
            high = row["option_entry_price"]
            errors += 1

        # Worst-case must be >= original entry (can't get better than original fill)
        high = max(high, row["option_entry_price"])
        cache[cache_key] = high
        worst_prices.append(high)
        fetched += 1

        # Save cache periodically
        if fetched % 50 == 0:
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)
            pct = (i + 1) / n * 100
            print(f"  [{pct:5.1f}%] {i+1}/{n} done — "
                  f"{fetched} fetched, {fallbacks} 1min fallbacks, {errors} errors")

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Final cache save
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    df["worst_entry_price"]    = worst_prices
    df["original_entry_price"] = df["option_entry_price"]

    # ── Recompute P&L with worst-case entry ───────────────────────────────
    df["wc_contracts"] = (df["budget"] / (df["worst_entry_price"] * 100)).astype(int).clip(lower=1)
    df["wc_premium"]   = df["wc_contracts"] * df["worst_entry_price"] * 100
    df["wc_exit_adj"]  = (df["option_exit_price"] * (1 - SLIPPAGE)).clip(lower=0)
    df["wc_pnl"]       = (df["wc_exit_adj"] - df["worst_entry_price"] * (1 + SLIPPAGE)) * df["wc_contracts"] * 100
    df["wc_pnl"]       = df.apply(lambda r: max(r["wc_pnl"], -r["wc_premium"]), axis=1)
    df["wc_pnl_pct"]   = df["wc_pnl"] / df["wc_premium"].replace(0, np.nan)

    out = df.copy()
    out["option_entry_price"] = out["worst_entry_price"]
    out["num_contracts"]      = out["wc_contracts"]
    out["premium_paid"]       = out["wc_premium"]
    out["pnl"]                = out["wc_pnl"]
    out["pnl_pct"]            = out["wc_pnl_pct"]

    drop_cols = ["worst_entry_price", "original_entry_price",
                 "wc_contracts", "wc_premium", "wc_exit_adj", "wc_pnl", "wc_pnl_pct",
                 "_strat", "_entry_hhmm"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    out.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    orig_wr  = (df["pnl"]  > 0).mean() * 100
    wc_wr    = (out["pnl"] > 0).mean() * 100
    orig_tot = df["pnl"].sum()
    wc_tot   = out["pnl"].sum()

    ratio = df["worst_entry_price"] / df["original_entry_price"]
    slip_avg = (ratio - 1).mean() * 100
    slip_p50 = (ratio - 1).median() * 100
    slip_p90 = (ratio - 1).quantile(0.9) * 100

    out['strat'] = out['rule'].map(name_map)
    df['strat']  = df['rule'].map(name_map)

    print(f"\n{'='*65}")
    print(f"WORST-CASE FILL RESULTS  (10-sec window, actual Polygon 1-sec data)")
    print(f"{'='*65}")
    print(f"  1-sec data fetched:       {fetched}")
    print(f"  1-min fallbacks:          {fallbacks}")
    print(f"  Original entry fallbacks: {errors}")
    print(f"  Avg worst vs original:    +{slip_avg:.2f}%  |  median +{slip_p50:.2f}%  |  p90 +{slip_p90:.2f}%")
    print(f"\n  {'':14} {'Orig WR':>8} {'WC WR':>8} {'Orig P&L':>13} {'WC P&L':>13} {'Slip%':>7}")
    print(f"  {'-'*67}")
    for s in ['GapLarge','HighVolWR','QQQ Short','SPY Short','QQQ Weak','SPY Weak','50Hi Weak']:
        go = df[df['strat']==s];  gw = out[out['strat']==s]
        if len(go)==0: continue
        s_ratio = go["worst_entry_price"] / go["original_entry_price"]
        print(f"  {s:<14} {(go['pnl']>0).mean()*100:>7.1f}% {(gw['pnl']>0).mean()*100:>7.1f}% "
              f"${go['pnl'].sum():>12,.0f} ${gw['pnl'].sum():>12,.0f} "
              f"{(s_ratio.mean()-1)*100:>+6.2f}%")
    print(f"  {'-'*67}")
    print(f"  {'TOTAL':<14} {orig_wr:>7.1f}% {wc_wr:>7.1f}% ${orig_tot:>12,.0f} ${wc_tot:>12,.0f} {slip_avg:>+6.2f}%")
    print(f"\n  Slippage cost: ${wc_tot - orig_tot:,.0f}")
    print(f"  MaxDD (worst-case):  see calendar")
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
