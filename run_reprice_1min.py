#!/usr/bin/env python3
"""
Reprice all trades using 1-minute Polygon bars
===============================================
The original backtest priced entries/exits off 5-minute bar CLOSEs, meaning
a 9:31 signal got the bar close at ~9:34:59. This script re-fetches 1-minute
bars and reprices each trade at the CLOSE of the 1-minute bar that contains
the actual signal time — so a 9:31 entry gets the 9:31 close, a 10:00 exit
gets the 10:00 close, etc.

Input:  output/options_931filter_pt50.csv   (914 trades, 9:31-filtered)
Output: output/options_931filter_1min.csv   (same trades, 1-min repriced)
Cache:  output/reprice_1min_cache.json      (resumable)

Run on your machine:
    python run_reprice_1min.py
"""

import os, time, json
import requests
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

from src.config import OUTPUT_DIR

INPUT_CSV   = os.path.join(OUTPUT_DIR, "options_931filter_pt50.csv")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "options_931filter_1min.csv")
CACHE_FILE  = os.path.join(OUTPUT_DIR, "reprice_1min_cache.json")

API_KEY  = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
BASE_URL = "https://api.polygon.io"
SLIPPAGE = 0.01          # 1% per side (same as original backtest)
SLEEP    = 0.2           # seconds between API calls (adjust for your plan)

ET = ZoneInfo("America/New_York")


def ts_to_ms(ts_str: str) -> int:
    """Convert entry/exit time string to Unix milliseconds."""
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    return int(ts.timestamp() * 1000)


def fetch_1min_close(option_ticker: str, trade_date: str,
                     signal_time_str: str, session: requests.Session) -> float:
    """
    Fetch 1-minute bars for option_ticker on trade_date.
    Returns the CLOSE of the 1-minute bar that contains signal_time.

    Bar timestamp from Polygon = start of bar (e.g. 9:31:00 bar closes at 9:31:59).
    We find the last bar whose start time <= signal_time.
    """
    url = (f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/minute"
           f"/{trade_date}/{trade_date}"
           f"?adjusted=false&sort=asc&limit=1000&apiKey={API_KEY}")
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [ERR] {option_ticker} {trade_date}: {e}")
        return float("nan")

    results = data.get("results", [])
    if not results:
        return float("nan")

    # Convert signal time to ms
    signal_ms = ts_to_ms(signal_time_str)

    # Find last bar whose start <= signal time
    valid = [r for r in results if r["t"] <= signal_ms]
    if not valid:
        # Signal is before first bar — use first bar
        valid = results[:1]

    bar = valid[-1]
    return float(bar["c"])


def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    n  = len(df)
    print(f"  {n} trades\n")

    # Load cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached prices\n")

    session = requests.Session()
    session.headers["User-Agent"] = "0DTE-Reprice/1.0"

    new_entry_prices = []
    new_exit_prices  = []
    fetched = 0
    errors  = 0

    print(f"Fetching 1-min bars for {n} trades (entry + exit each)...")
    print(f"Sleep between calls: {SLEEP}s\n")

    for i, row in df.iterrows():
        ticker    = row["option_ticker"]
        date      = str(row["trade_date"])[:10]
        entry_str = str(row["entry_time"])
        exit_str  = str(row["exit_time"])
        entry_hm  = entry_str[11:16]
        exit_hm   = exit_str[11:16]

        entry_key = f"{ticker}_{date}_entry_{entry_hm}"
        exit_key  = f"{ticker}_{date}_exit_{exit_hm}"

        # ── Entry price ───────────────────────────────────────────────
        if entry_key in cache:
            ep = cache[entry_key]
        else:
            ep = fetch_1min_close(ticker, date, entry_str, session)
            if np.isnan(ep) or ep <= 0:
                ep = row["option_entry_price"]   # fallback to original
                errors += 1
            cache[entry_key] = ep
            fetched += 1
            time.sleep(SLEEP)

        # ── Exit price ────────────────────────────────────────────────
        if exit_key in cache:
            xp = cache[exit_key]
        else:
            xp = fetch_1min_close(ticker, date, exit_str, session)
            if np.isnan(xp) or xp <= 0:
                xp = row["option_exit_price"]    # fallback to original
                errors += 1
            cache[exit_key] = xp
            fetched += 1
            time.sleep(SLEEP)

        new_entry_prices.append(ep)
        new_exit_prices.append(xp)

        # Save cache and log progress every 50 fetched
        if fetched > 0 and fetched % 50 == 0:
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)
            pct = (i + 1) / n * 100
            print(f"  [{pct:5.1f}%] trade {i+1}/{n} — {fetched} fetched, {errors} errors")

    # Final cache save
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    # ── Recompute P&L ────────────────────────────────────────────────
    out = df.copy()
    out["option_entry_price"] = new_entry_prices
    out["option_exit_price"]  = new_exit_prices

    out["num_contracts"] = (out["budget"] / (out["option_entry_price"] * 100)).astype(int).clip(lower=1)
    out["premium_paid"]  = out["num_contracts"] * out["option_entry_price"] * 100

    entry_adj = out["option_entry_price"] * (1 + SLIPPAGE)
    exit_adj  = (out["option_exit_price"]  * (1 - SLIPPAGE)).clip(lower=0)
    raw_pnl   = (exit_adj - entry_adj) * out["num_contracts"] * 100
    out["pnl"] = raw_pnl.clip(lower=-out["premium_paid"])
    out["pnl_pct"] = (out["pnl"] / out["premium_paid"].replace(0, np.nan)).round(4)

    out.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ──────────────────────────────────────────────────────
    name_map = {
        'GapLarge_First30min_SPY':          'GapLarge',
        'HighVolWR_30min_SPY_filtered':     'HighVolWR',
        'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
        'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
        'PriorDayWeak_30min_QQQ':           'QQQ Weak',
        'PriorDayWeak_30min_SPY_filtered':  'SPY Weak',
        'PriorDayWeak_50Hi_SPY_filtered':   '50Hi Weak',
    }
    df0 = pd.read_csv(INPUT_CSV)
    df0["strat"] = df0["rule"].map(name_map)
    out["strat"] = out["rule"].map(name_map)

    # Entry price comparison
    ep_ratio = out["option_entry_price"].values / df0["option_entry_price"].values
    xp_ratio = out["option_exit_price"].values  / df0["option_exit_price"].values

    daily = out.groupby("trade_date")["pnl"].sum().sort_index()
    cum   = daily.cumsum()
    maxdd = (cum - cum.cummax()).min()
    win_d = (daily > 0).sum()

    print(f"\n{'='*68}")
    print(f"REPRICE RESULTS  (1-min bar CLOSEs at actual signal times)")
    print(f"{'='*68}")
    print(f"  API calls made:  {fetched}   |  fallbacks: {errors}")
    print(f"  Entry shift:  mean {(ep_ratio.mean()-1)*100:+.2f}%  median {(np.median(ep_ratio)-1)*100:+.2f}%")
    print(f"  Exit  shift:  mean {(xp_ratio.mean()-1)*100:+.2f}%  median {(np.median(xp_ratio)-1)*100:+.2f}%")
    print()
    print(f"  {'Strategy':<14} {'N':>5}  {'Orig WR':>8} {'New WR':>8}  {'Orig PnL':>13} {'New PnL':>13}")
    print(f"  {'-'*65}")
    for s in ['GapLarge','HighVolWR','QQQ Short','SPY Short','QQQ Weak','SPY Weak','50Hi Weak']:
        go = df0[df0['strat']==s];  gn = out[out['strat']==s]
        if len(go) == 0: continue
        print(f"  {s:<14} {len(gn):>5}  {(go['pnl']>0).mean()*100:>7.1f}% {(gn['pnl']>0).mean()*100:>7.1f}%  "
              f"${go['pnl'].sum():>12,.0f} ${gn['pnl'].sum():>12,.0f}")
    print(f"  {'-'*65}")
    print(f"  {'TOTAL':<14} {len(out):>5}  {(df0['pnl']>0).mean()*100:>7.1f}% {(out['pnl']>0).mean()*100:>7.1f}%  "
          f"${df0['pnl'].sum():>12,.0f} ${out['pnl'].sum():>12,.0f}")
    print()
    print(f"  Win day rate:  {win_d}/{len(daily)} ({win_d/len(daily)*100:.0f}%)")
    print(f"  Avg / day:     ${daily.mean():,.0f}")
    print(f"  Max DrawDown:  ${maxdd:,.0f}")
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
