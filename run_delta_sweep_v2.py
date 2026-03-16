#!/usr/bin/env python3
"""
Delta Sweep v2 — Corrected Underlying Price Estimation
=======================================================
FIX vs v1: The original sweep used daily bar CLOSE prices to estimate the
underlying at entry. On days where the underlying moved significantly from
open to close (e.g. QQQ opened at $489, closed at $481 on 2024-10-01), the
wrong strike was selected — a deeply OTM option was treated as ATM.

This version back-solves the actual 9:31 AM underlying price from:
  1. The original δ0.70 option entry price (from reprice_1min_cache.json)
  2. The known strike K and expiry
  3. Black-Scholes: iterates over S until the computed put/call price matches
     the observed 1-min price while satisfying the original delta target

If the back-solve fails (option not in cache or iteration fails), falls back
to orig_strike (which is more accurate than the daily close for 0DTE options,
since for 0DTE the strike is always near the underlying).

Input:  output/options_931filter_pt50.csv
Output: output/delta_sweep_v2_results.csv
Cache:  output/delta_sweep_v2_cache.json  (separate from v1 cache)
Reuses: output/reprice_1min_cache.json (for original option prices)

Run on your machine:
    python run_delta_sweep_v2.py
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
from src.black_scholes import put_delta, call_delta, implied_vol
from src.options_client import OptionsClient

INPUT_CSV   = os.path.join(OUTPUT_DIR, "options_931filter_pt50.csv")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "delta_sweep_v2_results.csv")
CACHE_FILE  = os.path.join(OUTPUT_DIR, "delta_sweep_v2_cache.json")
CACHE_DIR   = os.path.join(DATA_DIR, "options_cache")
REPRICE_CACHE = os.path.join(OUTPUT_DIR, "reprice_1min_cache.json")

API_KEY  = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
BASE_URL = "https://api.polygon.io"
SLIPPAGE = 0.01
SLEEP    = 0.2

SWEEP_CONFIG = {
    'PriorDayStrong_AboveOR_QQQ_short': {
        'label':   'QQQ Short',
        'deltas':  [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        'budget':  104000,
        'is_call': False,
    },
    'PriorDayStrong_AboveOR_SPY_short': {
        'label':   'SPY Short',
        'deltas':  [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        'budget':  99000,
        'is_call': False,
    },
    'PriorDayWeak_50Hi_SPY_filtered': {
        'label':   '50Hi Weak',
        'deltas':  [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        'budget':  48000,
        'is_call': True,
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


def estimate_underlying_price(orig_option_ticker: str, orig_strike: float,
                               entry_str: str, trade_date: str,
                               orig_entry_hm: str, expiry_date: date,
                               is_call: bool, reprice_cache: dict) -> tuple:
    """
    Back-solve the actual underlying price at entry from the original option's
    1-min market price.

    Steps:
    1. Look up the original option's 1-min entry price from reprice_cache
    2. Compute T (time to expiry at entry time)
    3. Use Black-Scholes implied_vol + iterative solve for S

    Returns: (underlying_price, sigma, method)
      method = 'back_solve' if we successfully solved, 'orig_strike' if fallback
    """
    # Look up the cached original option price
    cache_key = f"{orig_option_ticker}_{trade_date}_entry_{orig_entry_hm}"
    orig_price = reprice_cache.get(cache_key)

    if orig_price is None or orig_price <= 0:
        return orig_strike, 0.20, 'orig_strike'

    entry_ts = pd.Timestamp(entry_str)
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize("America/New_York")
    T = time_to_expiry_years(entry_ts, expiry_date)
    if T <= 0:
        T = 1 / (252 * 6.5)

    K  = orig_strike
    r  = RISK_FREE_RATE
    P  = orig_price

    # Strategy: iterate over S in [K*0.80, K*1.20]
    # For each candidate S, compute implied_vol from the market price,
    # then check if the computed price matches (it will by construction).
    # The correct S is the one where the BS put_delta matches the
    # expected delta for a ~0.70 delta option.
    #
    # Since for 0DTE options: delta(put) = N(d1) - 1
    # and the original target was 0.70, we want put_delta ≈ -0.70.
    #
    # We binary-search over S to find where put_delta ≈ -0.70.

    delta_fn = call_delta if is_call else put_delta
    target_d = 0.70 if not is_call else 0.70   # original sweep used 0.70

    lo, hi = K * 0.85, K * 1.15
    best_S  = K
    best_err = float('inf')

    for _ in range(60):
        S = (lo + hi) / 2

        # Get implied vol from the observed price
        sigma = implied_vol(P, S, K, T, r, is_call=is_call)
        if sigma is None or sigma <= 0.001:
            sigma = 0.20

        # Compute actual delta
        d = abs(delta_fn(S, K, T, r, sigma))
        err = abs(d - target_d)

        if err < best_err:
            best_err = err
            best_S   = S

        if err < 0.005:
            break

        # For a put: lower S → higher |delta| (more ITM)
        if is_call:
            if d < target_d:
                lo = S   # need higher S to increase call delta
            else:
                hi = S
        else:
            if d < target_d:
                hi = S   # need lower S to increase put delta
            else:
                lo = S

    # Verify the result makes sense (S within 5% of K for 0DTE)
    if abs(best_S - K) / K > 0.07:
        return orig_strike, 0.20, 'orig_strike'

    # Get sigma at best_S
    sigma_final = implied_vol(P, best_S, K, T, r, is_call=is_call)
    if sigma_final is None or sigma_final <= 0:
        sigma_final = 0.20

    return best_S, sigma_final, 'back_solve'


def load_contracts(ticker: str, expiry: str, cp: str) -> list:
    fname = os.path.join(CACHE_DIR, f"contracts_{ticker}_{expiry}_{cp.lower()}.json")
    if not os.path.exists(fname):
        return []
    with open(fname) as f:
        return json.load(f)


def main():
    print(f"Loading {INPUT_CSV}...")
    df_all = pd.read_csv(INPUT_CSV)
    df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])
    print(f"  {len(df_all)} total signals\n")

    # Load reprice cache (has original 1-min prices — key for back-solve)
    reprice_cache = {}
    if os.path.exists(REPRICE_CACHE):
        with open(REPRICE_CACHE) as f:
            reprice_cache = json.load(f)
        print(f"  Loaded {len(reprice_cache)} prices from reprice_1min_cache")
    else:
        print(f"  WARNING: {REPRICE_CACHE} not found — will use orig_strike fallback")

    # Load sweep cache (resumable)
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} prices from v2 sweep cache\n")

    session = requests.Session()
    session.headers["User-Agent"] = "0DTE-DeltaSweep-v2/1.0"

    client = OptionsClient(API_KEY)

    all_rows = []
    fetched  = 0
    errors   = 0
    back_solve_count = 0
    fallback_count   = 0

    for rule, cfg in SWEEP_CONFIG.items():
        df = df_all[df_all['rule'] == rule].copy()
        label   = cfg['label']
        deltas  = cfg['deltas']
        budget  = cfg['budget']
        is_call = cfg['is_call']
        cp_char = 'C' if is_call else 'P'
        cp_word = 'call' if is_call else 'put'
        cp_full = 'CALL' if is_call else 'PUT'

        print(f"\n{'='*60}")
        print(f"{label}  ({len(df)} trades, testing {len(deltas)} deltas)")
        print(f"{'='*60}")

        for target_delta in deltas:
            pnls    = []
            skipped = 0

            for _, row in df.iterrows():
                trade_date  = row['trade_date'].date()
                entry_str   = str(row['entry_time'])
                exit_str    = str(row['exit_time'])
                entry_hm    = entry_str[11:16]
                exit_hm     = exit_str[11:16]
                date_str    = str(trade_date)
                ticker      = row['ticker']
                orig_strike = float(row['strike'])
                expiry_str  = str(row['expiry_date'])[:10]
                expiry_date = date.fromisoformat(expiry_str)
                orig_ticker = str(row['option_ticker'])

                # ── Estimate actual underlying price at entry ─────────────
                # v2 FIX: back-solve from original 1-min option price
                # instead of using daily bar close
                entry_ts = pd.Timestamp(entry_str)
                if entry_ts.tzinfo is None:
                    entry_ts = entry_ts.tz_localize("America/New_York")
                T = time_to_expiry_years(entry_ts, expiry_date)
                if T <= 0:
                    T = 1 / (252 * 6.5)

                underlying_price, sigma, method = estimate_underlying_price(
                    orig_ticker, orig_strike, entry_str, date_str,
                    entry_hm, expiry_date, is_call, reprice_cache
                )

                if method == 'back_solve':
                    back_solve_count += 1
                else:
                    fallback_count += 1

                # ── Load available strikes ────────────────────────────────
                contracts = load_contracts(ticker, expiry_str, cp_word)
                if not contracts:
                    skipped += 1
                    continue
                strikes = get_strikes_for_type(contracts, cp_full)
                if not strikes:
                    skipped += 1
                    continue

                # ── Select strike at target delta ─────────────────────────
                new_strike, actual_delta = select_strike(
                    underlying_price=underlying_price,
                    available_strikes=strikes,
                    target_delta=target_delta,
                    T=T,
                    r=RISK_FREE_RATE,
                    sigma=sigma,
                    is_call=is_call,
                )
                if new_strike is None:
                    skipped += 1
                    continue

                # ── Construct option ticker ───────────────────────────────
                new_ticker = client.construct_ticker(
                    ticker, expiry_date, new_strike, cp_char
                )

                # ── Entry price (1-min bar) ───────────────────────────────
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

                # ── Exit price (1-min bar) ────────────────────────────────
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

                # ── P&L ───────────────────────────────────────────────────
                contracts_n = max(1, int(budget / (ep * 100)))
                premium     = contracts_n * ep * 100
                entry_adj   = ep * (1 + SLIPPAGE)
                exit_adj    = max(xp * (1 - SLIPPAGE), 0)
                pnl         = max((exit_adj - entry_adj) * contracts_n * 100, -premium)
                pnl_pct     = pnl / premium if premium > 0 else 0

                pnls.append({
                    'rule':          rule,
                    'label':         label,
                    'trade_date':    date_str,
                    'target_delta':  target_delta,
                    'actual_delta':  round(actual_delta, 3) if actual_delta else target_delta,
                    'underlying_est': round(underlying_price, 2),
                    'iv_est':        round(sigma, 3),
                    'ul_method':     method,
                    'strike':        new_strike,
                    'option_ticker': new_ticker,
                    'entry_price':   ep,
                    'exit_price':    xp,
                    'contracts':     contracts_n,
                    'premium':       round(premium, 2),
                    'pnl':           round(pnl, 2),
                    'pnl_pct':       round(pnl_pct, 4),
                    'win':           1 if pnl > 0 else 0,
                })

            if pnls:
                all_rows.extend(pnls)
                total = sum(r['pnl'] for r in pnls)
                wr    = sum(r['win'] for r in pnls) / len(pnls) * 100
                n_ok  = len(pnls)
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

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("DELTA SWEEP v2 SUMMARY  (corrected underlying price)")
    print(f"{'='*75}")
    print(f"Underlying price method: {back_solve_count} back-solved, {fallback_count} orig_strike fallback")
    print(f"Total API calls: {fetched}  |  errors: {errors}")

    all_dates = pd.bdate_range(results['trade_date'].min(), results['trade_date'].max())

    for rule, cfg in SWEEP_CONFIG.items():
        label = cfg['label']
        sub   = results[results['label'] == label]
        if sub.empty:
            continue
        print(f"\n{label}  (budget=${cfg['budget']:,})")
        print(f"  {'Delta':>7}  {'Trades':>7}  {'WR%':>7}  {'Total P&L':>13}  {'Avg/Trade':>10}  {'Sharpe':>7}")
        print(f"  {'-'*62}")

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
            daily_full = daily.reindex(
                pd.to_datetime(all_dates).strftime('%Y-%m-%d'), fill_value=0
            )
            mu, sig = daily_full.mean(), daily_full.std()
            sharpe = (mu / sig * np.sqrt(252)) if sig > 0 else 0
            star = '  ◄ best' if abs(total - best_pnl) < 1 else ''
            print(f"  {td:>7.2f}  {len(g):>7}  {wr:>6.1f}%  ${total:>12,.0f}  ${avg:>9,.0f}  {sharpe:>7.2f}{star}")

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
