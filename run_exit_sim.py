#!/usr/bin/env python3
"""
Exit Simulation — Time Cutoffs & Stop-Losses
=============================================
Tests what happens if we exit earlier on trades that currently
run to the time exit (optimal_exit_1000 / optimal_exit_1010).

Strategy:
  - Profit-target trades (profit_target_50): kept exactly as-is.
  - Time-exit trades (356 total): re-simulated with 1-min bars.

Scenarios tested:
  A) Hard time cutoff  — exit at HH:MM if PT not hit, no stop
  B) Stop-loss only    — exit if option drops X% from entry
  C) Combined          — stop-loss + time cutoff (whichever first)

Data:  1-min option bars from Polygon (adjusted=false, matches backtest)
Input: output/options_2strat.csv
Cache: output/exit_sim_1min_cache.json   (reused across runs)
Output: output/exit_sim_results.csv + printed summary
"""

import os, json, time, sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────

from src.config import POLYGON_API_KEY, OUTPUT_DIR

INPUT_CSV   = os.path.join(OUTPUT_DIR, "options_2strat.csv")
CACHE_FILE  = os.path.join(OUTPUT_DIR, "exit_sim_1min_cache.json")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "exit_sim_results.csv")

BASE_URL    = "https://api.polygon.io"
REQUEST_DELAY = 0.15   # seconds between API calls

# Scenarios
TIME_CUTOFFS = ["09:40", "09:45", "09:50", "09:55"]   # exit at this time if PT not hit
STOP_LOSSES  = [0.25, 0.30, 0.40, 0.50]               # fraction of entry price (e.g. 0.30 = exit if down 30%)
COMBINED     = [                                        # (stop_loss_pct, time_cutoff)
    (0.30, "09:45"),
    (0.30, "09:50"),
    (0.40, "09:45"),
    (0.40, "09:50"),
    (0.50, "09:45"),
    (0.50, "09:50"),
]

# ── API helpers ───────────────────────────────────────────────────────────────

_last_req = 0.0

def _get(url, params=None, retries=3):
    global _last_req
    for attempt in range(retries):
        elapsed = time.time() - _last_req
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        _last_req = time.time()
        try:
            resp = requests.get(url, params=params, headers={"Authorization": f"Bearer {POLYGON_API_KEY}"}, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {}


def fetch_1min_bars(option_ticker: str, trade_date: str) -> dict:
    """
    Fetch 1-min bars for option_ticker on trade_date.
    Returns dict: {"HH:MM": {"o":, "h":, "l":, "c":}} for each bar.
    Uses adjusted=false to match the existing backtest data.
    """
    url = f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/minute/{trade_date}/{trade_date}"
    params = {"adjusted": "false", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}

    all_bars = []
    page_url = url
    page = 0
    while page_url:
        page += 1
        data = _get(page_url, params=params if page == 1 else None)
        if not data or data.get("resultsCount", 0) == 0:
            break
        all_bars.extend(data.get("results", []))
        next_url = data.get("next_url")
        if next_url:
            page_url = next_url if "apiKey" in next_url else f"{next_url}&apiKey={POLYGON_API_KEY}"
            params = None
        else:
            break

    bars = {}
    for b in all_bars:
        ts_ms = b.get("t", 0)
        ts = datetime.fromtimestamp(ts_ms / 1000).strftime("%H:%M")
        bars[ts] = {"o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"]}
    return bars


# ── Simulation helpers ────────────────────────────────────────────────────────

def hhmm_to_dec(hhmm: str) -> float:
    h, m = int(hhmm[:2]), int(hhmm[3:5])
    return h + m / 60.0


def simulate_trade(bars: dict, entry_price: float, entry_hm: str,
                   original_exit_hm: str, original_exit_price: float,
                   stop_loss_pct: float = None, time_cutoff_hm: str = None) -> tuple:
    """
    Replay 1-min bars from entry to original exit (or cutoff, whichever first).
    Returns (exit_price, exit_hm, exit_reason).

    stop_loss_pct: float (e.g. 0.30 means exit if option drops to entry * 0.70)
    time_cutoff_hm: str (e.g. "09:45" — hard exit at this time if not stopped out)
    """
    stop_price  = entry_price * (1 - stop_loss_pct) if stop_loss_pct else None
    cutoff_dec  = hhmm_to_dec(time_cutoff_hm) if time_cutoff_hm else None
    entry_dec   = hhmm_to_dec(entry_hm)
    orig_dec    = hhmm_to_dec(original_exit_hm)

    # Walk bars in time order, skip entry bar itself
    for hm in sorted(bars.keys()):
        bar_dec = hhmm_to_dec(hm)
        if bar_dec <= entry_dec:
            continue  # skip entry bar and anything before it
        if bar_dec > orig_dec:
            break     # don't go past original exit

        bar = bars[hm]
        bar_low   = bar["l"]
        bar_close = bar["c"]

        # Time cutoff check (exit at this bar's close)
        if cutoff_dec and bar_dec >= cutoff_dec:
            return bar_close, hm, "time_cutoff"

        # Stop-loss check (bar low touched stop)
        if stop_price and bar_low <= stop_price:
            exit_px = max(stop_price, bar_low)  # filled at stop or worse
            return exit_px, hm, "stop_loss"

    # Neither triggered — use original exit
    return original_exit_price, original_exit_hm, "original"


def pnl_from_prices(entry_price, exit_price, num_contracts, premium_paid):
    """P&L = (exit - entry) * contracts * 100, floored at -premium_paid."""
    raw = (exit_price - entry_price) * num_contracts * 100
    return max(raw, -premium_paid)


def portfolio_stats(trade_rows):
    """Compute Sharpe, MaxDD, WR, total P&L from list of (date, pnl) tuples."""
    if not trade_rows:
        return {}
    df = pd.DataFrame(trade_rows, columns=["date", "pnl"])
    daily = df.groupby("date")["pnl"].sum().sort_index()
    all_bd = pd.bdate_range(daily.index.min(), daily.index.max())
    full = daily.reindex(all_bd.strftime("%Y-%m-%d"), fill_value=0)
    cum  = full.cumsum()
    maxdd = abs((cum - cum.cummax()).min())
    sharpe = full.mean() / full.std() * np.sqrt(252) if full.std() > 0 else 0
    wr = (df["pnl"] > 0).mean() * 100
    return {
        "total_pnl": df["pnl"].sum(),
        "trades": len(df),
        "win_rate": wr,
        "sharpe": sharpe,
        "max_dd": maxdd,
        "avg_pnl": df["pnl"].mean(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading trades...")
    df = pd.read_csv(INPUT_CSV)
    df["trade_date"] = df["trade_date"].astype(str).str[:10]

    # Parse entry/exit HH:MM
    def parse_hm(s):
        s = str(s)
        time_part = s.replace("T", " ").split(" ")[1][:5]
        return time_part  # "09:31"

    df["entry_hm"] = df["entry_time"].apply(parse_hm)
    df["exit_hm"]  = df["exit_time"].apply(parse_hm)

    time_exit_mask = df["exit_reason"].str.startswith("optimal_exit")
    pt_trades      = df[~time_exit_mask].copy()
    te_trades      = df[time_exit_mask].copy()

    print(f"  Profit-target trades (kept as-is): {len(pt_trades)}")
    print(f"  Time-exit trades (to simulate):    {len(te_trades)}")

    # ── Load / build 1-min bar cache ─────────────────────────────────────────
    print(f"\nLoading 1-min bar cache from {CACHE_FILE}...")
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            bar_cache = json.load(f)
        print(f"  {len(bar_cache)} entries already cached")
    else:
        bar_cache = {}

    # Fetch missing bars for time-exit trades only
    needed = set(te_trades["option_ticker"].astype(str) + "|" + te_trades["trade_date"].astype(str))
    missing = [k for k in needed if k not in bar_cache]
    print(f"  {len(missing)} bar sets to fetch from Polygon...")

    fetched = 0
    errors  = 0
    for i, key in enumerate(missing):
        ticker, trade_date = key.split("|")
        bars = fetch_1min_bars(ticker, trade_date)
        if bars:
            bar_cache[key] = bars
            fetched += 1
        else:
            bar_cache[key] = {}
            errors += 1
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(missing)} fetched ({errors} errors)...")
            with open(CACHE_FILE, "w") as f:
                json.dump(bar_cache, f)

    with open(CACHE_FILE, "w") as f:
        json.dump(bar_cache, f)
    print(f"  Done: {fetched} fetched, {errors} errors. Cache saved.")

    # ── Baseline stats (no changes) ───────────────────────────────────────────
    baseline_rows = [(r["trade_date"], r["pnl"]) for _, r in df.iterrows()]
    baseline = portfolio_stats(baseline_rows)
    print(f"\n{'='*72}")
    print(f"BASELINE (current exits, no changes)")
    print(f"  P&L=${baseline['total_pnl']:>+12,.0f}  WR={baseline['win_rate']:>5.1f}%  "
          f"Sharpe={baseline['sharpe']:>5.2f}  MaxDD=${baseline['max_dd']:>8,.0f}")

    # ── Run all scenarios ─────────────────────────────────────────────────────
    results = []

    def run_scenario(label, stop_pct=None, cutoff_hm=None):
        rows = []
        sim_exits = {"time_cutoff": 0, "stop_loss": 0, "original": 0}

        # Profit-target trades: unchanged
        for _, r in pt_trades.iterrows():
            rows.append((r["trade_date"], float(r["pnl"])))

        # Time-exit trades: simulate
        for _, r in te_trades.iterrows():
            key   = f"{r['option_ticker']}|{r['trade_date']}"
            bars  = bar_cache.get(key, {})
            entry = float(r["option_entry_price"])
            orig_exit_px = float(r["option_exit_price"])
            orig_exit_hm = r["exit_hm"]

            if not bars:
                # No bar data — keep original
                rows.append((r["trade_date"], float(r["pnl"])))
                sim_exits["original"] += 1
                continue

            exit_px, exit_hm, reason = simulate_trade(
                bars, entry, r["entry_hm"], orig_exit_hm, orig_exit_px,
                stop_loss_pct=stop_pct, time_cutoff_hm=cutoff_hm
            )
            pnl = pnl_from_prices(entry, exit_px, int(r["num_contracts"]), float(r["premium_paid"]))
            rows.append((r["trade_date"], pnl))
            sim_exits[reason] += 1

        stats = portfolio_stats(rows)
        stats["label"]        = label
        stats["stop_pct"]     = stop_pct
        stats["cutoff_hm"]    = cutoff_hm
        stats["n_time_cutoff"]= sim_exits["time_cutoff"]
        stats["n_stop"]       = sim_exits["stop_loss"]
        stats["n_original"]   = sim_exits["original"]
        delta_pnl   = stats["total_pnl"] - baseline["total_pnl"]
        delta_sharpe = stats["sharpe"] - baseline["sharpe"]
        print(f"  {label:<32}  P&L={stats['total_pnl']:>+10,.0f} ({delta_pnl:>+8,.0f})  "
              f"WR={stats['win_rate']:>5.1f}%  Sharpe={stats['sharpe']:>5.2f} ({delta_sharpe:>+.2f})  "
              f"MaxDD={stats['max_dd']:>8,.0f}  "
              f"[TC={sim_exits['time_cutoff']} SL={sim_exits['stop_loss']} Orig={sim_exits['original']}]")
        results.append(stats)

    # A) Time cutoffs only
    print(f"\n{'='*72}")
    print("A) HARD TIME CUTOFF (no stop-loss)")
    for cutoff in TIME_CUTOFFS:
        run_scenario(f"Exit @ {cutoff}", stop_pct=None, cutoff_hm=cutoff)

    # B) Stop-loss only
    print(f"\n{'='*72}")
    print("B) STOP-LOSS ONLY (no time cutoff, hold to original exit)")
    for sl in STOP_LOSSES:
        run_scenario(f"Stop -{sl*100:.0f}%", stop_pct=sl, cutoff_hm=None)

    # C) Combined
    print(f"\n{'='*72}")
    print("C) COMBINED (stop-loss + time cutoff)")
    for sl, cutoff in COMBINED:
        run_scenario(f"Stop -{sl*100:.0f}% + Exit @ {cutoff}", stop_pct=sl, cutoff_hm=cutoff)

    # ── Per-strategy breakdown for best scenarios ─────────────────────────────
    print(f"\n{'='*72}")
    print("PER-STRATEGY BREAKDOWN — Baseline vs best scenarios")

    best_sharpe = max(results, key=lambda x: x["sharpe"])
    best_pnl    = max(results, key=lambda x: x["total_pnl"])
    print(f"  Best Sharpe: {best_sharpe['label']}")
    print(f"  Best P&L:    {best_pnl['label']}")

    for rule in df["rule"].unique():
        label = "QQQ Short" if "QQQ" in rule else "SPY Short"
        sub = df[df["rule"] == rule]
        te_sub = sub[sub["exit_reason"].str.startswith("optimal_exit")]
        pt_sub = sub[~sub["exit_reason"].str.startswith("optimal_exit")]
        print(f"\n  {label}:")
        print(f"    Time-exit trades: {len(te_sub)} | PT trades: {len(pt_sub)}")
        print(f"    Time-exit WR={( te_sub['pnl']>0).mean()*100:.1f}%  avg P&L=${te_sub['pnl'].mean():>+,.0f}")
        print(f"    PT-exit   WR={( pt_sub['pnl']>0).mean()*100:.1f}%  avg P&L=${pt_sub['pnl'].mean():>+,.0f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved → {OUTPUT_CSV}")

    # ── Summary table sorted by Sharpe ────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY — sorted by Sharpe (descending)")
    print(f"  {'Scenario':<34} {'P&L':>12} {'vs Base':>9} {'WR':>6} {'Sharpe':>7} {'MaxDD':>10}")
    print(f"  {'-'*80}")
    print(f"  {'BASELINE':<34} {baseline['total_pnl']:>+12,.0f} {'':>9} "
          f"{baseline['win_rate']:>5.1f}% {baseline['sharpe']:>7.2f} {baseline['max_dd']:>10,.0f}")
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True)[:12]:
        dp = r['total_pnl'] - baseline['total_pnl']
        print(f"  {r['label']:<34} {r['total_pnl']:>+12,.0f} {dp:>+9,.0f} "
              f"{r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} {r['max_dd']:>10,.0f}")


if __name__ == "__main__":
    main()
