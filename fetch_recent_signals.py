#!/usr/bin/env python3
"""
Fetch recent signal data + option bars for dates after the V6 CSV ends (2026-02-23).
Uses historical options contracts + aggregates (not snapshots).

Run locally: python3 fetch_recent_signals.py
Output: output/recent_trades.json
"""
import json, time, os, math
from datetime import datetime, timedelta, timezone
import urllib.request

# Eastern Time offset (UTC-5 for EST, UTC-4 for EDT)
# Use fixed ET offset; for dates near DST boundary, ±1hr doesn't affect date or bar labels
ET = timezone(timedelta(hours=-5))

API = os.environ.get("POLYGON_KEY", "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF")
POLY = "https://api.polygon.io"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "recent_trades.json")

STRATEGIES = {
    "QQQ": {"delta": 0.55, "base_budget": 104000},
    "SPY": {"delta": 0.60, "base_budget": 99000},
}
SIGNAL_THRESHOLD = 0.75
GAP_UP_MULT = 0.75
GAP_DOWN_MULT = 1.5
COMMISSION = 1.10

def poly(path):
    sep = '&' if '?' in path else '?'
    url = POLY + path + sep + 'apiKey=' + API
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def get_daily_bars(ticker, start, end):
    d = poly(f'/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=50')
    return d.get('results', [])

def find_option_contract(ticker, date, delta_target):
    """
    Find the 0DTE put closest to target delta using the contracts endpoint
    + 1-min bars around 9:31 to estimate entry price.
    """
    # List available put contracts expiring on this date
    d = poly(f'/v3/reference/options/contracts?underlying_ticker={ticker}&expiration_date={date}&contract_type=put&expired=true&limit=100&order=desc&sort=strike_price')
    contracts = d.get('results', [])
    if not contracts:
        print(f"    No contracts found for {ticker} on {date}")
        return None

    # We don't have live greeks for historical dates, so estimate:
    # For puts, delta ~ -N(d1). ATM put delta ~ -0.50.
    # delta 0.55 put ≈ slightly ITM, delta 0.60 ≈ more ITM
    # Heuristic: for SPY/QQQ 0DTE, delta 0.55 put strike ≈ underlying * 1.001
    #            delta 0.60 put strike ≈ underlying * 1.003
    return contracts

def get_1min_bars(option_ticker, date):
    """Fetch 1-min bars for option on given date."""
    d = poly(f'/v2/aggs/ticker/{option_ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=500')
    bars = d.get('results', [])
    result = {}
    for b in bars:
        t = datetime.fromtimestamp(b['t']/1000, tz=ET)
        label = t.strftime("%H:%M")
        result[label] = {'o': b.get('o',0), 'h': b.get('h',0), 'l': b.get('l',0), 'c': b.get('c',0), 'v': b.get('v',0)}
    return result

def get_5min_bars(option_ticker, date):
    """Fetch 5-min bars for option on given date."""
    d = poly(f'/v2/aggs/ticker/{option_ticker}/range/5/minute/{date}/{date}?adjusted=true&sort=asc&limit=200')
    bars = d.get('results', [])
    result = {}
    for b in bars:
        t = datetime.fromtimestamp(b['t']/1000, tz=ET)
        label = t.strftime("%H:%M")
        result[label] = {'o': b.get('o',0), 'h': b.get('h',0), 'l': b.get('l',0), 'c': b.get('c',0), 'v': b.get('v',0)}
    return result

def find_best_strike(contracts, underlying_price, delta_target):
    """
    Without live greeks, estimate the right strike from the underlying price.
    For 0DTE puts:
      delta 0.55 → strike ≈ underlying * 1.000 to 1.002 (slightly ITM)
      delta 0.60 → strike ≈ underlying * 1.002 to 1.005 (more ITM)
    We'll pick the strike closest to underlying * (1 + offset).
    """
    if delta_target >= 0.60:
        target_strike = underlying_price * 1.003
    else:
        target_strike = underlying_price * 1.001

    best = None
    best_diff = float('inf')
    for c in contracts:
        strike = c.get('strike_price', 0)
        diff = abs(strike - target_strike)
        if diff < best_diff:
            best_diff = diff
            best = c
    return best

def main():
    if os.path.exists(OUT):
        with open(OUT) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing recent trades")
    else:
        existing = []

    existing_keys = set(f"{t['option_ticker']}|{t['trade_date']}" for t in existing)

    start = "2026-02-20"
    end = datetime.now().strftime("%Y-%m-%d")

    # Fetch daily bars
    daily = {}
    for tk in ['QQQ', 'SPY']:
        bars = get_daily_bars(tk, start, end)
        daily[tk] = bars
        print(f"{tk}: {len(bars)} daily bars from {start} to {end}")
        time.sleep(0.25)

    new_trades = list(existing)

    check_start = datetime(2026, 2, 24)
    check_end = datetime.now()

    d = check_start
    while d <= check_end:
        ds = d.strftime("%Y-%m-%d")
        dow = d.weekday()
        if dow >= 5:
            d += timedelta(days=1)
            continue

        print(f"\n--- {ds} ---")

        for tk in ['QQQ', 'SPY']:
            bars = daily[tk]
            prior = None
            today_bar = None
            for i, b in enumerate(bars):
                bdate = datetime.fromtimestamp(b['t']/1000, tz=ET).strftime("%Y-%m-%d")
                if bdate == ds:
                    today_bar = b
                    if i > 0:
                        prior = bars[i-1]
                    break

            if not prior:
                print(f"  {tk}: no prior day data, skip")
                continue

            rng = prior['h'] - prior['l']
            loc = (prior['c'] - prior['l']) / rng if rng > 0 else 0.5
            signal = loc > SIGNAL_THRESHOLD
            print(f"  {tk}: closeLoc={loc*100:.1f}% -> signal={'YES' if signal else 'no'}")

            if not signal:
                continue

            if not today_bar:
                print(f"  {tk}: no today bar, skip")
                continue

            gap_up = today_bar['o'] > prior['c']
            mult = GAP_UP_MULT if gap_up else GAP_DOWN_MULT
            budget = round(STRATEGIES[tk]['base_budget'] * mult / 1000) * 1000
            gap_pct = (today_bar['o'] - prior['c']) / prior['c']
            print(f"  {tk}: gap={'up' if gap_up else 'down'} {gap_pct*100:+.2f}%, mult={mult}x, budget=${budget:,}")

            # Get available contracts
            try:
                contracts = find_option_contract(tk, ds, STRATEGIES[tk]['delta'])
                time.sleep(0.25)
            except Exception as e:
                print(f"  {tk}: contracts error: {e}")
                continue

            if not contracts:
                continue

            # Find best strike based on underlying open price
            underlying_open = today_bar['o']
            best = find_best_strike(contracts, underlying_open, STRATEGIES[tk]['delta'])
            if not best:
                print(f"  {tk}: no best strike found")
                continue

            opt_ticker = best.get('ticker', '')
            strike = best.get('strike_price', 0)
            print(f"  {tk}: selected {opt_ticker} strike=${strike}")

            # Get 1-min bars to find entry price at 9:31
            try:
                bars_1m = get_1min_bars(opt_ticker, ds)
                time.sleep(0.25)
            except Exception as e:
                print(f"  {tk}: 1min bars error: {e}")
                bars_1m = {}

            # Entry price = 9:31 bar open, or first available bar
            entry_px = None
            for t_label in ['09:31', '09:30', '09:32']:
                if t_label in bars_1m:
                    entry_px = bars_1m[t_label].get('o') or bars_1m[t_label].get('c')
                    if entry_px and entry_px > 0:
                        break
            if not entry_px or entry_px <= 0:
                # Try first bar in trading hours
                for t_label in sorted(bars_1m.keys()):
                    if t_label >= '09:30' and t_label <= '09:35':
                        entry_px = bars_1m[t_label].get('o') or bars_1m[t_label].get('c')
                        if entry_px and entry_px > 0:
                            break

            if not entry_px or entry_px <= 0:
                print(f"  {tk}: no entry price from 1min bars ({len(bars_1m)} bars), skip")
                continue

            n_contracts = math.floor(budget / (entry_px * 100))
            if n_contracts <= 0:
                print(f"  {tk}: 0 contracts at entry=${entry_px:.4f}, skip")
                continue

            key = f"{opt_ticker}|{ds}"
            if key in existing_keys:
                print(f"  {tk}: already fetched {key}")
                continue

            # Get 5-min bars for trail simulation
            try:
                bars_5m = get_5min_bars(opt_ticker, ds)
                time.sleep(0.25)
            except Exception as e:
                print(f"  {tk}: 5min bars error: {e}")
                bars_5m = {}

            print(f"  {tk}: entry=${entry_px:.4f} cts={n_contracts} | {len(bars_1m)} 1m-bars, {len(bars_5m)} 5m-bars")

            trade = {
                "trade_date": ds,
                "ticker": tk,
                "option_ticker": opt_ticker,
                "strike": strike,
                "direction": "short",
                "target_delta": STRATEGIES[tk]['delta'],
                "option_entry_price": entry_px,
                "num_contracts": n_contracts,
                "budget": budget,
                "gap_up": gap_up,
                "gap_pct": gap_pct,
                "mult": mult,
                "prior_close_loc": loc,
                "bars_5m": bars_5m,
                "bars_1m": bars_1m,
            }
            new_trades.append(trade)
            existing_keys.add(key)

            with open(OUT, 'w') as f:
                json.dump(new_trades, f, indent=2)
            print(f"  Saved ({len(new_trades)} total)")

        d += timedelta(days=1)

    print(f"\n=== Done: {len(new_trades)} total recent trades saved to {OUT} ===")

if __name__ == '__main__':
    main()
