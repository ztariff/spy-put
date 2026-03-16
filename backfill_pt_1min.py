#!/usr/bin/env python3
"""
Backfill 1-min bars for PT trades that currently only have 5-min data.
Also backfills any time-exit trades missing from the 1-min cache.

Run locally: python3 backfill_pt_1min.py

Reads:  output/options_2strat_v6_costs.csv (trade list)
        output/exit_sim_1min_cache.json    (existing 1m cache)
Writes: output/exit_sim_1min_cache.json    (updated with new 1m bars)

Polygon rate limit: 5 req/s on free tier. Script paces at ~3 req/s.
164 trades = ~55 seconds.
"""
import json, csv, time, os, sys
from datetime import datetime
import urllib.request

API = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
POLY = "https://api.polygon.io"
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

def poly(path, retries=3):
    sep = '&' if '?' in path else '?'
    url = POLY + path + sep + 'apiKey=' + API
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            raise e

def get_1min_bars(option_ticker, date):
    """Fetch 1-min bars for option on given date."""
    d = poly(f'/v2/aggs/ticker/{option_ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=500')
    bars = d.get('results', [])
    result = {}
    for b in bars:
        t = datetime.fromtimestamp(b['t']/1000)
        label = t.strftime("%H:%M")
        # Only keep market hours 9:30 - 16:00
        if '09:30' <= label <= '16:00':
            result[label] = {
                'o': b.get('o', 0),
                'h': b.get('h', 0),
                'l': b.get('l', 0),
                'c': b.get('c', 0),
            }
    return result

def main():
    # Load trade list
    csv_path = os.path.join(BASE, "options_2strat_v6_costs.csv")
    with open(csv_path) as f:
        trades = list(csv.DictReader(f))
    print(f"Loaded {len(trades)} trades from CSV")

    # Load existing 1-min cache
    cache_path = os.path.join(BASE, "exit_sim_1min_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} entries in 1-min cache")
    else:
        cache = {}
        print("No existing 1-min cache, starting fresh")

    # Find trades missing 1-min bars
    missing = []
    for t in trades:
        key = f"{t['option_ticker']}|{t['trade_date']}"
        if key not in cache or not cache[key]:
            missing.append(t)

    print(f"Trades missing 1-min bars: {len(missing)}")
    if not missing:
        print("All trades already have 1-min bars!")
        return

    # Fetch 1-min bars for missing trades
    fetched = 0
    errors = 0
    for i, t in enumerate(missing):
        key = f"{t['option_ticker']}|{t['trade_date']}"
        option_ticker = t['option_ticker']
        date = t['trade_date']

        print(f"[{i+1}/{len(missing)}] {date} {option_ticker} ({t['exit_reason']})...", end=' ', flush=True)

        try:
            bars = get_1min_bars(option_ticker, date)
            if bars:
                cache[key] = bars
                fetched += 1
                print(f"{len(bars)} bars")
            else:
                print("0 bars (no data)")
                cache[key] = {}
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

        # Rate limit: ~3 requests/sec
        time.sleep(0.35)

        # Save every 20 fetches
        if (i + 1) % 20 == 0:
            with open(cache_path, 'w') as f:
                json.dump(cache, f)
            print(f"  [checkpoint] saved {len(cache)} entries")

    # Final save
    with open(cache_path, 'w') as f:
        json.dump(cache, f)

    print(f"\n{'='*60}")
    print(f"Done! Fetched {fetched} trades, {errors} errors")
    print(f"Total cache entries: {len(cache)}")
    print(f"Saved to: {cache_path}")

if __name__ == '__main__':
    main()
