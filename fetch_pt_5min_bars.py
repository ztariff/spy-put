"""
fetch_pt_5min_bars.py
─────────────────────
Fetches 5-minute option bar data from Polygon for the 164 PT trades
that are missing from the existing exit_sim_1min_cache.json.

Run from the "Momentum Trader" folder:
    python fetch_pt_5min_bars.py

Output: output/pt_5min_cache.json
"""

import json, os, time, urllib.request, urllib.error
from datetime import datetime, timezone

API_KEY  = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT      = os.path.join(SCRIPT_DIR, "output", "pt_trades_to_fetch.json")
OUTPUT     = os.path.join(SCRIPT_DIR, "output", "pt_5min_cache.json")


def fetch_5min_bars(ticker, date):
    url = (f"{BASE_URL}/{ticker}/range/5/minute/{date}/{date}"
           f"?adjusted=false&sort=asc&limit=50&apiKey={API_KEY}")
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        if data.get("status") not in ("OK", "DELAYED"):
            return None
        bars = {}
        for bar in data.get("results", []):
            t = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc)
            label = t.astimezone().strftime("%H:%M")
            bars[label] = {
                "o": bar["o"], "h": bar["h"],
                "l": bar["l"], "c": bar["c"], "v": bar.get("v", 0)
            }
        return bars
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    with open(INPUT) as f:
        trades = json.load(f)

    # Load existing cache if restarting mid-run
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            cache = json.load(f)
        print(f"Resuming — {len(cache)} already cached")
    else:
        cache = {}

    total   = len(trades)
    fetched = 0
    skipped = 0
    failed  = 0

    for i, t in enumerate(trades):
        key = f"{t['option_ticker']}|{t['trade_date']}"
        if key in cache:
            skipped += 1
            continue

        print(f"[{i+1:3}/{total}] {key} ... ", end="", flush=True)
        bars = fetch_5min_bars(t["option_ticker"], t["trade_date"])

        if bars is None or len(bars) == 0:
            print(f"no data")
            cache[key] = {}   # mark as attempted so we don't retry endlessly
            failed += 1
        else:
            print(f"OK  ({len(bars)} bars, "
                  f"{min(bars)} → {max(bars)})")
            cache[key] = bars
            fetched += 1

        # Save incrementally every 10 fetches
        if (fetched + failed) % 10 == 0:
            with open(OUTPUT, "w") as f:
                json.dump(cache, f)

        time.sleep(0.15)   # ~7 req/sec, well under free-tier limit

    # Final save
    with open(OUTPUT, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\n─── Done ───")
    print(f"  Fetched:  {fetched}")
    print(f"  Skipped:  {skipped}  (already cached)")
    print(f"  No data:  {failed}")
    print(f"  Output:   {OUTPUT}")


if __name__ == "__main__":
    main()
