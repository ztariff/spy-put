"""
fetch_tackon_5min_bars.py
─────────────────────────
Fetches 5-min option bar data for lower-delta tack-on puts.
Tests 3 candidate strikes per trade (roughly δ0.20 to δ0.30).

Run from the "Momentum Trader" folder:
    python fetch_tackon_5min_bars.py

Output: output/tackon_5min_cache.json
"""

import json, os, time, urllib.request
from datetime import datetime, timezone

API_KEY  = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT      = os.path.join(SCRIPT_DIR, "output", "tackon_targets.json")
OUTPUT     = os.path.join(SCRIPT_DIR, "output", "tackon_5min_cache.json")


def fetch_5min_bars(ticker, date):
    url = (f"{BASE_URL}/{ticker}/range/5/minute/{date}/{date}"
           f"?adjusted=false&sort=asc&limit=120&apiKey={API_KEY}")
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
        targets = json.load(f)

    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            cache = json.load(f)
        print(f"Resuming — {len(cache)} already cached")
    else:
        cache = {}

    total   = len(targets)
    fetched = 0
    skipped = 0
    failed  = 0

    for i, t in enumerate(targets):
        key = f"{t['option_ticker_tackon']}|{t['trade_date']}"
        if key in cache:
            skipped += 1
            continue

        print(f"[{i+1:4}/{total}] {key} ... ", end="", flush=True)
        bars = fetch_5min_bars(t["option_ticker_tackon"], t["trade_date"])

        if bars is None or len(bars) == 0:
            print(f"no data")
            cache[key] = {}
            failed += 1
        else:
            print(f"OK  ({len(bars)} bars)")
            cache[key] = bars
            fetched += 1

        # Save every 25 fetches
        if (fetched + failed) % 25 == 0:
            with open(OUTPUT, "w") as f:
                json.dump(cache, f)

        time.sleep(0.15)

    with open(OUTPUT, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\n─── Done ───")
    print(f"  Fetched:  {fetched}")
    print(f"  Skipped:  {skipped}  (already cached)")
    print(f"  No data:  {failed}")
    print(f"  Output:   {OUTPUT}")


if __name__ == "__main__":
    main()
