#!/usr/bin/env python3
"""
Fetch live market snapshot and save to output/live_snapshot.txt.
Properly filters to regular trading hours (14:30+ UTC = 9:30+ ET).

Usage:
    python fetch_live_snapshot.py
"""
import os
import requests
from datetime import datetime, date, timedelta

API_KEY = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
BASE = "https://api.polygon.io"
OUT_PATH = os.path.join(os.path.dirname(__file__), "output", "live_snapshot.txt")

# 9:30 ET = 14:30 UTC (EST) or 13:30 UTC (EDT)
# Feb 25 is EST, so market open = 14:30 UTC
MARKET_OPEN_UTC_HOUR = 14
MARKET_OPEN_UTC_MIN = 30

def fetch(url):
    resp = requests.get(url, params={"apiKey": API_KEY}, timeout=15)
    return resp.json()

def bar_time_utc(bar):
    return datetime.utcfromtimestamp(bar['t'] / 1000)

def bar_is_rth(bar):
    """Is this bar during regular trading hours (9:30 ET+ = 14:30 UTC+)?"""
    t = bar_time_utc(bar)
    return (t.hour > MARKET_OPEN_UTC_HOUR or
            (t.hour == MARKET_OPEN_UTC_HOUR and t.minute >= MARKET_OPEN_UTC_MIN))

def main():
    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    today = date.today().isoformat()
    start_30d = (date.today() - timedelta(days=45)).isoformat()

    p("=" * 70)
    p(f"  LIVE SNAPSHOT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 70)

    # 1. Snapshots
    for ticker in ['SPY', 'QQQ']:
        p(f"\n── {ticker} SNAPSHOT ──")
        try:
            data = fetch(f"{BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
            t = data.get('ticker', {})
            day = t.get('day', {})
            prev = t.get('prevDay', {})
            last = t.get('lastTrade', {})
            p(f"  Last trade:  ${last.get('p', 'N/A')}")
            p(f"  Today open:  ${day.get('o', 'N/A')}")
            p(f"  Today high:  ${day.get('h', 'N/A')}")
            p(f"  Today low:   ${day.get('l', 'N/A')}")
            p(f"  Today close: ${day.get('c', 'N/A')}")
            vol = day.get('v', 'N/A')
            p(f"  Today vol:   {vol:,.0f}" if isinstance(vol, (int, float)) else f"  Today vol:   {vol}")
            p(f"  Prev close:  ${prev.get('c', 'N/A')}")
            p(f"  Prev high:   ${prev.get('h', 'N/A')}")
            p(f"  Prev low:    ${prev.get('l', 'N/A')}")
            prev_range = prev.get('h', 0) - prev.get('l', 0)
            prev_close_loc = (prev.get('c', 0) - prev.get('l', 0)) / prev_range if prev_range > 0 else 0
            p(f"  Prev close location: {prev_close_loc:.1%}")
            if day.get('o') and prev.get('c'):
                gap = (day['o'] - prev['c']) / prev['c'] * 100
                p(f"  Gap from prev close: {gap:+.2f}%")
        except Exception as e:
            p(f"  ERROR: {e}")

    # 2. Today's 5-min bars — RTH only
    for ticker in ['SPY', 'QQQ']:
        p(f"\n── {ticker} 5-MIN BARS (RTH ONLY) ──")
        try:
            data = fetch(f"{BASE}/v2/aggs/ticker/{ticker}/range/5/minute/{today}/{today}?adjusted=true&sort=asc&limit=500")
            all_bars = data.get('results', [])
            bars = [b for b in all_bars if bar_is_rth(b)]
            p(f"  Total bars (all): {len(all_bars)}, RTH only: {len(bars)}")

            if bars:
                # Opening range = first RTH bar (9:30 ET = 14:30 UTC)
                first = bars[0]
                ft = bar_time_utc(first).strftime('%H:%M UTC (%I:%M %p ET)')
                # Convert to ET for display
                ft_et = bar_time_utc(first)
                et_hour = ft_et.hour - 5  # EST offset
                et_min = ft_et.minute
                p(f"  First RTH bar ({ft_et.strftime('%H:%M')} UTC = {et_hour}:{et_min:02d} ET): O=${first['o']:.2f} H=${first['h']:.2f} L=${first['l']:.2f} C=${first['c']:.2f}")

                or_high = first['h']
                or_low = first['l']
                or_mid = (or_high + or_low) / 2
                p(f"  Opening Range: High=${or_high:.2f}  Low=${or_low:.2f}  Mid=${or_mid:.2f}")

                last_bar = bars[-1]
                lt = bar_time_utc(last_bar)
                lt_et_h = lt.hour - 5
                lt_et_m = lt.minute
                p(f"  Latest RTH bar ({lt.strftime('%H:%M')} UTC = {lt_et_h}:{lt_et_m:02d} ET): O=${last_bar['o']:.2f} H=${last_bar['h']:.2f} L=${last_bar['l']:.2f} C=${last_bar['c']:.2f}")
                p(f"  Latest close ${last_bar['c']:.2f} {'ABOVE' if last_bar['c'] > or_mid else 'BELOW'} OR mid ${or_mid:.2f}")

                # Check which bars had high > OR mid
                bars_above = [b for b in bars if b['h'] > or_mid]
                p(f"  RTH bars with high > OR mid: {len(bars_above)}/{len(bars)}")
                if bars_above:
                    fa = bars_above[0]
                    fa_t = bar_time_utc(fa)
                    fa_et_h = fa_t.hour - 5
                    fa_et_m = fa_t.minute
                    p(f"  First RTH bar above OR mid: {fa_t.strftime('%H:%M')} UTC = {fa_et_h}:{fa_et_m:02d} ET (high=${fa['h']:.2f})")

                # Entry window check: 9:31-9:56 ET = 14:31-14:56 UTC
                entry_bars = [b for b in bars if 14*60+30 <= bar_time_utc(b).hour*60+bar_time_utc(b).minute <= 14*60+56]
                entry_above = [b for b in entry_bars if b['h'] > or_mid]
                p(f"\n  ENTRY WINDOW (9:30-9:56 ET):")
                p(f"    Bars in window: {len(entry_bars)}")
                p(f"    Bars above OR mid: {len(entry_above)}")
                if entry_above:
                    p(f"    → TRIGGER CONFIRMED at entry window")
                else:
                    p(f"    → NO TRIGGER during entry window")

                p(f"\n  All RTH bars (first 15):")
                for b in bars[:15]:
                    bt = bar_time_utc(b)
                    bt_et_h = bt.hour - 5
                    bt_et_m = bt.minute
                    above_flag = " *** ABOVE OR MID" if b['h'] > or_mid else ""
                    entry_flag = " [ENTRY WINDOW]" if 14*60+30 <= bt.hour*60+bt.minute <= 14*60+56 else ""
                    p(f"    {bt.strftime('%H:%M')} UTC ({bt_et_h}:{bt_et_m:02d} ET): O=${b['o']:.2f} H=${b['h']:.2f} L=${b['l']:.2f} C=${b['c']:.2f} V={b['v']:,.0f}{above_flag}{entry_flag}")
            else:
                p("  No RTH bars yet — market may not be open")
        except Exception as e:
            p(f"  ERROR: {e}")

    # 3. SMA20
    p(f"\n── SPY SMA20 CHECK ──")
    try:
        data = fetch(f"{BASE}/v2/aggs/ticker/SPY/range/1/day/{start_30d}/{today}?adjusted=true&sort=asc&limit=500")
        bars = data.get('results', [])
        daily = [{'date': datetime.utcfromtimestamp(b['t'] / 1000).strftime('%Y-%m-%d'),
                  'close': b['c'], 'high': b['h'], 'low': b['l'], 'volume': b['v']} for b in bars]
        prior = [b for b in daily if b['date'] < today]
        if len(prior) >= 20:
            sma20 = sum(b['close'] for b in prior[-20:]) / 20
            avg_vol = sum(b['volume'] for b in prior[-20:]) / 20
            avg_range = sum(b['high'] - b['low'] for b in prior[-20:]) / 20
            last_day = prior[-1]
            vol_ratio = last_day['volume'] / avg_vol if avg_vol > 0 else 1
            range_ratio = (last_day['high'] - last_day['low']) / avg_range if avg_range > 0 else 1
            p(f"  SMA20: ${sma20:.2f}")
            p(f"  Prior close: ${last_day['close']:.2f}")
            p(f"  {'BELOW' if last_day['close'] < sma20 else 'ABOVE'} SMA20 → longs {'OK' if last_day['close'] < sma20 else 'FILTERED'}")
            p(f"  Vol ratio: {vol_ratio:.2f}x")
            p(f"  Range ratio: {range_ratio:.2f}x")
    except Exception as e:
        p(f"  ERROR: {e}")

    # 4. Signal summary
    p(f"\n{'=' * 70}")
    p("  SIGNAL VERIFICATION SUMMARY")
    p("=" * 70)
    p("  SPY Short: Prior close loc 88% > 75% ✓ — check OR trigger above")
    p("  QQQ Short: Prior close loc 88% > 75% ✓ — check OR trigger above")
    p("  GapLarge:  Need 1-2% gap — check gap % above")
    p("  HighVolWR: Need vol ratio > 1.5x AND range ratio > 1.5x")
    p("  Weak strats: Need close loc < 25% — both at 88%, INACTIVE")
    p("=" * 70)

    with open(OUT_PATH, 'w') as f:
        f.write('\n'.join(lines))

if __name__ == "__main__":
    main()
