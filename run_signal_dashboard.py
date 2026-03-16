#!/usr/bin/env python3
"""
Daily Signal Dashboard
=======================
Fetches latest SPY/QQQ data from Polygon and checks which of the
7 strategies are signaling for today's session.

Generates an HTML dashboard showing:
  - Which signals are ACTIVE / INACTIVE
  - Prior day stats that drive the signals
  - Recommended option parameters (delta, budget, entry/exit times)
  - Regime filter status (SMA20)
  - Historical win rate and avg P&L for each active signal

Usage:
    python run_signal_dashboard.py
"""
import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, time
from collections import defaultdict

from src.config import POLYGON_API_KEY, POLYGON_BASE_URL, OUTPUT_DIR, DATA_DIR


# ── Strategy Configuration ──────────────────────────────────────────────────

STRATEGIES = {
    'GapLarge_First30min_SPY': {
        'name': 'GapLarge',
        'ticker': 'SPY',
        'direction': 'LONG',
        'delta': 0.50,
        'budget': 50_000,
        'entry': '9:31 AM',
        'exit': '2:50 PM',
        'description': 'Large gap up (1-2%) fade — buy the dip after gap exhaustion',
        'color': '#8b5cf6',  # purple
    },
    'HighVolWR_30min_SPY_filtered': {
        'name': 'HighVolWR',
        'ticker': 'SPY',
        'direction': 'LONG',
        'delta': 0.10,
        'budget': 20_000,
        'entry': '9:31 AM',
        'exit': '2:45 PM',
        'description': 'High vol wide range day follow-through — needs wide opening range',
        'color': '#f59e0b',  # amber
    },
    'PriorDayStrong_AboveOR_QQQ_short': {
        'name': 'QQQ Short',
        'ticker': 'QQQ',
        'direction': 'SHORT',
        'delta': 0.70,
        'budget': 75_000,
        'entry': '9:31-9:56 AM',
        'exit': '10:00 AM',
        'description': 'Prior day strong + above OR midpoint — fade the strength',
        'color': '#ef4444',  # red
    },
    'PriorDayStrong_AboveOR_SPY_short': {
        'name': 'SPY Short',
        'ticker': 'SPY',
        'direction': 'SHORT',
        'delta': 0.70,
        'budget': 75_000,
        'entry': '9:31-9:56 AM',
        'exit': '10:10 AM',
        'description': 'Prior day strong + above OR midpoint — fade the strength',
        'color': '#f87171',  # red-light
    },
    'PriorDayWeak_30min_QQQ': {
        'name': 'QQQ Weak',
        'ticker': 'QQQ',
        'direction': 'LONG',
        'delta': 0.50,
        'budget': 50_000,
        'entry': '9:31 AM',
        'exit': '3:20 PM',
        'description': 'Prior day weak (close near low) — mean reversion long',
        'regime_filter': True,
        'color': '#22d3ee',  # cyan
    },
    'PriorDayWeak_30min_SPY_filtered': {
        'name': 'SPY Weak',
        'ticker': 'SPY',
        'direction': 'LONG',
        'delta': 0.50,
        'budget': 50_000,
        'entry': '9:31 AM',
        'exit': '3:05 PM',
        'description': 'Prior day weak + high RVOL bar — mean reversion long',
        'regime_filter': True,
        'color': '#34d399',  # emerald
    },
    'PriorDayWeak_50Hi_SPY_filtered': {
        'name': '50Hi Weak',
        'ticker': 'SPY',
        'direction': 'LONG',
        'delta': 0.50,
        'budget': 50_000,
        'entry': 'Intraday (50-bar high)',
        'exit': '3:05 PM',
        'description': 'Prior day weak + new 50-bar high + high RVOL — breakout long',
        'regime_filter': True,
        'color': '#a3e635',  # lime
    },
}

# Historical stats from backtest
HISTORICAL_STATS = {
    'GapLarge_First30min_SPY':            {'trades': 44, 'wr': 54.5, 'avg_pnl': 14564, 'roi': 29.3},
    'HighVolWR_30min_SPY_filtered':       {'trades': 34, 'wr': 32.4, 'avg_pnl': 11081, 'roi': 128.4},
    'PriorDayStrong_AboveOR_QQQ_short':   {'trades': 470, 'wr': 53.4, 'avg_pnl': 5344, 'roi': 7.2},
    'PriorDayStrong_AboveOR_SPY_short':   {'trades': 464, 'wr': 51.1, 'avg_pnl': 4734, 'roi': 6.3},
    'PriorDayWeak_30min_QQQ':             {'trades': 161, 'wr': 42.2, 'avg_pnl': 5042, 'roi': 10.1},
    'PriorDayWeak_30min_SPY_filtered':    {'trades': 76, 'wr': 46.1, 'avg_pnl': 14838, 'roi': 29.7},
    'PriorDayWeak_50Hi_SPY_filtered':     {'trades': 79, 'wr': 44.3, 'avg_pnl': 12429, 'roi': 29.8},
}


def fetch_daily_bars(ticker, lookback_days=30):
    """Fetch recent daily bars from Polygon."""
    end = date.today()
    start = end - timedelta(days=lookback_days + 10)  # extra buffer for weekends

    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}

    resp = requests.get(url, params=params, headers=headers, timeout=30)
    data = resp.json()

    if not data.get('results'):
        print(f"  WARNING: No data for {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(data['results'])
    df = df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low',
                            'c': 'close', 'v': 'volume', 'vw': 'vwap', 'n': 'trades'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['date'] = df['timestamp'].dt.tz_convert('America/New_York').dt.date
    df = df.sort_values('date').reset_index(drop=True)
    return df


def check_signals(spy_daily, qqq_daily, today):
    """Check which strategies are signaling for today based on prior day data."""
    signals = {}

    # Get prior day data
    spy_prior = spy_daily[spy_daily['date'] < today].tail(1)
    qqq_prior = qqq_daily[qqq_daily['date'] < today].tail(1)

    if spy_prior.empty or qqq_prior.empty:
        print("  ERROR: No prior day data available")
        return signals

    spy_row = spy_prior.iloc[0]
    qqq_row = qqq_prior.iloc[0]

    # Compute prior day close location
    spy_range = spy_row['high'] - spy_row['low']
    spy_close_loc = (spy_row['close'] - spy_row['low']) / spy_range if spy_range > 0 else 0.5

    qqq_range = qqq_row['high'] - qqq_row['low']
    qqq_close_loc = (qqq_row['close'] - qqq_row['low']) / qqq_range if qqq_range > 0 else 0.5

    # Compute gap (need today's open — use latest bar if available, otherwise estimate)
    spy_today = spy_daily[spy_daily['date'] == today]
    qqq_today = qqq_daily[qqq_daily['date'] == today]

    spy_gap_pct = None
    if not spy_today.empty:
        spy_gap_pct = (spy_today.iloc[0]['open'] - spy_row['close']) / spy_row['close']

    # SMA20 for regime filter
    spy_sma20 = spy_daily[spy_daily['date'] < today].tail(20)['close'].mean()
    spy_below_sma20 = spy_row['close'] < spy_sma20

    # Volume and range averages (20-day lookback)
    spy_20d = spy_daily[spy_daily['date'] < today].tail(20)
    spy_avg_vol = spy_20d['volume'].mean()
    spy_avg_range = (spy_20d['high'] - spy_20d['low']).mean()
    spy_vol_ratio = spy_row['volume'] / spy_avg_vol if spy_avg_vol > 0 else 1
    spy_range_ratio = spy_range / spy_avg_range if spy_avg_range > 0 else 1

    # Store prior day metrics
    prior_metrics = {
        'spy_date': str(spy_row['date']),
        'spy_close': spy_row['close'],
        'spy_high': spy_row['high'],
        'spy_low': spy_row['low'],
        'spy_close_loc': spy_close_loc,
        'spy_range_pct': spy_range / spy_row['close'] * 100,
        'spy_vol_ratio': spy_vol_ratio,
        'spy_range_ratio': spy_range_ratio,
        'spy_sma20': spy_sma20,
        'spy_below_sma20': spy_below_sma20,
        'spy_gap_pct': spy_gap_pct,
        'qqq_date': str(qqq_row['date']),
        'qqq_close': qqq_row['close'],
        'qqq_high': qqq_row['high'],
        'qqq_low': qqq_row['low'],
        'qqq_close_loc': qqq_close_loc,
    }

    # ── Check each strategy ─────────────────────────────────────────────────

    # 1. GapLarge: gap up 1-2%
    if spy_gap_pct is not None and 0.01 < spy_gap_pct <= 0.02:
        signals['GapLarge_First30min_SPY'] = {
            'active': True,
            'reason': f"SPY gap up {spy_gap_pct:.2%} (between 1-2%)",
        }
    else:
        gap_str = f"{spy_gap_pct:.2%}" if spy_gap_pct is not None else "unknown (market not open)"
        signals['GapLarge_First30min_SPY'] = {
            'active': False,
            'reason': f"SPY gap {gap_str} — need 1-2% gap up",
        }

    # 2. HighVolWR: prior day high vol + wide range
    hvwr_active = spy_vol_ratio > 1.5 and spy_range_ratio > 1.5
    signals['HighVolWR_30min_SPY_filtered'] = {
        'active': hvwr_active,
        'reason': f"Vol ratio {spy_vol_ratio:.1f}x {'>' if spy_vol_ratio>1.5 else '<'} 1.5x, "
                  f"Range ratio {spy_range_ratio:.1f}x {'>' if spy_range_ratio>1.5 else '<'} 1.5x"
                  + (" — also needs wide opening range at entry" if hvwr_active else ""),
    }

    # 3. QQQ Short: prior day strong (close near high)
    qqq_strong = qqq_close_loc > 0.75
    signals['PriorDayStrong_AboveOR_QQQ_short'] = {
        'active': qqq_strong,
        'reason': f"QQQ close location {qqq_close_loc:.0%} {'>' if qqq_strong else '<'} 75% — "
                  + ("strong day, wait for above OR midpoint to short" if qqq_strong
                     else "not strong enough"),
    }

    # 4. SPY Short: prior day strong (close near high)
    spy_strong = spy_close_loc > 0.75
    signals['PriorDayStrong_AboveOR_SPY_short'] = {
        'active': spy_strong,
        'reason': f"SPY close location {spy_close_loc:.0%} {'>' if spy_strong else '<'} 75% — "
                  + ("strong day, wait for above OR midpoint to short" if spy_strong
                     else "not strong enough"),
    }

    # 5. QQQ Weak: prior day weak + regime filter
    qqq_weak = qqq_close_loc < 0.25
    qqq_regime_ok = spy_below_sma20  # uses SPY SMA20 for regime
    signals['PriorDayWeak_30min_QQQ'] = {
        'active': qqq_weak and qqq_regime_ok,
        'reason': f"QQQ close location {qqq_close_loc:.0%} {'<' if qqq_weak else '>'} 25%"
                  + (f", SPY {'below' if spy_below_sma20 else 'above'} SMA20 (${spy_sma20:.2f})"
                     if qqq_weak else ""),
        'regime_blocked': qqq_weak and not qqq_regime_ok,
    }

    # 6. SPY Weak: prior day weak + regime filter
    spy_weak = spy_close_loc < 0.25
    signals['PriorDayWeak_30min_SPY_filtered'] = {
        'active': spy_weak and spy_below_sma20,
        'reason': f"SPY close location {spy_close_loc:.0%} {'<' if spy_weak else '>'} 25%"
                  + (f", SPY {'below' if spy_below_sma20 else 'above'} SMA20 (${spy_sma20:.2f})"
                     if spy_weak else "")
                  + (" — also needs RVOL > 1.5 on entry bar" if (spy_weak and spy_below_sma20) else ""),
        'regime_blocked': spy_weak and not spy_below_sma20,
    }

    # 7. 50Hi Weak: prior day weak + regime filter (intraday trigger)
    signals['PriorDayWeak_50Hi_SPY_filtered'] = {
        'active': spy_weak and spy_below_sma20,
        'reason': f"SPY close location {spy_close_loc:.0%} {'<' if spy_weak else '>'} 25%"
                  + (f", SPY {'below' if spy_below_sma20 else 'above'} SMA20"
                     if spy_weak else "")
                  + (" — watch for 50-bar high + RVOL > 1.5" if (spy_weak and spy_below_sma20) else ""),
        'regime_blocked': spy_weak and not spy_below_sma20,
    }

    return signals, prior_metrics


def build_dashboard(signals, prior_metrics, today):
    """Generate HTML dashboard."""

    active_signals = [k for k, v in signals.items() if v['active']]
    regime_blocked = [k for k, v in signals.items() if v.get('regime_blocked')]
    n_active = len(active_signals)

    # Total premium at risk
    total_premium = sum(STRATEGIES[k]['budget'] for k in active_signals)

    # Status color
    if n_active == 0:
        status_color = '#666'
        status_text = 'NO SIGNALS'
    elif n_active <= 2:
        status_color = '#fbbf24'
        status_text = f'{n_active} SIGNAL{"S" if n_active > 1 else ""} ACTIVE'
    else:
        status_color = '#4ade80'
        status_text = f'{n_active} SIGNALS ACTIVE'

    day_name = datetime.strptime(str(today), '%Y-%m-%d').strftime('%A')

    # Build strategy cards HTML
    cards = ''
    for rule_id, strat in STRATEGIES.items():
        sig = signals.get(rule_id, {})
        active = sig.get('active', False)
        blocked = sig.get('regime_blocked', False)
        hist = HISTORICAL_STATS.get(rule_id, {})

        if active:
            border = strat['color']
            bg = strat['color'] + '15'
            badge = f'<span style="background:{strat["color"]};color:#fff;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700">ACTIVE</span>'
        elif blocked:
            border = '#f59e0b'
            bg = '#f59e0b10'
            badge = '<span style="background:#78350f;color:#fbbf24;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:600">REGIME BLOCKED</span>'
        else:
            border = '#333'
            bg = '#12121e'
            badge = '<span style="color:#555;font-size:11px">INACTIVE</span>'

        dir_badge = f'<span style="background:{"#064e3b" if strat["direction"]=="LONG" else "#4c1d1d"};color:{"#34d399" if strat["direction"]=="LONG" else "#f87171"};padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600">{strat["direction"]}</span>'
        delta_badge = f'<span style="background:#1e1b4b;color:#a5b4fc;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600">&Delta;{strat["delta"]}</span>'

        cards += f'''
        <div style="background:{bg};border:1px solid {border};border-radius:10px;padding:16px;margin-bottom:10px;{'opacity:0.5' if not active and not blocked else ''}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <div>
              <span style="font-size:16px;font-weight:700;color:#fff">{strat['name']}</span>
              {dir_badge} {delta_badge}
              <span style="color:#666;font-size:11px;margin-left:8px">{strat['ticker']}</span>
            </div>
            {badge}
          </div>
          <div style="color:#999;font-size:12px;margin-bottom:8px">{strat['description']}</div>
          <div style="color:#aaa;font-size:11px;margin-bottom:6px"><b>Signal:</b> {sig.get('reason','')}</div>
          {'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:8px;margin-top:10px;padding-top:10px;border-top:1px solid #ffffff10">' if active else ''}
          {'<div><div style="color:#555;font-size:9px;text-transform:uppercase">Entry</div><div style="color:#fff;font-size:13px;font-weight:600">'+strat['entry']+'</div></div>' if active else ''}
          {'<div><div style="color:#555;font-size:9px;text-transform:uppercase">Exit</div><div style="color:#fff;font-size:13px;font-weight:600">'+strat['exit']+'</div></div>' if active else ''}
          {'<div><div style="color:#555;font-size:9px;text-transform:uppercase">Budget</div><div style="color:#fff;font-size:13px;font-weight:600">$'+f"{strat['budget']:,}"+'</div></div>' if active else ''}
          {'<div><div style="color:#555;font-size:9px;text-transform:uppercase">Win Rate</div><div style="color:#4ade80;font-size:13px;font-weight:600">'+f"{hist.get('wr',0):.0f}%"+'</div></div>' if active else ''}
          {'<div><div style="color:#555;font-size:9px;text-transform:uppercase">Avg P&L</div><div style="color:#4ade80;font-size:13px;font-weight:600">$'+f"{hist.get('avg_pnl',0):+,.0f}"+'</div></div>' if active else ''}
          {'<div><div style="color:#555;font-size:9px;text-transform:uppercase">ROI</div><div style="color:#fbbf24;font-size:13px;font-weight:600">'+f"{hist.get('roi',0):+.0f}%"+'</div></div>' if active else ''}
          {'</div>' if active else ''}
        </div>'''

    # Prior day metrics
    pm = prior_metrics
    spy_cl = pm['spy_close_loc']
    spy_cl_label = 'STRONG' if spy_cl > 0.75 else ('WEAK' if spy_cl < 0.25 else 'NEUTRAL')
    spy_cl_color = '#f87171' if spy_cl > 0.75 else ('#4ade80' if spy_cl < 0.25 else '#888')

    qqq_cl = pm['qqq_close_loc']
    qqq_cl_label = 'STRONG' if qqq_cl > 0.75 else ('WEAK' if qqq_cl < 0.25 else 'NEUTRAL')
    qqq_cl_color = '#f87171' if qqq_cl > 0.75 else ('#4ade80' if qqq_cl < 0.25 else '#888')

    regime_color = '#4ade80' if pm['spy_below_sma20'] else '#f87171'
    regime_label = 'BELOW SMA20 (longs OK)' if pm['spy_below_sma20'] else 'ABOVE SMA20 (longs filtered)'

    gap_str = f"{pm['spy_gap_pct']:.2%}" if pm['spy_gap_pct'] is not None else 'N/A (pre-market)'

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>0DTE Signal Dashboard — {today}</title>
<meta http-equiv="refresh" content="60">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0f;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:20px;max-width:900px;margin:0 auto}}
</style>
</head><body>

<div style="text-align:center;margin-bottom:24px">
  <div style="font-size:12px;color:#555;text-transform:uppercase;letter-spacing:1px">0DTE Options Signal Dashboard</div>
  <div style="font-size:28px;font-weight:700;color:#fff;margin:4px 0">{day_name}, {today}</div>
  <div style="display:inline-block;background:{status_color}20;border:1px solid {status_color};color:{status_color};padding:6px 20px;border-radius:20px;font-size:14px;font-weight:700;margin-top:8px">{status_text}</div>
  {f'<div style="color:#888;font-size:13px;margin-top:8px">Total premium at risk: <b style="color:#fff">${total_premium:,}</b></div>' if n_active > 0 else ''}
</div>

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-bottom:24px">
  <div style="background:#12121e;border-radius:8px;padding:10px;text-align:center">
    <div style="font-size:9px;color:#555;text-transform:uppercase">SPY Close</div>
    <div style="font-size:18px;font-weight:700;color:#fff">${pm['spy_close']:.2f}</div>
    <div style="font-size:10px;color:{spy_cl_color}">{spy_cl_label} ({spy_cl:.0%})</div>
  </div>
  <div style="background:#12121e;border-radius:8px;padding:10px;text-align:center">
    <div style="font-size:9px;color:#555;text-transform:uppercase">QQQ Close</div>
    <div style="font-size:18px;font-weight:700;color:#fff">${pm['qqq_close']:.2f}</div>
    <div style="font-size:10px;color:{qqq_cl_color}">{qqq_cl_label} ({qqq_cl:.0%})</div>
  </div>
  <div style="background:#12121e;border-radius:8px;padding:10px;text-align:center">
    <div style="font-size:9px;color:#555;text-transform:uppercase">SPY Gap</div>
    <div style="font-size:18px;font-weight:700;color:#fff">{gap_str}</div>
  </div>
  <div style="background:#12121e;border-radius:8px;padding:10px;text-align:center">
    <div style="font-size:9px;color:#555;text-transform:uppercase">Vol Ratio</div>
    <div style="font-size:18px;font-weight:700;color:#fff">{pm['spy_vol_ratio']:.1f}x</div>
    <div style="font-size:10px;color:#888">vs 20d avg</div>
  </div>
  <div style="background:#12121e;border-radius:8px;padding:10px;text-align:center">
    <div style="font-size:9px;color:#555;text-transform:uppercase">Range Ratio</div>
    <div style="font-size:18px;font-weight:700;color:#fff">{pm['spy_range_ratio']:.1f}x</div>
    <div style="font-size:10px;color:#888">vs 20d avg</div>
  </div>
  <div style="background:#12121e;border-radius:8px;padding:10px;text-align:center">
    <div style="font-size:9px;color:#555;text-transform:uppercase">Regime</div>
    <div style="font-size:13px;font-weight:700;color:{regime_color}">{regime_label}</div>
    <div style="font-size:10px;color:#888">SMA20: ${pm['spy_sma20']:.2f}</div>
  </div>
</div>

<div style="font-size:14px;font-weight:600;color:#fff;margin-bottom:12px">Strategies</div>
{cards}

<div style="text-align:center;margin-top:20px;color:#444;font-size:10px">
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Auto-refreshes every 60s · Prior day: {pm['spy_date']}
</div>

</body></html>'''

    return html


def run():
    today = date.today()
    print(f"Signal Dashboard for {today}")
    print(f"{'='*60}")

    # Fetch data
    print("\nFetching data from Polygon...")
    spy = fetch_daily_bars('SPY', lookback_days=30)
    qqq = fetch_daily_bars('QQQ', lookback_days=30)

    if spy.empty or qqq.empty:
        print("ERROR: Could not fetch data")
        return

    print(f"  SPY: {len(spy)} bars, latest {spy['date'].max()}")
    print(f"  QQQ: {len(qqq)} bars, latest {qqq['date'].max()}")

    # Check signals
    print("\nChecking signals...")
    signals, metrics = check_signals(spy, qqq, today)

    # Print summary
    print(f"\n{'='*60}")
    active = [k for k, v in signals.items() if v['active']]
    blocked = [k for k, v in signals.items() if v.get('regime_blocked')]

    if active:
        print(f"ACTIVE SIGNALS ({len(active)}):")
        for k in active:
            s = STRATEGIES[k]
            print(f"  {s['name']:<16} {s['direction']:<6} {s['ticker']:<4} D{s['delta']}  ${s['budget']:>6,}  Entry {s['entry']:<16} Exit {s['exit']}")
    else:
        print("NO ACTIVE SIGNALS TODAY")

    if blocked:
        print(f"\nREGIME BLOCKED ({len(blocked)}):")
        for k in blocked:
            s = STRATEGIES[k]
            print(f"  {s['name']:<16} — signal present but SPY above SMA20")

    # Build dashboard
    html = build_dashboard(signals, metrics, today)
    out_path = os.path.join(OUTPUT_DIR, "signal_dashboard.html")
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"\nDashboard saved to {out_path}")


if __name__ == "__main__":
    run()
