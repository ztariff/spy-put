#!/usr/bin/env python3
"""
V9 Calendar Builder — +65% Full Exit, 1-min bar simulation

Simple strategy: exit 100% at +65% profit target, or at 9:50 close if PT not hit.
No trail, no partial exits.

Run: python3 build_calendar.py

Data sources:
  output/options_2strat_v6_costs.csv   - trade list
  output/exit_sim_1min_cache.json      - 1-min bars
  output/pt_5min_cache.json            - 5-min bars (fallback)
  output/recent_trades.json            - new trades with 1-min bars

Output:
  output/options_pnl_calendar_v9.html  - interactive calendar dashboard
"""

import json, csv, math, statistics, os, sys
from collections import defaultdict
from datetime import datetime, timedelta

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# Also try the VM path as fallback
if not os.path.exists(BASE):
    BASE = "/sessions/stoic-brave-ptolemy/mnt/Momentum Trader/output"

COMMISSION = 1.10
PT_MULT = 1.65   # +65% profit target
BASE_TIME = datetime(1900, 1, 1, 9, 30)

# ── Load data ──
with open(os.path.join(BASE, "options_2strat_v6_costs.csv")) as f:
    trades_raw = list(csv.DictReader(f))

with open(os.path.join(BASE, "exit_sim_1min_cache.json")) as f:
    cache_1m = json.load(f)

with open(os.path.join(BASE, "pt_5min_cache.json")) as f:
    cache_5m_pt = json.load(f)

try:
    with open(os.path.join(BASE, "recent_trades.json")) as f:
        recent_trades = json.load(f)
except:
    recent_trades = []

print(f"Loaded {len(trades_raw)} CSV trades, {len(cache_1m)} 1m-cached, {len(cache_5m_pt)} 5m-PT-cached, {len(recent_trades)} recent")

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def time_to_min(t_str):
    try:
        t = datetime.strptime(t_str[:5], "%H:%M")
        return int((t - BASE_TIME).total_seconds() / 60)
    except:
        return -1

def sort_1m_bars(bars_dict):
    result = []
    for t_str, bar in bars_dict.items():
        m = time_to_min(t_str)
        if m < 0:
            continue
        result.append((m, t_str, bar))
    result.sort(key=lambda x: x[0])
    return result

def normalize_5min(raw):
    result = []
    for label, bar in sorted(raw.items(), key=lambda x: x[0]):
        m = time_to_min(label)
        if m < 0:
            continue
        bi = m // 5
        result.append({
            'idx': bi, 'label': label, 'min': m,
            'h': bar.get('h', bar.get('c', 0)),
            'l': bar.get('l', bar.get('c', 0)),
            'c': bar.get('c', 0)
        })
    return sorted(result, key=lambda x: x['idx'])

# ── PT / 9:50 finders ──
def find_pt_1m(sorted_bars, pt_target):
    for m, t_str, bar in sorted_bars:
        if m < 1:  # skip 9:30 bar — entry is at 9:31
            continue
        if bar.get('h', bar.get('c', 0)) >= pt_target:
            return m, t_str, bar
    return None

def find_950_1m(sorted_bars):
    for m, t_str, bar in sorted_bars:
        if t_str[:5] == '09:50':
            return m, t_str, bar
    for m, t_str, bar in sorted_bars:
        if m >= 20:
            return m, t_str, bar
    return None

def find_pt_5m(sorted_5m, pt_target):
    for bar in sorted_5m:
        if bar['idx'] < 1:  # skip 9:30 bar — entry is at 9:31
            continue
        if bar['h'] >= pt_target:
            return bar
    return None

def find_950_5m(sorted_5m):
    for bar in sorted_5m:
        if bar['label'] == '09:50' or bar['idx'] == 4:
            return bar
    return None

# ═══════════════════════════════════════════════════════════════
# SIMULATION — Full exit at PT or at 9:50
# ═══════════════════════════════════════════════════════════════
def sim_full_exit_1m(sorted_bars, entry_px, n_contracts, pt_target):
    """Full exit at exact PT price (limit order fill) or at 9:50 close."""
    pt = find_pt_1m(sorted_bars, pt_target)
    if pt:
        m, t_str, bar = pt
        if m <= 20:
            exit_px = pt_target  # limit order fills at exact PT price
            pnl = (exit_px - entry_px) * n_contracts * 100 - n_contracts * COMMISSION
            return pnl, exit_px, 'profit_target', t_str
    b950 = find_950_1m(sorted_bars)
    if b950:
        m, t_str, bar = b950
        exit_px = bar['c']
        pnl = (exit_px - entry_px) * n_contracts * 100 - n_contracts * COMMISSION
        return pnl, exit_px, 'time_exit_950', t_str
    if sorted_bars:
        _, t_str, bar = sorted_bars[-1]
        exit_px = bar['c']
        pnl = (exit_px - entry_px) * n_contracts * 100 - n_contracts * COMMISSION
        return pnl, exit_px, 'last_bar', t_str
    return 0, entry_px, 'no_data', '??'

def sim_full_exit_5m(sorted_5m, entry_px, n_contracts, pt_target):
    """Full exit at exact PT price (limit order fill) or at 9:50 close — 5-min fallback."""
    pt = find_pt_5m(sorted_5m, pt_target)
    if pt:
        b950 = find_950_5m(sorted_5m)
        if not b950 or pt['idx'] <= b950['idx']:
            exit_px = pt_target  # limit order fills at exact PT price
            pnl = (exit_px - entry_px) * n_contracts * 100 - n_contracts * COMMISSION
            return pnl, exit_px, 'profit_target', pt['label']
    b950 = find_950_5m(sorted_5m)
    if b950:
        exit_px = b950['c']
        pnl = (exit_px - entry_px) * n_contracts * 100 - n_contracts * COMMISSION
        return pnl, exit_px, 'time_exit_950', b950['label']
    exit_px = sorted_5m[-1]['c'] if sorted_5m else entry_px
    pnl = (exit_px - entry_px) * n_contracts * 100 - n_contracts * COMMISSION
    return pnl, exit_px, 'last_bar', sorted_5m[-1]['label'] if sorted_5m else '??'

# ═══════════════════════════════════════════════════════════════
# PROCESS ALL TRADES
# ═══════════════════════════════════════════════════════════════
results = []
stats = {'1m': 0, '5m': 0, 'fallback': 0}

for tr in trades_raw:
    entry_px = float(tr['option_entry_price'])
    v6_recorded_exit = float(tr['option_exit_price'])
    n_contracts = int(tr['num_contracts'])
    v6_recorded_pnl = float(tr['pnl'])
    exit_reason = tr['exit_reason']
    trade_date = tr['trade_date']
    option_ticker = tr['option_ticker']
    key = f"{option_ticker}|{trade_date}"
    ticker_ul = tr['ticker']
    strike = tr['strike']
    direction = tr['direction']
    target_delta = tr['target_delta']
    pt_target = entry_px * PT_MULT

    # Priority: 1-min bars > 5-min bars > recorded fallback
    bars_1m_dict = cache_1m.get(key)
    bars_5m_raw = cache_5m_pt.get(key) if exit_reason == 'profit_target_50' else None

    if bars_1m_dict and len(bars_1m_dict) >= 2:
        sorted_bars = sort_1m_bars(bars_1m_dict)
        if len(sorted_bars) >= 2:
            v9_pnl, v9_exit, v9_reason, v9_time = sim_full_exit_1m(sorted_bars, entry_px, n_contracts, pt_target)
            stats['1m'] += 1
            results.append({
                'trade_date': trade_date, 'ticker': ticker_ul, 'option_ticker': option_ticker,
                'strike': strike, 'direction': direction, 'target_delta': target_delta,
                'entry_px': entry_px,
                'v6_exit_px': v6_recorded_exit, 'v6_pnl': v6_recorded_pnl, 'v6_pnl_recorded': v6_recorded_pnl,
                'v9_exit_px': v9_exit, 'n_contracts': n_contracts,
                'v9_pnl': v9_pnl,
                'exit_type': v9_reason, 'exit_time': v9_time,
                'v6_exit_reason': exit_reason, 'bar_gran': '1m',
            })
            continue

    if bars_5m_raw and len(bars_5m_raw) >= 2:
        bars_5m = normalize_5min(bars_5m_raw)
        if len(bars_5m) >= 2:
            v9_pnl, v9_exit, v9_reason, v9_time = sim_full_exit_5m(bars_5m, entry_px, n_contracts, pt_target)
            stats['5m'] += 1
            results.append({
                'trade_date': trade_date, 'ticker': ticker_ul, 'option_ticker': option_ticker,
                'strike': strike, 'direction': direction, 'target_delta': target_delta,
                'entry_px': entry_px,
                'v6_exit_px': v6_recorded_exit, 'v6_pnl': v6_recorded_pnl, 'v6_pnl_recorded': v6_recorded_pnl,
                'v9_exit_px': v9_exit, 'n_contracts': n_contracts,
                'v9_pnl': v9_pnl,
                'exit_type': v9_reason, 'exit_time': v9_time,
                'v6_exit_reason': exit_reason, 'bar_gran': '5m',
            })
            continue

    # No bar data — use recorded V6
    stats['fallback'] += 1
    results.append({
        'trade_date': trade_date, 'ticker': ticker_ul, 'option_ticker': option_ticker,
        'strike': strike, 'direction': direction, 'target_delta': target_delta,
        'entry_px': entry_px,
        'v6_exit_px': v6_recorded_exit, 'v6_pnl': v6_recorded_pnl, 'v6_pnl_recorded': v6_recorded_pnl,
        'v9_exit_px': v6_recorded_exit, 'n_contracts': n_contracts,
        'v9_pnl': v6_recorded_pnl,
        'exit_type': 'v6_fallback', 'exit_time': tr.get('exit_hm', '??'),
        'v6_exit_reason': exit_reason, 'bar_gran': 'none',
    })

# ── Process recent trades ──
csv_keys = set(f"{tr['option_ticker']}|{tr['trade_date']}" for tr in trades_raw)

for rt in recent_trades:
    key = f"{rt['option_ticker']}|{rt['trade_date']}"
    if key in csv_keys:
        continue

    bars_1m_dict = rt.get('bars_1m', {})
    if not bars_1m_dict or not isinstance(bars_1m_dict, dict) or len(bars_1m_dict) < 2:
        continue

    entry_px = rt.get('option_entry_price') or rt.get('entry_price')
    if not entry_px:
        for t_label in ['09:31', '09:30', '09:32']:
            if t_label in bars_1m_dict:
                entry_px = bars_1m_dict[t_label].get('o') or bars_1m_dict[t_label].get('c')
                if entry_px and entry_px > 0:
                    break
    if not entry_px or entry_px <= 0:
        continue

    n_contracts = rt.get('num_contracts', 100)
    pt_target = entry_px * PT_MULT

    sorted_bars = sort_1m_bars(bars_1m_dict)
    if len(sorted_bars) < 2:
        continue

    v9_pnl, v9_exit, v9_reason, v9_time = sim_full_exit_1m(sorted_bars, entry_px, n_contracts, pt_target)
    v6_pnl, v6_exit, v6_reason, v6_time = sim_full_exit_1m(sorted_bars, entry_px, n_contracts, entry_px * 1.5)
    stats['1m'] += 1

    results.append({
        'trade_date': rt['trade_date'], 'ticker': rt.get('ticker', '??'),
        'option_ticker': rt['option_ticker'],
        'strike': str(rt.get('strike', '')), 'direction': 'short', 'target_delta': str(rt.get('target_delta', '')),
        'entry_px': entry_px,
        'v6_exit_px': v6_exit, 'v6_pnl': v6_pnl, 'v6_pnl_recorded': v6_pnl,
        'v9_exit_px': v9_exit, 'n_contracts': n_contracts,
        'v9_pnl': v9_pnl,
        'exit_type': v9_reason, 'exit_time': v9_time,
        'v6_exit_reason': v6_reason, 'bar_gran': '1m', 'is_new': True,
    })

results.sort(key=lambda r: r['trade_date'])

# ═══════════════════════════════════════════════════════════════
# COMPUTE KPIs
# ═══════════════════════════════════════════════════════════════
total_v9 = sum(r['v9_pnl'] for r in results)
total_v6 = sum(r['v6_pnl'] for r in results)
edge = total_v9 - total_v6
n_trades = len(results)

day_map = defaultdict(list)
for r in results:
    day_map[r['trade_date']].append(r)

day_pnls_v9 = [(d, sum(r['v9_pnl'] for r in day_map[d])) for d in sorted(day_map)]
day_pnls_v6 = [(d, sum(r['v6_pnl'] for r in day_map[d])) for d in sorted(day_map)]

n_days = len(day_pnls_v9)
win_days_v9 = sum(1 for _, p in day_pnls_v9 if p > 0)
win_rate = win_days_v9 / n_days * 100 if n_days else 0
avg_day = total_v9 / n_days if n_days else 0

daily_v9 = [p for _, p in day_pnls_v9]
daily_v6 = [p for _, p in day_pnls_v6]

mn9 = statistics.mean(daily_v9) if daily_v9 else 0
sd9 = statistics.stdev(daily_v9) if len(daily_v9) > 1 else 1
sharpe9 = (mn9 / sd9) * (252**0.5) if sd9 > 0 else 0

mn6 = statistics.mean(daily_v6) if daily_v6 else 0
sd6 = statistics.stdev(daily_v6) if len(daily_v6) > 1 else 1
sharpe6 = (mn6 / sd6) * (252**0.5) if sd6 > 0 else 0

def calc_dd(pnls):
    cum = peak = mdd = 0
    for _, p in pnls:
        cum += p; peak = max(peak, cum); mdd = max(mdd, peak - cum)
    return mdd

max_dd_v9 = calc_dd(day_pnls_v9)
max_dd_v6 = calc_dd(day_pnls_v6)

n_pt_hits = sum(1 for r in results if r['exit_type'] == 'profit_target')
n_time_exits = sum(1 for r in results if r['exit_type'] == 'time_exit_950')
n_fallback = sum(1 for r in results if r['exit_type'] == 'v6_fallback')

# Monthly
monthly_v9 = defaultdict(float)
monthly_v6 = defaultdict(float)
for d, p in day_pnls_v9: monthly_v9[d[:7]] += p
for d, p in day_pnls_v6: monthly_v6[d[:7]] += p

pt_pct = round((PT_MULT - 1) * 100)
print(f"\n{'='*70}")
print(f"V9 CALENDAR — +{pt_pct}% Full Exit, 1-min bar simulation")
print(f"{'='*70}")
print(f"Bar sources: {stats['1m']} on 1-min, {stats['5m']} on 5-min, {stats['fallback']} fallback")
print(f"")
print(f"{'':30s}{'V6 (+50%)':>16s}{'V9 (+' + str(pt_pct) + '%)':>16s}")
print(f"{'-'*62}")
print(f"{'Total P&L':30s}${total_v6:>14,.0f} ${total_v9:>14,.0f}")
print(f"{'Edge':30s}{'':>16s} ${edge:>14,.0f}")
print(f"{'Sharpe':30s}{sharpe6:>15.2f} {sharpe9:>15.2f}")
print(f"{'Max Drawdown':30s}${max_dd_v6:>14,.0f} ${max_dd_v9:>14,.0f}")
print(f"{'Daily Std Dev':30s}${sd6:>14,.0f} ${sd9:>14,.0f}")
print(f"{'Win Day Rate':30s}{sum(1 for p in daily_v6 if p>0)/n_days*100:>14.0f}% {win_rate:>14.0f}%")
print(f"{'Avg Day P&L':30s}${mn6:>14,.0f} ${avg_day:>14,.0f}")
print(f"{'Trades':30s}{n_trades:>15d} {n_trades:>15d}")
print(f"{'Days':30s}{n_days:>15d} {n_days:>15d}")
print(f"V9 Exit Types: {n_pt_hits} PT hits (+{pt_pct}%), {n_time_exits} time exits (9:50), {n_fallback} fallback")

# ═══════════════════════════════════════════════════════════════
# BUILD HTML
# ═══════════════════════════════════════════════════════════════
eq_v9 = []
eq_v6 = []
cum9 = cum6 = 0
for (d9, p9), (d6, p6) in zip(day_pnls_v9, day_pnls_v6):
    cum9 += p9; cum6 += p6
    eq_v9.append({'x': d9, 'y': round(cum9)})
    eq_v6.append({'x': d6, 'y': round(cum6)})

months_sorted = sorted(set(list(monthly_v9.keys()) + list(monthly_v6.keys())))
monthly_data = json.dumps([{'ym': ym, 'v9': round(monthly_v9.get(ym, 0)), 'v6': round(monthly_v6.get(ym, 0))} for ym in months_sorted])

# ── Build bar data lookup for recent trades ──
recent_bars_lookup = {}
for rt in recent_trades:
    key = f"{rt['option_ticker']}|{rt['trade_date']}"
    if rt.get('bars_1m') and isinstance(rt['bars_1m'], dict):
        recent_bars_lookup[key] = rt['bars_1m']

def get_trimmed_bars(option_ticker, trade_date):
    """Get 1-min bars trimmed to 9:31-10:15 window, as compact array."""
    key = f"{option_ticker}|{trade_date}"
    bars_dict = cache_1m.get(key) or recent_bars_lookup.get(key)
    if not bars_dict:
        return []
    trimmed = []
    for t_str in sorted(bars_dict.keys()):
        if t_str[:5] < '09:31' or t_str[:5] > '10:15':
            continue
        b = bars_dict[t_str]
        # Compact: [time, open, high, low, close]
        trimmed.append([
            t_str[:5],
            round(b.get('o', b.get('c', 0)), 4),
            round(b.get('h', b.get('c', 0)), 4),
            round(b.get('l', b.get('c', 0)), 4),
            round(b.get('c', 0), 4),
        ])
    return trimmed

cal_days = {}
for d in sorted(day_map):
    trades_today = day_map[d]
    day_v9 = sum(r['v9_pnl'] for r in trades_today)
    day_v6 = sum(r['v6_pnl'] for r in trades_today)
    cards = []
    for r in trades_today:
        card = {
            'ticker': r['ticker'], 'option': r['option_ticker'],
            'strike': r['strike'], 'delta': r['target_delta'], 'dir': r['direction'],
            'entry': round(r['entry_px'], 4),
            'v6_exit': round(r['v6_exit_px'], 4), 'v9_exit': round(r['v9_exit_px'], 4),
            'contracts': r['n_contracts'],
            'v6_pnl': round(r['v6_pnl']), 'v9_pnl': round(r['v9_pnl']),
            'delta_pnl': round(r['v9_pnl'] - r['v6_pnl']),
            'exit_type': r['exit_type'], 'exit_time': r['exit_time'],
            'v6_reason': r['v6_exit_reason'],
            'gran': r['bar_gran'],
            'bars': get_trimmed_bars(r['option_ticker'], r['trade_date']),
        }
        if r.get('is_new'):
            card['is_new'] = True
        cards.append(card)
    cal_days[d] = {'total_v9': round(day_v9), 'total_v6': round(day_v6), 'trades': cards}

# Gran breakdown for subtitle
gran_1m_pct = stats['1m'] / n_trades * 100 if n_trades else 0

html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>V9 Strategy — +{pt_pct}% Full Exit — P&L Calendar</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0f;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:16px 24px;max-width:1400px;margin:0 auto}}
.hdr{{text-align:center;margin-bottom:20px}}
.hdr h1{{font-size:24px;color:#fff;margin-bottom:4px;letter-spacing:-0.5px}}
.hdr .sub{{color:#666;font-size:12px}}
.hdr .vtag{{display:inline-block;background:#1e3a2a;color:#4ade80;border:1px solid #16a34a;border-radius:6px;padding:3px 10px;font-size:11px;font-weight:600;margin-top:6px;letter-spacing:0.5px}}
.kpi{{display:flex;justify-content:center;gap:4px;margin-bottom:20px;flex-wrap:wrap}}
.kpi .k{{background:#12121e;border-radius:8px;padding:10px 18px;text-align:center;min-width:110px}}
.kpi .k .v{{font-size:20px;font-weight:700}}
.kpi .k .l{{font-size:10px;color:#666;margin-top:2px;text-transform:uppercase;letter-spacing:0.5px}}
.kpi .k .sub2{{font-size:10px;color:#444;margin-top:2px}}
.g{{color:#4ade80}}.r{{color:#f87171}}.w{{color:#fbbf24}}.b{{color:#60a5fa}}.p{{color:#c084fc}}
.eq{{background:#12121e;border-radius:10px;padding:16px;margin-bottom:20px}}
.eq h3{{font-size:13px;color:#888;margin-bottom:8px;font-weight:500}}
.eq .chart-wrap{{position:relative;height:220px;width:100%}}
.mo{{background:#12121e;border-radius:10px;padding:16px;margin-bottom:20px}}
.mo h3{{font-size:13px;color:#888;margin-bottom:8px;font-weight:500}}
.mo .chart-wrap{{position:relative;height:160px;width:100%}}
.nav{{display:flex;justify-content:center;align-items:center;gap:12px;margin-bottom:14px}}
.nav button{{background:#1a1a2e;border:1px solid #2a2a40;color:#ccc;padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12px;transition:all .15s}}
.nav button:hover{{background:#252545;border-color:#444}}
.nav .ym{{font-size:15px;font-weight:600;min-width:280px;text-align:center;color:#fff}}
.cal{{display:grid;grid-template-columns:repeat(7,1fr);gap:2px;margin-bottom:20px}}
.dh{{text-align:center;font-size:10px;color:#555;padding:4px 0;font-weight:600;text-transform:uppercase}}
.dc{{min-height:68px;background:#12121e;border-radius:6px;padding:5px;cursor:pointer;position:relative;transition:all .15s;border:1px solid transparent}}
.dc:hover{{background:#1a1a30;border-color:#333}}
.dc.empty{{background:transparent;cursor:default;border:none}}
.dc.today{{border-color:#fbbf24}}
.dc.newdata{{border-color:#7c3aed}}
.dc .dn{{font-size:10px;color:#555;margin-bottom:1px}}
.dc .dp{{font-size:13px;font-weight:700;text-align:center;margin-top:6px}}
.dc .dt{{font-size:8px;color:#666;text-align:center;margin-top:1px}}
.dc .db{{display:flex;gap:2px;flex-wrap:wrap;margin-top:3px;justify-content:center}}
.dc .db span{{width:6px;height:6px;border-radius:50%;display:inline-block}}
.dc .db .dg{{background:#4ade80}}.dc .db .dr{{background:#f87171}}.dc .db .dp2{{background:#c084fc}}
.modal-bg{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.75);z-index:100;justify-content:center;align-items:flex-start;padding-top:3vh}}
.modal-bg.show{{display:flex}}
.modal{{background:#14142a;border:1px solid #2a2a40;border-radius:14px;padding:0;max-width:1100px;width:96%;max-height:92vh;overflow:hidden;box-shadow:0 25px 60px rgba(0,0,0,.5)}}
.modal-hdr{{padding:18px 24px 14px;border-bottom:1px solid #1e1e35;display:flex;justify-content:space-between;align-items:center}}
.modal-hdr h2{{font-size:16px;color:#fff}}
.modal-hdr .close{{cursor:pointer;font-size:22px;color:#666;padding:0 4px;transition:color .15s}}
.modal-hdr .close:hover{{color:#fff}}
.modal-body{{padding:16px 24px 20px;overflow-y:auto;max-height:calc(90vh - 60px)}}
.day-stats{{display:flex;gap:16px;margin-bottom:14px;flex-wrap:wrap}}
.day-stats .ds{{background:#1a1a30;border-radius:6px;padding:8px 14px}}
.day-stats .ds .dv{{font-size:16px;font-weight:700}}
.day-stats .ds .dl{{font-size:10px;color:#666;text-transform:uppercase}}
.tc{{background:#1a1a2e;border-radius:8px;padding:12px 14px;margin-bottom:8px;border-left:3px solid #333}}
.tc.win{{border-left-color:#4ade80}}.tc.loss{{border-left-color:#f87171}}
.tc-top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
.tc-strat{{font-size:13px;font-weight:600;color:#fff;display:flex;align-items:center;gap:4px;flex-wrap:wrap}}
.tc-pnl{{font-size:15px;font-weight:700}}
.tc-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:6px 12px;font-size:11px}}
.tc-grid .tl{{color:#555;font-size:10px;text-transform:uppercase;margin-bottom:1px}}
.tc-grid .tv{{color:#ccc}}
.tc-grid .tv.hi{{color:#4ade80}}
.candle-wrap{{background:#131722;border-radius:8px;margin-bottom:10px;position:relative;overflow:hidden}}
.candle-wrap .tv-chart{{width:100%;height:420px;position:relative}}
.ohlc-tip{{position:absolute;top:8px;left:12px;background:rgba(19,23,34,0.85);border:1px solid #2a2e3d;border-radius:5px;padding:4px 10px;font-size:12px;pointer-events:none;z-index:10;white-space:nowrap;backdrop-filter:blur(4px)}}
.candle-legend{{display:flex;gap:12px;padding:4px 14px 8px;font-size:10px;color:#666;flex-wrap:wrap;justify-content:center;background:#131722}}
.candle-legend span{{display:flex;align-items:center;gap:4px}}
.candle-legend .dot{{width:8px;height:8px;border-radius:50%;display:inline-block}}
.badge{{display:inline-block;padding:2px 7px;border-radius:3px;font-size:10px;font-weight:600}}
.badge.short{{background:#4c1d1d;color:#f87171}}
.badge.long{{background:#064e3b;color:#34d399}}
.badge.delta{{background:#1e1b4b;color:#a5b4fc}}
.badge.trail-pt{{background:#1a3a1a;color:#4ade80;border:1px solid #16a34a}}
.badge.trail-950{{background:#1a2a3a;color:#60a5fa;border:1px solid #2563eb}}
.badge.partial{{background:#2d1a3a;color:#c084fc;border:1px solid #7c3aed}}
.badge.nodata{{background:#2a2a2a;color:#888}}
.badge.gran-1m{{background:#0a2a0a;color:#4ade80;font-size:9px}}
.badge.gran-5m{{background:#2a2a0a;color:#fbbf24;font-size:9px}}
.v8-legend{{background:#12121e;border-radius:8px;padding:10px 16px;margin-bottom:16px;font-size:11px;color:#888;display:flex;gap:16px;flex-wrap:wrap;align-items:center}}
.v8-legend span{{display:flex;align-items:center;gap:5px}}
.strat-box{{background:#12121e;border-radius:10px;padding:16px;margin-bottom:20px}}
.strat-box h3{{font-size:13px;color:#888;margin-bottom:10px;font-weight:500}}
.strat-desc{{display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:12px}}
.strat-desc .sd{{background:#1a1a2e;border-radius:6px;padding:10px 14px;border-left:3px solid #333}}
.strat-desc .sd h4{{font-size:11px;color:#fff;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px}}
.strat-desc .sd p{{color:#888;line-height:1.5}}
.strat-desc .sd.pt{{border-left-color:#c084fc}}
.strat-desc .sd.trail{{border-left-color:#4ade80}}
</style></head><body>

<div class="hdr">
<h1>0DTE Options — V9 · +{pt_pct}% Full Exit</h1>
<div class="sub">QQQ Short (d0.55) · SPY Short (d0.60) · Full exit at +{pt_pct}% PT or 9:50 close · $1.10/ct commission · {gran_1m_pct:.0f}% on 1-min data</div>
<div class="vtag">V9 · +{pt_pct}% Full Exit · ${total_v9/1e6:.2f}M total · Sharpe {sharpe9:.2f} · {n_trades} trades · {n_days} days</div>
</div>

<div class="kpi">
<div class="k"><div class="v g">${total_v9:,.0f}</div><div class="l">V9 Total P&L</div><div class="sub2">vs V6 ${total_v6:,.0f}</div></div>
<div class="k"><div class="v {'g' if edge >= 0 else 'r'}">${edge:+,.0f}</div><div class="l">V9 Edge</div></div>
<div class="k"><div class="v b">{sharpe9:.2f}</div><div class="l">V9 Sharpe</div><div class="sub2">V6: {sharpe6:.2f}</div></div>
<div class="k"><div class="v r">${max_dd_v9:,.0f}</div><div class="l">V9 Max DD</div><div class="sub2">V6: ${max_dd_v6:,.0f}</div></div>
<div class="k"><div class="v w">${sd9:,.0f}</div><div class="l">V9 Daily Std</div><div class="sub2">V6: ${sd6:,.0f}</div></div>
<div class="k"><div class="v g">{win_rate:.0f}%</div><div class="l">Win Day Rate</div></div>
<div class="k"><div class="v w">${avg_day:,.0f}</div><div class="l">Avg Day</div></div>
<div class="k"><div class="v b">{n_trades}</div><div class="l">Trades</div></div>
</div>

<div class="strat-box">
<h3>V9 Exit Strategy</h3>
<div class="strat-desc">
<div class="sd pt"><h4>When +{pt_pct}% PT Hit</h4><p>Exit <strong>100%</strong> at +{pt_pct}% profit target (bar close).</p></div>
<div class="sd"><h4>When 9:50 (No PT)</h4><p>Exit <strong>100%</strong> at the 9:50 bar close. Simple time-based exit.</p></div>
</div>
</div>

<div class="v9-legend" style="background:#12121e;border-radius:8px;padding:10px 16px;margin-bottom:16px;font-size:11px;color:#888;display:flex;gap:16px;flex-wrap:wrap;align-items:center">
<span style="display:flex;align-items:center;gap:5px"><span class="badge" style="background:#1a3a1a;color:#4ade80;border:1px solid #16a34a">PT +{pt_pct}%</span> Hit profit target</span>
<span style="display:flex;align-items:center;gap:5px"><span class="badge" style="background:#1a2a3a;color:#60a5fa;border:1px solid #2563eb">9:50 Exit</span> Time-based exit</span>
<span style="display:flex;align-items:center;gap:5px"><span class="badge gran-1m">1m</span> 1-min bar data</span>
</div>

<div class="eq"><h3>Equity Curve — V9 (green) vs V6 (grey dashed)</h3><div class="chart-wrap"><canvas id="eqChart"></canvas></div></div>
<div class="mo"><h3>Monthly P&L — V9 vs V6</h3><div class="chart-wrap"><canvas id="moChart"></canvas></div></div>

<div class="nav">
<button onclick="chMo(-12)">&laquo; Year</button>
<button onclick="chMo(-1)">&lsaquo; Mo</button>
<div class="ym" id="ymLabel"></div>
<button onclick="chMo(1)">Mo &rsaquo;</button>
<button onclick="chMo(12)">Year &raquo;</button>
</div>

<div id="calGrid" class="cal"></div>

<div class="modal-bg" id="modalBg" onclick="if(event.target===this)closeModal()">
<div class="modal">
<div class="modal-hdr"><h2 id="modalTitle"></h2><span class="close" onclick="closeModal()">x</span></div>
<div class="modal-body" id="modalBody"></div>
</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<script>
const EQ_V9 = {json.dumps(eq_v9)};
const EQ_V6 = {json.dumps(eq_v6)};
const MO = {monthly_data};
const CAL = {json.dumps(cal_days)};

new Chart(document.getElementById('eqChart').getContext('2d'), {{
  type:'line',
  data:{{
    labels: EQ_V9.map(d=>d.x),
    datasets:[
      {{label:'V9 (+{pt_pct}%)',data:EQ_V9.map(d=>d.y),borderColor:'#4ade80',borderWidth:1.8,pointRadius:0,fill:false,tension:0.1}},
      {{label:'V6 Baseline',data:EQ_V6.map(d=>d.y),borderColor:'#555',borderWidth:1.2,pointRadius:0,fill:false,borderDash:[4,3],tension:0.1}}
    ]
  }},
  options:{{responsive:true,maintainAspectRatio:false,animation:false,
    plugins:{{legend:{{labels:{{color:'#888',font:{{size:10}}}}}}}},
    scales:{{x:{{display:false}},y:{{ticks:{{color:'#666',callback:v=>'$'+(v/1e6).toFixed(1)+'M'}},grid:{{color:'#1a1a2a'}}}}}}
  }}
}});

new Chart(document.getElementById('moChart').getContext('2d'), {{
  type:'bar',
  data:{{
    labels: MO.map(d=>d.ym),
    datasets:[
      {{label:'V9',data:MO.map(d=>d.v9),backgroundColor:MO.map(d=>d.v9>=0?'#4ade80':'#f87171'),borderRadius:3}},
      {{label:'V6',data:MO.map(d=>d.v6),backgroundColor:'rgba(100,100,140,0.35)',borderRadius:3}}
    ]
  }},
  options:{{responsive:true,maintainAspectRatio:false,animation:false,
    plugins:{{legend:{{labels:{{color:'#888',font:{{size:10}}}}}}}},
    scales:{{x:{{ticks:{{color:'#555',font:{{size:8}},maxRotation:90}},grid:{{display:false}}}},
      y:{{ticks:{{color:'#666',callback:v=>'$'+(v/1e3).toFixed(0)+'K'}},grid:{{color:'#1a1a2a'}}}}}}
  }}
}});

const allDates = Object.keys(CAL).sort();
let curY, curM;
if (allDates.length) {{
  const last = allDates[allDates.length-1].split('-');
  curY = parseInt(last[0]); curM = parseInt(last[1]);
}} else {{ curY=2024; curM=1; }}

function chMo(d) {{
  curM += d;
  while(curM > 12) {{ curM -= 12; curY++; }}
  while(curM < 1) {{ curM += 12; curY--; }}
  renderCal();
}}

function renderCal() {{
  const ym = curY+'-'+String(curM).padStart(2,'0');
  let mV9 = 0, mV6 = 0, mTrades = 0;
  for (const [d, info] of Object.entries(CAL)) {{
    if (d.startsWith(ym)) {{ mV9 += info.total_v9; mV6 += info.total_v6; mTrades += info.trades.length; }}
  }}
  const cls = mV9 >= 0 ? 'g' : 'r';
  const delta = mV9 - mV6;
  const dCls = delta >= 0 ? 'g' : 'r';
  document.getElementById('ymLabel').innerHTML =
    new Date(curY, curM-1).toLocaleString('en',{{month:'long',year:'numeric'}}) +
    ` <span class="${{cls}}" style="font-size:13px">V9: ${{mV9>=0?'+':''}}${{(mV9/1e3).toFixed(1)}}K</span>` +
    ` <span style="color:#888;font-size:11px">V6: ${{mV6>=0?'+':''}}${{(mV6/1e3).toFixed(1)}}K</span>` +
    ` <span class="${{dCls}}" style="font-size:10px">(${{delta>=0?'+':''}}${{(delta/1e3).toFixed(1)}}K)</span>` +
    ` <span style="font-size:11px;color:#666">${{mTrades}} trades</span>`;

  const grid = document.getElementById('calGrid');
  grid.innerHTML = '';
  ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'].forEach(d => {{
    grid.innerHTML += `<div class="dh">${{d}}</div>`;
  }});

  const first = new Date(curY, curM-1, 1);
  const daysInMonth = new Date(curY, curM, 0).getDate();
  const startDow = first.getDay();
  for (let i = 0; i < startDow; i++) grid.innerHTML += '<div class="dc empty"></div>';

  const today = new Date().toISOString().split('T')[0];

  for (let day = 1; day <= daysInMonth; day++) {{
    const ds = ym + '-' + String(day).padStart(2,'0');
    const info = CAL[ds];
    const isToday = ds === today;
    const hasNew = info && info.trades.some(t => t.is_new);

    if (!info) {{
      grid.innerHTML += `<div class="dc empty${{isToday?' today':''}}"><div class="dn">${{day}}</div></div>`;
      continue;
    }}

    const p = info.total_v9;
    const pV6 = info.total_v6;
    const pcls = p >= 0 ? 'g' : 'r';
    const dots = info.trades.map(t => {{
      return t.v9_pnl >= 0 ? '<span class="dg"></span>' : '<span class="dr"></span>';
    }}).join('');

    const edg = p - pV6;
    const eCls = edg >= 0 ? 'g' : 'r';

    grid.innerHTML += `<div class="dc${{isToday?' today':''}}${{hasNew?' newdata':''}}" onclick="showDay('${{ds}}')">
      <div class="dn">${{day}}</div>
      <div class="dp ${{pcls}}">${{p>=0?'+':''}}${{Math.abs(p)>=1000?(p/1e3).toFixed(1)+'K':Math.round(p)}}</div>
      <div class="dt">V6: ${{pV6>=0?'+':''}}${{Math.abs(pV6)>=1000?(pV6/1e3).toFixed(1)+'K':Math.round(pV6)}}</div>
      <div class="db">${{dots}}</div>
    </div>`;
  }}
}}

// ── TradingView Lightweight Charts renderer ──
let _chartId = 0;
const _tvCharts = [];  // track for cleanup

function _destroyCharts() {{
  for (const ch of _tvCharts) {{ try {{ ch.remove(); }} catch(e) {{}} }}
  _tvCharts.length = 0;
}}

function drawCandles(containerId, bars, trade, ds) {{
  const el = document.getElementById(containerId);
  if (!el || !bars || bars.length < 2) return;

  // Convert bar times to Unix timestamps
  // bars: [[HH:MM, o, h, l, c], ...]
  // Store ET times directly as UTC so getUTCHours() returns the ET hour
  const [yr, mo, dy] = ds.split('-').map(Number);
  function toTs(timeStr) {{
    const [h, m] = timeStr.split(':').map(Number);
    return Math.floor(Date.UTC(yr, mo - 1, dy, h, m, 0) / 1000);
  }}

  const candleData = bars.map(b => ({{
    time: toTs(b[0]),
    open: b[1], high: b[2], low: b[3], close: b[4]
  }}));

  // Create chart
  const chart = LightweightCharts.createChart(el, {{
    width: el.clientWidth,
    height: 420,
    layout: {{ background: {{ color: '#131722' }}, textColor: '#DDD' }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    handleScroll: {{ mouseWheel: false }},
    handleScale: {{ mouseWheel: false }},
    timeScale: {{
      borderColor: '#2a2e3d',
      timeVisible: true,
      secondsVisible: false,
    }},
    rightPriceScale: {{ borderColor: '#2a2e3d', mode: 0 }},
    grid: {{
      vertLines: {{ color: 'rgba(255,255,255,0.04)' }},
      horzLines: {{ color: 'rgba(255,255,255,0.04)' }},
    }},
    localization: {{
      timeFormatter: (ts) => {{
        const d = new Date(ts * 1000);
        const h = d.getUTCHours().toString().padStart(2, '0');
        const m = d.getUTCMinutes().toString().padStart(2, '0');
        return h + ':' + m;
      }},
    }},
  }});
  _tvCharts.push(chart);

  const candleSeries = chart.addCandlestickSeries({{
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    lastValueVisible: false, priceLineVisible: false,
  }});
  candleSeries.setData(candleData);

  // ── Price lines: Entry and +50% PT ──
  const ptLevel = trade.entry * {PT_MULT};
  candleSeries.createPriceLine({{
    price: trade.entry,
    color: '#2962ff',
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    axisLabelVisible: true,
    title: 'Entry',
  }});
  candleSeries.createPriceLine({{
    price: ptLevel,
    color: '#c084fc',
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    axisLabelVisible: true,
    title: '+{pt_pct}% PT',
  }});

  // ── Markers ──
  const markers = [];

  // Helper: find timestamp for a time string (exact or first bar >=)
  function findTs(timeStr) {{
    for (const b of bars) {{
      if (b[0] >= timeStr) return toTs(b[0]);
    }}
    return null;
  }}

  // Entry marker at 9:31
  const entryTs = findTs('09:31');
  if (entryTs) {{
    markers.push({{
      time: entryTs,
      position: 'belowBar',
      color: '#2962ff',
      shape: 'arrowUp',
      text: 'ENTRY $' + trade.entry.toFixed(2),
    }});
  }}

  // 9:50 vertical dashed line (drawn as DOM overlay)
  const ts950 = findTs('09:50');
  if (ts950) {{
    const updateLine950 = () => {{
      let lineEl = el.querySelector('.line-950');
      const x = chart.timeScale().timeToCoordinate(ts950);
      if (x === null || x < 0) {{
        if (lineEl) lineEl.style.display = 'none';
        return;
      }}
      if (!lineEl) {{
        lineEl = document.createElement('div');
        lineEl.className = 'line-950';
        lineEl.style.cssText = 'position:absolute;top:0;width:0;height:100%;border-left:1.5px dashed #fbbf2488;pointer-events:none;z-index:5;';
        const label = document.createElement('div');
        label.style.cssText = 'position:absolute;top:4px;left:4px;font-size:10px;color:#fbbf24;white-space:nowrap;';
        label.textContent = '9:50';
        lineEl.appendChild(label);
        el.appendChild(lineEl);
      }}
      lineEl.style.display = '';
      lineEl.style.left = x + 'px';
    }};
    chart.timeScale().subscribeVisibleLogicalRangeChange(updateLine950);
    setTimeout(updateLine950, 50);
  }}

  // Exit marker
  const exitTimeStr = (trade.exit_time || '').substring(0, 5);
  const exitTs = findTs(exitTimeStr);
  if (exitTs) {{
    const isPT = trade.exit_type === 'profit_target';
    markers.push({{
      time: exitTs,
      position: 'aboveBar',
      color: isPT ? '#26a69a' : '#ef5350',
      shape: 'arrowDown',
      text: (isPT ? 'PT EXIT $' : 'EXIT $') + trade.v9_exit.toFixed(2),
    }});
  }}

  // Sort markers by time (required by lightweight-charts)
  markers.sort((a, b) => a.time - b.time);
  candleSeries.setMarkers(markers);

  // Fit content
  chart.timeScale().fitContent();

  // Watermark with ticker info
  chart.applyOptions({{ watermark: {{
    visible: true, fontSize: 32, horzAlign: 'center', vertAlign: 'center',
    color: 'rgba(171,71,188,0.15)',
    text: trade.ticker + ' · ' + trade.strike,
  }} }});

  // ── OHLC tooltip on crosshair hover ──
  const tooltip = document.createElement('div');
  tooltip.className = 'ohlc-tip';
  tooltip.style.display = 'none';
  el.appendChild(tooltip);

  chart.subscribeCrosshairMove((param) => {{
    if (!param || !param.time || !param.seriesData) {{
      tooltip.style.display = 'none';
      return;
    }}
    const candle = param.seriesData.get(candleSeries);
    if (!candle) {{ tooltip.style.display = 'none'; return; }}

    const isUp = candle.close >= candle.open;
    const cls = isUp ? 'color:#26a69a' : 'color:#ef5350';
    const pf = (v) => v != null ? '$' + v.toFixed(2) : '-';
    const d = new Date(param.time * 1000);
    const h = d.getUTCHours().toString().padStart(2, '0');
    const m = d.getUTCMinutes().toString().padStart(2, '0');

    tooltip.innerHTML = `<span style="color:#888">${{h}}:${{m}}</span> &nbsp;`
      + `<span style="color:#888">O</span> <span style="${{cls}}">${{pf(candle.open)}}</span> &nbsp;`
      + `<span style="color:#888">H</span> <span style="${{cls}}">${{pf(candle.high)}}</span> &nbsp;`
      + `<span style="color:#888">L</span> <span style="${{cls}}">${{pf(candle.low)}}</span> &nbsp;`
      + `<span style="color:#888">C</span> <span style="${{cls}}">${{pf(candle.close)}}</span>`;
    tooltip.style.display = 'block';
  }});

  // Resize observer
  const ro = new ResizeObserver(() => {{
    chart.applyOptions({{ width: el.clientWidth }});
  }});
  ro.observe(el);
}}

function showDay(ds) {{
  _destroyCharts();
  const info = CAL[ds];
  if (!info) return;
  document.getElementById('modalTitle').textContent = new Date(ds+'T12:00:00').toLocaleDateString('en',{{weekday:'long',month:'long',day:'numeric',year:'numeric'}});

  const tCls = info.total_v9 >= 0 ? 'g' : 'r';
  const v6Cls = info.total_v6 >= 0 ? 'g' : 'r';
  const edg = info.total_v9 - info.total_v6;
  const eCls = edg >= 0 ? 'g' : 'r';

  let html = `<div class="day-stats">
    <div class="ds"><div class="dv ${{tCls}}">${{info.total_v9>=0?'+':''}}$${{info.total_v9.toLocaleString()}}</div><div class="dl">V9 Day P&L</div></div>
    <div class="ds"><div class="dv ${{v6Cls}}">${{info.total_v6>=0?'+':''}}$${{info.total_v6.toLocaleString()}}</div><div class="dl">V6 Day P&L</div></div>
    <div class="ds"><div class="dv ${{eCls}}">${{edg>=0?'+':''}}$${{edg.toLocaleString()}}</div><div class="dl">V9 Edge</div></div>
    <div class="ds"><div class="dv b">${{info.trades.length}}</div><div class="dl">Trades</div></div>
  </div>`;

  const chartIds = [];

  for (let ti = 0; ti < info.trades.length; ti++) {{
    const t = info.trades[ti];
    const tCls = t.v9_pnl >= 0 ? 'win' : 'loss';
    const pCls = t.v9_pnl >= 0 ? 'g' : 'r';
    const dPnl = t.delta_pnl;
    const dCls = dPnl >= 0 ? 'g' : 'r';
    const granBadge = t.gran === '1m' ? '<span class="badge gran-1m">1m</span>' : '<span class="badge gran-5m">5m</span>';

    let badgeHtml = '';
    if (t.exit_type === 'profit_target') badgeHtml = `<span class="badge" style="background:#1a3a1a;color:#4ade80;border:1px solid #16a34a">PT +{pt_pct}%</span>`;
    else if (t.exit_type === 'time_exit_950') badgeHtml = `<span class="badge" style="background:#1a2a3a;color:#60a5fa;border:1px solid #2563eb">9:50 Exit</span>`;
    else badgeHtml = `<span class="badge nodata">Fallback</span>`;

    const cid = 'candle_' + (++_chartId);
    if (t.bars && t.bars.length > 2) chartIds.push({{id: cid, bars: t.bars, trade: t, ds: ds}});

    const chartHtml = (t.bars && t.bars.length > 2) ? `
      <div class="candle-wrap">
        <div id="${{cid}}" class="tv-chart"></div>
        <div class="candle-legend">
          <span style="color:#2962ff">▼ Entry</span>
          <span style="color:#c084fc">— +{pt_pct}% PT</span>
          <span style="color:#fbbf24">— 9:50</span>
          <span style="color:#26a69a">▼ PT Exit</span>
          <span style="color:#ef5350">▼ Time Exit</span>
        </div>
      </div>` : '';

    html += `<div class="tc ${{tCls}}">
      <div class="tc-top">
        <div class="tc-strat">
          <span class="badge ${{t.dir}}">${{t.dir.toUpperCase()}}</span>
          ${{badgeHtml}} ${{granBadge}}
          <span style="color:#ccc">${{t.ticker}} · ${{t.strike}}</span>
        </div>
        <div class="tc-pnl ${{pCls}}">${{t.v9_pnl>=0?'+':''}}$${{t.v9_pnl.toLocaleString()}}</div>
      </div>
      ${{chartHtml}}
      <div class="tc-grid">
        <div><div class="tl">Entry</div><div class="tv">$${{t.entry.toFixed(2)}}</div></div>
        <div><div class="tl">V9 Exit</div><div class="tv">$${{t.v9_exit.toFixed(2)}}</div></div>
        <div><div class="tl">V6 Exit</div><div class="tv" style="color:#666">$${{t.v6_exit.toFixed(2)}}</div></div>
        <div><div class="tl">Contracts</div><div class="tv">${{t.contracts}}</div></div>
        <div><div class="tl">V9 P&L</div><div class="tv ${{pCls}}">${{t.v9_pnl>=0?'+':''}}$${{t.v9_pnl.toLocaleString()}}</div></div>
        <div><div class="tl">V6 P&L</div><div class="tv">${{t.v6_pnl>=0?'+':''}}$${{t.v6_pnl.toLocaleString()}}</div></div>
        <div><div class="tl">Entry Time</div><div class="tv">09:31</div></div>
        <div><div class="tl">Exit Time</div><div class="tv">${{t.exit_time}}</div></div>
        <div><div class="tl">V9 vs V6</div><div class="tv ${{dCls}}">${{dPnl>=0?'+':''}}$${{dPnl.toLocaleString()}}</div></div>
      </div>
    </div>`;
  }}

  document.getElementById('modalBody').innerHTML = html;
  document.getElementById('modalBg').classList.add('show');

  // Render candlestick charts after DOM is ready
  requestAnimationFrame(() => {{
    for (const c of chartIds) {{
      drawCandles(c.id, c.bars, c.trade, c.ds);
    }}
  }});
}}

function closeModal() {{ _destroyCharts(); document.getElementById('modalBg').classList.remove('show'); }}
document.addEventListener('keydown', e => {{ if(e.key==='Escape') closeModal(); }});
renderCal();
</script>
</body></html>'''

out_path = os.path.join(BASE, "options_pnl_calendar_v9.html")
with open(out_path, 'w') as f:
    f.write(html)
print(f"\nSaved: {out_path}")
print(f"File size: {len(html)/1024:.0f} KB")
