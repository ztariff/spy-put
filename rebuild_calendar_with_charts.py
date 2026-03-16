#!/usr/bin/env python3
"""
Rebuild Calendar with Intraday Charts
======================================
Regenerates the options P&L calendar with intraday price charts
embedded in the day modal. When you click a day, you see:
  - Intraday SPY/QQQ 5-min charts
  - Entry points marked with green arrows (longs) / red arrows (shorts)
  - Exit points marked with white dots
  - Trade P&L annotations

Reads 5-min equity bars from data/5m/*.parquet and the regime-filtered trades.

Usage:
    python rebuild_calendar_with_charts.py
"""
import os
import sys
import json
import base64
import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from collections import defaultdict

from src.config import OUTPUT_DIR, DATA_DIR

# ── Load Data ────────────────────────────────────────────────────────────────

INPUT_CSV = os.path.join(OUTPUT_DIR, "options_eqloss20k_pt50.csv")

def load_trades():
    df = pd.read_csv(INPUT_CSV)
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    return df

def load_5m_bars():
    """Load 5-min equity bars for SPY and QQQ."""
    bars = {}
    for ticker in ['SPY', 'QQQ']:
        path = os.path.join(DATA_DIR, '5m', f'{ticker}.parquet')
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Ensure timezone-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York')
            else:
                df.index = df.index.tz_convert('America/New_York')
            bars[ticker] = df
            print(f"  {ticker}: {len(df)} bars, {df.index.min().date()} to {df.index.max().date()}")
        else:
            print(f"  WARNING: {path} not found")
    return bars

def get_intraday_bars(bars_dict, ticker, trade_date):
    """Extract intraday 5-min bars for a specific date."""
    if ticker not in bars_dict:
        return None
    df = bars_dict[ticker]
    mask = df.index.date == trade_date
    day_bars = df[mask].copy()
    if day_bars.empty:
        return None
    # Filter to regular trading hours (9:30 - 16:00)
    day_bars = day_bars.between_time('09:30', '16:00')
    return day_bars


# ── Build Calendar Data ──────────────────────────────────────────────────────

def build_calendar_data(trades, bars_dict):
    """Build the calendar JSON including intraday chart data."""

    # Group trades by date
    daily_trades = defaultdict(list)
    for _, row in trades.iterrows():
        td = row['trade_date'].strftime('%Y-%m-%d')

        # Parse entry/exit times as HH:MM strings
        et = row['entry_time']
        xt = row['exit_time']
        entry_hhmm = et.strftime('%H:%M') if pd.notna(et) else '09:31'
        exit_hhmm = xt.strftime('%H:%M') if pd.notna(xt) else '15:00'

        # Strategy short name
        rule = row['rule']
        name_map = {
            'GapLarge_First30min_SPY': 'GapLarge',
            'HighVolWR_30min_SPY_filtered': 'HighVolWR',
            'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
            'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
            'PriorDayWeak_30min_QQQ': 'QQQ Weak',
            'PriorDayWeak_30min_SPY_filtered': 'SPY Weak',
            'PriorDayWeak_50Hi_SPY_filtered': '50Hi Weak',
        }
        sname = name_map.get(rule, rule)
        direction = row['direction']
        ticker = row['ticker']

        # Option label
        try:
            ot = row['option_ticker']
            strike = row['strike']
            expiry = row['expiry_date']
            otype = row['option_type']
            olabel = f"{ticker} {expiry} ${strike:.0f} {otype[0] if isinstance(otype, str) else 'C'}"
        except:
            olabel = str(row.get('option_ticker', ''))

        exit_reason = str(row.get('exit_reason', ''))
        is_pt = 'profit_target' in exit_reason

        daily_trades[td].append({
            'r': sname,
            'd': direction,
            'dl': row['target_delta'],
            'o': olabel,
            'ep': round(row['option_entry_price'], 2),
            'xp': round(row['option_exit_price'], 2),
            'et': et.strftime('%I:%M %p').lstrip('0') if pd.notna(et) else '',
            'xt': xt.strftime('%I:%M %p').lstrip('0') if pd.notna(xt) else '',
            'c': int(row['num_contracts']),
            'pr': int(row['premium_paid']),
            'pnl': round(row['pnl']),
            'pp': round(row['pnl_pct'] * 100, 1),
            'tk': ticker,
            'ehm': entry_hhmm,
            'xhm': exit_hhmm,
            'pt': 1 if is_pt else 0,
        })

    # Build daily data dict
    D = {}
    for dk, tlist in daily_trades.items():
        D[dk] = {
            'p': sum(t['pnl'] for t in tlist),
            't': tlist,
        }

    # Build intraday chart data per date per ticker
    # Only include tickers that have trades on that day
    C = {}  # date -> { ticker: [[time_minutes, open, high, low, close], ...] }
    all_dates = sorted(daily_trades.keys())
    print(f"\nBuilding intraday chart data for {len(all_dates)} trading days...")

    for dk in all_dates:
        td = datetime.strptime(dk, '%Y-%m-%d').date()
        tickers_today = set(t['tk'] for t in daily_trades[dk])
        chart_data = {}

        for ticker in tickers_today:
            day_bars = get_intraday_bars(bars_dict, ticker, td)
            if day_bars is None or day_bars.empty:
                continue

            # Convert to list of [minutes_since_930, open, high, low, close]
            candles = []
            for ts, bar in day_bars.iterrows():
                minutes = (ts.hour * 60 + ts.minute) - (9 * 60 + 30)
                if minutes < 0:
                    continue
                candles.append([
                    minutes,
                    round(float(bar['open']), 2),
                    round(float(bar['high']), 2),
                    round(float(bar['low']), 2),
                    round(float(bar['close']), 2),
                ])

            if candles:
                chart_data[ticker] = candles

        if chart_data:
            C[dk] = chart_data

    print(f"  Chart data available for {len(C)}/{len(all_dates)} days")

    # Monthly stats
    monthly = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'wins': 0})
    for dk, dd in D.items():
        mk = dk[:7]
        monthly[mk]['pnl'] += dd['p']
        monthly[mk]['trades'] += len(dd['t'])
        monthly[mk]['wins'] += sum(1 for t in dd['t'] if t['pnl'] > 0)

    M = {mk: {'pnl': v['pnl'], 'trades': v['trades'], 'wins': v['wins']}
         for mk, v in monthly.items()}

    return D, M, C


# ── Generate HTML ────────────────────────────────────────────────────────────

def build_html(D, M, C, trades):
    """Generate the complete HTML calendar with intraday charts."""

    # KPI calculations
    all_pnls = [dd['p'] for dd in D.values()]
    total_pnl = sum(all_pnls)
    n_trades = sum(len(dd['t']) for dd in D.values())
    n_days = len(D)
    win_days = sum(1 for p in all_pnls if p > 0)
    avg_day = total_pnl / n_days if n_days > 0 else 0
    total_premium = trades['premium_paid'].sum()
    total_contracts = int(trades['num_contracts'].sum())

    # Max drawdown
    cum = 0
    peak = 0
    max_dd = 0
    for dk in sorted(D.keys()):
        cum += D[dk]['p']
        if cum > peak:
            peak = cum
        dd = cum - peak
        if dd < max_dd:
            max_dd = dd

    # Encode data
    d_json = json.dumps(D, separators=(',', ':'))
    m_json = json.dumps(M, separators=(',', ':'))
    c_json = json.dumps(C, separators=(',', ':'))

    d_b64 = base64.b64encode(d_json.encode()).decode()
    m_b64 = base64.b64encode(m_json.encode()).decode()
    c_b64 = base64.b64encode(c_json.encode()).decode()

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>0DTE Options Strategy — P&L Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0f;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:16px 24px;max-width:1400px;margin:0 auto}}
.hdr{{text-align:center;margin-bottom:20px}}
.hdr h1{{font-size:24px;color:#fff;margin-bottom:4px;letter-spacing:-0.5px}}
.hdr .sub{{color:#666;font-size:12px}}
/* KPI strip */
.kpi{{display:flex;justify-content:center;gap:4px;margin-bottom:20px;flex-wrap:wrap}}
.kpi .k{{background:#12121e;border-radius:8px;padding:10px 18px;text-align:center;min-width:110px}}
.kpi .k .v{{font-size:20px;font-weight:700}}
.kpi .k .l{{font-size:10px;color:#666;margin-top:2px;text-transform:uppercase;letter-spacing:0.5px}}
.g{{color:#4ade80}}.r{{color:#f87171}}.w{{color:#fbbf24}}.b{{color:#60a5fa}}
/* Equity curve */
.eq{{background:#12121e;border-radius:10px;padding:16px;margin-bottom:20px}}
.eq h3{{font-size:13px;color:#888;margin-bottom:8px;font-weight:500}}
.eq canvas{{width:100%;height:220px}}
/* Monthly bar chart */
.mo{{background:#12121e;border-radius:10px;padding:16px;margin-bottom:20px}}
.mo h3{{font-size:13px;color:#888;margin-bottom:8px;font-weight:500}}
.mo canvas{{width:100%;height:160px}}
/* Nav */
.nav{{display:flex;justify-content:center;align-items:center;gap:12px;margin-bottom:14px}}
.nav button{{background:#1a1a2e;border:1px solid #2a2a40;color:#ccc;padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12px;transition:all .15s}}
.nav button:hover{{background:#252545;border-color:#444}}
.nav .ym{{font-size:15px;font-weight:600;min-width:280px;text-align:center;color:#fff}}
/* Calendar grid */
.cal{{display:grid;grid-template-columns:repeat(7,1fr);gap:2px;margin-bottom:20px}}
.dh{{text-align:center;font-size:10px;color:#555;padding:4px 0;font-weight:600;text-transform:uppercase}}
.dc{{min-height:68px;background:#12121e;border-radius:6px;padding:5px;cursor:pointer;position:relative;transition:all .15s;border:1px solid transparent}}
.dc:hover{{background:#1a1a30;border-color:#333}}
.dc.empty{{background:transparent;cursor:default;border:none}}
.dc.today{{border-color:#fbbf24}}
.dc .dn{{font-size:10px;color:#555;margin-bottom:1px}}
.dc .dp{{font-size:13px;font-weight:700;text-align:center;margin-top:6px}}
.dc .dt{{font-size:8px;color:#666;text-align:center;margin-top:1px}}
.dc .db{{display:flex;gap:2px;flex-wrap:wrap;margin-top:3px;justify-content:center}}
.dc .db span{{width:6px;height:6px;border-radius:50%;display:inline-block}}
.dc .db .dg{{background:#4ade80}}.dc .db .dr{{background:#f87171}}
/* Modal */
.modal-bg{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.75);z-index:100;justify-content:center;align-items:flex-start;padding-top:3vh}}
.modal-bg.show{{display:flex}}
.modal{{background:#14142a;border:1px solid #2a2a40;border-radius:14px;padding:0;max-width:960px;width:95%;max-height:90vh;overflow:hidden;box-shadow:0 25px 60px rgba(0,0,0,.5)}}
.modal-hdr{{padding:18px 24px 14px;border-bottom:1px solid #1e1e35;display:flex;justify-content:space-between;align-items:center}}
.modal-hdr h2{{font-size:16px;color:#fff}}
.modal-hdr .close{{cursor:pointer;font-size:22px;color:#666;padding:0 4px;transition:color .15s}}
.modal-hdr .close:hover{{color:#fff}}
.modal-body{{padding:16px 24px 20px;overflow-y:auto;max-height:calc(90vh - 60px)}}
/* Day summary in modal */
.day-stats{{display:flex;gap:16px;margin-bottom:14px;flex-wrap:wrap}}
.day-stats .ds{{background:#1a1a30;border-radius:6px;padding:8px 14px}}
.day-stats .ds .dv{{font-size:16px;font-weight:700}}
.day-stats .ds .dl{{font-size:10px;color:#666;text-transform:uppercase}}
/* Chart container */
.chart-wrap{{margin-bottom:16px}}
.chart-wrap h4{{font-size:12px;color:#888;margin-bottom:6px;font-weight:500}}
.chart-canvas{{background:#0d0d1a;border-radius:8px;border:1px solid #1e1e35}}
/* Trade cards */
.tc{{background:#1a1a2e;border-radius:8px;padding:12px 14px;margin-bottom:8px;border-left:3px solid #333}}
.tc.win{{border-left-color:#4ade80}}
.tc.loss{{border-left-color:#f87171}}
.tc-top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}}
.tc-strat{{font-size:13px;font-weight:600;color:#fff}}
.tc-pnl{{font-size:15px;font-weight:700}}
.tc-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:4px 12px;font-size:11px}}
.tc-grid .tl{{color:#555;font-size:10px;text-transform:uppercase}}
.tc-grid .tv{{color:#ccc}}
.badge{{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;margin-left:6px}}
.badge.long{{background:#064e3b;color:#34d399}}
.badge.short{{background:#4c1d1d;color:#f87171}}
.badge.delta{{background:#1e1b4b;color:#a5b4fc;margin-left:4px}}
.badge.pt{{background:#4a3728;color:#fbbf24;margin-left:4px}}
</style></head><body>
<div class="hdr">
<h1>0DTE Options Strategy Dashboard</h1>
<div class="sub">7 Strategies · EqLoss $20K Sizing · +50% Profit Target · Per-Rule Optimal Exits · Regime-Filtered Longs (Below SMA20) · Click any day for intraday charts</div>
</div>
<div class="kpi">
<div class="k"><div class="v g">$+{total_pnl:,.0f}</div><div class="l">Total P&L</div></div>
<div class="k"><div class="v b">{n_trades:,}</div><div class="l">Trades</div></div>
<div class="k"><div class="v">{n_days}</div><div class="l">Trading Days</div></div>
<div class="k"><div class="v g">{win_days/n_days*100:.0f}%</div><div class="l">Win Day Rate</div></div>
<div class="k"><div class="v w">$+{avg_day:,.0f}</div><div class="l">Avg Day</div></div>
<div class="k"><div class="v">${total_premium:,.0f}</div><div class="l">Total Premium</div></div>
<div class="k"><div class="v r">${abs(max_dd):,.0f}</div><div class="l">Max Drawdown</div></div>
<div class="k"><div class="v">{total_contracts:,}</div><div class="l">Contracts</div></div>
</div>
<div class="eq"><h3>Cumulative Equity Curve</h3><canvas id="eqChart"></canvas></div>
<div class="mo"><h3>Monthly P&L</h3><canvas id="moChart"></canvas></div>
<div class="nav">
<button onclick="chMo(-12)">&laquo; Year</button>
<button onclick="chMo(-1)">&lsaquo; Mo</button>
<div class="ym" id="ymLabel"></div>
<button onclick="chMo(1)">Mo &rsaquo;</button>
<button onclick="chMo(12)">Year &raquo;</button>
</div>
<div class="cal" id="cal"></div>
<div class="modal-bg" id="modalBg" onclick="if(event.target===this)this.classList.remove('show')">
<div class="modal">
<div class="modal-hdr"><h2 id="modalTitle"></h2><span class="close" onclick="document.getElementById('modalBg').classList.remove('show')">&times;</span></div>
<div class="modal-body" id="modalBody"></div>
</div>
</div>
<script>
const D=JSON.parse(atob("{d_b64}"));
const M=JSON.parse(atob("{m_b64}"));
const CH=JSON.parse(atob("{c_b64}"));
let cY,cM;
const MN=["January","February","March","April","May","June","July","August","September","October","November","December"];
const MNS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const fmt=v=>v>=0?'$+'+Math.abs(v).toLocaleString():'$-'+Math.abs(v).toLocaleString();

function init(){{const ks=Object.keys(D).sort();const last=ks[ks.length-1].split('-');cY=+last[0];cM=+last[1]-1;render()}}

function chMo(d){{cM+=d;while(cM>11){{cM-=12;cY++}}while(cM<0){{cM+=12;cY--}}render()}}

function render(){{
const cal=document.getElementById('cal');
cal.innerHTML='';
['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].forEach(d=>{{
  cal.innerHTML+='<div class="dh">'+d+'</div>';
}});
const fd=new Date(cY,cM,1);const ld=new Date(cY,cM+1,0);
let startDow=fd.getDay();if(startDow===0)startDow=7;
for(let i=1;i<startDow;i++)cal.innerHTML+='<div class="dc empty"></div>';
let mPnl=0,mTrades=0,mWins=0;
for(let d=1;d<=ld.getDate();d++){{
  const dk=cY+'-'+String(cM+1).padStart(2,'0')+'-'+String(d).padStart(2,'0');
  const dd=D[dk];
  const dow=new Date(cY,cM,d).getDay();
  if(dow===0||dow===6){{cal.innerHTML+='<div class="dc empty"></div>';continue}}
  let cls='dc';
  const today=new Date().toISOString().slice(0,10);
  if(dk===today)cls+=' today';
  if(dd){{
    mPnl+=dd.p;mTrades+=dd.t.length;mWins+=dd.t.filter(t=>t.pnl>0).length;
    const pcls=dd.p>=0?'g':'r';
    const dots=dd.t.map(t=>'<span class="'+(t.pnl>=0?'dg':'dr')+'"></span>').join('');
    cal.innerHTML+='<div class="'+cls+'" onclick=\\'showDay("'+dk+'",D["'+dk+'"])\\'>'+
      '<div class="dn">'+d+'</div>'+
      '<div class="dp '+pcls+'">'+fmt(dd.p)+'</div>'+
      '<div class="dt">'+dd.t.length+' trade'+(dd.t.length>1?'s':'')+'</div>'+
      '<div class="db">'+dots+'</div></div>';
  }}else{{
    cal.innerHTML+='<div class="'+cls+' empty" style="background:#12121e;opacity:0.4"><div class="dn">'+d+'</div></div>';
  }}
}}
const mWR=mTrades>0?Math.round(mWins/mTrades*100)+'%':'—';
document.getElementById('ymLabel').innerHTML=MN[cM]+' '+cY+' &nbsp;|&nbsp; <span class="'+(mPnl>=0?'g':'r')+'">'+fmt(mPnl)+'</span> &nbsp; '+mTrades+' trades &nbsp; WR '+mWR;
}}

function showDay(dk,dd){{
document.getElementById('modalTitle').textContent=dk+' — '+fmt(dd.p)+' ('+dd.t.length+' trades)';
const wins=dd.t.filter(t=>t.pnl>0).length;
const prem=dd.t.reduce((s,t)=>s+t.pr,0);
const con=dd.t.reduce((s,t)=>s+t.c,0);
let h='<div class="day-stats">';
h+='<div class="ds"><div class="dv '+(dd.p>=0?'g':'r')+'">'+fmt(dd.p)+'</div><div class="dl">Day P&L</div></div>';
h+='<div class="ds"><div class="dv">'+dd.t.length+'</div><div class="dl">Trades</div></div>';
h+='<div class="ds"><div class="dv g">'+wins+'/'+dd.t.length+'</div><div class="dl">Wins</div></div>';
h+='<div class="ds"><div class="dv">$'+prem.toLocaleString()+'</div><div class="dl">Premium</div></div>';
h+='<div class="ds"><div class="dv">'+con.toLocaleString()+'</div><div class="dl">Contracts</div></div>';
h+='</div>';

// ── Intraday Charts ──
const chartData=CH[dk];
if(chartData){{
  // Get unique tickers traded today
  const tickers=Object.keys(chartData);
  tickers.forEach(tk=>{{
    const pts=chartData[tk];
    if(!pts||pts.length<2)return;
    // Get trades for this ticker
    const tkTrades=dd.t.filter(t=>t.tk===tk);
    const canvasId='chart_'+tk+'_'+dk.replace(/-/g,'');
    h+='<div class="chart-wrap">';
    h+='<h4>'+tk+' Intraday (5-min)</h4>';
    h+='<canvas id="'+canvasId+'" class="chart-canvas" width="880" height="260"></canvas>';
    h+='</div>';
  }});
}}

// ── Trade Cards ──
h+='<div style="font-size:12px;font-weight:600;color:#888;margin:12px 0 8px;text-transform:uppercase;letter-spacing:0.5px">Trades</div>';
dd.t.sort((a,b)=>b.pnl-a.pnl);
dd.t.forEach(t=>{{
const cls=t.pnl>=0?'win':'loss';
const pcls=t.pnl>=0?'g':'r';
h+='<div class="tc '+cls+'">';
h+='<div class="tc-top"><div class="tc-strat">'+t.r+'<span class="badge '+t.d+'">'+t.d.toUpperCase()+'</span><span class="badge delta">&Delta;'+t.dl+'</span>'+(t.pt?'<span class="badge pt">PT +50%</span>':'')+'</div>';
h+='<div class="tc-pnl '+pcls+'">'+fmt(t.pnl)+' <span style="font-size:11px;font-weight:400">('+(t.pp>=0?'+':'')+t.pp.toFixed(1)+'%)</span></div></div>';
h+='<div class="tc-grid">';
h+='<div><div class="tl">Option</div><div class="tv">'+t.o+'</div></div>';
h+='<div><div class="tl">Entry</div><div class="tv">$'+t.ep.toFixed(2)+' @ '+t.et+'</div></div>';
h+='<div><div class="tl">Exit</div><div class="tv">$'+t.xp.toFixed(2)+' @ '+t.xt+'</div></div>';
h+='<div><div class="tl">Contracts</div><div class="tv">'+t.c.toLocaleString()+'</div></div>';
h+='<div><div class="tl">Premium</div><div class="tv">$'+t.pr.toLocaleString()+'</div></div>';
h+='<div><div class="tl">P&L / Contract</div><div class="tv '+pcls+'">'+fmt(Math.round(t.pnl/t.c))+'</div></div>';
h+='</div></div>';
}});

document.getElementById('modalBody').innerHTML=h;
document.getElementById('modalBg').classList.add('show');

// ── Draw charts after DOM update ──
if(chartData){{
  setTimeout(()=>{{
    Object.keys(chartData).forEach(tk=>{{
      const canvasId='chart_'+tk+'_'+dk.replace(/-/g,'');
      const canvas=document.getElementById(canvasId);
      if(!canvas)return;
      const tkTrades=dd.t.filter(t=>t.tk===tk);
      drawIntradayChart(canvas, chartData[tk], tkTrades, tk);
    }});
  }}, 50);
}}
}}

function drawIntradayChart(canvas, candles, trades, ticker){{
  const dpr=window.devicePixelRatio||1;
  const W=canvas.parentElement.offsetWidth-4;
  const H=260;
  canvas.width=W*dpr;canvas.height=H*dpr;
  canvas.style.width=W+'px';canvas.style.height=H+'px';
  const ctx=canvas.getContext('2d');
  ctx.scale(dpr,dpr);

  const pad={{l:58,r:20,t:24,b:28}};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;

  // candles: [[minutes, open, high, low, close], ...]
  const allHighs=candles.map(c=>c[2]);
  const allLows=candles.map(c=>c[3]);
  const mn=Math.min(...allLows);
  const mx=Math.max(...allHighs);
  const range=mx-mn||1;
  const pxMin=mn-range*0.04;
  const pxMax=mx+range*0.04;
  const totalMin=6.5*60; // 9:30-4:00

  // Candle width based on number of candles
  const candleW=Math.max(2, Math.min(8, (pw/candles.length)*0.7));
  const wickW=1;

  function toX(minutes){{return pad.l+pw*(minutes/totalMin)}}
  function toY(price){{return pad.t+ph*(1-(price-pxMin)/(pxMax-pxMin))}}

  // Background
  ctx.fillStyle='#0d0d1a';
  ctx.fillRect(0,0,W,H);

  // Grid lines
  ctx.strokeStyle='#1a1a2e';ctx.lineWidth=0.5;
  const nGrid=6;
  for(let i=0;i<=nGrid;i++){{
    const y=pad.t+ph*(1-i/nGrid);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
    const val=pxMin+(pxMax-pxMin)*i/nGrid;
    ctx.fillStyle='#555';ctx.font='10px sans-serif';ctx.textAlign='right';
    ctx.fillText('$'+val.toFixed(2),pad.l-6,y+3);
  }}

  // Time labels
  ctx.fillStyle='#444';ctx.font='10px sans-serif';ctx.textAlign='center';
  const timeLabels=[
    [0,'9:30'],[30,'10:00'],[60,'10:30'],[90,'11:00'],[120,'11:30'],
    [150,'12:00'],[180,'12:30'],[210,'1:00'],[240,'1:30'],[270,'2:00'],
    [300,'2:30'],[330,'3:00'],[360,'3:30'],[390,'4:00']
  ];
  timeLabels.forEach(tl=>{{
    const x=toX(tl[0]);
    if(x>=pad.l&&x<=W-pad.r){{
      ctx.fillText(tl[1],x,H-pad.b+14);
      ctx.strokeStyle='#15152a';ctx.lineWidth=0.3;
      ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,H-pad.b);ctx.stroke();
    }}
  }});

  // ── Draw Candlesticks ──
  candles.forEach(c=>{{
    const [min,o,h,l,cl]=c;
    const x=toX(min);
    const bullish=cl>=o;
    const bodyColor=bullish?'#22c55e':'#ef4444';
    const wickColor=bullish?'#16a34a':'#dc2626';

    // Wick (high-low line)
    ctx.beginPath();
    ctx.strokeStyle=wickColor;
    ctx.lineWidth=wickW;
    ctx.moveTo(x,toY(h));
    ctx.lineTo(x,toY(l));
    ctx.stroke();

    // Body (open-close rect)
    const bodyTop=toY(Math.max(o,cl));
    const bodyBot=toY(Math.min(o,cl));
    const bodyH=Math.max(1, bodyBot-bodyTop);

    if(bullish){{
      // Hollow or filled green
      ctx.fillStyle=bodyColor;
      ctx.fillRect(x-candleW/2, bodyTop, candleW, bodyH);
    }}else{{
      // Filled red
      ctx.fillStyle=bodyColor;
      ctx.fillRect(x-candleW/2, bodyTop, candleW, bodyH);
    }}
  }});

  // ── Draw entry/exit markers ──
  // Offset labels vertically to avoid overlap
  let entryLabelSlots=[];
  let exitLabelSlots=[];

  trades.forEach((t,ti)=>{{
    const entryMin=parseTimeToMinutes(t.ehm);
    const exitMin=parseTimeToMinutes(t.xhm);
    const isLong=t.d==='long';
    const isWin=t.pnl>=0;

    const entryPrice=findCandlePriceAtMinute(candles, entryMin);
    const exitPrice=findCandlePriceAtMinute(candles, exitMin);

    if(entryPrice!==null){{
      const ex=toX(entryMin);const ey=toY(entryPrice);

      // Vertical dashed line at entry
      ctx.beginPath();
      ctx.setLineDash([2,2]);
      ctx.strokeStyle=isLong?'rgba(74,222,128,0.4)':'rgba(248,113,113,0.4)';
      ctx.lineWidth=1;
      ctx.moveTo(ex,pad.t);ctx.lineTo(ex,H-pad.b);
      ctx.stroke();ctx.setLineDash([]);

      // Entry arrow
      ctx.beginPath();
      if(isLong){{
        ctx.moveTo(ex,ey+14);ctx.lineTo(ex-7,ey+24);ctx.lineTo(ex+7,ey+24);
        ctx.fillStyle='#4ade80';
      }}else{{
        ctx.moveTo(ex,ey-14);ctx.lineTo(ex-7,ey-24);ctx.lineTo(ex+7,ey-24);
        ctx.fillStyle='#f87171';
      }}
      ctx.closePath();ctx.fill();

      // Entry label with background
      const label=t.r;
      ctx.font='bold 9px sans-serif';
      const tw=ctx.measureText(label).width+8;
      const labelY=isLong?ey+30+ti*14:ey-30-ti*14;
      ctx.fillStyle='rgba(0,0,0,0.7)';
      ctx.fillRect(ex-tw/2,labelY-8,tw,12);
      ctx.fillStyle=isLong?'#4ade80':'#f87171';
      ctx.textAlign='center';
      ctx.fillText(label,ex,labelY);
    }}

    if(exitPrice!==null){{
      const xx=toX(exitMin);const xy=toY(exitPrice);

      // Vertical dashed line at exit
      ctx.beginPath();
      ctx.setLineDash([2,2]);
      ctx.strokeStyle='rgba(255,255,255,0.15)';
      ctx.lineWidth=1;
      ctx.moveTo(xx,pad.t);ctx.lineTo(xx,H-pad.b);
      ctx.stroke();ctx.setLineDash([]);

      // Exit marker: diamond for normal exit, circle for PT exit
      const isPT=t.pt===1;
      if(isPT){{
        ctx.beginPath();
        ctx.arc(xx,xy,6,0,Math.PI*2);
        ctx.fillStyle='#fbbf24';
        ctx.fill();
        ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke();
      }}else{{
        ctx.beginPath();
        ctx.moveTo(xx,xy-6);ctx.lineTo(xx+6,xy);ctx.lineTo(xx,xy+6);ctx.lineTo(xx-6,xy);
        ctx.closePath();
        ctx.fillStyle=isWin?'#4ade80':'#f87171';
        ctx.fill();
        ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke();
      }}

      // Exit P&L label with background
      const plabel=fmt(t.pnl);
      ctx.font='bold 9px sans-serif';
      const tw2=ctx.measureText(plabel).width+8;
      const exitLabelY=xy-12-ti*14;
      ctx.fillStyle='rgba(0,0,0,0.7)';
      ctx.fillRect(xx-tw2/2,exitLabelY-8,tw2,12);
      ctx.fillStyle=isWin?'#4ade80':'#f87171';
      ctx.textAlign='center';
      ctx.fillText(plabel,xx,exitLabelY);
    }}

    // Shaded region between entry and exit
    if(entryPrice!==null&&exitPrice!==null){{
      ctx.fillStyle=isWin?'rgba(74,222,128,0.06)':'rgba(248,113,113,0.06)';
      const x1=toX(entryMin);const x2=toX(exitMin);
      ctx.fillRect(Math.min(x1,x2),pad.t,Math.abs(x2-x1),ph);
    }}
  }});

  // Ticker label
  ctx.font='bold 16px sans-serif';ctx.fillStyle='#fff';ctx.textAlign='left';
  ctx.fillText(ticker,pad.l+8,pad.t+16);

  // Open/Close labels
  if(candles.length>0){{
    const first=candles[0];const last=candles[candles.length-1];
    const dayOpen=first[1];const dayClose=last[4];
    const dayChg=((dayClose-dayOpen)/dayOpen*100).toFixed(2);
    const chgColor=dayClose>=dayOpen?'#4ade80':'#f87171';
    ctx.font='12px sans-serif';ctx.fillStyle='#888';
    ctx.fillText('O: $'+dayOpen.toFixed(2),pad.l+8,pad.t+32);
    ctx.fillStyle=chgColor;
    ctx.fillText('C: $'+dayClose.toFixed(2)+'  ('+( dayClose>=dayOpen?'+':'')+dayChg+'%)',pad.l+100,pad.t+32);
  }}
}}

function parseTimeToMinutes(hhmm){{
  // "HH:MM" 24h format -> minutes since 9:30
  const parts=hhmm.split(':');
  const h=parseInt(parts[0]);const m=parseInt(parts[1]);
  return (h*60+m)-(9*60+30);
}}

function findPriceAtMinute(pts, targetMin){{
  let best=null;let bestDist=Infinity;
  pts.forEach(p=>{{
    const dist=Math.abs(p[0]-targetMin);
    if(dist<bestDist){{bestDist=dist;best=p[1]}}
  }});
  return best;
}}

function findCandlePriceAtMinute(candles, targetMin){{
  // Find closest candle and return its close price
  let best=null;let bestDist=Infinity;
  candles.forEach(c=>{{
    const dist=Math.abs(c[0]-targetMin);
    if(dist<bestDist){{bestDist=dist;best=c[4]}}  // c[4] = close
  }});
  return best;
}}

// ── Equity Curve ──
function drawEq(){{
const canvas=document.getElementById('eqChart');
const ctx=canvas.getContext('2d');
const dpr=window.devicePixelRatio||1;
const W=canvas.parentElement.offsetWidth-32;
const H=220;
canvas.width=W*dpr;canvas.height=H*dpr;
canvas.style.width=W+'px';canvas.style.height=H+'px';
ctx.scale(dpr,dpr);
const ks=Object.keys(D).sort();
let cum=0,peak=0;const pts=[];const dds=[];
ks.forEach(k=>{{cum+=D[k].p;if(cum>peak)peak=cum;pts.push({{x:k,y:cum}});dds.push(cum-peak)}});
const mn=Math.min(...pts.map(p=>p.y),0);
const mx=Math.max(...pts.map(p=>p.y));
const pad2={{l:55,r:15,t:15,b:25}};
const pw2=W-pad2.l-pad2.r, ph2=H-pad2.t-pad2.b;
function toX2(i){{return pad2.l+pw2*i/(pts.length-1)}}
function toY2(v){{return pad2.t+ph2*(1-(v-mn)/(mx-mn||1))}}
ctx.fillStyle='#12121e';ctx.fillRect(0,0,W,H);
ctx.strokeStyle='#1a1a2e';ctx.lineWidth=0.5;
for(let i=0;i<=5;i++){{
  const y=pad2.t+ph2*(1-i/5);
  ctx.beginPath();ctx.moveTo(pad2.l,y);ctx.lineTo(W-pad2.r,y);ctx.stroke();
  const val=mn+(mx-mn)*i/5;
  ctx.fillStyle='#555';ctx.font='10px sans-serif';ctx.textAlign='right';
  ctx.fillText('$'+(val/1e6).toFixed(1)+'M',pad2.l-6,y+3);
}}
ctx.beginPath();ctx.strokeStyle='#4ade80';ctx.lineWidth=1.5;
pts.forEach((p,i)=>{{const x=toX2(i);const y=toY2(p.y);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)}});
ctx.stroke();
if(mn<0){{const zy=toY2(0);ctx.strokeStyle='#333';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(pad2.l,zy);ctx.lineTo(W-pad2.r,zy);ctx.stroke();ctx.setLineDash([])}}
ctx.fillStyle='#555';ctx.font='10px sans-serif';ctx.textAlign='center';
let lastYr='';
pts.forEach((p,i)=>{{const yr=p.x.slice(0,4);if(yr!==lastYr){{lastYr=yr;const x=toX2(i);ctx.fillText(yr,x,H-pad2.b+14)}}}});
}}

// ── Monthly Bar Chart ──
function drawMo(){{
const canvas=document.getElementById('moChart');
const ctx=canvas.getContext('2d');
const dpr=window.devicePixelRatio||1;
const W=canvas.parentElement.offsetWidth-32;
const H=160;
canvas.width=W*dpr;canvas.height=H*dpr;
canvas.style.width=W+'px';canvas.style.height=H+'px';
ctx.scale(dpr,dpr);
const ks=Object.keys(M).sort();
const vals=ks.map(k=>M[k].pnl);
const mx2=Math.max(...vals.map(Math.abs));
const pad3={{l:55,r:10,t:10,b:22}};
const pw3=W-pad3.l-pad3.r, ph3=H-pad3.t-pad3.b;
const bw=Math.max(2,pw3/ks.length-2);
const zeroY=pad3.t+ph3/2;
ctx.fillStyle='#12121e';ctx.fillRect(0,0,W,H);
ctx.strokeStyle='#333';ctx.lineWidth=0.5;
ctx.beginPath();ctx.moveTo(pad3.l,zeroY);ctx.lineTo(W-pad3.r,zeroY);ctx.stroke();
ks.forEach((k,i)=>{{
  const x=pad3.l+(pw3*i/ks.length)+1;
  const v=M[k].pnl;
  const bh=Math.abs(v)/mx2*(ph3/2);
  ctx.fillStyle=v>=0?'#4ade80':'#f87171';
  if(v>=0)ctx.fillRect(x,zeroY-bh,bw,bh);
  else ctx.fillRect(x,zeroY,bw,bh);
  if(i%6===0){{
    ctx.fillStyle='#555';ctx.font='9px sans-serif';ctx.textAlign='center';
    ctx.fillText(k,x+bw/2,H-pad3.b+12);
  }}
}});
ctx.fillStyle='#555';ctx.font='10px sans-serif';ctx.textAlign='right';
ctx.fillText('$+'+(mx2/1e6).toFixed(1)+'M',pad3.l-6,pad3.t+8);
ctx.fillText('$-'+(mx2/1e6).toFixed(1)+'M',pad3.l-6,H-pad3.b-2);
}}

init();drawEq();drawMo();
window.addEventListener('resize',()=>{{drawEq();drawMo()}});
</script>
</body></html>'''

    return html


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("Loading trades...")
    trades = load_trades()
    print(f"  {len(trades)} trades loaded")

    print("\nLoading 5-min equity bars...")
    bars = load_5m_bars()

    print("\nBuilding calendar data...")
    D, M, C = build_calendar_data(trades, bars)

    print("\nGenerating HTML...")
    html = build_html(D, M, C, trades)

    out_path = os.path.join(OUTPUT_DIR, "options_pnl_calendar.html")
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"\nSaved to {out_path} ({len(html)/1024:.0f} KB)")
    print("Done!")


if __name__ == "__main__":
    run()
