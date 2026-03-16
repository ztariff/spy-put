#!/usr/bin/env python3
"""
Rebuild P&L calendar with updated delta changes.
=================================================
Reads options_updated_deltas.csv, filters to 5 active strategies,
rebuilds the D blob and KPI strip, and saves a new HTML calendar.

Input:  output/options_updated_deltas.csv
        output/options_pnl_calendar_5strat.html  (template)
Output: output/options_pnl_calendar_updated.html
"""

import os, base64, json, re
import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR

INPUT_CSV  = os.path.join(OUTPUT_DIR, "options_updated_deltas.csv")
TEMPLATE   = os.path.join(OUTPUT_DIR, "options_pnl_calendar_5strat.html")
OUTPUT     = os.path.join(OUTPUT_DIR, "options_pnl_calendar_updated.html")

# Active strategies only (hibernated ones excluded from calendar)
ACTIVE_RULES = {
    'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
    'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
    'PriorDayWeak_50Hi_SPY_filtered':   '50Hi Weak',
    'GapLarge_First30min_SPY':          'GapLarge',
    'PriorDayWeak_30min_SPY_filtered':  'SPY Weak',
}

DIRECTION = {
    'PriorDayStrong_AboveOR_QQQ_short': 'short',
    'PriorDayStrong_AboveOR_SPY_short': 'short',
    'PriorDayWeak_50Hi_SPY_filtered':   'long',
    'GapLarge_First30min_SPY':          'long',
    'PriorDayWeak_30min_SPY_filtered':  'long',
}


def parse_option_desc(option_ticker: str, strike: float, expiry_date: str) -> str:
    """
    Build human-readable option description.
    e.g. O:QQQ210108P00310000 → 'QQQ 2021-01-08 $310 P'
    """
    # Extract from option_ticker: O:XYZ{YYMMDD}{C/P}{STRIKE8}
    # Use the expiry_date and strike columns directly (already parsed)
    try:
        # Parse ticker symbol from option_ticker
        parts = option_ticker.replace('O:', '')   # QQQ210108P00310000
        # Find the underlying ticker — everything before 6 digits (date)
        m = re.match(r'^([A-Z]+)(\d{6})([CP])(\d+)$', parts)
        if m:
            sym   = m.group(1)
            cp    = m.group(3)
        else:
            sym   = option_ticker.split('210')[0].replace('O:', '') if '210' in option_ticker else '???'
            cp    = 'P' if 'P' in option_ticker else 'C'
        return f"{sym} {expiry_date} ${int(strike)} {cp}"
    except Exception:
        return str(option_ticker)


def fmt_time(ts_str: str) -> str:
    """Format entry/exit time as '9:31 AM'."""
    try:
        ts = pd.Timestamp(ts_str)
        return ts.strftime('%-I:%M %p')
    except Exception:
        return str(ts_str)[11:16]


def fmt_hm(ts_str: str) -> str:
    """Format as 'HH:MM' for the ehm/xhm fields."""
    try:
        ts = pd.Timestamp(ts_str)
        return ts.strftime('%H:%M')
    except Exception:
        return str(ts_str)[11:16]


def main():
    # ── Load trades ──────────────────────────────────────────────────────
    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df['trade_date'] = df['trade_date'].astype(str).str[:10]

    # Filter to active strategies only
    df = df[df['rule'].isin(ACTIVE_RULES)].copy()
    df['label'] = df['rule'].map(ACTIVE_RULES)
    print(f"  {len(df)} active trades loaded")

    # ── Build D blob ─────────────────────────────────────────────────────
    D = {}

    for _, row in df.iterrows():
        td  = row['trade_date']
        pnl = round(float(row['pnl']))

        # Build compact trade
        trade = {
            'r':   row['label'],
            'd':   DIRECTION[row['rule']],
            'dl':  float(row['target_delta']),
            'o':   parse_option_desc(
                       str(row['option_ticker']),
                       float(row['strike']),
                       str(row['expiry_date'])[:10]
                   ),
            'ep':  round(float(row['option_entry_price']), 2),
            'xp':  round(float(row['option_exit_price']),  2),
            'et':  fmt_time(str(row['entry_time'])),
            'xt':  fmt_time(str(row['exit_time'])),
            'c':   int(row['num_contracts']),
            'pr':  round(float(row['premium_paid'])),
            'pnl': pnl,
            'pp':  round(float(row['pnl_pct']) * 100, 1),
            'tk':  str(row['ticker']),
            'ehm': fmt_hm(str(row['entry_time'])),
            'xhm': fmt_hm(str(row['exit_time'])),
            'pt':  1 if pnl > 0 else 0,
        }

        if td not in D:
            D[td] = {'p': 0, 't': []}
        D[td]['p'] += pnl
        D[td]['t'].append(trade)

    # Round daily P&L
    for td in D:
        D[td]['p'] = round(D[td]['p'])

    print(f"  {len(D)} trading days in D blob")

    # ── Compute KPIs ─────────────────────────────────────────────────────
    daily_pnl  = pd.Series({td: D[td]['p'] for td in D}).sort_index()
    all_bdates = pd.bdate_range(daily_pnl.index.min(), daily_pnl.index.max())
    daily_full = daily_pnl.reindex(
        pd.DatetimeIndex(all_bdates).strftime('%Y-%m-%d'), fill_value=0
    )

    total_pnl  = int(df['pnl'].sum())
    n_trades   = len(df)
    n_days     = len(daily_pnl)
    n_bdates   = len(all_bdates)
    win_days   = int((daily_pnl > 0).sum())
    win_day_pct = round(win_days / n_days * 100)
    avg_day    = round(daily_pnl.mean())
    win_rate   = round((df['pnl'] > 0).mean() * 100, 1)

    cum        = daily_full.cumsum()
    maxdd      = int((cum - cum.cummax()).min())
    mu         = daily_full.mean()
    sig        = daily_full.std()
    sharpe     = round(mu / sig * np.sqrt(252), 2) if sig > 0 else 0.0

    # Sortino (downside std)
    neg = daily_full[daily_full < 0]
    sortino = round(mu / neg.std() * np.sqrt(252), 2) if len(neg) > 1 and neg.std() > 0 else 0.0

    print(f"\nKPIs:")
    print(f"  Total P&L:    ${total_pnl:,}")
    print(f"  Trades:       {n_trades}")
    print(f"  Trading Days: {n_days}")
    print(f"  Win Day Rate: {win_day_pct}%")
    print(f"  Avg/Day:      ${avg_day:,}")
    print(f"  Win Rate:     {win_rate}%")
    print(f"  Max DD:       ${maxdd:,}")
    print(f"  Sharpe:       {sharpe}")
    print(f"  Sortino:      {sortino}")

    # ── Load template HTML ────────────────────────────────────────────────
    print(f"\nLoading template: {TEMPLATE}")
    with open(TEMPLATE) as f:
        html = f.read()

    # ── Replace D blob ────────────────────────────────────────────────────
    new_b64 = base64.b64encode(json.dumps(D, separators=(',', ':')).encode()).decode()
    old_start = html.find('const D=JSON.parse(atob("')
    old_end   = html.find('"))', old_start) + len('"))')
    if old_start == -1:
        raise ValueError("Could not find 'const D=JSON.parse(atob(\"' in template")

    html = html[:old_start] + f'const D=JSON.parse(atob("{new_b64}"))' + html[old_end:]
    print("  D blob replaced")

    # ── Update KPI values ─────────────────────────────────────────────────
    # Replace by label text patterns — find the value div before each label div
    def replace_kpi_by_label(html: str, label: str, new_val: str) -> str:
        """Find <div class="l">label</div> and replace the preceding <div class="v ...">value</div>."""
        pattern = r'(<div class="v[^"]*">)[^<]*(</div><div class="l">)' + re.escape(label) + r'(</div>)'
        replacement = r'\g<1>' + new_val + r'\g<2>' + label + r'\g<3>'
        new_html, n = re.subn(pattern, replacement, html)
        if n == 0:
            print(f"  [WARN] KPI label not found: '{label}'")
        return new_html

    # Compute additional KPIs the template shows
    total_premium = int(df['premium_paid'].sum())
    total_contracts = int(df['num_contracts'].sum())

    pnl_str  = f"${total_pnl/1e6:.2f}M" if abs(total_pnl) >= 1e6 else f"${total_pnl:,}"
    pnl_sign = '+' if total_pnl >= 0 else ''

    # Exact label names as they appear in the HTML template
    html = replace_kpi_by_label(html, 'Total P&L',    f'{pnl_sign}{pnl_str}')
    html = replace_kpi_by_label(html, 'Trades',       str(n_trades))
    html = replace_kpi_by_label(html, 'Trading Days', str(n_days))
    html = replace_kpi_by_label(html, 'Win Day Rate', f'{win_day_pct}%')
    html = replace_kpi_by_label(html, 'Avg Day',      f'$+{avg_day:,}')
    html = replace_kpi_by_label(html, 'Total Premium',f'${total_premium:,}')
    html = replace_kpi_by_label(html, 'Max Drawdown', f'${abs(maxdd):,}')
    html = replace_kpi_by_label(html, 'Contracts',    f'{total_contracts:,}')

    # Update page title to reflect new deltas
    html = html.replace(
        '0DTE Options Strategy — P&L Dashboard',
        '0DTE Options Strategy — P&L Dashboard (Updated Deltas)'
    )

    # ── Update equity curve data ──────────────────────────────────────────
    # The equity curve is built from D in JS — no change needed (uses D blob directly)

    # ── Save ──────────────────────────────────────────────────────────────
    with open(OUTPUT, 'w') as f:
        f.write(html)

    print(f"\nSaved: {OUTPUT}")
    print(f"File size: {os.path.getsize(OUTPUT)/1024:.0f} KB")


if __name__ == "__main__":
    main()
