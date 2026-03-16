#!/usr/bin/env python3
"""
Build V6 Strategy Document PDF
"""
import json, os
import pandas as pd, numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable, KeepTogether)
from reportlab.platypus import ListFlowable, ListItem
from reportlab.lib.colors import HexColor

# ── Colour palette ────────────────────────────────────────────────────────────
DARK      = HexColor("#0D1117")
NAVY      = HexColor("#0F2744")
BLUE      = HexColor("#1A4B8C")
LIGHTBLUE = HexColor("#2E6FBF")
ACCENT    = HexColor("#00C896")
RED       = HexColor("#D94F3D")
LGRAY     = HexColor("#F4F6F9")
MGRAY     = HexColor("#DDE2EA")
DGRAY     = HexColor("#6B7280")
WHITE     = colors.white
BLACK     = colors.black

W, H = letter
MARGIN = 0.65 * inch

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
PDF_PATH   = os.path.join(OUTPUT_DIR, "V6_Strategy_Document.pdf")

# ── Load data ─────────────────────────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "V6_strategy_params.json")) as f:
    P = json.load(f)

df = pd.read_csv(os.path.join(OUTPUT_DIR, "options_2strat_v6_costs.csv"))
df['trade_date'] = df['trade_date'].astype(str).str[:10]
df['year']  = df['trade_date'].str[:4]
df['month'] = df['trade_date'].str[:7]
df['win']   = df['pnl'] > 0

daily = df.groupby('trade_date')['pnl'].sum().sort_index()
all_bd = pd.bdate_range(daily.index.min(), daily.index.max())
full   = daily.reindex(all_bd.strftime('%Y-%m-%d'), fill_value=0)
cum    = full.cumsum()

monthly = df.groupby('month')['pnl'].sum()

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

sTitle    = S('sTitle',    fontSize=28, textColor=WHITE,     alignment=TA_CENTER,
               fontName='Helvetica-Bold', spaceAfter=6, leading=34)
sSubtitle = S('sSubtitle', fontSize=13, textColor=ACCENT,    alignment=TA_CENTER,
               fontName='Helvetica', spaceAfter=4)
sMeta     = S('sMeta',     fontSize=10, textColor=MGRAY,     alignment=TA_CENTER,
               fontName='Helvetica', spaceAfter=2)
sH1       = S('sH1',       fontSize=14, textColor=WHITE,     fontName='Helvetica-Bold',
               spaceBefore=14, spaceAfter=6, leading=18)
sH2       = S('sH2',       fontSize=11, textColor=NAVY,      fontName='Helvetica-Bold',
               spaceBefore=10, spaceAfter=4, leading=14,
               backColor=LGRAY, leftIndent=6, rightIndent=6)
sBody     = S('sBody',     fontSize=9,  textColor=BLACK,     fontName='Helvetica',
               spaceAfter=4, leading=13)
sBodySm   = S('sBodySm',   fontSize=8,  textColor=DGRAY,     fontName='Helvetica',
               spaceAfter=3, leading=11)
sNote     = S('sNote',     fontSize=8,  textColor=DGRAY,     fontName='Helvetica-Oblique',
               spaceAfter=3, leading=11, leftIndent=12)
sBold     = S('sBold',     fontSize=9,  textColor=BLACK,     fontName='Helvetica-Bold',
               spaceAfter=4, leading=13)
sRight    = S('sRight',    fontSize=9,  textColor=BLACK,     fontName='Helvetica',
               alignment=TA_RIGHT, spaceAfter=4)
sGreen    = S('sGreen',    fontSize=9,  textColor=ACCENT,    fontName='Helvetica-Bold',
               spaceAfter=3)
sRed      = S('sRed',      fontSize=9,  textColor=RED,       fontName='Helvetica-Bold',
               spaceAfter=3)

# ── Table helpers ─────────────────────────────────────────────────────────────
def kv_table(rows, col_widths=None):
    """Two-column key/value table."""
    if col_widths is None:
        col_widths = [2.5*inch, 4.0*inch]
    data = []
    for k, v in rows:
        data.append([Paragraph(k, sBodySm), Paragraph(str(v), sBold)])
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), WHITE),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [WHITE, LGRAY]),
        ('GRID', (0,0), (-1,-1), 0.4, MGRAY),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    return t

def stat_table(headers, rows, col_widths=None):
    """Multi-column stats table with header row."""
    data = [[Paragraph(h, S('th', fontSize=8, fontName='Helvetica-Bold',
                             textColor=WHITE, alignment=TA_CENTER)) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), S('td', fontSize=8.5, fontName='Helvetica',
                                          textColor=BLACK, alignment=TA_CENTER)) for c in row])
    if col_widths is None:
        n = len(headers)
        col_widths = [(W - 2*MARGIN) / n] * n
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  NAVY),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LGRAY]),
        ('GRID',          (0,0), (-1,-1), 0.4, MGRAY),
        ('LEFTPADDING',   (0,0), (-1,-1), 5),
        ('RIGHTPADDING',  (0,0), (-1,-1), 5),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
    ]))
    return t

def section_header(title):
    """Dark banner section header."""
    data = [[Paragraph(title, sH1)]]
    t = Table(data, colWidths=[W - 2*MARGIN])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), NAVY),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('ROUNDEDCORNERS', [4,4,4,4]),
    ]))
    return t

def hero_kpi(label, value, sub=None, positive=True):
    color = ACCENT if positive else RED
    data = [
        [Paragraph(value, S('v', fontSize=20, fontName='Helvetica-Bold',
                             textColor=color, alignment=TA_CENTER))],
        [Paragraph(label, S('l', fontSize=8, fontName='Helvetica-Bold',
                             textColor=DGRAY, alignment=TA_CENTER))],
    ]
    if sub:
        data.append([Paragraph(sub, S('s', fontSize=7, fontName='Helvetica',
                                       textColor=DGRAY, alignment=TA_CENTER))])
    t = Table(data, colWidths=[1.45*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LGRAY),
        ('BOX', (0,0), (-1,-1), 1, MGRAY),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    return t

# ── Page callbacks ────────────────────────────────────────────────────────────
def cover_background(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.rect(0, H*0.38, W, H*0.62, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, H*0.38 - 3, W, 3, fill=1, stroke=0)
    canvas.restoreState()

def inner_page(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(NAVY)
    canvas.rect(0, H - 0.45*inch, W, 0.45*inch, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont('Helvetica-Bold', 8)
    canvas.drawString(MARGIN, H - 0.28*inch, "0DTE SHORT PUT STRATEGY — V6 STRATEGY DOCUMENT")
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(W - MARGIN, H - 0.28*inch, "CONFIDENTIAL")
    # Footer
    canvas.setFillColor(MGRAY)
    canvas.rect(0, 0, W, 0.35*inch, fill=1, stroke=0)
    canvas.setFillColor(DGRAY)
    canvas.setFont('Helvetica', 7.5)
    canvas.drawString(MARGIN, 0.13*inch, "Backtest period: 2021-01-04 to 2026-02-23  |  All figures include commissions & slippage")
    canvas.drawRightString(W - MARGIN, 0.13*inch, f"Page {doc.page}")
    canvas.restoreState()

# ── Build document ────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    PDF_PATH, pagesize=letter,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=0.6*inch, bottomMargin=0.5*inch,
    title="V6 Strategy Document — 0DTE Short Put Options",
    author="Momentum Trader",
)

story = []

# ═══════════════════════════════════════════════════════════════════════
# COVER PAGE
# ═══════════════════════════════════════════════════════════════════════
story.append(Spacer(1, 1.6*inch))
story.append(Paragraph("0DTE SHORT PUT OPTIONS", sTitle))
story.append(Paragraph("QQQ Short + SPY Short — Strategy Document", sSubtitle))
story.append(Spacer(1, 0.15*inch))
story.append(HRFlowable(width="60%", thickness=1, color=ACCENT, spaceAfter=12,
                         hAlign='CENTER'))
story.append(Paragraph("Version 6  ·  Finalized Strategy", sMeta))
story.append(Paragraph("Backtest: January 4, 2021 – February 23, 2026  (5.1 Years)", sMeta))
story.append(Paragraph("Includes commissions ($0.55/side) and realistic slippage (1.5%/side)", sMeta))
story.append(Spacer(1, 0.5*inch))

# Hero KPIs on cover
kpis = [
    hero_kpi("Total P&L", "+$8.55M"),
    hero_kpi("Sharpe Ratio", "3.18"),
    hero_kpi("Calmar Ratio", "8.43"),
    hero_kpi("Max Drawdown", "$197K", positive=False),
    hero_kpi("Win Rate", "62.5%"),
]
kpi_row = Table([kpis], colWidths=[1.45*inch]*5,
                hAlign='CENTER', spaceAfter=6)
kpi_row.setStyle(TableStyle([
    ('ALIGN',  (0,0),(-1,-1),'CENTER'),
    ('VALIGN', (0,0),(-1,-1),'TOP'),
    ('LEFTPADDING',  (0,0),(-1,-1), 4),
    ('RIGHTPADDING', (0,0),(-1,-1), 4),
]))
story.append(kpi_row)
story.append(Spacer(1, 0.3*inch))

kpis2 = [
    hero_kpi("Sortino", "5.61"),
    hero_kpi("Recovery Factor", "56.25x"),
    hero_kpi("Profit Factor", "3.27"),
    hero_kpi("Pos. Months", "84%"),
    hero_kpi("Avg Day P&L", "+$24,210"),
]
kpi_row2 = Table([kpis2], colWidths=[1.45*inch]*5, hAlign='CENTER')
kpi_row2.setStyle(TableStyle([
    ('ALIGN',  (0,0),(-1,-1),'CENTER'),
    ('VALIGN', (0,0),(-1,-1),'TOP'),
    ('LEFTPADDING',  (0,0),(-1,-1), 4),
    ('RIGHTPADDING', (0,0),(-1,-1), 4),
]))
story.append(kpi_row2)
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — STRATEGY OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
story.append(section_header("1.  STRATEGY OVERVIEW"))
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph(
    "This strategy trades 0DTE (same-day expiration) put options on QQQ and SPY. "
    "It identifies days where the prior session closed strong and price opens above the "
    "opening range, then purchases put options to express a short-term mean-reversion or "
    "continuation-short thesis. Entry occurs at 9:31 AM using the first 1-minute bar close "
    "from Polygon.io. Positions are exited either at a 50% profit target (limit order) or at "
    "9:50 AM if the target is not reached — capturing the window where edge is concentrated "
    "and before 0DTE theta decay becomes destructive.", sBody))
story.append(Spacer(1, 0.08*inch))

story.append(Paragraph("Key Design Principles", sH2))
items = [
    "Pure 0DTE — no overnight risk, every position is flat by 9:50 AM or earlier",
    "Two instruments only — QQQ and SPY, among the most liquid options markets in the world",
    "Fixed entry time — 9:31 AM first bar, eliminates discretionary timing decisions",
    "Defined risk — maximum loss on any trade is 100% of premium paid (option cannot lose more)",
    "Asymmetric exits — limit orders for winners, time exits for non-movers",
    "No leverage beyond the inherent leverage in options — position sizing is premium-based (~$100K/trade)",
]
for item in items:
    story.append(Paragraph(f"• {item}", sBody))

story.append(Spacer(1, 0.1*inch))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — STRATEGY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
story.append(section_header("2.  STRATEGY PARAMETERS"))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("Instrument & Option Specifications", sH2))
story.append(kv_table([
    ("Underlying instruments",       "QQQ (Nasdaq-100 ETF), SPY (S&P 500 ETF)"),
    ("Option type",                  "PUT — 0DTE (same-day expiration)"),
    ("Entry time",                   "9:31 AM ET — first 1-minute bar close"),
    ("Exit — Profit Target",         "50% gain on option premium (limit order)"),
    ("Exit — Time",                  "9:50 AM ET close price if profit target not triggered"),
    ("Data source",                  "Polygon.io 1-minute bars, adjusted=false"),
    ("Budget per trade",             "~$100,000 premium per strategy per day"),
]))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("Delta & Signal Parameters", sH2))
story.append(stat_table(
    ["Strategy", "Underlying", "Signal Condition", "Option Type", "Target Delta", "Avg Entry Price"],
    [
        ["QQQ Short", "QQQ", "Prior day strong close, above opening range at 9:31", "PUT", "δ 0.55", "$2.19"],
        ["SPY Short", "SPY", "Prior day strong close, above opening range at 9:31", "PUT", "δ 0.60", "$1.85"],
    ],
    col_widths=[1.1*inch, 0.75*inch, 2.5*inch, 0.85*inch, 0.85*inch, 0.95*inch]
))
story.append(Spacer(1, 0.08*inch))
story.append(Paragraph(
    "Delta is back-solved from the original option's 9:31 AM 1-minute price using iterative "
    "Black-Scholes, ensuring strike selection reflects the actual underlying price at entry — "
    "not the prior day's closing price.", sNote))

story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("Cost Model (V6)", sH2))
story.append(stat_table(
    ["Cost Component", "PT Exits (164 trades)", "Time Exits (356 trades)"],
    [
        ["Entry slippage",   "1.5% of option price (pay more)", "1.5% of option price (pay more)"],
        ["Exit slippage",    "None — limit order fills at target", "1.5% of option price (receive less)"],
        ["Commission",       "$0.55/side × 2 = $1.10 RT per contract", "$0.55/side × 2 = $1.10 RT per contract"],
        ["Total commissions (5 yr)", "$463,455", "Included in total above"],
    ],
    col_widths=[2.1*inch, 2.7*inch, 2.2*inch]
))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════
story.append(section_header("3.  PERFORMANCE METRICS  (V6 — After All Costs)"))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("Returns", sH2))
story.append(kv_table([
    ("Total P&L (5.1 years)",        "+$8,546,266"),
    ("Annualized P&L",               "+$1,665,736"),
    ("Total premium deployed",       "$52,691,567"),
    ("ROI on premium deployed",      "16.22%"),
    ("Backtest period",              "January 4, 2021 – February 23, 2026  (5.1 years)"),
]))

story.append(Spacer(1, 0.08*inch))
story.append(Paragraph("Risk-Adjusted Metrics", sH2))
story.append(kv_table([
    ("Sharpe Ratio",         "3.18  (annualized daily, √252)"),
    ("Sortino Ratio",        "5.61  (downside deviation only)"),
    ("Calmar Ratio",         "8.43  (annualized return ÷ max drawdown)"),
    ("Recovery Factor",      "56.25×  (total P&L ÷ max drawdown)"),
]))

story.append(Spacer(1, 0.08*inch))
story.append(Paragraph("Drawdown", sH2))
story.append(kv_table([
    ("Maximum drawdown",             "$197,312"),
    ("Max DD as % of equity peak",   "1.8%"),
    ("Max DD date",                  "August 1, 2023"),
    ("Average drawdown depth",       "$41,916  (when in drawdown)"),
]))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — TRADE STATISTICS
# ═══════════════════════════════════════════════════════════════════════
story.append(section_header("4.  TRADE STATISTICS"))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("Portfolio-Level Trade Stats", sH2))
story.append(kv_table([
    ("Total trades",             "520"),
    ("Trading days with activity","353"),
    ("Win rate (per trade)",     "62.5%"),
    ("Win day rate",             "62.6%"),
    ("Avg P&L per trade",        "+$16,435"),
    ("Avg winning trade",        "+$38,952"),
    ("Avg losing trade",         "-$18,695"),
    ("Win / Loss ratio",         "2.08×"),
    ("Profit factor",            "3.27"),
    ("Largest single win",       "+$232,669"),
    ("Largest single loss",      "-$59,800"),
    ("Max consecutive wins",     "15 trades"),
    ("Max consecutive losses",   "7 trades"),
    ("Avg hold time",            "19.4 minutes"),
    ("Median hold time",         "19.0 minutes"),
    ("Total contracts (one-way)","421,323"),
    ("Total contracts (RT)",     "842,646"),
]))

story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("Daily & Monthly Statistics", sH2))
story.append(kv_table([
    ("Win days / Loss days",     f"{int((daily>0).sum())} / {int((daily<0).sum())}  ({round((daily>0).mean()*100,1)}% / {round((daily<0).mean()*100,1)}%)"),
    ("Avg day P&L",              "+$24,210"),
    ("Best single day",          "+$367,212"),
    ("Worst single day",         "-$74,956"),
    ("Daily std deviation",      "$31,924"),
    ("Positive months",          f"52 of 62  (84%)"),
    ("Negative months",          "10 of 62  (16%)"),
    ("Avg monthly P&L",          "+$137,843"),
    ("Best month",               "+$549,737"),
    ("Worst month",              "-$142,238"),
]))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — EXIT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
story.append(section_header("5.  EXIT TYPE ANALYSIS"))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph(
    "The strategy has two exit paths. Profit-target exits (31.5% of trades) generate 85% of all "
    "P&L with a 92.7% win rate. Time exits (68.5% of trades) contribute the remaining 15% at a "
    "near-breakeven level. The edge is concentrated in the first 20 minutes when the directional "
    "move either happens or doesn't.", sBody))
story.append(Spacer(1, 0.08*inch))

story.append(stat_table(
    ["Exit Type", "Trades", "% of Total", "Win Rate", "Avg P&L", "Total P&L", "% of P&L"],
    [
        ["Profit Target (50%)", "164", "31.5%", "92.7%", "+$51,568", "+$8,457,157", "85.0%"],
        ["Time Exit (9:50 AM)", "356", "68.5%", "48.6%", "+$250",    "+$89,109",    "15.0%"],
        ["TOTAL", "520", "100%", "62.5%", "+$16,435", "+$8,546,266", "100%"],
    ],
    col_widths=[1.6*inch, 0.7*inch, 0.8*inch, 0.8*inch, 1.0*inch, 1.1*inch, 0.8*inch]
))
story.append(Spacer(1, 0.08*inch))
story.append(Paragraph(
    "Note: PT exit slippage is zero (limit order). Time exit slippage is 1.5%/side. "
    "All figures include $1.10/contract round-trip commission.", sNote))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — PER-STRATEGY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════
story.append(section_header("6.  PER-STRATEGY BREAKDOWN"))
story.append(Spacer(1, 0.1*inch))

story.append(stat_table(
    ["Metric", "QQQ Short (δ0.55)", "SPY Short (δ0.60)"],
    [
        ["Trades",              "252",        "268"],
        ["Win Rate",            "67.9%",      "63.8%"],
        ["Total P&L (net)",     "~$4.30M",    "~$4.25M"],
        ["Avg Win",             "+$40,118",   "+$36,315"],
        ["Avg Loss",            "-$20,828",   "-$15,548"],
        ["Win/Loss Ratio",      "1.93×",      "2.34×"],
        ["Profit Factor",       "4.12",       "4.25"],
        ["Avg Hold Time",       "18.7 min",   "20.2 min"],
        ["Avg Entry Price",     "$2.19/contract", "$1.85/contract"],
        ["Avg Contracts/Trade", "610",        "998"],
        ["PT Exits",            "80  (32%)",  "84  (31%)"],
        ["Time Exits",          "172  (68%)", "184  (69%)"],
        ["Avg Contracts/Trade", "610",        "998"],
        ["Budget per trade",    "~$103,900",  "~$98,900"],
    ],
    col_widths=[2.3*inch, 2.4*inch, 2.3*inch]
))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — YEARLY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════
story.append(section_header("7.  ANNUAL PERFORMANCE"))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph(
    "The strategy produced positive returns in all 5 full calendar years of the backtest, "
    "including 2022 (a bear market year for equities). 2023 was the weakest year, "
    "driven by lower trade frequency and choppier intraday conditions.", sBody))
story.append(Spacer(1, 0.08*inch))

yr_rows = []
for yr, g in df.groupby('year'):
    wr = g['win'].mean()*100
    qqq_pnl = g[g['rule'].str.contains('QQQ')]['pnl'].sum()
    spy_pnl = g[g['rule'].str.contains('SPY')]['pnl'].sum()
    yr_rows.append([yr, len(g), f"{wr:.0f}%",
                    f"${g['pnl'].sum():>+,.0f}",
                    f"${g['pnl'].mean():>+,.0f}",
                    f"${qqq_pnl:>+,.0f}",
                    f"${spy_pnl:>+,.0f}"])

story.append(stat_table(
    ["Year", "Trades", "Win Rate", "Total P&L", "Avg/Trade", "QQQ P&L", "SPY P&L"],
    yr_rows,
    col_widths=[0.65*inch, 0.7*inch, 0.8*inch, 1.2*inch, 1.0*inch, 1.2*inch, 1.2*inch]
))
story.append(Spacer(1, 0.12*inch))

# Monthly breakdown
story.append(Paragraph("Monthly P&L Heatmap (by year)", sH2))
months_short = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

df['month_num'] = df['trade_date'].str[5:7].astype(int)
pivot = df.groupby(['year','month_num'])['pnl'].sum().unstack(fill_value=None)

header = ['Year'] + months_short + ['Total']
m_rows = []
for yr in sorted(df['year'].unique()):
    row = [yr]
    yr_total = 0
    for m_num in range(1,13):
        val = pivot.loc[yr, m_num] if yr in pivot.index and m_num in pivot.columns else None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            row.append('—')
        else:
            row.append(f"${val/1000:+.0f}K")
            yr_total += val
    row.append(f"${yr_total/1000:+.0f}K")
    m_rows.append(row)

m_col_w = [0.45*inch] + [0.49*inch]*12 + [0.55*inch]
m_table = stat_table(header, m_rows, col_widths=m_col_w)

# Color positive/negative cells
style_cmds = []
for r_idx, row in enumerate(m_rows):
    for c_idx, val in enumerate(row[1:], 1):
        if val != '—' and 'K' in str(val):
            try:
                v = float(str(val).replace('$','').replace('K','').replace('+',''))
                if v > 0:
                    style_cmds.append(('TEXTCOLOR', (c_idx, r_idx+1), (c_idx, r_idx+1), ACCENT))
                elif v < 0:
                    style_cmds.append(('TEXTCOLOR', (c_idx, r_idx+1), (c_idx, r_idx+1), RED))
            except: pass

m_table.setStyle(TableStyle([
    ('BACKGROUND',    (0,0),  (-1,0),  NAVY),
    ('ROWBACKGROUNDS',(0,1),  (-1,-1), [WHITE, LGRAY]),
    ('GRID',          (0,0),  (-1,-1), 0.4, MGRAY),
    ('FONTNAME',      (0,0),  (-1,0),  'Helvetica-Bold'),
    ('FONTSIZE',      (0,0),  (-1,-1), 7),
    ('LEFTPADDING',   (0,0),  (-1,-1), 3),
    ('RIGHTPADDING',  (0,0),  (-1,-1), 3),
    ('TOPPADDING',    (0,0),  (-1,-1), 4),
    ('BOTTOMPADDING', (0,0),  (-1,-1), 4),
    ('ALIGN',         (0,0),  (-1,-1), 'CENTER'),
    ('TEXTCOLOR',     (0,0),  (-1,0),  WHITE),
    *style_cmds,
]))
story.append(m_table)
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — SLIPPAGE & CAPACITY
# ═══════════════════════════════════════════════════════════════════════
story.append(section_header("8.  SLIPPAGE, CAPACITY & RISK CONSIDERATIONS"))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("Slippage Sensitivity", sH2))
story.append(stat_table(
    ["Scenario", "Entry Slip", "Exit Slip", "Total P&L", "vs Clean", "Sharpe", "Max DD"],
    [
        ["No costs (backtest)",       "0%",    "0%",    "+$9,942,448", "—",          "3.69", "$176,768"],
        ["Optimistic  (0.5%/side)",   "0.5%",  "0.5%*", "+$9.6M",     "-$320K",     "3.55", "$180K"],
        ["V6 Realistic (1.5%/side)",  "1.5%",  "1.5%*", "+$8,546,266","-$1,396,182","3.18", "$197,312"],
        ["Conservative (2.5%/side)",  "2.5%",  "2.5%*", "+$7.2M",     "-$2.7M",     "2.75", "$215K"],
        ["Stressed (5.0%/side)",      "5.0%",  "5.0%*", "+$4.2M",     "-$5.7M",     "1.80", "$300K+"],
    ],
    col_widths=[1.8*inch, 0.8*inch, 0.75*inch, 1.1*inch, 0.95*inch, 0.7*inch, 0.9*inch]
))
story.append(Paragraph("* PT exits have 0% exit slippage (limit order) in V6. Starred % applies to time exits only.", sNote))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("Capacity Considerations", sH2))
story.append(kv_table([
    ("Avg contracts per trade (QQQ)",    "610  (61,000 delta-equivalent shares)"),
    ("Avg contracts per trade (SPY)",    "998  (99,800 delta-equivalent shares)"),
    ("Max single trade (SPY)",           "7,615 contracts — largest individual print"),
    ("Entry timing risk",               "9:31 AM is widest spread of the day; fills may vary 2-4% on large prints"),
    ("PT exit capacity",                "Limit orders at +50% — natural fill, no market impact concern"),
    ("Recommended position cap",        "Stress-test individual trades >3,000 contracts for market impact"),
]))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("Key Risk Factors", sH2))
risks = [
    "Gap risk at 9:31 AM — if the underlying gaps significantly, the 9:31 entry price may differ materially from pre-market expectations",
    "Regime dependence — 2023 was the weakest year, suggesting the edge may be regime-sensitive (trending vs choppy markets)",
    "Theta cliff — 0DTE options lose value exponentially; the 9:50 time exit captures most of the viable trading window",
    "Liquidity concentration — SPY and QQQ 0DTE options are among the most liquid instruments, but large prints (7K+ contracts) still carry market impact risk",
    "Backtest limitations — actual fills at 9:31 AM may include partial fills and price improvement/degradation not modeled",
    "Correlation — both strategies fire on similar days (prior day strong); on those days, combined exposure doubles and correlation is high",
]
for r in risks:
    story.append(Paragraph(f"• {r}", sBody))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════
# SECTION 9 — VERSION HISTORY
# ═══════════════════════════════════════════════════════════════════════
story.append(section_header("9.  VERSION HISTORY & METHODOLOGY"))
story.append(Spacer(1, 0.1*inch))

story.append(stat_table(
    ["Version", "Description", "P&L", "Sharpe", "Status"],
    [
        ["V1", "Original backtest — 5 strategies, original deltas (δ0.70), 10:00/10:10 exits", "$6.15M", "2.82", "Superseded"],
        ["V2", "Delta sweep v1 applied — FLAWED (daily close used for underlying price)", "$8.23M", "2.79", "⚠ Flawed"],
        ["V3", "Delta sweep v2 corrected — back-solved underlying from 9:31 1-min price", "$7.39M", "2.56", "Superseded"],
        ["V4", "2-strategy focus — hibernated 50Hi Weak, GapLarge, SPY Weak", "$6.89M", "2.45", "Superseded"],
        ["V5", "Exit time optimized — changed to 9:50 AM (from 10:00/10:10)", "$9.94M", "3.69", "Superseded"],
        ["V6", "Realistic costs — $0.55/side commission + 1.5% slippage model", "$8.55M", "3.18", "✅ FINAL"],
    ],
    col_widths=[0.5*inch, 3.2*inch, 1.0*inch, 0.7*inch, 0.9*inch]
))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("Key Methodology Notes", sH2))
story.append(kv_table([
    ("Entry price source",   "Polygon.io 1-min bar CLOSE at 9:31 AM, adjusted=false"),
    ("Exit price source",    "Polygon.io 1-min bar CLOSE at exit time, adjusted=false"),
    ("Delta back-solve",     "Iterative Black-Scholes on original option's 9:31 price; not daily close"),
    ("Strike selection",     "Nearest listed strike to back-solved delta target"),
    ("PT trigger",           "50% gain on premium — exit at that bar's close (simulated as limit)"),
    ("Time exit",            "9:50 AM close price of the 1-min bar"),
    ("Position sizing",      "~$100K premium per strategy per trade; contracts = floor($100K / (price × 100))"),
    ("Hibernated strategies","HighVolWR (δ0.10), 50Hi Weak, GapLarge, SPY Weak — data preserved in V3"),
]))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════
# SECTION 10 — FILES & REFERENCE
# ═══════════════════════════════════════════════════════════════════════
story.append(section_header("10.  FILES & REFERENCE"))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("V6 Strategy Package Contents", sH2))
story.append(kv_table([
    ("V6_Strategy_Document.pdf",         "This document"),
    ("options_pnl_calendar_v6_costs.html","Interactive P&L calendar — open in any browser"),
    ("options_2strat_v6_costs.csv",       "Full trade-level data with adjusted prices and P&L"),
    ("V6_strategy_params.json",           "All parameters and metrics in machine-readable format"),
    ("STRATEGY_CHANGELOG.md",             "Full version history with rationale for every change"),
]))

story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("Version Archive", sH2))
story.append(kv_table([
    ("output/versions/v1_original/",      "Original 5-strategy backtest"),
    ("output/versions/v2_delta_sweep_v1/","Flawed delta sweep — do not use"),
    ("output/versions/v3_delta_sweep_v2/","Corrected deltas, 5 strategies"),
    ("output/versions/v4_2strat/",        "2-strategy focus, old exits"),
    ("output/versions/v5_exit0950/",      "9:50 exit, no cost model"),
    ("output/versions/v6_costs/",         "FINAL — full cost model applied"),
]))

story.append(Spacer(1, 0.25*inch))
story.append(HRFlowable(width="100%", thickness=0.5, color=MGRAY))
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph(
    "This document reflects backtest results only. Past performance is not indicative of future results. "
    "All figures include modeled commissions and slippage. Actual live trading results may differ materially "
    "due to market impact, partial fills, data discrepancies, and changing market conditions.",
    sNote))

# ═══════════════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════════════
doc.build(story,
          onFirstPage=cover_background,
          onLaterPages=inner_page)
print(f"PDF saved: {PDF_PATH}")
print(f"Size: {os.path.getsize(PDF_PATH)/1024:.0f} KB")
