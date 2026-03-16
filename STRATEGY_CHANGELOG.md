# Strategy Changelog
All versions are archived in `output/versions/`. To rewind, use the CSV and calendar from that folder.

---

## v5 — Exit Time Changed to 9:50 ✅ CURRENT
**Date:** 2026-02-26
**Files:** `output/options_2strat_v5_exit0950.csv` · `output/options_pnl_calendar_v5_exit0950.html`
**Archived:** `output/versions/v5_exit0950/`

### Strategies Active
| Strategy | Delta | Exit Rule |
|---|---|---|
| QQQ Short | δ0.55 | Profit target 50% OR hard exit 9:50 AM |
| SPY Short | δ0.60 | Profit target 50% OR hard exit 9:50 AM |

### Change
- QQQ Short: exit time changed from **10:00 AM → 9:50 AM** for non-PT trades
- SPY Short: exit time changed from **10:10 AM → 9:50 AM** for non-PT trades
- 354 of 356 time-exit trades updated using 1-min bar close at 9:50
- 164 profit-target trades unchanged

### Why
Exit time analysis showed time-exit trades (holding to 10:00/10:10) had WR 42-45% and avg P&L of -$2K to -$7K. 0DTE theta decay accelerates past 9:50 with no edge remaining for non-moving trades.

### Stats vs v4
| Metric | v4 (before) | v5 (after) | Change |
|---|---|---|---|
| Total P&L | $6,886,624 | **$9,942,448** | +$3,055,824 (+44%) |
| Sharpe | 2.45 | **3.69** | +1.24 |
| Max Drawdown | $259,801 | **$176,769** | -$83,032 (-32%) |
| Win Rate | 58.8% | **65.8%** | +7.0pp |
| Avg Day | $19,509 | **$28,166** | +$8,657 |
| Trades | 520 | 520 | — |
| Trading Days | 353 | 353 | — |

---

## v4 — Hibernated 3 Strategies, 2-Strategy Focus
**Date:** 2026-02-26
**Files:** `output/options_2strat.csv` · `output/options_pnl_calendar_2strat.html`
**Archived:** `output/versions/v4_2strat/`

### Strategies Active
| Strategy | Delta | Exit Rule |
|---|---|---|
| QQQ Short | δ0.55 | Profit target 50% OR 10:00 AM |
| SPY Short | δ0.60 | Profit target 50% OR 10:10 AM |

### Hibernated (not deleted, data preserved in v3)
- 50Hi Weak (δ0.60) — low WR (39%), regime-dependent
- GapLarge (δ0.50) — too few trades (44), low edge
- SPY Weak (δ0.50) — breakeven, 0.59x win/loss ratio

### Why
QQQ Short + SPY Short generate 93% of all P&L. Removing the 3 weaker strategies simplifies the system and removes noise. Each can be re-added later.

### Stats vs v3
| Metric | v3 (before) | v4 (after) | Change |
|---|---|---|---|
| Total P&L | $7,394,190 | $6,886,624 | -$507,566 |
| Sharpe | 2.56 | 2.45 | -0.11 |
| Max Drawdown | $287,392 | $259,801 | -$27,591 |
| Trades | 719 | 520 | -199 |

---

## v3 — Delta Sweep v2 (Corrected Underlying Price) ✅ RECOMMENDED BASE FOR FULL PORTFOLIO
**Date:** 2026-02-25
**Files:** `output/options_updated_deltas_v2.csv` · `output/options_pnl_calendar_updated_v2.html`
**Archived:** `output/versions/v3_delta_sweep_v2/`

### Strategies Active
| Strategy | Delta | Change from v1 |
|---|---|---|
| QQQ Short | δ0.55 | ↑ from δ0.70 |
| SPY Short | δ0.60 | ↑ from δ0.70 |
| 50Hi Weak | δ0.60 | ↑ from δ0.50 |
| GapLarge | δ0.50 | unchanged |
| SPY Weak | δ0.50 | unchanged |
| HighVolWR | hibernated | δ0.10 → hibernated |

### Change
Fixed the delta sweep. v2 back-solves the actual underlying price at 9:31 AM from the original option's 1-min entry price (instead of using daily bar close). All 5,888 sweep results used `ul_method=back_solve`.

**Root cause of v2 flaw:** `load_daily_bars()` returned end-of-day CLOSE as underlying. On 2024-10-01, QQQ closed $481 but opened $489 → wrong strike selected ($482 vs correct $488) → 89% of QQQ Short strikes were wrong.

### Stats vs v1
| Metric | v1 | v3 | Change |
|---|---|---|---|
| Total P&L | $6,145,397 | $7,394,190 | +$1,248,793 |
| Sharpe | 2.82 | 2.56 | -0.26 |
| Max Drawdown | $235,583 | $287,392 | +$51,809 |
| Trades | 719 | 719 | — |

---

## v2 — Delta Sweep v1 ⚠️ FLAWED — DO NOT USE
**Date:** 2026-02-25
**Files:** `output/options_updated_deltas.csv` · `output/options_pnl_calendar_updated.html`
**Archived:** `output/versions/v2_delta_sweep_v1/`

### Change
First delta sweep applied: QQQ Short δ0.70→0.55, 50Hi Weak δ0.50→0.65.
SPY Short kept at δ0.70 (sweep incorrectly showed it was optimal).

### Why Flawed
Delta sweep used daily bar CLOSE price (~$481) as underlying instead of 9:31 open (~$489) for QQQ on 2024-10-01. 89% of QQQ Short strikes were wrong (224/252 trades). The $8.23M figure is unreliable. Replaced by v3.

---

## v1 — Original 1-Min Backtest
**Date:** 2026-02-25
**Files:** `output/options_931filter_1min.csv` · `output/options_pnl_calendar_5strat.html`
**Archived:** `output/versions/v1_original/`

### Strategies Active
| Strategy | Delta | Exit Rule |
|---|---|---|
| QQQ Short | δ0.70 | Profit target 50% OR 10:00 AM |
| SPY Short | δ0.70 | Profit target 50% OR 10:10 AM |
| 50Hi Weak | δ0.50 | Profit target 50% OR various |
| GapLarge | δ0.50 | Profit target 50% OR various |
| SPY Weak | δ0.50 | Profit target 50% OR various |
| HighVolWR | δ0.10 | hibernated |

### Notes
- Entry: 9:31 AM 1-min bar close (Polygon, adjusted=false)
- Exit: 1-min bar close at exit time (Polygon, adjusted=false)
- Budget: ~$75K target per strategy
- All 719 trades verified against Polygon cache (0 mismatches)

### Stats
| P&L | Sharpe | MaxDD | WR | Trades | Days |
|---|---|---|---|---|---|
| $6,145,397 | 2.82 | $235,583 | 61.6% | 719 | 501 |

---

## Hibernated Strategies (data preserved in v3 CSV)
| Strategy | Reason | Last seen |
|---|---|---|
| HighVolWR (δ0.10) | -$176K total P&L, losing every year since 2022, 44% WR | v1 |
| 50Hi Weak (δ0.60) | Regime-dependent, 39% WR, needs regime filter before reactivation | v3 |
| GapLarge (δ0.50) | 44 trades only, 75% WR but insufficient sample size | v3 |
| SPY Weak (δ0.50) | Breakeven ($31K), 0.59x win/loss ratio, near-total losses on bad days | v3 |

---

## How to Rewind
Each version folder contains the exact CSV and calendar HTML used at that point.

```
output/versions/
  v1_original/          ← original backtest
  v2_delta_sweep_v1/    ← flawed (do not use)
  v3_delta_sweep_v2/    ← corrected deltas, 5 strategies
  v4_2strat/            ← 2-strategy focus
  v5_exit0950/          ← CURRENT: 9:50 exit
```

To reload any version, open the `.html` file for the calendar or load the `.csv` into analysis scripts.

---

## v6 — Realistic Cost Model Applied ✅ CURRENT
**Date:** 2026-02-26
**Files:** `output/options_2strat_v6_costs.csv` · `output/options_pnl_calendar_v6_costs.html`
**Archived:** `output/versions/v6_costs/`

### Cost Model
| Cost | PT exits (164) | Time exits (356) |
|---|---|---|
| Entry slippage | 1.5% (pay more) | 1.5% (pay more) |
| Exit slippage | None (limit order) | 1.5% (receive less) |
| Commission | $0.55/side × 2 = $1.10 RT | $0.55/side × 2 = $1.10 RT |

### Stats vs v5 (no costs)
| Metric | v5 (clean) | v6 (with costs) | Change |
|---|---|---|---|
| Total P&L | $9,942,448 | **$8,546,266** | -$1,396,182 (-14%) |
| Sharpe | 3.69 | **3.18** | -0.51 |
| Sortino | 7.16 | **5.61** | -1.55 |
| Calmar | 10.95 | **8.43** | -2.52 |
| Max Drawdown | $176,768 | **$197,312** | +$20,544 |
| Win Rate | 65.8% | **62.5%** | -3.3pp |
| Win Day Rate | 66.3% | **62.6%** | -3.7pp |
| Avg Day P&L | $28,166 | **$24,210** | -$3,956 |
| Profit Factor | 4.18 | **3.27** | -0.91 |
| Total Commissions | — | $463,455 | — |
