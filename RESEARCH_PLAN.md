# Intraday Momentum Strategy — Research Plan

## Thesis

Intraday momentum is not random — it clusters around names and conditions where higher-timeframe forces create directional pressure. The goal is to systematically discover which higher-timeframe contexts produce the best intraday momentum opportunities, then identify the optimal lower-timeframe entry signals within those contexts.

We are not fitting a strategy to our beliefs. We are running a structured research pipeline and letting statistical evidence guide every decision.

---

## Architecture: Top-Down Multi-Timeframe

```
LAYER 1: Universe Selection (Daily)
  "Which tickers should I even look at today?"
  → Filters: liquidity, gap, relative volume, sector momentum, trend

LAYER 2: Higher-Timeframe Context (Daily / 60m)
  "Is this a momentum setup or noise?"
  → Signals: trend direction, breakout, relative strength, VWAP position

LAYER 3: Intraday Entry (1m / 5m / 15m / 30m)
  "When do I get in?"
  → Signals: momentum bursts, flag breaks, pullback entries, ORB

LAYER 4: Risk Management
  "How much, and when do I get out?"
  → Position sizing, stops, targets, time-based exits
```

---

## Data Requirements

### Source: Polygon.io (Developer+ Plan)

| Data Type | Timeframe | History | Purpose |
|-----------|-----------|---------|---------|
| Aggregated bars | 1m, 5m, 15m, 30m, 60m | 5 years | Intraday signal testing |
| Daily bars | 1D | 10 years | HTF context, trend, relative strength |
| Weekly bars | 1W | 10 years | Macro trend context |
| Pre-market bars | 1m | 5 years | Gap classification, pre-market momentum |
| Volume data | Per bar | 5 years | Relative volume, VWAP computation |

### Universe — Staged Expansion

The research follows a **prove-then-expand** approach. We start with the most liquid, cleanest instruments and only widen the universe after confirming edge.

**Stage 1 — Proof of Concept (SPY + QQQ only)**

- The two most liquid ETFs on earth: tightest spreads, deepest books, cleanest data
- If a momentum signal doesn't work here, it won't work anywhere
- This stage runs through Phases 1-3 in full before expanding
- Goal: identify which HTF contexts and intraday signals produce statistically significant edge

**Stage 2 — Cross-Asset Validation**

- Expand to: IWM, DIA, GLD, TLT, IBIT
- Test: does the edge found in Stage 1 generalize to small-caps, gold, bonds, crypto?
- If yes → the signal captures a real market dynamic, not SPY/QQQ-specific noise
- If no → the edge may be specific to large-cap equity momentum (still tradeable, just narrower)

**Stage 3 — Sector Rotation**

- Add sector ETFs: XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLRE, XLB, XLC
- Test intraday sector rotation momentum: does rotating into the strongest intraday sector work?
- This introduces a cross-sectional element on top of the time-series momentum from Stages 1-2

**Stage 4 — Full Universe + Dynamic Scanner**

- Add remaining ETFs: EFA, EEM, SLV, USO, DBC, ETHA, BITO, leveraged (TQQQ, SOXL)
- Activate dynamic "in-play" scanner for individual stocks:
  - Mid-cap+ ($2B+), pre-market gap > 2%, RVOL > 2x
  - Polygon snapshot endpoint for pre-market identification
  - Target: 10-30 additional names per day

**Expansion rule:** A stage only unlocks if Stage 1 produces out-of-sample Sharpe > 1.0 on SPY/QQQ. If it doesn't, we iterate on the strategy before expanding.

### Storage

- Format: Parquet files (columnar, fast reads, compression)
- Structure: one file per ticker per timeframe (e.g., `data/1m/SPY.parquet`)
- Cache layer to avoid redundant API calls
- Estimated size: Stage 1 ~500 MB, full dataset ~5-10 GB

---

## Phase 1: Higher-Timeframe Context Discovery

**Question: Which daily/weekly conditions predict strong intraday momentum?**

We test each HTF factor independently first, then combine.

### Factors to Test

**Stage 1 factors (SPY + QQQ):**

**A. Trend & Regime**

| # | Factor | Computation | Hypothesis |
|---|--------|-------------|------------|
| 1 | **Trend direction** | Price vs 10/20/50/200-day EMA | Intraday momentum is stronger in the direction of the daily trend |
| 2 | **Trend strength** | ADX(14) on daily | Trending regimes produce better momentum continuation |
| 3 | **Volatility regime** | ATR(14) percentile, VIX level | Momentum strategies behave differently in high vs low vol |
| 4 | **SPY vs QQQ divergence** | Relative return SPY vs QQQ over 1/5/10 days | Divergence or convergence as a regime signal |

**B. Prior Period Range Context**

This is critical. Institutions watch these levels. Price leaving a prior period range is a high-information event.

| # | Factor | Computation | Hypothesis |
|---|--------|-------------|------------|
| 5 | **Position vs prior day range** | Price relative to prior day's high/low. Above = leaving range up. Below = leaving range down. Inside = contained. | Breaking out of yesterday's range creates follow-through momentum |
| 6 | **Position vs prior week range** | Price relative to prior week's high/low (Mon-Fri) | Weekly range breaks are higher-conviction, attract more flow |
| 7 | **Position vs prior month range** | Price relative to prior calendar month's high/low | Monthly range breaks are major structural events |
| 8 | **Multi-period range alignment** | Is price breaking the daily AND weekly range in the same direction? Or just one? | Aligned multi-period breaks = strongest momentum |
| 9 | **Distance from prior range edge** | How far past the prior day/week/month high or low (as % of that period's range) | Fresh breakouts vs extended ones behave differently |
| 10 | **Days since range break** | How many days has price been outside the prior week/month range? | Early breakout days may have more momentum than day 5 of the move |

**C. Prior Bar / Prior Day Character**

The character of the prior bar tells you about conviction and what to expect next.

| # | Factor | Computation | Hypothesis |
|---|--------|-------------|------------|
| 11 | **Prior day range width** | Prior day's (high - low) as % of close, ranked vs 20-day average | Wide range days may exhaust or signal trend; narrow range days set up expansion |
| 12 | **Prior day range percentile** | Prior day range vs last 50 days (NR4, NR7, WR4, WR7 classification) | NR days (compression) → expansion; WR days → exhaustion or continuation |
| 13 | **Prior day volume** | Prior day volume vs 20-day average volume | High volume days = institutional involvement; low volume = drift |
| 14 | **Prior day volume × range** | Was it wide range on high volume (conviction) or wide range on low volume (thin market)? | High vol + wide range = real move. Low vol + wide range = gap fill candidate |
| 15 | **Prior day close location** | Close in top 25% / bottom 25% / middle 50% of the day's range | Close near high = buyers in control going into next day |
| 16 | **Prior day body vs wick ratio** | (close - open) / (high - low) — how much of the range was "body" | Large body = conviction. Small body (doji) = indecision |
| 17 | **Consecutive day direction** | N days closing in same direction (up or down streak) | Streaks may mean trending (continuation) or overextension (reversal) |
| 18 | **Inside day / outside day** | Today's range fully inside yesterday's? Or engulfing? | Inside day = coil, outside day = expansion event |

**D. Gap & Opening Conditions**

| # | Factor | Computation | Hypothesis |
|---|--------|-------------|------------|
| 19 | **Gap classification** | Overnight gap % (close-to-open), bucketed: <0.5%, 0.5-1%, 1-2%, 2%+ | Gap size affects fill probability and continuation |
| 20 | **Gap vs trend alignment** | Gap direction matches daily trend direction or not | Aligned gaps continue; counter-trend gaps fade |
| 21 | **Gap vs prior range** | Does the gap open above/below the prior day/week/month range? | Gapping out of a range = breakaway gap = strongest continuation |
| 22 | **Relative volume at open** | First 5/15/30 min volume vs same-window 20-day avg | High RVOL at open = institutional interest = momentum |

**E. Key Levels**

| # | Factor | Computation | Hypothesis |
|---|--------|-------------|------------|
| 23 | **Distance from 52-week high/low** | % from 52-wk high and low | Near-high breakouts vs mid-range vs near-low bounces |
| 24 | **Round number proximity** | Distance to nearest $5/$10/$50 round number | Round numbers act as psychological support/resistance |

**Factors added in later stages:**

| Factor | Stage | Rationale |
|--------|-------|-----------|
| Relative strength vs SPY | Stage 2+ | Needs multiple assets to compute cross-sectional ranking |
| Sector relative strength | Stage 3+ | Requires sector ETF universe |
| Cross-asset momentum | Stage 4+ | Full universe needed |

### Methodology — Same P&L Snapshot Approach

Phase 1 uses the same P&L snapshot engine as Phase 2, just at a different granularity. For each HTF factor:

1. Classify each ticker-day into bins (e.g., "above 20 EMA" vs "below 20 EMA", "leaving prior week range up" vs "inside prior week range" vs "leaving down")
2. For each bin, build a **daily P&L snapshot matrix**: enter at the open of that day, snapshot the intraday P&L at every 5m/15m/30m bar through end of day
3. Compare the average P&L curve shape across bins — does "leaving prior day range up" produce a meaningfully different forward path than "inside prior day range"?
4. Statistical tests:
   - t-test for mean difference between bins
   - Mann-Whitney U for non-parametric confirmation
   - Effect size (Cohen's d)
   - Bootstrap confidence intervals
5. Minimum threshold: p < 0.01 AND effect size > 0.2 to be considered meaningful

This means Phase 1 doesn't just tell us "this HTF context is good or bad" — it shows us the *shape* of the intraday P&L path under each HTF regime, which feeds directly into Phase 2's condition combinations.

### Output

A ranked list of HTF factors by the quality of forward intraday P&L curves they produce. Each factor gets a P&L curve per bin, so we can see exactly how different regimes shape intraday price action. Interaction effects tested as well (e.g., "leaving prior week range + gap up + RVOL high" may produce a dramatically steeper curve than any factor alone).

---

## Phase 2: Intraday Condition Scanning via P&L Snapshots

**Question: Which observable market conditions, when they occur, produce favorable forward P&L paths?**

We do NOT pre-define "entry signals." Instead, we define a broad set of **measurable market conditions** — things we can observe at any bar close — and then snapshot the forward P&L path after every occurrence. The conditions that consistently produce upward-sloping P&L curves with statistical significance *become* our entry signals. The data defines the entries, not us.

### Observable Conditions to Measure

These are not trading signals — they are things we can compute at every bar close. We measure what happens next.

**A. Price vs HTF Range Levels (intraday view of Phase 1 range context)**

Every intraday bar is aware of where it sits relative to the higher-timeframe ranges. This is the bridge between Phase 1 and Phase 2.

| # | Condition | Computation |
|---|-----------|-------------|
| 1 | Price vs prior day range | Current bar relative to yesterday's high/low: above, below, or inside |
| 2 | Price vs prior week range | Current bar relative to last week's high/low |
| 3 | Price vs prior month range | Current bar relative to last month's high/low |
| 4 | Breakout bar detection | Is this the bar that just crossed above/below a prior day/week/month high or low? |
| 5 | Distance past range edge | If outside a prior range, how far (as % of that range)? Fresh breakout vs extended |
| 6 | Multi-range alignment | Breaking daily AND weekly in same direction vs just one |

**B. Price-based conditions:**

| # | Condition | Computation |
|---|-----------|-------------|
| 7 | Price vs VWAP | Bar close above/below VWAP, and by how much (z-score of distance) |
| 8 | Price vs intraday EMAs | Position relative to 9/20/50 EMA on the current timeframe |
| 9 | New N-bar high/low | Price making highest close of last 5/10/20/50 bars |
| 10 | Opening range position | Price relative to first 5/15/30/60 min high and low |
| 11 | Consecutive directional bars | N bars closing in same direction (1, 2, 3, 4, 5+) |
| 12 | Return over last N bars | ROC percentile over trailing 5/10/20/50 bars vs its own history |
| 13 | Distance from session high/low | % away from the day's high or low at this point |

**C. Prior bar character (intraday mirror of the daily prior-bar analysis in Phase 1):**

Just like we analyze the prior day's character in Phase 1, we analyze the prior bar's character on the intraday timeframe. A wide-range, high-volume bar followed by a narrow, low-volume bar is a very different setup than two consecutive wide-range bars.

| # | Condition | Computation |
|---|-----------|-------------|
| 14 | Prior bar range width | Prior bar's (high - low) vs 20-bar average range. Wide (>1.5x), normal, narrow (<0.5x) |
| 15 | Prior bar range percentile | Prior bar range ranked vs last 50 bars (narrowest 10%? widest 10%?) |
| 16 | Prior bar close location | Did the prior bar close near its high (top 25%), low (bottom 25%), or middle? |
| 17 | Prior bar body vs wick | (close - open) / (high - low) of prior bar — conviction vs indecision |
| 18 | Prior bar volume | Prior bar volume vs 20-bar average. High (>2x), normal, low (<0.5x) |
| 19 | Prior bar volume × range | Was it wide range + high volume (conviction) or wide range + low volume (thin/fake)? |
| 20 | Range contraction sequence | Number of consecutive bars with decreasing range (1, 2, 3+) — coiling |
| 21 | Range expansion after contraction | Current bar range > prior bar range, after N bars of contraction — spring releasing |

**D. Volume-based conditions:**

| # | Condition | Computation |
|---|-----------|-------------|
| 22 | Relative volume (bar) | Current bar volume vs 20-bar average (1x, 2x, 3x+) |
| 23 | Relative volume (session) | Cumulative session volume vs same-time 20-day average |
| 24 | Volume trend | Volume increasing or decreasing over last N bars |
| 25 | Volume × direction | High-volume bar with strong directional close vs weak close |
| 26 | Volume climax | Volume > 3x 20-bar avg with bar closing near low (exhaustion?) or near high (thrust?) |

**E. Volatility/range conditions:**

| # | Condition | Computation |
|---|-----------|-------------|
| 27 | ATR expansion/compression | Current N-bar ATR vs 50-bar ATR (ratio) |
| 28 | Bar range vs average | Current bar's range relative to recent average bar range |
| 29 | Bollinger bandwidth | Squeeze (narrow bands) vs expansion |

**F. Time-of-day conditions:**

| # | Condition | Computation |
|---|-----------|-------------|
| 30 | Time bucket | First 30 min, mid-morning, lunch, afternoon, last 30 min |
| 31 | Minutes since open | Continuous variable, binned |

**G. Cross-asset conditions (SPY vs QQQ in Stage 1):**

| # | Condition | Computation |
|---|-----------|-------------|
| 32 | Intraday relative performance | QQQ return vs SPY return from open to current bar |
| 33 | Lead/lag | Which one moved first in current session? Does the laggard catch up? |

### The P&L Snapshot Engine

This is the core of the entire research pipeline. The same engine is used for both entry discovery (this phase) and exit discovery.

**For every occurrence of every condition, on every timeframe (1m, 5m, 15m, 30m, 60m):**

1. Record the bar close price at the moment the condition is observed
2. Snapshot the mark-to-market P&L at every subsequent bar close through end of day
3. Store as a row in the P&L matrix for that condition

**P&L Matrix structure (one per condition × timeframe × ticker):**

```
                Bar+1   Bar+2   Bar+3   Bar+4  ...  Bar+N (EOD)
Occurrence 1    +0.05%  +0.12%  +0.08%  +0.15% ...  +0.22%
Occurrence 2    -0.03%  -0.08%  -0.12%  -0.06% ...  +0.04%
Occurrence 3    +0.10%  +0.18%  +0.25%  +0.30% ...  +0.19%
...
Occurrence M    ...

HTF Context:    [trend_up, gap_up, rvol_high, ...]  ← tags from Phase 1
```

Each occurrence also carries its HTF context tags from Phase 1, so we can slice the matrix later.

### What We Extract

**Per-condition analysis:**

- **Average P&L curve** — mean P&L at each bar-after-occurrence. Upward slope = momentum continuation. Downward = mean reversion. Flat = no edge.
- **Median P&L curve** — if mean and median diverge, the edge is driven by outliers (fragile).
- **P&L percentile bands** — 10th/25th/75th/90th at each bar. How wide is the distribution?
- **Peak P&L timing** — histogram of which bar each occurrence hits its max P&L. Tight cluster = predictable hold time. Dispersed = unreliable.
- **Win rate evolution** — % positive at bar+1, bar+2, ... bar+N. Rising = trend continuation. Falling = fading edge.
- **Max runup / max drawdown per occurrence** — MFE and MAE distributions, derived naturally.

**Cross-condition comparison:**

- Rank all conditions by: average P&L at peak bar, Sharpe of the P&L curve, consistency (median vs mean), sample size
- Identify conditions with the best *shape*: steep early rise, tight bands, high win rate

**HTF-conditioned analysis:**

- Slice every P&L matrix by HTF context buckets from Phase 1
- Compare: same intraday condition, different HTF regime → different P&L curve?
- This reveals the interaction: which intraday conditions are enhanced by which HTF contexts?

**Condition combinations:**

- After identifying individually strong conditions, test combinations: "new 20-bar high AND volume > 2x average AND above VWAP"
- Does combining conditions sharpen the P&L curve (steeper, tighter bands)?
- Track sample size — combinations filter aggressively, need enough occurrences for significance

### Defining Entries and Exits From the Data

The P&L matrices produce entries and exits simultaneously:

**Entries emerge as:** conditions (or combinations) where the forward P&L curve is significantly positive with high consistency across the sample. The entry "signal" is simply: this condition is present.

**Exits emerge as:**
- **Target** = bar where the average P&L curve peaks
- **Stop** = max drawdown that 90% of eventually-profitable occurrences never exceeded
- **Time stop** = bar after which the average P&L decays below peak by a threshold
- **No exit pre-definition needed** — the optimal hold period is visible in the curve shape

### Why This Approach

- No pattern-matching bias: we're not testing "ORB breakouts" because we think they work. We're measuring every observable condition and seeing which ones *actually* predict forward returns.
- Conditions we might never have thought to test emerge naturally from the data.
- Entries and exits are derived from the same framework — no Frankenstein of different methodologies.
- Easy to add new conditions later without restructuring anything.
- The P&L matrix is the single source of truth for everything.

### Output

A ranked list of conditions (and combinations) by forward P&L quality, with their associated optimal hold time and exit parameters, sliced by HTF context.

---

## Phase 3: Multi-Timeframe Combination

**Question: What is the optimal combination of HTF context + intraday signal?**

### Approach

1. Take the top HTF factors from Phase 1 (those with p < 0.01, effect size > 0.2)
2. Take the top intraday signals from Phase 2
3. Build a scoring system:
   - HTF context score (0-100): weighted sum of active HTF factors
   - Intraday signal score (0-100): signal strength + confirmation
   - Combined score = f(HTF, intraday) — test linear and non-linear combinations
4. Evaluate combined strategies:
   - Only trade when HTF score > threshold AND intraday signal fires
   - Compare: combined vs HTF-only vs signal-only vs random baseline

### Walk-Forward Validation

**Critical to avoid overfitting:**

- Split data: 60% in-sample (2019-2022), 20% validation (2022-2023), 20% out-of-sample (2023-2025)
- Train on in-sample, tune on validation, report on out-of-sample only
- Walk-forward: retrain every 6 months with expanding window
- If out-of-sample Sharpe < 50% of in-sample Sharpe → likely overfit, discard

---

## Phase 4: Robustness and Stress Testing

Before committing capital, we need to break the strategy:

1. **Transaction cost sensitivity**: test at 5, 10, 20, 50 bps one-way
2. **Slippage modeling**: estimate realistic fills using bid-ask spreads from Polygon
3. **Regime analysis**: does the strategy work in bull markets, bear markets, and sideways?
4. **Correlation to benchmark**: is this just leveraged beta or genuine alpha?
5. **Capacity analysis**: at what AUM does market impact erode the edge?
6. **Bootstrap**: resample returns 10,000x to get confidence intervals on all metrics
7. **Survivorship bias check**: include delisted tickers if possible
8. **Out-of-sample performance**: the only number that matters

---

## Phase 5: Signal Dashboard (Post-Research)

Once we have a validated strategy, we build the live system:

- Real-time Polygon WebSocket for price/volume
- HTF context computed daily pre-market
- Intraday signals computed on each bar close
- Alert system: push notifications or dashboard when a setup triggers
- Position tracker with P&L
- Risk limits and daily loss caps

This phase only begins after we have statistically validated edge from Phases 1-4.

---

## Success Criteria

A strategy is worth trading if:

- Out-of-sample Sharpe > 1.0 (after costs)
- t-stat on mean daily return > 2.0 (p < 0.05)
- Bootstrap 90% CI for Sharpe does not include 0
- Edge persists across market regimes
- Turnover and costs don't eat the alpha
- Minimum 500+ trade sample size for statistical power

---

## Execution Order

```
1. Polygon API key → Build data pipeline for SPY + QQQ
2. Phase 1 on SPY + QQQ → Which HTF contexts matter?
3. Phase 2 on SPY + QQQ → Which intraday signals work within those contexts?
4. Phase 3 on SPY + QQQ → Combine and walk-forward validate
5. GATE: Out-of-sample Sharpe > 1.0?
   → YES: Expand to Stage 2 (IWM, GLD, TLT, IBIT) and repeat
   → NO: Iterate on strategy design before expanding
6. Phase 4 robustness testing on validated strategy
7. Phase 5 live signal dashboard
```

## Next Step

Provide Polygon API key → Build data pipeline → Begin Phase 1 on SPY + QQQ.
