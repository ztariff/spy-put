# Project: SPX 0DTE Opening Print Strategy

## Polygon API
- **API Key:** `YOUR_POLYGON_API_KEY`
- **Plan:** Top-tier paid plan. Always assume full access to all endpoints, tick-level data, unlimited calls, and options data. Never throttle, downsample, or limit requests based on free-tier assumptions.

## Core Rules

### Never fabricate data
Never generate synthetic, placeholder, or simulated data to stand in for real market data — not even temporarily, not even as a fallback, not even "until the real data loads." This includes seeded random numbers, normal distributions, dummy P&L values, fake price series, or any other invented numbers presented as if they reflect reality. If real data is unavailable (API error, missing contract, rate limit), surface that failure clearly rather than silently filling in made-up values. The user must always be able to trust that every number on screen came from a real data source.

### Never use theoretical pricing models as a substitute for real data
Do not use Black-Scholes, binomial models, Greeks-based estimation, or any other theoretical pricing model to generate option prices, P&L, or trade outcomes when real market data (actual prints, OHLC bars, trades) is available or obtainable. Theoretical models are acceptable only for supplementary analysis (e.g., estimating Greeks for context) — never as the source of truth for P&L, entry/exit prices, or backtest results.

### Be thorough — never cut corners
Always prioritize completeness and precision in data analysis and collection. Never skip steps, truncate datasets, use approximations, or reduce granularity to save compute, tokens, API calls, or time. If a task requires processing every row, fetching every contract, or checking every date — do exactly that. Do not summarize when the user expects exhaustive output. Do not sample when the user expects the full population.

### Never silently accept missing data when it can be obtained
If a backtest, scan, or analysis identifies dates/contracts/signals that should be priced or evaluated but the required data is not in the local cache, do not silently skip those dates and present partial results as if they are complete. Instead: (1) quantify exactly how many dates/contracts are missing, (2) surface this gap to the user immediately, and (3) offer to fetch the missing data from the API before proceeding. Partial results are acceptable only if the user explicitly chooses to proceed without the missing data. Never present a backtest with more than 5% of signal days missing as a finished product.

### Surface problems, don't hide them
If something looks wrong — P&L doesn't match, data is missing, a calculation contradicts expectations — flag it immediately. Never silently "fix" discrepancies by smoothing over them, and never present results that paper over known issues. The user would rather see an ugly truth than a polished lie.

### Scripts meant for local execution must write output to a shared file
When generating a script that the user will run outside the sandbox (e.g., because outbound API calls are blocked), always write all results, logs, and summaries to a file inside the project directory (e.g., `backtest_results/` or a clearly named `.txt`/`.json` alongside the script). Never rely on the user to copy-paste terminal output back. The script's output file should contain everything needed to continue analysis — so that on the next turn, the results can simply be read from disk.

## Project-Specific Context
- SPX 0DTE options use SPXW ticker format: `O:SPXW{YYMMDD}{C/P}{strike*1000:08d}`
- Polygon requires `I:SPX` prefix for index tickers on aggs endpoint
- Bear call spread P&L: `(credit_received - debit_to_close) * contracts * 100`
- Single-leg long P&L: `(exit_price - entry_price) * contracts * 100`
- Single-leg short (bearish) P&L: `-(exit_price - entry_price) * contracts * 100`
- Forward-walking signal detection only — never use hindsight/peak detection for entry/exit timing
- All backtest logic must match live strategy logic exactly
