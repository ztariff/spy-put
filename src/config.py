"""
Central configuration for the intraday momentum research framework.
"""
import os

# ─── Polygon API ─────────────────────────────────────────────────────────────
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF")
POLYGON_BASE_URL = "https://api.polygon.io"

# ─── Universe (Staged Expansion) ────────────────────────────────────────────
STAGE_1_TICKERS = ["SPY", "QQQ"]
STAGE_2_TICKERS = ["IWM", "DIA", "GLD", "TLT", "IBIT"]
STAGE_3_TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLRE", "XLB", "XLC"]
STAGE_4_TICKERS = ["EFA", "EEM", "SLV", "USO", "DBC", "ETHA", "BITO", "TQQQ", "SOXL"]

# Active tickers for current stage
ACTIVE_TICKERS = STAGE_1_TICKERS

# ─── Data Parameters ────────────────────────────────────────────────────────
# Intraday bar sizes to download
INTRADAY_TIMEFRAMES = {
    "1m":  {"multiplier": 1,  "timespan": "minute"},
    "5m":  {"multiplier": 5,  "timespan": "minute"},
    "15m": {"multiplier": 15, "timespan": "minute"},
    "30m": {"multiplier": 30, "timespan": "minute"},
    "60m": {"multiplier": 60, "timespan": "minute"},
}

# HTF bar sizes
HTF_TIMEFRAMES = {
    "1D": {"multiplier": 1, "timespan": "day"},
    "1W": {"multiplier": 1, "timespan": "week"},
}

# Date ranges
INTRADAY_START = "2021-01-01"   # 5 years of intraday data
DAILY_START    = "2016-01-01"   # 10 years of daily data
DATA_END       = "2026-02-24"   # Today

# ─── Data Storage ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# ─── Session Times (Eastern) ────────────────────────────────────────────────
PREMARKET_START = "04:00"
MARKET_OPEN     = "09:30"
MARKET_CLOSE    = "16:00"
POSTMARKET_END  = "20:00"

# ─── Analysis Parameters ────────────────────────────────────────────────────
# Minimum occurrences for a condition to be considered statistically valid
MIN_OCCURRENCES = 200
MIN_COMBO_OCCURRENCES = 100

# Statistical significance thresholds
P_VALUE_THRESHOLD = 0.01
EFFECT_SIZE_THRESHOLD = 0.2

# ─── Rate Limiting ──────────────────────────────────────────────────────────
# Polygon Developer+ plan: unlimited calls, but be respectful
REQUESTS_PER_SECOND = 50
REQUEST_DELAY = 1.0 / REQUESTS_PER_SECOND
