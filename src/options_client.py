"""
Polygon.io API client for options data.
Handles contract lookup, 5-min bar fetching, and ticker construction.
Includes disk caching to avoid redundant API calls.
"""
import os
import time
import json
import hashlib
import requests
import pandas as pd
from datetime import datetime, date, timedelta

from src.config import POLYGON_API_KEY, POLYGON_BASE_URL, REQUEST_DELAY, DATA_DIR


CACHE_DIR = os.path.join(DATA_DIR, "options_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class OptionsClient:
    """Polygon.io options API client with caching."""

    def __init__(self, api_key: str = POLYGON_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._last_request_time = 0
        self._memory_cache = {}

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict = None, retries: int = 3) -> dict:
        for attempt in range(retries):
            self._rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                    continue
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue
        return {}

    # ── Ticker Construction ──────────────────────────────────────────────────

    @staticmethod
    def construct_ticker(underlying: str, expiry_date: date, strike: float, call_put: str) -> str:
        """
        Build Polygon options ticker in OCC format.
        Example: O:SPY250224C00580000
          underlying = SPY
          expiry_date = 2025-02-24
          strike = 580.0
          call_put = 'C' or 'P'
        """
        yy = expiry_date.strftime("%y")
        mm = expiry_date.strftime("%m")
        dd = expiry_date.strftime("%d")
        cp = call_put[0].upper()
        # Strike in OCC format: dollars * 1000, zero-padded to 8 digits
        strike_int = int(round(strike * 1000))
        strike_str = f"{strike_int:08d}"
        return f"O:{underlying}{yy}{mm}{dd}{cp}{strike_str}"

    # ── Contract Lookup ──────────────────────────────────────────────────────

    def get_option_contracts(self, underlying: str, expiry_date: date,
                             contract_type: str = None) -> list:
        """
        Fetch available option contracts for an underlying on a specific expiry.

        Returns list of dicts: [{strike, type, ticker, expiry}, ...]
        """
        cache_key = f"contracts_{underlying}_{expiry_date}_{contract_type}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Check disk cache
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                data = json.load(f)
                self._memory_cache[cache_key] = data
                return data

        url = f"{POLYGON_BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "expiration_date": str(expiry_date),
            "limit": 1000,
            "expired": "true",
        }
        if contract_type:
            params["contract_type"] = contract_type.lower()

        all_contracts = []
        while url:
            data = self._get(url, params=params if not all_contracts else None)
            if not data:
                break
            results = data.get("results", [])
            for c in results:
                all_contracts.append({
                    "strike": c.get("strike_price"),
                    "type": c.get("contract_type", "").upper(),
                    "ticker": c.get("ticker"),
                    "expiry": c.get("expiration_date"),
                })
            next_url = data.get("next_url")
            if next_url:
                url = next_url
                if "apiKey" not in next_url:
                    url = f"{next_url}&apiKey={self.api_key}"
                params = None
            else:
                url = None

        # Cache to disk
        with open(cache_file, "w") as f:
            json.dump(all_contracts, f)
        self._memory_cache[cache_key] = all_contracts
        return all_contracts

    def get_available_expiries(self, underlying: str, trade_date: date) -> list:
        """
        Get available expiration dates for an underlying around a trade date.
        Uses a single ATM strike to quickly find which expiries exist (avoids
        downloading thousands of contracts across all strikes).
        Returns sorted list of date objects.
        """
        cache_key = f"expiries_{underlying}_{trade_date}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                dates = [date.fromisoformat(d) for d in json.load(f)]
                self._memory_cache[cache_key] = dates
                return dates

        # Query with a narrow strike range to limit results and find expiries fast
        url = f"{POLYGON_BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "expiration_date.gte": str(trade_date),
            "expiration_date.lte": str(trade_date + timedelta(days=8)),
            "contract_type": "call",
            "expired": "true",
            "limit": 1000,
            "sort": "expiration_date",
            "order": "asc",
        }

        expiries = set()
        page_url = url
        page = 0
        while page_url:
            page += 1
            data = self._get(page_url, params=params if page == 1 else None)
            if not data:
                break
            for c in data.get("results", []):
                exp = c.get("expiration_date")
                if exp:
                    expiries.add(exp)
            # If we have expiries already, no need to paginate further
            if expiries:
                break
            next_url = data.get("next_url")
            if next_url:
                page_url = next_url
                if "apiKey" not in next_url:
                    page_url = f"{next_url}&apiKey={self.api_key}"
                params = None
            else:
                page_url = None

        sorted_expiries = sorted([date.fromisoformat(e) for e in expiries])

        with open(cache_file, "w") as f:
            json.dump([str(d) for d in sorted_expiries], f)
        self._memory_cache[cache_key] = sorted_expiries
        return sorted_expiries

    # ── Options Bars ─────────────────────────────────────────────────────────

    def get_options_bars(self, option_ticker: str, from_date: str, to_date: str,
                         multiplier: int = 5, timespan: str = "minute") -> pd.DataFrame:
        """
        Fetch OHLCV bars for a specific option contract.
        Returns DataFrame with columns: open, high, low, close, volume, vwap
        """
        # Disk cache
        safe_ticker = option_ticker.replace(":", "_").replace("/", "_")
        cache_file = os.path.join(CACHE_DIR, f"bars_{safe_ticker}_{multiplier}m_{from_date}_{to_date}.parquet")

        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        url = (f"{POLYGON_BASE_URL}/v2/aggs/ticker/{option_ticker}"
               f"/range/{multiplier}/{timespan}/{from_date}/{to_date}")
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }

        all_results = []
        page_url = url
        page = 0
        while page_url:
            page += 1
            data = self._get(page_url, params=params if page == 1 else None)
            if not data or data.get("resultsCount", 0) == 0:
                break
            all_results.extend(data.get("results", []))
            next_url = data.get("next_url")
            if next_url:
                page_url = next_url
                if "apiKey" not in next_url:
                    page_url = f"{next_url}&apiKey={self.api_key}"
                params = None
            else:
                page_url = None

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        col_map = {"t": "timestamp", "o": "open", "h": "high", "l": "low",
                    "c": "close", "v": "volume", "vw": "vwap", "n": "trades"}
        df = df.rename(columns=col_map)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
        df = df.set_index("timestamp").sort_index()
        keep = [c for c in ["open", "high", "low", "close", "volume", "vwap", "trades"] if c in df.columns]
        df = df[keep]

        # Cache to disk
        df.to_parquet(cache_file)
        return df

    # ── Convenience ──────────────────────────────────────────────────────────

    def get_option_price_at_time(self, option_ticker: str, target_time: pd.Timestamp,
                                  trade_date: date, multiplier: int = 5) -> float:
        """
        Get the option close price at or near a specific timestamp.
        Returns NaN if no data available.
        """
        bars = self.get_options_bars(
            option_ticker,
            from_date=str(trade_date),
            to_date=str(trade_date),
            multiplier=multiplier,
        )
        if bars.empty:
            return float("nan")

        # Find the bar at or just before target_time
        target = target_time
        if target.tzinfo is None:
            target = target.tz_localize("America/New_York")
        elif str(target.tzinfo) != "America/New_York":
            target = target.tz_convert("America/New_York")

        # Get bars at or before target time
        valid = bars[bars.index <= target]
        if valid.empty:
            # Try the first available bar after target
            valid = bars[bars.index >= target]
            if valid.empty:
                return float("nan")
            return float(valid.iloc[0]["close"])

        return float(valid.iloc[-1]["close"])
