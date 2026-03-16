"""
Polygon.io API client with pagination, rate limiting, and retry logic.
"""
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from src.config import POLYGON_API_KEY, POLYGON_BASE_URL, REQUEST_DELAY


class PolygonClient:
    """Thin wrapper around Polygon REST API for aggregate bars."""

    def __init__(self, api_key: str = POLYGON_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict = None, retries: int = 3) -> dict:
        """Make a GET request with retry logic."""
        for attempt in range(retries):
            self._rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    # Rate limited — back off
                    wait = 2 ** (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                    continue

            except requests.exceptions.RequestException as e:
                print(f"    Request error: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue

        return {}

    def get_aggregates(self,
                       ticker: str,
                       multiplier: int,
                       timespan: str,
                       from_date: str,
                       to_date: str,
                       limit: int = 50000,
                       ) -> pd.DataFrame:
        """
        Fetch aggregate bars with automatic pagination.

        Parameters
        ----------
        ticker : str
        multiplier : int (e.g., 5 for 5-minute bars)
        timespan : str ("minute", "hour", "day", "week")
        from_date, to_date : str (YYYY-MM-DD)
        limit : int (max results per page, Polygon max = 50000)

        Returns
        -------
        DataFrame with columns: timestamp, open, high, low, close, volume, vwap, trades
        """
        all_results = []
        url = (f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}"
               f"/range/{multiplier}/{timespan}/{from_date}/{to_date}")
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit,
        }

        page = 0
        while url:
            page += 1
            data = self._get(url, params=params if page == 1 else None)

            if not data or data.get("resultsCount", 0) == 0:
                break

            results = data.get("results", [])
            all_results.extend(results)

            # Check for next page
            next_url = data.get("next_url")
            if next_url:
                # Polygon includes the API key in next_url for v2,
                # but we use Bearer auth so just use the URL
                url = next_url
                if "apiKey" not in next_url:
                    url = f"{next_url}&apiKey={self.api_key}"
                params = None  # params are in the next_url
            else:
                url = None

            if page % 10 == 0:
                print(f"    Page {page}, {len(all_results):,} bars so far...")

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)

        # Rename columns to standard names
        col_map = {
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "trades",
        }
        df = df.rename(columns=col_map)

        # Convert timestamp (Polygon uses milliseconds)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
        df = df.set_index("timestamp").sort_index()

        # Keep only the columns we need
        keep = ["open", "high", "low", "close", "volume", "vwap", "trades"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep]

        return df

    def test_connection(self) -> bool:
        """Test API connectivity and key validity."""
        data = self._get(
            f"{POLYGON_BASE_URL}/v2/aggs/ticker/SPY/range/1/day/2025-01-02/2025-01-03"
        )
        if data and data.get("resultsCount", 0) > 0:
            print("  Polygon API connection OK")
            return True
        print(f"  Polygon API connection FAILED: {data}")
        return False
