"""
Data download orchestrator.
Downloads all required data from Polygon and saves to Parquet files.
Supports incremental updates (only downloads missing date ranges).
"""
import os
import time
import pandas as pd
from datetime import datetime, timedelta

from src.config import (
    ACTIVE_TICKERS, INTRADAY_TIMEFRAMES, HTF_TIMEFRAMES,
    INTRADAY_START, DAILY_START, DATA_END, DATA_DIR,
)
from src.polygon_client import PolygonClient


def _parquet_path(ticker: str, timeframe: str) -> str:
    """Get the parquet file path for a ticker/timeframe combo."""
    tf_dir = os.path.join(DATA_DIR, timeframe)
    os.makedirs(tf_dir, exist_ok=True)
    return os.path.join(tf_dir, f"{ticker}.parquet")


def _load_existing(path: str) -> pd.DataFrame:
    """Load existing parquet file if it exists."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def _save_parquet(df: pd.DataFrame, path: str):
    """Save DataFrame to parquet."""
    df.to_parquet(path, engine="pyarrow")


def download_ticker_timeframe(client: PolygonClient,
                               ticker: str,
                               timeframe: str,
                               multiplier: int,
                               timespan: str,
                               start_date: str,
                               end_date: str,
                               chunk_days: int = 30,
                               ) -> pd.DataFrame:
    """
    Download data for one ticker/timeframe, chunking by date range
    to stay within Polygon's result limits.

    For minute data, we chunk into 30-day windows.
    For daily/weekly, we download in one shot.
    """
    path = _parquet_path(ticker, timeframe)
    existing = _load_existing(path)

    # Determine actual start date (resume from where we left off)
    actual_start = pd.Timestamp(start_date).tz_localize(None)
    if not existing.empty:
        last_date = existing.index.max()
        # Strip timezone if present for comparison
        if hasattr(last_date, 'tz') and last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        # Start from the day after the last data point
        actual_start = max(actual_start, last_date + timedelta(days=1))
        print(f"    Resuming from {actual_start.date()} (have {len(existing):,} bars)")

    actual_end = pd.Timestamp(end_date).tz_localize(None)

    if actual_start >= actual_end:
        print(f"    Already up to date ({len(existing):,} bars)")
        return existing

    # For intraday data, chunk into windows to avoid hitting result limits
    if timespan == "minute":
        all_chunks = []
        chunk_start = actual_start

        while chunk_start < actual_end:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), actual_end)
            print(f"    Fetching {chunk_start.date()} → {chunk_end.date()}...", end=" ")

            df = client.get_aggregates(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=chunk_start.strftime("%Y-%m-%d"),
                to_date=chunk_end.strftime("%Y-%m-%d"),
            )

            if not df.empty:
                all_chunks.append(df)
                print(f"{len(df):,} bars")
            else:
                print("0 bars")

            chunk_start = chunk_end + timedelta(days=1)

        if all_chunks:
            new_data = pd.concat(all_chunks)
        else:
            new_data = pd.DataFrame()
    else:
        # Daily/weekly: single request
        print(f"    Fetching {actual_start.date()} → {actual_end.date()}...", end=" ")
        new_data = client.get_aggregates(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_date=actual_start.strftime("%Y-%m-%d"),
            to_date=actual_end.strftime("%Y-%m-%d"),
        )
        if not new_data.empty:
            print(f"{len(new_data):,} bars")
        else:
            print("0 bars")

    # Combine existing + new
    if not existing.empty and not new_data.empty:
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
    elif not new_data.empty:
        combined = new_data.sort_index()
    else:
        combined = existing

    # Save
    if not combined.empty:
        _save_parquet(combined, path)
        print(f"    Saved → {path} ({len(combined):,} bars)")

    return combined


def download_all(tickers: list[str] = None):
    """
    Download all data for all tickers and timeframes.
    This is the main entry point for data collection.
    """
    if tickers is None:
        tickers = ACTIVE_TICKERS

    client = PolygonClient()

    # Test connection first
    print("\n" + "=" * 70)
    print("TESTING POLYGON CONNECTION")
    print("=" * 70)
    if not client.test_connection():
        print("  FATAL: Cannot connect to Polygon API. Check your API key.")
        return

    total_start = time.time()

    # ── Download HTF (daily/weekly) data ─────────────────────────────────
    print("\n" + "=" * 70)
    print("DOWNLOADING HTF DATA (Daily / Weekly)")
    print("=" * 70)

    for ticker in tickers:
        for tf_name, tf_config in HTF_TIMEFRAMES.items():
            print(f"\n  [{ticker}] {tf_name}:")
            download_ticker_timeframe(
                client=client,
                ticker=ticker,
                timeframe=tf_name,
                multiplier=tf_config["multiplier"],
                timespan=tf_config["timespan"],
                start_date=DAILY_START,
                end_date=DATA_END,
            )

    # ── Download intraday data ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DOWNLOADING INTRADAY DATA")
    print("=" * 70)

    for ticker in tickers:
        for tf_name, tf_config in INTRADAY_TIMEFRAMES.items():
            print(f"\n  [{ticker}] {tf_name}:")

            # For 1-minute data, use smaller chunks (more data per day)
            chunk_days = 7 if tf_config["multiplier"] == 1 else 30

            download_ticker_timeframe(
                client=client,
                ticker=ticker,
                timeframe=tf_name,
                multiplier=tf_config["multiplier"],
                timespan=tf_config["timespan"],
                start_date=INTRADAY_START,
                end_date=DATA_END,
                chunk_days=chunk_days,
            )

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"DOWNLOAD COMPLETE ({elapsed/60:.1f} minutes)")
    print("=" * 70)

    # Print data inventory
    print("\n  Data Inventory:")
    for tf_dir in sorted(os.listdir(DATA_DIR)):
        tf_path = os.path.join(DATA_DIR, tf_dir)
        if os.path.isdir(tf_path):
            for f in sorted(os.listdir(tf_path)):
                if f.endswith(".parquet"):
                    fp = os.path.join(tf_path, f)
                    df = pd.read_parquet(fp)
                    ticker = f.replace(".parquet", "")
                    print(f"    {tf_dir:6s} / {ticker:5s}: "
                          f"{len(df):>10,} bars  "
                          f"({df.index.min().date()} → {df.index.max().date()})")


def load_data(ticker: str, timeframe: str) -> pd.DataFrame:
    """Load cached data for a ticker/timeframe."""
    path = _parquet_path(ticker, timeframe)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data file: {path}. Run download_all() first.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    download_all()
