#!/usr/bin/env python3
"""
Data Download Script
=====================
Downloads all required data from Polygon.io for the momentum research.

Usage:
    pip install -r requirements.txt
    python download_data.py

This downloads:
    - 10 years of daily + weekly bars for HTF context
    - 5 years of 1m, 5m, 15m, 30m, 60m bars for intraday analysis
    - For SPY and QQQ (Stage 1)

Data is cached to Parquet files in the data/ directory.
Subsequent runs only download missing date ranges (incremental).
"""
from src.downloader import download_all

if __name__ == "__main__":
    download_all()
