"""
P&L Snapshot Engine
====================
The core of the research framework.

Given a set of condition occurrences (bar indices where a condition is true),
builds the forward P&L matrix by tracking mark-to-market at every subsequent
bar close through end of session.

This single engine is used for:
- Phase 1: HTF context discovery (enter at open, track intraday P&L)
- Phase 2: Intraday condition scanning (enter at condition bar, track forward P&L)
- Exit derivation: optimal hold time, stops, targets all read from the matrix
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class SnapshotResult:
    """Container for P&L snapshot analysis of a single condition."""
    condition_name: str
    timeframe: str
    ticker: str
    n_occurrences: int

    # The raw P&L matrix: rows = occurrences, columns = bars after entry
    pnl_matrix: np.ndarray          # shape: (n_occurrences, max_bars_forward)
    bar_labels: list                 # column labels: ["Bar+1", "Bar+2", ...]

    # HTF context tags for each occurrence (for slicing later)
    htf_tags: pd.DataFrame = None    # shape: (n_occurrences, n_htf_factors)

    # Entry timestamps for reference
    entry_times: list = field(default_factory=list)
    entry_prices: list = field(default_factory=list)

    @property
    def mean_curve(self) -> np.ndarray:
        """Average P&L at each bar-after-entry."""
        return np.nanmean(self.pnl_matrix, axis=0)

    @property
    def median_curve(self) -> np.ndarray:
        """Median P&L at each bar-after-entry."""
        return np.nanmedian(self.pnl_matrix, axis=0)

    @property
    def std_curve(self) -> np.ndarray:
        return np.nanstd(self.pnl_matrix, axis=0)

    @property
    def pct_25(self) -> np.ndarray:
        return np.nanpercentile(self.pnl_matrix, 25, axis=0)

    @property
    def pct_75(self) -> np.ndarray:
        return np.nanpercentile(self.pnl_matrix, 75, axis=0)

    @property
    def pct_10(self) -> np.ndarray:
        return np.nanpercentile(self.pnl_matrix, 10, axis=0)

    @property
    def pct_90(self) -> np.ndarray:
        return np.nanpercentile(self.pnl_matrix, 90, axis=0)

    @property
    def win_rate_curve(self) -> np.ndarray:
        """% of occurrences that are profitable at each bar."""
        return np.nanmean(self.pnl_matrix > 0, axis=0)

    @property
    def peak_bar_mean(self) -> int:
        """Bar index where the mean P&L peaks."""
        curve = self.mean_curve
        if len(curve) == 0 or np.all(np.isnan(curve)):
            return 0
        return int(np.nanargmax(curve))

    @property
    def peak_bar_distribution(self) -> np.ndarray:
        """For each occurrence, which bar had the max P&L."""
        mat = self.pnl_matrix
        if mat.size == 0 or np.all(np.isnan(mat)):
            return np.array([])
        # Mask rows that are entirely NaN
        valid = ~np.all(np.isnan(mat), axis=1)
        result = np.full(mat.shape[0], np.nan)
        if valid.any():
            result[valid] = np.nanargmax(mat[valid], axis=1)
        return result

    @property
    def mfe_distribution(self) -> np.ndarray:
        """Max favorable excursion for each occurrence."""
        mat = self.pnl_matrix
        if mat.size == 0 or np.all(np.isnan(mat)):
            return np.array([])
        return np.nanmax(mat, axis=1)

    @property
    def mae_distribution(self) -> np.ndarray:
        """Max adverse excursion for each occurrence."""
        mat = self.pnl_matrix
        if mat.size == 0 or np.all(np.isnan(mat)):
            return np.array([])
        return np.nanmin(mat, axis=1)

    def slice_by_htf(self, factor_name: str, value) -> "SnapshotResult":
        """
        Return a new SnapshotResult filtered to occurrences where
        htf_tags[factor_name] == value.
        """
        if self.htf_tags is None or factor_name not in self.htf_tags.columns:
            raise ValueError(f"HTF factor '{factor_name}' not found in tags.")

        mask = self.htf_tags[factor_name] == value
        indices = np.where(mask)[0]

        return SnapshotResult(
            condition_name=f"{self.condition_name} | {factor_name}={value}",
            timeframe=self.timeframe,
            ticker=self.ticker,
            n_occurrences=len(indices),
            pnl_matrix=self.pnl_matrix[indices],
            bar_labels=self.bar_labels,
            htf_tags=self.htf_tags.iloc[indices].reset_index(drop=True),
            entry_times=[self.entry_times[i] for i in indices] if self.entry_times else [],
            entry_prices=[self.entry_prices[i] for i in indices] if self.entry_prices else [],
        )

    def summary_stats(self) -> dict:
        """Compute summary statistics for this condition."""
        if self.n_occurrences == 0:
            return {}

        mean_c = self.mean_curve
        if len(mean_c) == 0 or np.all(np.isnan(mean_c)):
            return {"condition": self.condition_name, "timeframe": self.timeframe,
                    "ticker": self.ticker, "n_occurrences": self.n_occurrences,
                    "peak_bar": 0, "mean_pnl_at_peak": 0, "t_stat": 0, "skip": True}

        peak_bar = self.peak_bar_mean
        peak_pnl = float(mean_c[peak_bar]) if len(mean_c) > peak_bar else 0

        # Sharpe of the mean P&L curve (daily)
        curve_diff = np.diff(mean_c)
        curve_sharpe = (np.mean(curve_diff) / np.std(curve_diff) * np.sqrt(252)
                        if np.std(curve_diff) > 0 else 0)

        # Mean/median divergence (fragility indicator)
        med_c = self.median_curve
        divergence = abs(peak_pnl - med_c[peak_bar]) / abs(peak_pnl) if peak_pnl != 0 else 0

        # MFE/MAE stats
        mfe = self.mfe_distribution
        mae = self.mae_distribution

        # t-stat on mean P&L at peak bar
        peak_pnls = self.pnl_matrix[:, peak_bar]
        peak_pnls_clean = peak_pnls[~np.isnan(peak_pnls)]
        if len(peak_pnls_clean) > 1:
            t_stat = np.mean(peak_pnls_clean) / (np.std(peak_pnls_clean) / np.sqrt(len(peak_pnls_clean)))
        else:
            t_stat = 0

        return {
            "condition": self.condition_name,
            "timeframe": self.timeframe,
            "ticker": self.ticker,
            "n_occurrences": self.n_occurrences,
            "peak_bar": peak_bar,
            "mean_pnl_at_peak": peak_pnl,
            "median_pnl_at_peak": med_c[peak_bar] if len(med_c) > peak_bar else 0,
            "mean_median_divergence": divergence,
            "win_rate_at_peak": self.win_rate_curve[peak_bar] if len(self.win_rate_curve) > peak_bar else 0,
            "curve_sharpe": curve_sharpe,
            "avg_mfe": np.nanmean(mfe),
            "avg_mae": np.nanmean(mae),
            "mae_90pct": np.nanpercentile(mae, 10),  # 10th pct of MAE = worst 10%
            "t_stat": t_stat,
            "eod_mean_pnl": mean_c[-1] if len(mean_c) > 0 else 0,
        }


def build_pnl_matrix(bars: pd.DataFrame,
                     condition_mask: pd.Series,
                     max_bars_forward: int = None,
                     session_dates: pd.Series = None,
                     direction: str = "long",
                     ) -> SnapshotResult:
    """
    Build the forward P&L snapshot matrix for a given condition.

    Parameters
    ----------
    bars : DataFrame
        OHLCV bar data with DatetimeIndex (must be a single ticker, single timeframe).
    condition_mask : Series
        Boolean mask, same index as bars. True where the condition is observed.
    max_bars_forward : int or None
        Maximum number of bars to track forward. If None, tracks to end of session.
    session_dates : Series or None
        Date component of each bar's timestamp, used to identify session boundaries.
        If None, computed from bars.index.
    direction : str
        "long" or "short" — determines P&L sign convention.

    Returns
    -------
    SnapshotResult
    """
    if session_dates is None:
        session_dates = bars.index.date

    # Find all bars where condition is True
    condition_indices = np.where(condition_mask.values)[0]
    n_occurrences = len(condition_indices)

    if n_occurrences == 0:
        return SnapshotResult(
            condition_name="",
            timeframe="",
            ticker="",
            n_occurrences=0,
            pnl_matrix=np.array([]).reshape(0, 0),
            bar_labels=[],
        )

    # Determine max bars forward if not specified
    if max_bars_forward is None:
        # Estimate: max bars in a session
        # For 1m bars: ~390 bars/session, for 5m: ~78, 15m: ~26, 30m: ~13, 60m: ~7
        dates_arr = np.array(session_dates)
        unique_dates = np.unique(dates_arr)
        if len(unique_dates) > 0:
            bars_per_day = []
            for d in unique_dates[:20]:  # sample first 20 days
                bars_per_day.append(np.sum(dates_arr == d))
            max_bars_forward = max(bars_per_day) if bars_per_day else 390
        else:
            max_bars_forward = 390

    # Build the P&L matrix
    close_prices = bars["close"].values
    dates_arr = np.array(session_dates)
    sign = 1.0 if direction == "long" else -1.0

    pnl_matrix = np.full((n_occurrences, max_bars_forward), np.nan)
    entry_times = []
    entry_prices = []

    for row_idx, bar_idx in enumerate(condition_indices):
        entry_price = close_prices[bar_idx]
        entry_date = dates_arr[bar_idx]
        entry_times.append(bars.index[bar_idx])
        entry_prices.append(entry_price)

        # Walk forward bar by bar until end of session or max_bars_forward
        for fwd in range(max_bars_forward):
            future_idx = bar_idx + 1 + fwd
            if future_idx >= len(close_prices):
                break
            # Stop at session boundary
            if dates_arr[future_idx] != entry_date:
                break

            pnl = sign * (close_prices[future_idx] - entry_price) / entry_price
            pnl_matrix[row_idx, fwd] = pnl

    bar_labels = [f"Bar+{i+1}" for i in range(max_bars_forward)]

    return SnapshotResult(
        condition_name="",
        timeframe="",
        ticker="",
        n_occurrences=n_occurrences,
        pnl_matrix=pnl_matrix,
        bar_labels=bar_labels,
        entry_times=entry_times,
        entry_prices=entry_prices,
    )


def build_daily_pnl_matrix(daily_bars: pd.DataFrame,
                           intraday_bars: pd.DataFrame,
                           condition_mask: pd.Series,
                           direction: str = "long",
                           ) -> SnapshotResult:
    """
    Phase 1 variant: condition is evaluated on daily bars,
    P&L is tracked on intraday bars.

    For each day where the condition is True, enter at market open
    and track intraday P&L at every intraday bar close.

    Parameters
    ----------
    daily_bars : DataFrame with daily OHLCV
    intraday_bars : DataFrame with intraday OHLCV (e.g., 5m bars)
    condition_mask : Series, boolean, indexed like daily_bars
    direction : str, "long" or "short"
    """
    sign = 1.0 if direction == "long" else -1.0

    condition_dates = daily_bars.index[condition_mask].date if hasattr(daily_bars.index[condition_mask], 'date') else daily_bars.index[condition_mask]
    condition_dates = set(pd.Timestamp(d).date() if not isinstance(d, type(pd.Timestamp("2020-01-01").date())) else d for d in condition_dates)

    intraday_dates = intraday_bars.index.date

    # Find all intraday bars for each condition date
    all_rows = []
    entry_times = []
    entry_prices_list = []

    for d in sorted(condition_dates):
        day_mask = intraday_dates == d
        day_bars = intraday_bars[day_mask]
        if len(day_bars) < 2:
            continue

        # Enter at open of first bar
        entry_price = day_bars["open"].iloc[0]
        entry_times.append(day_bars.index[0])
        entry_prices_list.append(entry_price)

        # P&L at each subsequent bar close
        pnls = sign * (day_bars["close"].values - entry_price) / entry_price
        all_rows.append(pnls)

    if not all_rows:
        return SnapshotResult(
            condition_name="", timeframe="", ticker="",
            n_occurrences=0,
            pnl_matrix=np.array([]).reshape(0, 0),
            bar_labels=[],
        )

    # Pad rows to same length (different days may have different bar counts)
    max_len = max(len(r) for r in all_rows)
    pnl_matrix = np.full((len(all_rows), max_len), np.nan)
    for i, row in enumerate(all_rows):
        pnl_matrix[i, :len(row)] = row

    bar_labels = [f"Bar+{i+1}" for i in range(max_len)]

    return SnapshotResult(
        condition_name="",
        timeframe="",
        ticker="",
        n_occurrences=len(all_rows),
        pnl_matrix=pnl_matrix,
        bar_labels=bar_labels,
        entry_times=entry_times,
        entry_prices=entry_prices_list,
    )
