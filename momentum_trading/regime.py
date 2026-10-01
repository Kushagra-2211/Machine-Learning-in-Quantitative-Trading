"""
Data-driven regime classification: trending vs. range-bound.

Adapted from pairs_trading/regime.py -- identical Efficiency Ratio
mechanics, applied here to the crypto momentum strand. Classifies EVERY
CALENDAR DAY (crypto trades every day, no weekend/holiday gaps) as
"Trending" or "Range-bound" using Kaufman's Efficiency Ratio (Kaufman,
"Trading Systems and Methods", 1995).

main.py applies this to the equal-weighted crypto Buy & Hold portfolio's
cumulative return (the "crypto market" trendiness proxy) -- the SAME choice
pairs_trading/main.py makes (classifying regime from the sector-level Buy &
Hold proxy rather than each individual asset's own price), for direct
methodological consistency and comparability between the two strands'
regime-conditional analyses. This is a genuinely interesting point of
CONTRAST between the two strands worth stating explicitly in the write-up:
pairs trading (mean-reversion) is expected to do BEST in Range-bound
regimes and worst in Trending ones, while trend-following momentum is
expected to show the OPPOSITE pattern -- doing best when the market is
genuinely trending and worst when it whipsaws back and forth without net
progress (exactly the conditions that generate repeated false crossover
signals for an SMA-crossover strategy).

Efficiency Ratio over a rolling window of length `window` (default 60
calendar days -- see config.REGIME_WINDOW):

    ER(t) = |P(t) - P(t-window)| / sum_{i=t-window+1}^{t} |P(i) - P(i-1)|

ER close to 1 means price moved in a straight line over that window (a
strong trend); ER close to 0 means price churned back and forth without net
progress (range-bound / choppy conditions). Days are split into two buckets
by the MEDIAN Efficiency Ratio over the whole out-of-sample window -- see
pairs_trading/regime.py's docstring for the full rationale on why a median
split is used instead of a fixed technical-analysis threshold.
"""

import numpy as np
import pandas as pd


def efficiency_ratio(price_series, window=60):
    """Kaufman's Efficiency Ratio, computed on a rolling window of length
    `window` calendar days. Returns a Series aligned to price_series.index,
    NaN for the first `window` days (insufficient history to compute a full
    window)."""
    price_series = price_series.dropna()
    net_change = (price_series - price_series.shift(window)).abs()
    daily_moves = price_series.diff().abs()
    path_length = daily_moves.rolling(window).sum()
    er = net_change / path_length
    return er


def classify_regime(price_series, window=60):
    """Returns a Series of "Trending" / "Range-bound" string labels, one per
    day, split at the MEDIAN Efficiency Ratio over the whole series."""
    er = efficiency_ratio(price_series, window).dropna()
    if len(er) == 0:
        raise ValueError(
            f"No Efficiency Ratio values computed -- price_series has fewer than "
            f"{window + 1} usable days. Reduce `window` or check the input series."
        )
    median_er = er.median()
    labels = pd.Series(
        np.where(er >= median_er, "Trending", "Range-bound"),
        index=er.index,
    )
    return labels


def regime_summary(labels):
    """Quick sanity-check summary: how many days fall in each regime bucket,
    and the contiguous date ranges ("runs") each bucket is made up of.
    Returns (counts, runs) where counts is a Series and runs is a list of
    (label, start_date, end_date) tuples in chronological order."""
    counts = labels.value_counts()

    runs = []
    current_label, run_start, prev_date = None, None, None
    for date, label in labels.items():
        if label != current_label:
            if current_label is not None:
                runs.append((current_label, run_start, prev_date))
            current_label, run_start = label, date
        prev_date = date
    if current_label is not None:
        runs.append((current_label, run_start, prev_date))

    return counts, runs
