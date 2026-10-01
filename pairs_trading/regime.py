"""
Data-driven regime classification: trending vs. range-bound.

Motivation: the calendar-year breakdown in performance.annual_breakdown()
suggested a regime-dependence story (mean-reversion strategies -- Classical
and ML: SVM -- do best in 2022, then decay through 2023-2025 as Buy & Hold's
Sharpe rises), but a calendar year is only a rough proxy for "regime", and
four data points per strategy isn't enough to actually TEST whether SVM's
edge over Classical is statistically concentrated in mean-reverting
conditions, as opposed to just eyeballing the shape of four annual bars.

This module classifies EVERY TRADING DAY (not just every year) as "Trending"
or "Range-bound" using Kaufman's Efficiency Ratio (Kaufman, "Trading Systems
and Methods", 1995), computed on a sector-level price proxy: the cumulative
equal-weighted Buy & Hold portfolio index. Buy & Hold has no trading rule of
its own (it never enters/exits a position based on the spread), so its
cumulative return is a clean, strategy-independent stand-in for "how much is
the underlying sector actually trending, day by day" -- uncontaminated by any
of the strategies this regime label is later used to evaluate.

Efficiency Ratio over a rolling window of length `window` (default 60
trading days, ~1 quarter -- the same window used for ml_common's rolling
half-life feature, kept consistent across the codebase):

    ER(t) = |P(t) - P(t-window)| / sum_{i=t-window+1}^{t} |P(i) - P(i-1)|

ER close to 1 means price moved in a straight line over that window (a
strong trend -- little of the day-to-day movement was "wasted" reversing);
ER close to 0 means price churned back and forth without net progress
(range-bound / mean-reverting conditions -- exactly what a pairs-trading
spread needs to do for the strategy family to be profitable).

Days are split into two buckets by the MEDIAN Efficiency Ratio over the whole
out-of-sample window, rather than a fixed technical-analysis threshold (e.g.
"ER > 0.3 = trending") -- fixed thresholds from the technical-analysis
literature are heuristic and not calibrated to this specific market/period.
A median split is fully data-driven and reproducible, and gives two roughly
balanced regime buckets by construction. The tradeoff, worth stating
explicitly wherever these labels are used in the write-up: "Trending" here
means "the more trending half of this sample", not an absolute, universally
comparable cutoff -- the labels are relative to this dataset, not a general
claim about what ER value constitutes a "real" trend.
"""

import numpy as np
import pandas as pd


def efficiency_ratio(price_series, window=60):
    """Kaufman's Efficiency Ratio, computed on a rolling window of length
    `window` trading days. Returns a Series aligned to price_series.index,
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
    day (the first `window` days, where Efficiency Ratio is undefined, are
    dropped -- not labelled either way), split at the MEDIAN Efficiency
    Ratio over the whole series. See module docstring for why a median split
    is used instead of a fixed technical-analysis threshold."""
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
    and the contiguous date ranges ("runs") each bucket is made up of --
    worth printing once so the data-driven regime split can be eyeballed
    against the calendar-year story (e.g. "is 2022 mostly classified as
    Range-bound, consistent with what the annual breakdown suggested?").
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
