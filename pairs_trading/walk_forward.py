"""
Walk-forward retraining orchestration.

Every strategy backtest so far (classical_strategy.py, ml_random_forest.py,
ml_svm.py, ml_neural_net.py, ml_lstm.py, ml_gradient_boosting.py) fits ONE
model on the formation period (2016-2021) and then applies it, completely
unchanged, across FOUR YEARS of trading (2022-2025). Diagnosing why NN/LSTM
underperformed on the trading period despite fine formation-period
validation loss pointed at exactly this: a single static fit has to stay
valid for the entire multi-year trading window, and the actual relationship
between two stocks' prices can drift over that much time -- see
ml_neural_net.py's module docstring for the fuller diagnosis (validation
loss was still improving at epoch 500 on every pair, i.e. not classic
overfitting -- more consistent with formation-period patterns not fully
carrying over to the trading period).

This module re-fits every strategy periodically instead: split
TRADING_START..TRADING_END into sub-periods (annual by default -- see
config.WALK_FORWARD_FREQUENCY), and before each sub-period, refit every
model on an EXPANDING formation window (always starting at
config.FORMATION_START, growing to include everything up to the day before
that sub-period begins), then trade forward through that sub-period only,
using its own fresh fit. Daily returns from every sub-period are then
stitched into one continuous series covering the whole trading window, so
performance.py can score it exactly like the static (single-fit) result.

This is deliberately a thin orchestration layer, NOT a rewrite of any
individual strategy: every strategy module already accepts
(formation_end, trading_start, trading_end) as override arguments (that's
what made the early-stopping validation split possible in the first place),
so walk-forward here just means calling that SAME run_strategy() repeatedly
with different period boundaries and concatenating the results. Nothing
about any individual model's training code changes.

config.WALK_FORWARD toggles this on/off so the static and walk-forward
results can be compared directly and reported side by side -- that
comparison IS the evidence for whether regime shift explains the gap
between formation-period validation loss and trading-period performance.
"""

import pandas as pd

import config


def generate_walk_forward_periods(trading_start=None, trading_end=None, freq=None):
    """Returns a list of (formation_end, period_start, period_end) tuples
    covering trading_start..trading_end in non-overlapping sub-periods of
    length `freq` (a pandas offset alias -- "YS" for annual, "QS" for
    quarterly). formation_end for each sub-period is the day before that
    sub-period starts, so formation always EXPANDS: the first period's
    formation_end matches the original static config.FORMATION_END exactly,
    and every later period's model gets strictly more data (including
    realized results from earlier trading sub-periods, which by then are
    just "the past")."""
    trading_start = pd.Timestamp(trading_start or config.TRADING_START)
    trading_end = pd.Timestamp(trading_end or config.TRADING_END)
    freq = freq or config.WALK_FORWARD_FREQUENCY

    period_starts = list(pd.date_range(trading_start, trading_end, freq=freq))
    if not period_starts or period_starts[0] != trading_start:
        period_starts = [trading_start] + period_starts
    period_starts = sorted(set(s for s in period_starts if s <= trading_end))

    periods = []
    for i, period_start in enumerate(period_starts):
        period_end = (
            period_starts[i + 1] - pd.Timedelta(days=1) if i + 1 < len(period_starts) else trading_end
        )
        formation_end = period_start - pd.Timedelta(days=1)
        periods.append((formation_end, period_start, period_end))
    return periods


def run_walk_forward(strategy_run_fn, price1, price2, trading_start=None, trading_end=None,
                      freq=None, verbose=True):
    """Runs strategy_run_fn(price1, price2, formation_end=..., trading_start=...,
    trading_end=...) once per walk-forward sub-period, refitting on an
    expanding formation window each time, and concatenates the resulting
    daily_returns into one continuous Series spanning the whole
    trading_start..trading_end range.

    strategy_run_fn must accept (price1, price2, formation_end, trading_start,
    trading_end) and return a dict with at least a "daily_returns" key --
    every run_strategy() / run_classical_strategy() in this package already
    matches this. buy_and_hold.run_buy_and_hold() has a different signature
    (it doesn't fit anything, so has no formation_end) -- see
    make_buy_and_hold_adapter() below for a thin wrapper.

    Every Series-valued key present in the per-period result dicts (not just
    "daily_returns" -- also "predicted_spread"/"actual_spread" for the ML
    strategies, "zscore"/"position1"/"position2" for all of them) is
    concatenated the same way, so RMSE diagnostics and any other per-day
    analysis work identically whether WALK_FORWARD is on or off. The raw
    per-period dicts are also kept (key "period_results") for anything
    non-Series, e.g. each period's own fitted "model" object."""
    periods = generate_walk_forward_periods(trading_start, trading_end, freq)
    period_results = []
    for formation_end, period_start, period_end in periods:
        if verbose:
            print(f"    [walk-forward] refit on formation through {formation_end.date()}, "
                  f"trade {period_start.date()} to {period_end.date()}")
        result = strategy_run_fn(
            price1, price2,
            formation_end=formation_end, trading_start=period_start, trading_end=period_end,
        )
        period_results.append(result)

    combined = {}
    series_keys = [k for k, v in period_results[0].items() if isinstance(v, pd.Series)]
    for key in series_keys:
        combined[key] = pd.concat([r[key] for r in period_results if key in r]).sort_index()
    combined["period_results"] = period_results
    return combined


def make_buy_and_hold_adapter(run_buy_and_hold_fn):
    """buy_and_hold.run_buy_and_hold(price1, price2, trading_start, trading_end)
    has no formation_end (buy-and-hold doesn't fit anything) -- wrap it so it
    matches the (price1, price2, formation_end, trading_start, trading_end)
    signature run_walk_forward() expects, simply ignoring formation_end.

    NOT used by default in main.py: routing buy-and-hold through
    run_walk_forward would call pct_change() separately within each
    sub-period, and run_buy_and_hold() fillna(0)'s the first return of
    whatever series it's given -- under walk-forward that would zero out one
    day's return at EVERY period boundary (4 times a year) instead of just
    once at the very start, slightly understating cumulative buy-and-hold
    performance. Since there's no model to refit for a passive benchmark
    anyway, main.py just calls run_buy_and_hold() once over the full
    trading range regardless of config.WALK_FORWARD. Kept here in case a
    genuinely time-varying benchmark is ever wanted."""
    def adapter(price1, price2, formation_end=None, trading_start=None, trading_end=None):
        return run_buy_and_hold_fn(price1, price2, trading_start, trading_end)
    return adapter
