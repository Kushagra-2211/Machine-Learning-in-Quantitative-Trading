"""
Walk-forward retraining orchestration.

Single-asset analog of pairs_trading/walk_forward.py: every strategy
backtest (classical_strategy.py, ml_random_forest.py, ml_svm.py,
ml_neural_net.py, ml_lstm.py, ml_gradient_boosting.py) accepts a single
`full_price` Series instead of a (price1, price2) pair, but the
orchestration logic -- refit periodically on an EXPANDING formation window,
trade forward through each sub-period, stitch the results into one
continuous series -- is otherwise identical. See
pairs_trading/walk_forward.py's module docstring for the fuller rationale
(a single static formation-period fit has to stay valid across the whole
multi-year trading window; refitting periodically guards against the
fitted relationship drifting over that much time).

config.WALK_FORWARD toggles this on/off so the static and walk-forward
results can be compared directly.
"""

import pandas as pd

import config


def generate_walk_forward_periods(trading_start=None, trading_end=None, freq=None):
    """Returns a list of (formation_end, period_start, period_end) tuples
    covering trading_start..trading_end in non-overlapping sub-periods of
    length `freq` (a pandas offset alias -- "YS" for annual, "QS" for
    quarterly). formation_end for each sub-period is the day before that
    sub-period starts, so formation always EXPANDS."""
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


def run_walk_forward(strategy_run_fn, price, trading_start=None, trading_end=None,
                      freq=None, verbose=True):
    """Runs strategy_run_fn(price, formation_end=..., trading_start=...,
    trading_end=...) once per walk-forward sub-period, refitting on an
    expanding formation window each time, and concatenates the resulting
    daily_returns into one continuous Series spanning the whole
    trading_start..trading_end range.

    strategy_run_fn must accept (price, formation_end, trading_start,
    trading_end) and return a dict with at least a "daily_returns" key --
    every run_strategy() / run_classical_strategy() in this package already
    matches this. buy_and_hold.run_buy_and_hold() has a different signature
    (no formation_end) -- see make_buy_and_hold_adapter() below.

    Every Series-valued key present in the per-period result dicts is
    concatenated the same way, so RMSE diagnostics and any other per-day
    analysis work identically whether WALK_FORWARD is on or off."""
    periods = generate_walk_forward_periods(trading_start, trading_end, freq)
    period_results = []
    for formation_end, period_start, period_end in periods:
        if verbose:
            print(f"    [walk-forward] refit on formation through {formation_end.date()}, "
                  f"trade {period_start.date()} to {period_end.date()}")
        result = strategy_run_fn(
            price,
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
    """buy_and_hold.run_buy_and_hold(price, trading_start, trading_end) has
    no formation_end (buy-and-hold doesn't fit anything) -- wrap it so it
    matches the (price, formation_end, trading_start, trading_end) signature
    run_walk_forward() expects, simply ignoring formation_end.

    NOT used by default in main.py -- same reasoning as
    pairs_trading/walk_forward.py: main.py just calls run_buy_and_hold()
    once over the full trading range regardless of config.WALK_FORWARD."""
    def adapter(price, formation_end=None, trading_start=None, trading_end=None):
        return run_buy_and_hold_fn(price, trading_start, trading_end)
    return adapter
