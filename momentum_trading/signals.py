"""
Shared signal-to-return logic used by BOTH the classical and ML strategies.

Single-asset analog of pairs_trading/signals.py: kept in one place so the
classical and ML strategies are genuinely comparable -- the only difference
between them should be how the +1/-1 signal is generated (from the actual
current SMA crossover vs. from the sign of a model's predicted next-day
return), not the mechanism that turns a signal into a realized return.

Trading rule follows Peng Liu, "Quantitative Trading Strategies Using
Python" (2023), Ch.5 "Trend-Following Strategy": always in the market, long
(+1) or short (-1), no flat/neutral state.

IMPLEMENTATION NOTE (deliberate deviation from the book, for codebase
consistency): the book pre-shifts the SMAs themselves by one day before
computing the signal, then multiplies signal * same-day return directly with
no further shift. Here, the signal is instead computed from the ACTUAL
same-day indicator (SMA crossover, or predicted-return sign), and the
one-day causal delay is applied to the SIGNAL itself in strategy_returns()
below (signal.shift(1) * realized_return) -- exactly the convention already
used in pairs_trading/signals.py (pos.shift(1) * ret). This lets classical
and ML strategies share one identical strategy_returns() implementation
(the ML model's predicted-return target is naturally "forward-looking"
relative to its feature date, so shifting the signal forward by one day
lines it up with the return it was actually predicting -- see
ml_common.py's build_target() docstring for the full alignment argument).
Both conventions are causally equivalent; this one was chosen to match the
rest of this codebase rather than introduce a second shifting convention.
"""

import numpy as np
import pandas as pd

import config


def generate_signal(short_indicator, long_indicator):
    """Given two same-index Series (short-term and long-term SMAs, or any
    other short-vs-long comparison), return a +1/-1 signal Series: +1 where
    short > long (bullish crossover), -1 otherwise (bearish or equal).
    Matches Peng Liu Ch.5's np.where(SMA_short > SMA_long, 1, -1) exactly,
    just without the book's separate pre-shift step (see module docstring)."""
    signal = np.where(short_indicator > long_indicator, 1.0, -1.0)
    return pd.Series(signal, index=short_indicator.index)


def position_turnover(signal):
    """===================== TURNOVER (for transaction costs) =====================
    Single-asset analog of pairs_trading/signals.py's position_turnover().
    Every strategy here is always fully invested at +1 or -1 (see module
    docstring), so turnover on any given day is |change in signal|: 0 while
    the position is held, and 2 on a full long-to-short or short-to-long
    flip (economically two trades -- closing the old side, opening the
    new one). The very first day is treated as a fresh entry from flat
    (turnover = |signal[0]|, i.e. 1), matching the pairs strand's convention.
    ================================================================================"""
    diff = signal.diff().abs()
    if len(signal) > 0:
        diff.iloc[0] = abs(signal.iloc[0])
    return diff.fillna(0.0)


def transaction_cost_drag(signal, cost_bps=None):
    """Daily cost drag, in return terms: cost_bps basis points of notional
    per unit of same-day turnover (see position_turnover()). Returns an
    all-zero Series (no-op) if config.INCLUDE_TRANSACTION_COSTS is False and
    no explicit cost_bps override was passed -- this is what makes every
    existing no-cost call site unaffected by this function's addition."""
    if cost_bps is None:
        cost_bps = config.TRANSACTION_COST_BPS if config.INCLUDE_TRANSACTION_COSTS else 0.0
    if cost_bps == 0.0:
        return pd.Series(0.0, index=signal.index)
    turnover = position_turnover(signal)
    return turnover * (cost_bps / 10000.0)


def strategy_returns(simple_return, signal, cost_bps=None):
    """Daily strategy return: YESTERDAY's signal (decided using information
    available through yesterday) applied to TODAY's realized SIMPLE return
    -- signal.shift(1) * simple_return, identical shift-then-multiply
    pattern to pairs_trading/signals.py's strategy_returns(pos.shift(1)*ret)
    -- net of transaction_cost_drag(), which is a no-op unless
    config.INCLUDE_TRANSACTION_COSTS is True or cost_bps is passed explicitly.

    Uses simple (percentage) returns rather than the book's log returns --
    a deliberate deviation so daily_returns here can be fed directly into
    performance.py's cumulative_return()/annualized_return()/sharpe_ratio(),
    all of which assume simple, directly-compoundable returns (this is the
    same convention pairs_trading already uses throughout). The signal
    mechanism (SMA crossover -> +1/-1, or ML predicted-return sign -> +1/-1)
    is unaffected either way; only the return arithmetic changes."""
    daily_return = signal.shift(1) * simple_return
    cost = transaction_cost_drag(signal, cost_bps).reindex(daily_return.index).fillna(0.0)
    return (daily_return - cost).fillna(0.0)
