"""
Shared position/return logic used by BOTH the classical and ML strategies.

Kept in one place so the two strategies are genuinely comparable -- the only
difference between them should be how the z-score is computed (from the
actual current spread vs. from a model's predicted next-day spread), not the
trading rule itself.

Trading rule follows Peng Liu, "Quantitative Trading Strategies Using Python"
(2023), Ch.8, Listing 8-12: enter at |z| > entry_threshold, exit at
|z| < exit_threshold, otherwise hold the existing position.
"""

import numpy as np
import pandas as pd
import config


def generate_positions(zscore, entry_threshold=None, exit_threshold=None):
    """Given a z-score series, return (position1, position2) series where
    +1/-1 means long/short that leg of the pair. position1 and position2 are
    always opposite signs and equal magnitude (dollar-neutral long/short)."""
    entry_threshold = entry_threshold if entry_threshold is not None else config.ENTRY_THRESHOLD
    exit_threshold = exit_threshold if exit_threshold is not None else config.EXIT_THRESHOLD

    # Freshly allocated numpy arrays (NOT `.values` off a pandas Series) --
    # on newer pandas versions with copy-on-write enabled, `.values` can
    # return a read-only view, which breaks in-place assignment in the loop
    # below. Allocating our own arrays here sidesteps that regardless of
    # pandas version.
    n = len(zscore)
    z = zscore.values          # only ever READ from, so read-only is fine here
    p1 = np.zeros(n, dtype=float)
    p2 = np.zeros(n, dtype=float)

    # ===================== ENTRY / EXIT TRADING RULE =====================
    # |z| > entry_threshold  -> open a new position (spread is unusually
    #                           wide/narrow, bet on it reverting)
    # |z| < exit_threshold   -> close out (spread has reverted enough)
    # in between             -> hold whatever position we already have
    # =======================================================================
    for i in range(1, len(z)):
        if pd.isna(z[i]):
            p1[i], p2[i] = p1[i - 1], p2[i - 1]
            continue
        if z[i] < -entry_threshold and p1[i - 1] == 0:
            p1[i], p2[i] = 1, -1       # spread too low -> long leg 1, short leg 2
        elif z[i] > entry_threshold and p2[i - 1] == 0:
            p1[i], p2[i] = -1, 1       # spread too high -> short leg 1, long leg 2
        elif abs(z[i]) < exit_threshold:
            p1[i], p2[i] = 0, 0        # converged -> flat
        else:
            p1[i], p2[i] = p1[i - 1], p2[i - 1]  # hold

    return pd.Series(p1, index=zscore.index), pd.Series(p2, index=zscore.index)


def position_turnover(pos1, pos2):
    """===================== TURNOVER (for transaction costs) =====================
    Per-day total turnover across both legs: |change in pos1| + |change in
    pos2| from the previous day. The FIRST day of whatever series is passed
    in is treated as a fresh entry from flat (turnover = |pos1[0]| + |pos2[0]|)
    -- see config.py's note on walk-forward boundary handling for why this is
    a deliberate, documented simplification (it makes the very first day of
    every walk-forward sub-period incur an entry cost, even on internal
    annual boundaries where the "true" prior-day position may not actually
    have been flat) rather than a silent oversight.

    Returns 0 turnover for an empty pos1/pos2 (e.g. a walk-forward window
    with no usable signal -- see ml_common.no_signal_result()) instead of
    raising on the .iloc[0] assignment below.
    ================================================================================"""
    d1 = pos1.diff()
    d2 = pos2.diff()
    if len(pos1) > 0:
        d1.iloc[0] = pos1.iloc[0]
        d2.iloc[0] = pos2.iloc[0]
    return d1.abs() + d2.abs()


def transaction_cost_drag(pos1, pos2, cost_bps=None):
    """===================== TRANSACTION COST DRAG =====================
    Daily return DRAG (a non-negative number, in the same units as a daily
    return, meant to be SUBTRACTED from gross returns) from transaction
    costs: cost_bps basis points of notional per unit of same-day turnover
    (see position_turnover() and config.py's module-level note for the full
    rationale and the {-1,0,+1}-per-leg cost convention).

    Returns an all-zero Series (no-op) if config.INCLUDE_TRANSACTION_COSTS
    is False and no explicit cost_bps override was passed -- this is what
    lets the exact same strategy_returns() code path reproduce the original
    zero-cost n=10/20/40 headline runs when the flag is off, and the
    cost-sensitivity pass when it's on.
    ======================================================================"""
    if cost_bps is None:
        cost_bps = config.TRANSACTION_COST_BPS if config.INCLUDE_TRANSACTION_COSTS else 0.0
    if cost_bps == 0.0:
        return pd.Series(0.0, index=pos1.index)
    turnover = position_turnover(pos1, pos2)
    return turnover * (cost_bps / 10000.0)


def strategy_returns(price1, price2, pos1, pos2, cost_bps=None):
    """Daily strategy return: yesterday's position applied to today's simple
    return of each leg, summed across the two legs (each leg gets $1 of
    capital, i.e. a $1 long-$1 short dollar-neutral pair position -- matches
    Peng Liu Ch.8's convention), MINUS transaction cost drag from any
    same-day change in position (see transaction_cost_drag() -- a no-op,
    returning the original gross-return series unchanged, whenever
    config.INCLUDE_TRANSACTION_COSTS is False)."""
    ret1 = price1.pct_change()
    ret2 = price2.pct_change()
    gross_return = (pos1.shift(1) * ret1 + pos2.shift(1) * ret2).fillna(0.0)
    cost = transaction_cost_drag(pos1, pos2, cost_bps).reindex(gross_return.index).fillna(0.0)
    return gross_return - cost
