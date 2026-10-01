"""
Classical (non-ML) pairs trading strategy.

Methodology follows Peng Liu, "Quantitative Trading Strategies Using Python"
(2023), Ch.8:
  - spread = residual of an OLS regression of price1 (Y) on price2 (X)
  - z-score = rolling (spread - rolling_mean) / rolling_std
  - enter at |z| > entry_threshold, exit at |z| < exit_threshold (Listing 8-12)

One deliberate change from the book: the book fits the OLS regression and
the rolling z-score on the SAME window it trades on -- fine for a worked
example, but a real backtest needs the hedge ratio estimated out-of-sample.
Here, the OLS coefficients (const, beta) are estimated ONCE on the formation
period only, then held fixed and applied to the trading period. This is what
makes the trading-period results a genuine out-of-sample test rather than an
in-sample curve-fit -- see DISSERTATION_PLAN.md / config.py for the
formation/trading split.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config
import signals


def fit_spread_model(formation_price1, formation_price2):
    """OLS of price1 (Y) on price2 (X), fit on the formation period only.
    Returns (const, beta) -- the fixed hedge ratio used going forward."""
    common_idx = formation_price1.dropna().index.intersection(formation_price2.dropna().index)
    y = formation_price1.loc[common_idx]
    x = formation_price2.loc[common_idx]
    # ===================== SPREAD DEFINITION (OLS) =====================
    # The "spread" is the residual of regressing price1 on price2 -- the
    # part of price1's movement NOT explained by price2. If the pair is
    # genuinely cointegrated, this residual should be stationary (mean-
    # reverting) even though the two raw prices are not.
    # =====================================================================
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    const, beta = model.params.iloc[0], model.params.iloc[1]
    return const, beta


def compute_spread(price1, price2, const, beta):
    """Apply a FIXED (const, beta) hedge ratio to any period's prices."""
    return price1 - (const + beta * price2)


def compute_zscore(spread, window=None):
    """===================== Z-SCORE (standardized spread) =====================
    Turns the spread into a standardized signal: how many rolling standard
    deviations away from its recent rolling mean is the spread right now?
    This is what the entry/exit thresholds in signals.py actually act on.
    ============================================================================"""
    window = window if window is not None else config.ZSCORE_WINDOW
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    return (spread - mean) / std


def run_classical_strategy(full_price1, full_price2, formation_end=None, trading_start=None, trading_end=None):
    """Full pipeline for one pair: fit on formation, trade on trading period.

    full_price1 / full_price2: full-history price Series (formation +
    trading concatenated), indexed by date. Passing the full series (rather
    than just the trading slice) lets the rolling z-score window see back
    into the tail of the formation period on the first trading days, instead
    of starting with a window full of NaNs.
    """
    formation_end = formation_end or config.FORMATION_END
    trading_start = trading_start or config.TRADING_START
    trading_end = trading_end or config.TRADING_END

    formation_p1 = full_price1.loc[:formation_end]
    formation_p2 = full_price2.loc[:formation_end]
    const, beta = fit_spread_model(formation_p1, formation_p2)

    full_spread = compute_spread(full_price1, full_price2, const, beta)
    full_zscore = compute_zscore(full_spread)

    pos1_full, pos2_full = signals.generate_positions(full_zscore)

    trading_pos1 = pos1_full.loc[trading_start:trading_end]
    trading_pos2 = pos2_full.loc[trading_start:trading_end]
    trading_p1 = full_price1.loc[trading_start:trading_end]
    trading_p2 = full_price2.loc[trading_start:trading_end]

    daily_returns = signals.strategy_returns(trading_p1, trading_p2, trading_pos1, trading_pos2)

    return {
        "const": const,
        "beta": beta,
        "zscore": full_zscore.loc[trading_start:trading_end],
        "position1": trading_pos1,
        "position2": trading_pos2,
        "daily_returns": daily_returns,
    }


if __name__ == "__main__":
    import data_utils
    prices = data_utils.load_wide_prices()

    # Quick smoke test on a known formation-period-cointegrated pair; run
    # pair_selection.py first to get a real ranked list for your sector.
    permnos = data_utils.get_sector_permnos()
    p1, p2 = permnos[0], permnos[1]
    result = run_classical_strategy(prices[p1], prices[p2])
    cumret = (1 + result["daily_returns"]).cumprod()
    print(f"Classical strategy smoke test on PERMNO {p1}/{p2}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
