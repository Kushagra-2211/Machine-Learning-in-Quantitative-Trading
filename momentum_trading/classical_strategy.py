"""
Classical (non-ML) trend-following strategy: SMA crossover.

Methodology follows Peng Liu, "Quantitative Trading Strategies Using Python"
(2023), Ch.5 "Trend-Following Strategy" (Listings 5-9 through 5-16):
  - short-term SMA and long-term SMA of the coin's own closing price
  - signal = +1 if SMA_short > SMA_long, else -1 (Listing 5-15)
  - strategy return = signal * return (Listing 5-16, using SIMPLE returns
    here rather than the book's log returns -- see signals.py's module
    docstring for why), applied via signals.strategy_returns() with the
    one-day shift moved onto the signal instead of the SMAs.

Unlike the pairs-trading classical strategy (which fits an OLS hedge ratio
on the formation period only), there is nothing to "fit" here -- SMA windows
are fixed hyperparameters (config.SHORT_WINDOW / config.LONG_WINDOW), not
estimated from data. The strategy is computed over the FULL price history
(so the rolling SMA windows are never starved of history at the very start
of the trading period) and then scored on the trading period only, for a
fair, identical-window comparison against the ML and Buy & Hold strategies.
"""

import numpy as np
import pandas as pd

import config
import signals


def compute_sma(price, window):
    return price.rolling(window=window).mean()


def compute_return(price):
    return price.pct_change()


def run_classical_strategy(full_price, formation_end=None, trading_start=None, trading_end=None,
                            short_window=None, long_window=None, include_regime_features=None):
    """Full pipeline for one coin: compute SMA crossover signal over the
    full history, score daily returns on the trading period only.

    full_price: full-history Close price Series (formation + trading
    concatenated), indexed by date. formation_end is accepted for interface
    consistency with the ML strategies / walk_forward.py (which all expect a
    (formation_end, trading_start, trading_end)-shaped call) but is unused
    here -- the classical strategy has no formation-period fitting step.
    include_regime_features is likewise accepted-but-ignored, for interface
    consistency with main.py's maybe_walk_forward() wrapper (config.
    INCLUDE_REGIME_FEATURES only affects the ML models -- Classical has no
    features to add anything to).
    """
    trading_start = trading_start or config.TRADING_START
    trading_end = trading_end or config.TRADING_END
    short_window = short_window if short_window is not None else config.SHORT_WINDOW
    long_window = long_window if long_window is not None else config.LONG_WINDOW

    sma_short = compute_sma(full_price, short_window)
    sma_long = compute_sma(full_price, long_window)
    full_signal = signals.generate_signal(sma_short, sma_long)
    full_return = compute_return(full_price)

    full_daily_returns = signals.strategy_returns(full_return, full_signal)

    trading_signal = full_signal.loc[trading_start:trading_end]
    trading_daily_returns = full_daily_returns.loc[trading_start:trading_end]

    return {
        "sma_short": sma_short.loc[trading_start:trading_end],
        "sma_long": sma_long.loc[trading_start:trading_end],
        "signal": trading_signal,
        "daily_returns": trading_daily_returns,
    }


if __name__ == "__main__":
    import data_utils
    prices = data_utils.load_wide_prices()

    coin = config.COINS[0]
    result = run_classical_strategy(prices[coin])
    cumret = (1 + result["daily_returns"]).cumprod()
    print(f"Classical trend-following smoke test on {coin}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
