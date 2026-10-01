"""
Passive buy-and-hold benchmark: hold one coin for the entire trading period,
no rebalancing, no shorting. Single-asset analog of
pairs_trading/buy_and_hold.py's 50/50 split -- answers "would you have been
better off just holding this coin" rather than "does actively trading its
trend work", which is exactly why it's a useful point of comparison against
both the classical SMA-crossover strategy and the ML strategies.
"""

import pandas as pd


def run_buy_and_hold(price, trading_start, trading_end):
    p = price.loc[trading_start:trading_end]
    daily_returns = p.pct_change().fillna(0.0)
    return {"daily_returns": daily_returns}


if __name__ == "__main__":
    import data_utils
    import config
    prices = data_utils.load_wide_prices()
    coin = config.COINS[0]
    result = run_buy_and_hold(prices[coin], config.TRADING_START, config.TRADING_END)
    cumret = (1 + result["daily_returns"]).cumprod()
    print(f"Buy-and-hold smoke test on {coin}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
