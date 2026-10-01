"""
Passive buy-and-hold benchmark: 50/50 capital split across the pair's two
stocks, held for the entire trading period, no rebalancing. This answers a
different question than the pairs-trading strategies do -- "would you have
been better off just holding these two stocks" rather than "does actively
trading their spread work" -- which is exactly why it's a useful third point
of comparison, not a redundant one (a market-neutral pairs strategy can look
good on Sharpe ratio while a simple bull market makes buy-and-hold win on
raw return, or vice versa in a downturn).
"""

import pandas as pd

import config


def run_buy_and_hold(price1, price2, trading_start, trading_end):
    p1 = price1.loc[trading_start:trading_end]
    p2 = price2.loc[trading_start:trading_end]
    ret1 = p1.pct_change().fillna(0.0)
    ret2 = p2.pct_change().fillna(0.0)
    daily_returns = 0.5 * ret1 + 0.5 * ret2

    # ===================== ONE-TIME ENTRY COST =====================
    # Unlike Classical/ML (which rebalance -- see signals.transaction_cost_drag()
    # for the per-day turnover-based cost applied to those), Buy & Hold never
    # rebalances after its initial 50/50 entry, so it only ever incurs ONE
    # transaction cost: opening the position on the very first trading day.
    # Turnover on entry from flat is 0.5 + 0.5 = 1.0 unit (both legs, half
    # size each), charged the same TRANSACTION_COST_BPS as every other
    # strategy -- a no-op (subtracts 0) whenever
    # config.INCLUDE_TRANSACTION_COSTS is False.
    # =================================================================
    if config.INCLUDE_TRANSACTION_COSTS and len(daily_returns) > 0:
        entry_cost = 1.0 * (config.TRANSACTION_COST_BPS / 10000.0)
        daily_returns.iloc[0] -= entry_cost

    return {"daily_returns": daily_returns}


if __name__ == "__main__":
    import data_utils
    import config
    prices = data_utils.load_wide_prices()
    permnos = data_utils.get_sector_permnos()
    p1, p2 = permnos[0], permnos[1]
    result = run_buy_and_hold(prices[p1], prices[p2], config.TRADING_START, config.TRADING_END)
    cumret = (1 + result["daily_returns"]).cumprod()
    print(f"Buy-and-hold smoke test on PERMNO {p1}/{p2}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
