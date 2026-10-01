"""
Loading and slicing the cleaned crypto data for the momentum strand.

Expects the file produced during data collection (see DISSERTATION_PLAN.md's
"Crypto universe -- FINALIZED" note): crypto_clean_panel.csv, long format,
one row per (Symbol, Date), gap-free daily coverage of 2018-01-01 to
2025-12-31 for all 15 coins in config.COINS.
"""

import pandas as pd

import config


def load_panel():
    """Raw long-format panel: one row per (Symbol, Date)."""
    df = pd.read_csv(config.PANEL_PATH, parse_dates=["Date"])
    return df.sort_values(["Symbol", "Date"])


def load_wide_prices(field="Close"):
    """Date-indexed DataFrame, columns are coin symbols (str), values are the
    given price field (default: Close). Analogous to pairs_trading's
    load_wide_prices(), but pivoted from the long-format crypto panel rather
    than read directly from an already-wide CSV."""
    panel = load_panel()
    wide = panel.pivot(index="Date", columns="Symbol", values=field)
    return wide.sort_index()


def split_formation_trading(price_df):
    """Split a date-indexed price DataFrame into (formation, trading) halves
    using the date ranges in config.py. No fitting should ever touch the
    trading half."""
    formation = price_df.loc[config.FORMATION_START:config.FORMATION_END]
    trading = price_df.loc[config.TRADING_START:config.TRADING_END]
    return formation, trading


if __name__ == "__main__":
    # Quick sanity check when run directly.
    prices = load_wide_prices()
    formation, trading = split_formation_trading(prices)
    print(f"Loaded {prices.shape[1]} coins, {prices.shape[0]} calendar days total.")
    print(f"Coins: {sorted(prices.columns.tolist())}")
    print(f"Formation period: {formation.shape[0]} days ({config.FORMATION_START} to {config.FORMATION_END})")
    print(f"Trading period:   {trading.shape[0]} days ({config.TRADING_START} to {config.TRADING_END})")
    missing = [c for c in config.COINS if c not in prices.columns]
    if missing:
        print(f"WARNING: coins missing from panel: {missing}")
