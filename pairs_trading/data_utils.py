"""
Loading and slicing the cleaned equity data for pairs trading.

Expects the two files produced by code/clean_equity_data.py:
  - equity_adj_price_wide.csv : date x PERMNO matrix of split/dividend
    adjusted prices (this is what we actually trade on -- see
    data/EQUITY_DATA_QA_NOTES.md for why raw price is NOT usable directly).
  - equity_clean_panel.csv    : long format, one row per (PERMNO, date),
    with ticker / GICS sector metadata alongside the price data.
"""

import pandas as pd
import config


def load_wide_prices():
    """Date-indexed DataFrame, columns are PERMNO (int), values are AdjPrice."""
    df = pd.read_csv(config.WIDE_PRICE_PATH, index_col=0, parse_dates=True)
    df.columns = df.columns.astype(int)
    return df.sort_index()


def load_sector_map():
    """One row per PERMNO with its current ticker and GICS sector/sub-industry."""
    panel = pd.read_csv(config.PANEL_PATH, usecols=[
        "PERMNO", "CurrentTicker", "SecurityNm", "gics_sector", "gics_sub_industry"
    ])
    return panel.drop_duplicates(subset="PERMNO").set_index("PERMNO")


def get_sector_permnos(sector=None):
    """PERMNOs belonging to the given GICS sector (defaults to the first
    entry in config.SECTORS if not given explicitly -- main.py always passes
    an explicit sector when looping over config.SECTORS for a combined run)."""
    sector = sector or config.SECTORS[0]
    sector_map = load_sector_map()
    matches = sector_map[sector_map["gics_sector"] == sector]
    if matches.empty:
        available = sorted(sector_map["gics_sector"].dropna().unique())
        raise ValueError(
            f"No companies found for sector '{sector}'. Available sectors: {available}"
        )
    return matches.index.tolist()


def split_formation_trading(price_df):
    """Split a date-indexed price DataFrame into (formation, trading) halves
    using the date ranges in config.py. No fitting should ever touch the
    trading half."""
    formation = price_df.loc[config.FORMATION_START:config.FORMATION_END]
    trading = price_df.loc[config.TRADING_START:config.TRADING_END]
    return formation, trading


def ticker_for(permno, sector_map=None):
    """Convenience: look up a PERMNO's current ticker for readable output."""
    sector_map = sector_map if sector_map is not None else load_sector_map()
    if permno in sector_map.index:
        return sector_map.loc[permno, "CurrentTicker"]
    return str(permno)


if __name__ == "__main__":
    # Quick sanity check when run directly in Spyder.
    prices = load_wide_prices()
    sector_permnos = get_sector_permnos()
    formation, trading = split_formation_trading(prices)
    print(f"Loaded {prices.shape[1]} securities, {prices.shape[0]} trading days total.")
    print(f"{config.SECTORS[0]}: {len(sector_permnos)} companies.")
    print(f"Formation period: {formation.shape[0]} days ({config.FORMATION_START} to {config.FORMATION_END})")
    print(f"Trading period:   {trading.shape[0]} days ({config.TRADING_START} to {config.TRADING_END})")
