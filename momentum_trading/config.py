"""
Central configuration for the crypto momentum (trend-following) strand of
the dissertation.

Change values here rather than hunting through the other modules -- same
philosophy as pairs_trading/config.py.
"""

# --- Data paths (relative to this package's parent 'code' folder's sibling 'data' folder) ---
PANEL_PATH = "../../data/crypto_clean_panel.csv"

# --- Universe ---
# The 15 coins finalized after checking gap-free coverage of the full
# 2018-01-01 to 2025-12-31 window (see DISSERTATION_PLAN.md's "Crypto
# universe -- FINALIZED" note for the full exclusion list and rationale).
# Unlike the pairs-trading strand, there is no candidate-screening step here
# (no cointegration test, no "top N" selection) -- every coin in this list is
# traded, since momentum/trend-following is evaluated per-asset, not as a
# relationship between two assets.
COINS = [
    "BTC", "ETH", "BNB", "LTC", "NEO", "ADA", "QTUM", "XRP", "IOTA",
    "XLM", "ICX", "ETC", "TRX", "VET", "ONT",
]

# --- Formation / trading period split ---
# All model fitting (feature scaling, ML training) happens ONLY on the
# formation period; the trading period is held out and never touched during
# fitting -- same discipline as the pairs-trading strand. The classical SMA
# crossover strategy has no parameters to "fit" as such, but is still only
# scored on the trading period so every strategy is compared on identical
# out-of-sample days.
#
# Split chosen to fall on a meaningful structural break in the crypto market
# rather than an arbitrary midpoint: formation (2018-01-01 to 2022-12-31)
# spans the 2018 bear market, 2020 COVID crash/recovery, the 2021 bull run,
# AND the 2022 "crypto winter" (Terra/Luna collapse in May 2022, FTX collapse
# in November 2022) -- i.e. it already contains a full boom-bust cycle by the
# time trading starts. Trading (2023-01-01 to 2025-12-31) then captures the
# 2023-2024 recovery/bull run and 2025, giving a genuinely different market
# regime from most of formation -- the crypto-strand analog of the pairs
# strand's 2016-2021 formation / 2022-2025 trading split (which was chosen
# for the same reason: a regime shift sitting at the boundary).
FORMATION_START = "2018-01-01"
FORMATION_END = "2022-12-31"
TRADING_START = "2023-01-01"
TRADING_END = "2025-12-31"

# --- Trend-following rule (SMA crossover) ---
# Following Peng Liu, "Quantitative Trading Strategies Using Python" (2023),
# Ch.5 "Trend-Following Strategy": go long (+1) when the short-term SMA is
# above the long-term SMA, short (-1) when it's below -- always in the
# market, no flat/neutral state (matches the book's own signal convention:
# np.where(SMA_short > SMA_long, 1, -1)).
#
# The book's own worked example uses SMA-3/SMA-20 on a single year of Apple
# stock data -- a small, illustrative window pairing. Crypto trades 24/7
# (no weekend/holiday gaps) over an 8-year sample here, so a slower pairing
# gives more stable, less noise-dominated crossovers: SHORT_WINDOW=10 /
# LONG_WINDOW=50 (documented deviation from the book's 3/20, same spirit as
# pairs_trading/config.py's ZSCORE_WINDOW=20 vs the book's 10 -- larger
# windows for a longer, richer sample). Easy to set back to 3/20 to reproduce
# the book's exact pairing if wanted.
SHORT_WINDOW = 10
LONG_WINDOW = 50

# --- Statistical testing (see performance.py) ---
# Block size (periods) for the moving-block bootstrap on the Sharpe ratio --
# see pairs_trading/config.py's BLOCK_SIZE for the full rationale. 20 here
# means 20 CALENDAR days (crypto trades every day), matching REGIME_WINDOW's
# own calendar-day convention below.
BLOCK_SIZE = 20

# --- Regime classification (see regime.py) ---
# Rolling window (calendar days -- crypto trades every day, unlike equities'
# trading days) for Kaufman's Efficiency Ratio, used to split the trading
# period into "Trending" vs "Range-bound" days on a reproducible, data-driven
# basis (median split). Matches pairs_trading/config.py's REGIME_WINDOW=60
# for cross-strand consistency, even though here "60" means 60 calendar days
# instead of 60 trading days (crypto has no non-trading days to skip).
REGIME_WINDOW = 60

# --- Walk-forward retraining ---
# Same rationale as pairs_trading/config.py: refit periodically on an
# EXPANDING formation window (always starting at FORMATION_START, growing
# over time) instead of once on a single static formation period, to guard
# against the fitted relationship (here: the ML model's mapping from
# momentum features to next-day return) drifting over a multi-year trading
# window. Set to False to reproduce the original static single-fit
# behaviour, so the two can be compared directly.
WALK_FORWARD = True
WALK_FORWARD_FREQUENCY = "YS"  # pandas offset alias: "YS" = annual refits, "QS" = quarterly (4x the compute)

# --- ML strategy ---
RANDOM_SEED = 42

# Adds EfficiencyRatio (Kaufman, rolling/causal, same window as REGIME_WINDOW
# above) and ClassicalSignal (the classical strategy's own +1/-1 SMA-crossover
# signal) as explicit input features for every ML model (see
# ml_common.build_features()) -- a follow-up experiment testing whether
# giving the ML models direct access to regime/trend-strength information
# closes any of the gap with Classical in Trending days (the sanity-check
# subset found it narrows the gap for most models -- especially LSTM and
# Random Forest -- without closing it; see DISSERTATION_PLAN.md). Off by
# default so the original headline results reproduce exactly; set True and
# re-run main.py to reproduce the "with regime features" variant on the
# full 15-coin universe.
INCLUDE_REGIME_FEATURES = False

# --- Target horizon experiment ---
# Default target is TOMORROW's return (horizon=1). Setting this higher makes
# every ML model predict the CUMULATIVE return over the next N days instead
# (still trading daily on the sign of that prediction) -- a smoother,
# less noisy trend-continuation proxy that may suit trend-following better
# than a 1-day-ahead target. See ml_common.build_target()/build_sequences()
# for the exact shift arithmetic. Off (1) by default so headline results
# reproduce exactly.
TARGET_HORIZON = 1

# --- Position sizing experiment ---
# Default trading rule is flat +1/-1 (always fully long or fully short, per
# signals.py). Setting this True switches every ML model (not Classical) to
# a confidence-weighted position instead: predicted_return normalized by the
# formation period's own target volatility, clipped to [-1, 1] -- bigger
# predicted moves get bigger (up to fully-sized) positions, small/uncertain
# predictions get scaled-down ones, instead of always betting a full unit
# either way. See ml_common.finish_strategy_sized(). Off by default.
#
# Reset to False (flat baseline) for the headline re-run -- this flag was
# left on from an earlier isolated position-sizing experiment (results saved
# separately under "_n15_possize", untouched by this reset). Set back to True
# and re-run separately if you want to reproduce that experiment's numbers;
# the headline Methodology/Results write-up describes the flat +1/-1 signal,
# which requires this to be False.
POSITION_SIZING = False

# --- Regime-conditional training experiment ---
# Default trains ONE model per walk-forward period on the whole formation
# window. Setting this True (RF/SVM/Neural Net/Gradient Boosting only --
# LSTM's sequence-windowed data makes a clean regime split much more
# involved, so it's excluded from this experiment) instead trains TWO
# separate models per period -- one on formation rows classified Trending,
# one on formation rows classified Range-bound (via a FIXED, causal
# Efficiency-Ratio median threshold computed from the formation period only)
# -- and applies whichever model matches each trading day's own regime.
# See regime_conditional.py. Off by default.
REGIME_CONDITIONAL_TRAINING = False

# --- Transaction costs ---
# Single-asset analog of pairs_trading/config.py's cost model. The headline
# n=15 run was first produced with NO transaction costs modeled at all, to
# isolate whether ML adds anything over Classical on a frictionless
# comparison. INCLUDE_TRANSACTION_COSTS switches on a cost-sensitivity pass
# over the SAME 15-coin universe, so the two can be compared directly
# (run_label picks up a "_cost{N}bps" suffix automatically -- see main.py --
# so the no-cost results are never overwritten by turning this on).
#
# Costs are modeled as a fixed number of basis points of notional per unit of
# position TURNOVER -- the same simplification used in the pairs strand,
# just on a single leg instead of two. Every strategy here is always fully
# invested at +1 or -1 (no flat state -- see signals.py), so turnover on any
# given day is 0 (position held) or 2 (a full long-to-short or short-to-long
# flip); a flip is treated as closing the old position and opening the new
# one, i.e. two units of turnover, applied identically to Classical and
# every ML strategy since all of them route through signals.strategy_returns().
# Buy & Hold is untouched, consistent with the pairs strand's convention,
# since it never rebalances after its initial entry.
INCLUDE_TRANSACTION_COSTS = False  # reset to the no-cost baseline -- this is
                           # the headline n=15 rerun (needed to pick up the
                           # new daily_returns_by_strategy.csv /
                           # daily_turnover_by_strategy.csv persistence).
                           # Set True again for a fresh cost5bps pass if
                           # ever needed; that run already has its own
                           # results saved under n15_cost5bps/ untouched.
TRANSACTION_COST_BPS = 5   # 5 basis points (0.05%) of notional per unit of
                           # turnover, one-way -- identical illustrative
                           # assumption to the pairs strand's transaction-cost
                           # robustness check (Section 5.1.6), not calibrated
                           # to any specific exchange/venue.

# --- Performance ---
# Crypto trades every calendar day (365/year), unlike equities' ~252 trading
# days/year -- this changes annualization math (annualized_return/volatility/
# Sharpe in performance.py), so it's kept as its own config constant rather
# than reusing pairs_trading's TRADING_DAYS_PER_YEAR=252.
TRADING_DAYS_PER_YEAR = 365
RISK_FREE_RATE = 0.0  # annualized; set to a real T-bill rate later if wanted
