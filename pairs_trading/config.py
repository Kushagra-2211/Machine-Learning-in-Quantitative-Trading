"""
Central configuration for the pairs trading strand of the dissertation.

Change values here rather than hunting through the other modules.
"""

# --- Data paths (relative to this package's parent 'code' folder's sibling 'data' folder) ---
WIDE_PRICE_PATH = "../../data/equity_adj_price_wide.csv"
PANEL_PATH = "../../data/equity_clean_panel.csv"

# --- Universe ---
# HEADLINE DESIGN (as of this comment): three runs -- n=10, n=20, n=40 -- all
# using the IDENTICAL selection methodology, differing ONLY in TOTAL_N_PAIRS.
# This replaced an earlier, less consistent design where n=10 was a
# Financials-only run and n=20 was capped at 5 pairs from each of a fixed
# front-4-sectors list -- neither of those is comparable to a global
# top-N selection, which made "n=10 -> n=20 -> n=40" an apples-to-oranges
# progression (both N and sector mix changed at every step). Under the
# current design, cointegration SCREENING still happens STRICTLY WITHIN each
# sector (a Financials stock is never tested against a Utilities stock) --
# this preserves the multiple-testing/false-discovery/common-fundamentals
# rationale from Gatev, Goetzmann & Rouwenhorst (2006) that justifies
# within-sector screening in the first place. What changes across the three
# runs is ONLY how many of the resulting globally-ranked pairs are kept.
#
# SECTORS is ordered by descending GICS candidate-pool size (a purely
# mechanical rule, not cherry-picked, based on the actual company counts in
# equity_clean_panel.csv): Industrials (80), Financials (77), Information
# Technology (76), Health Care (60), Consumer Discretionary (49), Consumer
# Staples (39), Real Estate (32), Utilities (31), Materials (29),
# Communication Services (23), Energy (22). With GLOBAL_TOP_SELECTION=True,
# every sector in this list is screened, so the ordering itself has no
# effect on results -- it only affects the order sectors are printed/screened
# in the console log.
SECTORS = [
    "Industrials", "Financials", "Information Technology", "Health Care",
    "Consumer Discretionary", "Consumer Staples", "Real Estate", "Utilities",
    "Materials", "Communication Services", "Energy",
]
PAIRS_PER_SECTOR = 5   # top-N cointegrated pairs taken from EACH sector used this run
                       # (only used when GLOBAL_TOP_SELECTION is False -- see below;
                       # kept only for the older select_combined_pairs() code path,
                       # which is no longer part of the planned headline runs)
TOTAL_N_PAIRS = 40     # THE ONLY VALUE TO CHANGE between runs.
                       # Reset to 40 (the headline universe) after the n=80
                       # exploratory run -- needed so threshold_sensitivity.py
                       # reproduces the IDENTICAL 40-pair universe used in
                       # final_results/pairs_trading/globaltop_n40/ (pair
                       # selection is deterministic given TOTAL_N_PAIRS, so
                       # this must match exactly for that comparison to be
                       # fair). n=80 results remain saved under their own
                       # globaltop_n80/ folder, untouched by this reset.
                       # Each run saves under its own run_label folder in
                       # final_results/pairs_trading/, so nothing gets
                       # overwritten between runs.

# GLOBAL_TOP_SELECTION changes how the combined pool is built:
#   True (current design for ALL three planned runs, n=10/20/40) -- screen
#     EVERY sector in SECTORS (cointegration testing still never crosses
#     sector boundaries -- see the rationale above), pool ALL candidate pairs
#     from every sector into one list, sort by cointegration p-value
#     GLOBALLY, and take the overall top TOTAL_N_PAIRS regardless of which
#     industry they come from (see select_global_top_pairs() in main.py).
#     This answers: does the strategy generalize when pair selection is
#     allowed to concentrate in whichever industries happen to produce the
#     strongest cointegration, rather than being spread deliberately across a
#     fixed list of sectors? Because all three runs use this same rule, the
#     ONLY thing that changes across n=10/20/40 is the pair count -- a clean
#     basis for comparing how ML vs. classical performance moves as the
#     universe broadens, without also confounding it with a change in
#     selection methodology.
#   False (superseded, kept only for backward compatibility / robustness
#     checks) -- take sectors off the FRONT of SECTORS, PAIRS_PER_SECTOR
#     pairs from each, until TOTAL_N_PAIRS is reached (see
#     select_combined_pairs() in main.py). This is the ORIGINAL n=20 design
#     (5 pairs each from Industrials/Financials/IT/Health Care) -- its
#     results are already saved under "_combined_n20" and are NOT being
#     replaced, just no longer part of the three headline runs.
GLOBAL_TOP_SELECTION = True

# --- Formation / trading period split ---
# Cointegration screening and all model fitting (OLS spread regression, ML
# model training) happens ONLY on the formation period. The trading period
# is held out and never touched during fitting -- this is what makes the
# backtest genuinely out-of-sample rather than an in-sample curve-fit.
FORMATION_START = "2016-01-01"
FORMATION_END = "2021-12-31"
TRADING_START = "2022-01-01"
TRADING_END = "2025-12-31"

# --- Pair selection ---
# Engle-Granger cointegration test p-value threshold. Standard is 0.05, but
# see DISSERTATION_PLAN.md's methodology note on why screening many pairs at
# once creates a multiple-testing / false-discovery risk -- worth reporting
# expected false positives alongside results, not just the raw p-value cutoff.
COINT_PVALUE_THRESHOLD = 0.05
# NOTE: the earlier single-sector headline results (Financials n=10/n=25,
# Utilities n=10) used a single config.SECTOR + config.TOP_N_PAIRS pair of
# settings. Those have been superseded by SECTORS/PAIRS_PER_SECTOR/
# TOTAL_N_PAIRS above (a single-sector run is just the special case of that
# with one sector in the list), but the old single-sector CSVs/plots in
# results/ are untouched -- this combined-sector mode saves under its own
# "_combined_n{TOTAL_N_PAIRS}" filenames so nothing gets overwritten.
#
# Methodological note worth keeping in the write-up regardless of sector mix:
# within Financials alone, 109 of 2,701 testable pairs are significant at raw
# p<0.05 (actually *below* the ~135 expected by chance alone across that many
# simultaneous tests), and literally none of the 109 survive Benjamini-Hochberg
# FDR correction at q=0.05 or q=0.10. Read the p-value ranking used to pick the
# top N per sector as an economically-motivated ordering device (as in Gatev
# et al.'s original methodology), not a family-wise-error-controlled
# significance claim -- and expect the same to hold in every other sector
# screened here, not just Financials.

# --- Trading rule (z-score thresholds) ---
# Following Peng Liu, "Quantitative Trading Strategies Using Python" (2023),
# Ch.8: entry at +/-2 standard deviations, exit at +/-1 (not 0) -- positions
# are held, not exited, while |z| is between 1 and 2.
ZSCORE_WINDOW = 20  # Peng Liu's book uses a 10-day window; we use 20 for a
                    # more stable rolling estimate given our longer sample --
                    # documented deviation, easy to set back to 10 if desired.
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 1.0

# --- Statistical testing (see performance.py) ---
# Block size (trading days) for the moving-block bootstrap on the Sharpe
# ratio -- resamples contiguous BLOCK_SIZE-day chunks instead of individual
# days, so the day-to-day serial correlation that position-holding creates
# in daily strategy returns survives the resample (the i.i.d. bootstrap in
# bootstrap_sharpe_ci() does not preserve this -- see that function's
# docstring). 20 matches ZSCORE_WINDOW/the rolling-feature windows already
# used elsewhere in this codebase (~1 trading month).
BLOCK_SIZE = 20

# --- Regime classification (see regime.py) ---
# Rolling window (trading days) for Kaufman's Efficiency Ratio, used to split
# the trading period into "Trending" vs "Range-bound" days on a
# reproducible, data-driven basis (median split) rather than a fixed
# technical-analysis threshold. 60 trading days (~1 quarter) matches the
# window already used for ml_common's rolling half-life feature.
REGIME_WINDOW = 60

# --- Walk-forward retraining ---
# If True, every strategy is refit periodically on an EXPANDING formation
# window (always starting at FORMATION_START, growing over time) instead of
# once on a single static formation period -- see walk_forward.py for the
# full rationale (diagnosed regime-shift between a single static 2016-2021
# fit and 2022-2025 trading performance). Set to False to reproduce the
# original static single-fit behaviour, so the two can be compared directly.
WALK_FORWARD = True
WALK_FORWARD_FREQUENCY = "YS"  # pandas offset alias: "YS" = annual refits, "QS" = quarterly (4x the compute)

# --- ML strategy ---
ML_MODEL = "random_forest"  # "random_forest" | "svm" | "neural_net" (only
                            # random_forest is implemented for now -- see
                            # ml_strategy.py to add the others later)
RANDOM_SEED = 42

# --- Performance ---
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0  # annualized; set to a real T-bill rate later if wanted

# --- Transaction costs ---
# All three headline runs (n=10/20/40, see TOTAL_N_PAIRS above) were first
# produced with NO transaction costs modeled at all, purely to isolate
# whether ML adds anything over Classical on a frictionless comparison.
# INCLUDE_TRANSACTION_COSTS switches on a cost-sensitivity pass over the
# SAME three universes, so the two can be compared directly (run_label picks
# up a "_cost{N}bps" suffix automatically -- see main.py -- so no results
# folder is ever overwritten by turning this on/off).
#
# Costs are modeled as a fixed number of basis points of NOTIONAL per unit of
# position TURNOVER -- a standard simplification in pairs-trading
# cost-sensitivity studies that folds bid-ask spread, commission, and a small
# slippage allowance into one number rather than modeling each separately.
# Applied identically to Classical and every ML strategy since all of them
# route through signals.strategy_returns() (see that module); Buy & Hold gets
# a single entry-cost charge on day 1 only, since it never rebalances (see
# buy_and_hold.py).
#
# Position sizes here are always in {-1, 0, +1} per leg, so: entering from
# flat costs TRANSACTION_COST_BPS once (one unit of turnover on that leg);
# a full flip (e.g. -1 -> +1) costs 2x TRANSACTION_COST_BPS (two units of
# turnover -- economically two trades, closing the old leg and opening the
# new one); a leg that just holds its existing position costs nothing.
INCLUDE_TRANSACTION_COSTS = False  # reset to the no-cost baseline for this
                           # n=80 headline run (was left True after the
                           # n=40_cost5bps pass) -- flip back to True for a
                           # follow-up n=80 cost-sensitivity pass once the
                           # frictionless n=80 results are in.
TRANSACTION_COST_BPS = 5   # 5 basis points (0.05%) of notional per unit of
                           # turnover, one-way -- a commonly used, moderate
                           # assumption for liquid large-cap US equities
                           # (roughly: half the bid-ask spread + commission +
                           # a small slippage allowance). Not calibrated to
                           # any specific broker/venue -- intended as a
                           # reasonable illustrative cost level for a
                           # cost-sensitivity check, not a claim about actual
                           # achievable execution cost.
#
# NOTE on walk-forward boundaries (see walk_forward.py): each walk-forward
# sub-period's position series is generated fresh from that sub-period's
# own model fit and only sliced to that sub-period afterward (see
# classical_strategy.py / ml_common.finish_strategy()), so the FIRST day of
# every sub-period is treated as a fresh entry from flat for cost purposes,
# even on the ~3 internal annual boundaries where the "true" position may
# not actually have been flat the day before. This is a conservative
# (cost-INFLATING, never cost-deflating) simplification -- worth stating
# explicitly in the write-up rather than leaving it implicit.
