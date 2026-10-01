"""
Robustness check: classical z-score entry/exit threshold sensitivity.

Every headline pairs-trading result in the dissertation uses ENTRY_THRESHOLD
= 2.0 / EXIT_THRESHOLD = 1.0 (config.py). This script asks a single question:
does the Classical strategy's result change qualitatively if those two
numbers had been chosen differently? It re-runs the classical strategy ONLY
-- never any of the five ML models -- on the exact same n=40 global-top-N
pair universe as final_results/pairs_trading/globaltop_n40/, at two extra
threshold pairs bracketing the headline choice:

    baseline_2.0_1.0  (already reported -- reproduced here for a fair,
                       side-by-side comparison in one place)
    tight_1.5_0.5     (enters sooner, exits sooner -- more frequent trading)
    loose_2.5_1.5     (enters later, exits later -- more conservative)

WHY IT'S SAFE TO SKIP THE ML MODELS
------------------------------------
config.ENTRY_THRESHOLD / config.EXIT_THRESHOLD are read in exactly one
place: signals.generate_positions(), which only classical_strategy.py calls.
ml_common.py (the shared feature/training code behind every ML model) never
references either constant -- the ML models predict the spread itself and
are completely blind to how the classical rule later thresholds a z-score.
Changing these two numbers therefore cannot change any ML model's features,
training, or predictions; a classical-only re-run is a complete, faithful
test of threshold sensitivity.

WHY IT'S FAST
-------------
The n=40 headline run is slow almost entirely because of ML fitting (NN/LSTM
hyperparameter search, refit every walk-forward period, for every pair).
classical_strategy.run_classical_strategy() is just one OLS regression plus
a rolling z-score per walk-forward sub-period -- no iterative training at
all. The only genuinely slow step left in this script is re-screening all
11 sectors for cointegration (deterministic, a few minutes, same as every
previous run) to reproduce the identical 40-pair universe; the actual
backtest loop after that should finish in well under a minute.

WHY n=10 AND n=20 ARE INCLUDED "FOR FREE"
------------------------------------------
Pair selection is deterministic and NESTED (confirmed empirically earlier in
this project: n=10's pairs are exactly the first 10 rows of n=20's, which are
exactly the first 20 rows of n=40's, all ranked by global cointegration
p-value -- see Section 5.1.4/4.2). top_pairs here is already sorted by
p-value, so top_pairs.iloc[:10] and top_pairs.iloc[:20] ARE the n=10 and n=20
universes. Rather than re-screening and re-backtesting three separate times,
this script backtests each of the 40 pairs' classical strategy ONCE per
threshold set, then reports the n=10/20/40 portfolio built from the first
10/20/40 of those already-computed pairs -- zero extra screening or backtest
cost for the two smaller universes.

Run with: python threshold_sensitivity.py
"""
import os
import json
import time
import datetime

import pandas as pd

import config
import data_utils
import classical_strategy
import walk_forward
import buy_and_hold
import performance
import regime
from main import select_global_top_pairs

# ----- Threshold sets to test (entry, exit) -----
THRESHOLD_SETS = [
    ("baseline_2.0_1.0", 2.0, 1.0),
    ("tight_1.5_0.5", 1.5, 0.5),
    ("loose_2.5_1.5", 2.5, 1.5),
]

# ----- Universe sizes to report, all sliced from the same 40-pair backtest
# (see module docstring: nested selection means this costs nothing extra). -----
UNIVERSE_SIZES = [10, 20, 40]


def run_classical_only_for_pair(price1, price2):
    """Classical strategy only, walk-forward if config.WALK_FORWARD (matches
    main.py's maybe_walk_forward for the Classical entry exactly)."""
    if config.WALK_FORWARD:
        return walk_forward.run_walk_forward(
            classical_strategy.run_classical_strategy, price1, price2, verbose=False)
    return classical_strategy.run_classical_strategy(
        price1, price2, config.FORMATION_END, config.TRADING_START, config.TRADING_END)


def main():
    t0 = time.time()
    prices = data_utils.load_wide_prices()
    formation, trading = data_utils.split_formation_trading(prices)

    print(f"config.TOTAL_N_PAIRS = {config.TOTAL_N_PAIRS}, "
          f"GLOBAL_TOP_SELECTION = {config.GLOBAL_TOP_SELECTION} "
          f"(must match globaltop_n40 exactly for this to be a fair comparison)")
    top_pairs = select_global_top_pairs(prices, formation)
    t_screened = time.time()
    print(f"\n[timing] pair screening took {t_screened - t0:.1f}s")

    # ----- Buy & Hold portfolio + regime labels: computed ONCE, reused for
    # every threshold set below, since neither depends on the classical
    # entry/exit thresholds at all. -----
    bh_returns = []
    for _, row in top_pairs.iterrows():
        p1, p2 = row["permno1"], row["permno2"]
        bh = buy_and_hold.run_buy_and_hold(
            prices[p1], prices[p2], config.TRADING_START, config.TRADING_END)
        bh_returns.append(bh["daily_returns"])
    bh_portfolio = pd.concat(bh_returns, axis=1).mean(axis=1)
    regime_labels = regime.classify_regime(
        performance.cumulative_return(bh_portfolio), window=config.REGIME_WINDOW)

    # Labels for each universe size, in the same order/format used below --
    # top_pairs is already sorted by p-value (see select_global_top_pairs()),
    # so iloc[:N] IS the n=N universe (nested-selection property).
    labels_by_size = {
        n: [f"{row['sector']}:{row['ticker1']}-{row['ticker2']}"
            for _, row in top_pairs.iloc[:n].iterrows()]
        for n in UNIVERSE_SIZES
    }

    summary_rows = []

    for set_label, entry_threshold, exit_threshold in THRESHOLD_SETS:
        print(f"\n=== Threshold set: {set_label} (entry={entry_threshold}, exit={exit_threshold}) ===")
        config.ENTRY_THRESHOLD = entry_threshold
        config.EXIT_THRESHOLD = exit_threshold

        # Backtest the classical strategy on all 40 pairs ONCE per threshold
        # set -- the n=10/20 universes below are sliced from this, not
        # separately re-screened or re-backtested (see module docstring).
        pair_returns = {}
        for _, row in top_pairs.iterrows():
            p1, p2 = row["permno1"], row["permno2"]
            label = f"{row['sector']}:{row['ticker1']}-{row['ticker2']}"
            result = run_classical_only_for_pair(prices[p1], prices[p2])
            pair_returns[label] = result["daily_returns"]

        for n in UNIVERSE_SIZES:
            subset_labels = labels_by_size[n]
            portfolio_returns = pd.concat(
                [pair_returns[lbl] for lbl in subset_labels], axis=1).mean(axis=1)

            point, lower, upper = performance.block_bootstrap_sharpe_ci(portfolio_returns)
            per_pair_sharpe = {lbl: performance.sharpe_ratio(pair_returns[lbl]) for lbl in subset_labels}

            row_out = {
                "Threshold Set": set_label,
                "Universe Size (n)": n,
                "Entry Threshold": entry_threshold,
                "Exit Threshold": exit_threshold,
                "Full-period Sharpe": point,
                "CI Lower": lower,
                "CI Upper": upper,
                "Terminal Return": performance.terminal_return(portfolio_returns),
                "Annualized Volatility": performance.annualized_volatility(portfolio_returns),
                "Max Drawdown": performance.max_drawdown(portfolio_returns),
                "Mean Per-Pair Sharpe": pd.Series(per_pair_sharpe).mean(),
                "Std Per-Pair Sharpe": pd.Series(per_pair_sharpe).std(),
            }

            for regime_value in ("Trending", "Range-bound"):
                filtered = performance.filter_returns_by_regime(portfolio_returns, regime_labels, regime_value)
                r_point, r_lower, r_upper = performance.block_bootstrap_sharpe_ci(filtered)
                row_out[f"{regime_value} Sharpe"] = r_point
                row_out[f"{regime_value} CI Lower"] = r_lower
                row_out[f"{regime_value} CI Upper"] = r_upper

            print(f"  n={n}: Full-period Sharpe {point:.3f} [{lower:.3f}, {upper:.3f}] | "
                  f"Trending {row_out['Trending Sharpe']:.3f} "
                  f"[{row_out['Trending CI Lower']:.3f}, {row_out['Trending CI Upper']:.3f}] | "
                  f"Range-bound {row_out['Range-bound Sharpe']:.3f} "
                  f"[{row_out['Range-bound CI Lower']:.3f}, {row_out['Range-bound CI Upper']:.3f}]")

            summary_rows.append(row_out)

    summary_df = pd.DataFrame(summary_rows).set_index(["Threshold Set", "Universe Size (n)"])
    print("\n=== Summary across all threshold sets and universe sizes ===")
    print(summary_df.round(4))

    # ----- Save -----
    run_dir = os.path.join("..", "..", "final_results", "pairs_trading", "threshold_sensitivity")
    os.makedirs(run_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(run_dir, "summary.csv"))

    run_config = {
        "check": "classical_entry_exit_threshold_sensitivity",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": "ML models intentionally NOT re-run -- config.ENTRY_THRESHOLD/EXIT_THRESHOLD "
                "are read only inside signals.generate_positions(), which only the classical "
                "strategy calls; no ML model's features or training depend on either constant.",
        "universe": {
            "selection_mode": "global_top",
            "total_n_pairs_screened": config.TOTAL_N_PAIRS,
            "universe_sizes_reported": UNIVERSE_SIZES,
            "universe_sizes_note": "n=10/20 are the first 10/20 rows of the same n=40 backtest, "
                                    "not separately screened or backtested -- pair selection is "
                                    "deterministic and nested (see module docstring).",
            "sectors_used": sorted(top_pairs["sector"].unique().tolist()),
        },
        "threshold_sets_tested": [
            {"label": lbl, "entry_threshold": e, "exit_threshold": x} for lbl, e, x in THRESHOLD_SETS
        ],
        "walk_forward": {"enabled": config.WALK_FORWARD, "frequency": config.WALK_FORWARD_FREQUENCY},
        "regime_classification": {"window": config.REGIME_WINDOW, "method": "Kaufman Efficiency Ratio, median split"},
        "statistical_testing": {"block_bootstrap_size": config.BLOCK_SIZE, "bootstrap_reps": 2000},
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\n[timing] total elapsed: {time.time() - t0:.1f}s")
    print(f"Saved summary.csv and run_config.json to final_results/pairs_trading/threshold_sensitivity/")

    return summary_df


if __name__ == "__main__":
    summary_df = main()
