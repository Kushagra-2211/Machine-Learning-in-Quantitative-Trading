"""
Cluster D (ChatGPT-review Points 5 & 6 / RQ1 & RQ3): two new block-bootstrap
tests run directly on each existing run's saved daily_returns_by_strategy.csv
-- no rerun of main.py needed, since these are pure post-hoc statistics on
data already produced and saved.

Point 6 / RQ1: for each strategy, is its OWN Sharpe ratio itself
regime-dependent (Trending vs Range-bound)? See
performance.trend_range_delta_bootstrap_test().

Point 5 / RQ3: for each ML model vs Classical, does the SIZE of its edge
over Classical itself change between regimes (the interaction)? See
performance.ml_classical_regime_interaction_bootstrap_test().

Regime labels are recomputed on demand from each run's own Buy & Hold
column via regime.classify_regime() -- deterministic given the same data
and config.REGIME_WINDOW, so there is no need to have saved them separately.

Usage: run from inside code/pairs_trading/ (same convention as main.py,
since config.py's PANEL_PATH etc. are relative paths):
    python3 cluster_d_analysis.py
"""

import os

import pandas as pd

import config
import performance
import regime

STRATEGIES_FOR_DELTA_TABLE = [
    "Classical", "ML: Random Forest", "ML: SVM",
    "ML: Neural Net", "ML: LSTM", "ML: Gradient Boosting",
]

# Headline no-cost universe-size runs only (the cost5bps and n=80 exploratory
# runs predate the daily_returns_by_strategy.csv save step and are outside
# the scope of this analysis).
RUNS = {
    "n10": os.path.join("..", "..", "final_results", "pairs_trading", "globaltop_n10"),
    "n20": os.path.join("..", "..", "final_results", "pairs_trading", "globaltop_n20"),
    "n40": os.path.join("..", "..", "final_results", "pairs_trading", "globaltop_n40"),
}


def load_daily_returns(run_dir):
    path = os.path.join(run_dir, "daily_returns_by_strategy.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return {col: df[col] for col in df.columns}


def main():
    combined_delta = {}
    combined_interaction = {}

    for run_label, run_dir in RUNS.items():
        print(f"\n=== Cluster D: {run_label} ({run_dir}) ===")
        strategy_returns = load_daily_returns(run_dir)
        regime_labels = regime.classify_regime(
            performance.cumulative_return(strategy_returns["Buy & Hold"]),
            window=config.REGIME_WINDOW,
        )

        delta_table = performance.trend_range_delta_table(
            strategy_returns, regime_labels, strategies=STRATEGIES_FOR_DELTA_TABLE)
        print("\n--- Point 6 / RQ1: per-strategy Trend-Range Sharpe delta ---")
        print(delta_table.round(4))
        delta_table.to_csv(os.path.join(run_dir, "cluster_d_trend_range_delta.csv"))
        combined_delta[run_label] = delta_table["Delta Sharpe (Trend - Range)"]

        interaction_table = performance.ml_classical_interaction_table(
            strategy_returns, regime_labels)
        print("\n--- Point 5 / RQ3: ML-vs-Classical regime interaction ---")
        print(interaction_table.round(4))
        interaction_table.to_csv(os.path.join(run_dir, "cluster_d_ml_classical_interaction.csv"))
        combined_interaction[run_label] = interaction_table["Delta_regime (Trend gap - Range gap)"]

    combined_delta_df = pd.DataFrame(combined_delta)
    combined_interaction_df = pd.DataFrame(combined_interaction)
    out_dir = os.path.join("..", "..", "final_results", "pairs_trading", "cross_run_comparison")
    os.makedirs(out_dir, exist_ok=True)
    combined_delta_df.to_csv(os.path.join(out_dir, "cluster_d_trend_range_delta_n10_n20_n40.csv"))
    combined_interaction_df.to_csv(os.path.join(out_dir, "cluster_d_ml_classical_interaction_n10_n20_n40.csv"))

    print("\n=== Combined across n=10/20/40 (point estimates only, see per-run CSVs for full CI/p) ===")
    print("\nTrend-Range delta:")
    print(combined_delta_df.round(4))
    print("\nML-vs-Classical interaction:")
    print(combined_interaction_df.round(4))


if __name__ == "__main__":
    main()
