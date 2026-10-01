"""
Cluster D (ChatGPT-review Points 5 & 6 / RQ1 & RQ3) for the momentum strand
-- see pairs_trading/cluster_d_analysis.py for the full rationale. Runs on
the headline no-cost n=15 run's saved daily_returns_by_strategy.csv.

Usage: run from inside code/momentum_trading/:
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

RUN_DIR = os.path.join("..", "..", "final_results", "momentum_trading", "n15")


def load_daily_returns(run_dir):
    path = os.path.join(run_dir, "daily_returns_by_strategy.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return {col: df[col] for col in df.columns}


def main():
    print(f"\n=== Cluster D: momentum n15 ({RUN_DIR}) ===")
    strategy_returns = load_daily_returns(RUN_DIR)
    regime_labels = regime.classify_regime(
        performance.cumulative_return(strategy_returns["Buy & Hold"]),
        window=config.REGIME_WINDOW,
    )

    delta_table = performance.trend_range_delta_table(
        strategy_returns, regime_labels, strategies=STRATEGIES_FOR_DELTA_TABLE)
    print("\n--- Point 6 / RQ1: per-strategy Trend-Range Sharpe delta ---")
    print(delta_table.round(4))
    delta_table.to_csv(os.path.join(RUN_DIR, "cluster_d_trend_range_delta.csv"))

    interaction_table = performance.ml_classical_interaction_table(strategy_returns, regime_labels)
    print("\n--- Point 5 / RQ3: ML-vs-Classical regime interaction ---")
    print(interaction_table.round(4))
    interaction_table.to_csv(os.path.join(RUN_DIR, "cluster_d_ml_classical_interaction.csv"))


if __name__ == "__main__":
    main()
