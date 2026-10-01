"""
Merges the fast-strategy cache (results/cache/{coin}.pkl, from the earlier
full 15-coin run) with the per-period NN/LSTM cache
(results/cache_subset_nnlstm/{coin}_{tag}_p{i}.pkl) for the sanity-check
subset (BTC, ETH), then runs performance.summarize() per coin plus an
equal-weighted 2-coin portfolio -- a smaller-scale but complete
apples-to-apples comparison across ALL 7 series (Classical, RF, SVM, GB,
Buy & Hold, NN, LSTM) to sanity-check against the user's own full run.
"""

import pickle
from pathlib import Path

import pandas as pd

import config
import walk_forward
import performance

SUBSET_COINS = ["BTC", "ETH"]
CACHE_DIR = Path("results/cache")
NNLSTM_CACHE_DIR = Path("results/cache_subset_nnlstm")

periods = walk_forward.generate_walk_forward_periods()

all_results = {}
for coin in SUBSET_COINS:
    with open(CACHE_DIR / f"{coin}.pkl", "rb") as f:
        fast = pickle.load(f)["daily_returns"]

    coin_results = dict(fast)  # Classical, ML: Random Forest, ML: SVM, Buy & Hold, ML: Gradient Boosting

    for tag, label in [("nn", "ML: Neural Net"), ("lstm", "ML: LSTM")]:
        pieces = []
        for i in range(len(periods)):
            with open(NNLSTM_CACHE_DIR / f"{coin}_{tag}_p{i}.pkl", "rb") as f:
                pieces.append(pickle.load(f)["daily_returns"])
        coin_results[label] = pd.concat(pieces).sort_index()

    all_results[coin] = coin_results
    print(f"\n=== {coin} ===")
    print(performance.summarize(coin_results).round(4))

print(f"\n=== Equal-weighted portfolio across {SUBSET_COINS} (sanity-check subset) ===")
strategy_names = list(next(iter(all_results.values())).keys())
portfolio = {}
for name in strategy_names:
    portfolio[name] = pd.concat([all_results[c][name] for c in SUBSET_COINS], axis=1).mean(axis=1)
print(performance.summarize(portfolio).round(4))
