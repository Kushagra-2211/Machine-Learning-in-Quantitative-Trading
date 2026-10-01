"""
Compares the BTC/ETH subset ML results WITHOUT vs WITH the new
EfficiencyRatio/ClassicalSignal features (see ml_common.build_features()'s
include_regime_features flag), focusing specifically on Trending-regime
performance -- the question being tested: does giving the ML models
explicit regime information close (any of) the gap with Classical in
Trending days? One-off follow-up experiment script, not part of the
permanent pipeline.
"""

import pickle
from pathlib import Path

import pandas as pd

import config
import walk_forward
import performance
import regime

SUBSET_COINS = ["BTC", "ETH"]
FAST_CACHE = Path("results/cache")                    # Classical/RF/SVM/GB/Buy&Hold, no regime feats (full-run cache)
OLD_NNLSTM_CACHE = Path("results/cache_subset_nnlstm")  # NN/LSTM, no regime feats, reduced grid
NEW_CACHE = Path("results/cache_subset_regime")         # all 5 ML models, WITH regime feats, reduced grid

periods = walk_forward.generate_walk_forward_periods()


def load_old(coin):
    with open(FAST_CACHE / f"{coin}.pkl", "rb") as f:
        fast = pickle.load(f)["daily_returns"]
    out = dict(fast)  # Classical, ML: Random Forest, ML: SVM, Buy & Hold, ML: Gradient Boosting
    for tag, label in [("nn", "ML: Neural Net"), ("lstm", "ML: LSTM")]:
        pieces = [pickle.load(open(OLD_NNLSTM_CACHE / f"{coin}_{tag}_p{i}.pkl", "rb"))["daily_returns"]
                  for i in range(len(periods))]
        out[label] = pd.concat(pieces).sort_index()
    return out


def load_new(coin):
    out = {}
    tag_to_label = {
        "rf": "ML: Random Forest", "svm": "ML: SVM", "gb": "ML: Gradient Boosting",
        "nn": "ML: Neural Net", "lstm": "ML: LSTM",
    }
    for tag, label in tag_to_label.items():
        pieces = [pickle.load(open(NEW_CACHE / f"{coin}_{tag}_p{i}.pkl", "rb"))["daily_returns"]
                  for i in range(len(periods))]
        out[label] = pd.concat(pieces).sort_index()
    # Classical / Buy & Hold are unaffected by ML feature changes -- reuse the same series
    old_fast = pickle.load(open(FAST_CACHE / f"{coin}.pkl", "rb"))["daily_returns"]
    out["Classical"] = old_fast["Classical"]
    out["Buy & Hold"] = old_fast["Buy & Hold"]
    return out


old_by_coin = {c: load_old(c) for c in SUBSET_COINS}
new_by_coin = {c: load_new(c) for c in SUBSET_COINS}

strategy_order = ["Classical", "ML: Random Forest", "ML: SVM", "Buy & Hold",
                   "ML: Neural Net", "ML: LSTM", "ML: Gradient Boosting"]


def portfolio(by_coin):
    return {name: pd.concat([by_coin[c][name] for c in SUBSET_COINS], axis=1).mean(axis=1)
            for name in strategy_order}


old_portfolio = portfolio(old_by_coin)
new_portfolio = portfolio(new_by_coin)

# Regime labels from the (feature-invariant) Buy & Hold portfolio -- identical for old/new
regime_labels = regime.classify_regime(
    performance.cumulative_return(old_portfolio["Buy & Hold"]), window=config.REGIME_WINDOW
)

print("=== FULL PERIOD: BEFORE (no regime features) vs AFTER (with regime features) ===")
before_full = performance.summarize(old_portfolio)[["Sharpe Ratio", "Terminal Return"]]
after_full = performance.summarize(new_portfolio)[["Sharpe Ratio", "Terminal Return"]]
comparison_full = before_full.join(after_full, lsuffix=" (before)", rsuffix=" (after)")
print(comparison_full.round(4))

for regime_value in ("Trending", "Range-bound"):
    print(f"\n=== {regime_value.upper()} DAYS ONLY: BEFORE vs AFTER ===")
    old_filtered = {name: performance.filter_returns_by_regime(r, regime_labels, regime_value)
                    for name, r in old_portfolio.items()}
    new_filtered = {name: performance.filter_returns_by_regime(r, regime_labels, regime_value)
                    for name, r in new_portfolio.items()}
    before = performance.summarize(old_filtered)[["Sharpe Ratio", "Terminal Return"]]
    after = performance.summarize(new_filtered)[["Sharpe Ratio", "Terminal Return"]]
    comparison = before.join(after, lsuffix=" (before)", rsuffix=" (after)")
    comparison["Sharpe gap to Classical (before)"] = comparison["Sharpe Ratio (before)"].loc["Classical"] - comparison["Sharpe Ratio (before)"]
    comparison["Sharpe gap to Classical (after)"] = comparison["Sharpe Ratio (after)"].loc["Classical"] - comparison["Sharpe Ratio (after)"]
    print(comparison.round(4))
