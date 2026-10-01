"""
Follow-up experiment (NOT part of the permanent pipeline): re-runs all 5 ML
momentum variants on the BTC/ETH sanity-check subset with
include_regime_features=True (adds EfficiencyRatio + ClassicalSignal to
ml_common.build_features() -- see that module's docstring), to test whether
giving the ML models explicit access to regime/trend-strength information
closes the gap with Classical in Trending days.

Same per-(coin, model, walk-forward-period) checkpointing pattern as
run_nn_lstm_subset.py -- this sandbox can't fit LSTM's full walk-forward run
in one call. RF/SVM/GB are fast enough to just run inline without their own
checkpoint files.
"""

import pickle
import time
from pathlib import Path

import pandas as pd

import config
import data_utils
import walk_forward
import ml_random_forest
import ml_svm
import ml_neural_net
import ml_lstm
import ml_gradient_boosting

SUBSET_COINS = ["BTC", "ETH"]

# Same reduced LSTM search this sandbox used for the original subset check --
# HYPERPARAM_GRID trimming is the part of that reduction that actually takes
# effect at call time (EPOCHS/SEARCH_EPOCHS are bound as function defaults at
# import time, so reassigning the module attribute afterward doesn't change
# them -- early stopping with patience=15 caps actual training length anyway,
# so this doesn't materially change what's being compared).
ml_lstm.HYPERPARAM_GRID = ml_lstm.HYPERPARAM_GRID[:1]

CACHE_DIR = Path("results/cache_subset_regime")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TIME_BUDGET_SECONDS = 25

prices = data_utils.load_wide_prices()
periods = walk_forward.generate_walk_forward_periods()
start = time.time()

FAST_MODELS = [
    ("rf", lambda p, **kw: ml_random_forest.run_strategy(p, include_regime_features=True, **kw)),
    ("svm", lambda p, **kw: ml_svm.run_strategy(p, include_regime_features=True, **kw)),
    ("gb", lambda p, **kw: ml_gradient_boosting.run_strategy(p, include_regime_features=True, **kw)),
    ("nn", lambda p, **kw: ml_neural_net.run_strategy(p, include_regime_features=True, **kw)),
]
SLOW_MODELS = [
    ("lstm", lambda p, **kw: ml_lstm.run_strategy(p, include_regime_features=True, **kw)),
]

stopped_early = False
for coin in SUBSET_COINS:
    price = prices[coin]
    for tag, run_fn in FAST_MODELS + SLOW_MODELS:
        for i, (formation_end, period_start, period_end) in enumerate(periods):
            cache_path = CACHE_DIR / f"{coin}_{tag}_p{i}.pkl"
            if cache_path.exists():
                continue
            if time.time() - start > TIME_BUDGET_SECONDS:
                print(f"Time budget reached before {coin}/{tag}/period{i}. Re-run to continue.")
                stopped_early = True
                break
            t0 = time.time()
            result = run_fn(price, formation_end=formation_end, trading_start=period_start, trading_end=period_end)
            with open(cache_path, "wb") as f:
                pickle.dump({"daily_returns": result["daily_returns"]}, f)
            print(f"{coin} / {tag} / period {i} ({period_start.date()}-{period_end.date()}): done in {time.time() - t0:.1f}s")
        if stopped_early:
            break
    if stopped_early:
        break

total_needed = len(SUBSET_COINS) * (len(FAST_MODELS) + len(SLOW_MODELS)) * len(periods)
total_done = sum(
    1 for c in SUBSET_COINS for t, _ in FAST_MODELS + SLOW_MODELS for i in range(len(periods))
    if (CACHE_DIR / f"{c}_{t}_p{i}.pkl").exists()
)
print(f"\n{total_done}/{total_needed} (coin, model, period) units cached.")
if total_done == total_needed:
    print("All regime-feature units done -- ready to compare against the baseline subset cache.")
