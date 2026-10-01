"""
Finer-grained (per walk-forward PERIOD, not just per coin/model) checkpointed
runner for the slow NN/LSTM strategies on a small coin subset --
Classical/RF/SVM/GB/Buy&Hold/naive-baseline for these same coins are already
cached in results/cache/{coin}.pkl from the earlier full 15-coin run, so
there's no need to recompute those.

Checkpoints each (coin, model, walk-forward period index) individually in
results/cache_subset_nnlstm/{coin}_{tag}_p{i}.pkl, so progress survives even
when a single walk-forward period (let alone a whole coin) doesn't finish
within one time-limited call -- LSTM in particular is slow enough that even
ONE period's hyperparameter search can take longer than a single call here.
Sandbox workaround only -- see run_coin_cache.py's docstring for the fuller
explanation of why this exists.
"""

import pickle
import time
from pathlib import Path

import pandas as pd

import config
import data_utils
import walk_forward
import ml_neural_net
import ml_lstm

SUBSET_COINS = ["BTC", "ETH"]
MODELS = [("nn", ml_neural_net.run_strategy), ("lstm", ml_lstm.run_strategy)]

# Reduced hyperparameter-search budget for THIS sanity-check run only (my
# sandbox's per-command time limit can't fit LSTM's full search on a growing
# expanding-window formation set) -- ml_lstm.py itself is UNCHANGED; the
# user's own run uses the real EPOCHS/SEARCH_EPOCHS/grid. This only trims how
# long the search takes here, not the model architecture or trading logic.
ml_lstm.EPOCHS = 150
ml_lstm.SEARCH_EPOCHS = 40
ml_lstm.HYPERPARAM_GRID = ml_lstm.HYPERPARAM_GRID[:2]

CACHE_DIR = Path("results/cache_subset_nnlstm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TIME_BUDGET_SECONDS = 20

prices = data_utils.load_wide_prices()
periods = walk_forward.generate_walk_forward_periods()
start = time.time()

stopped_early = False
for coin in SUBSET_COINS:
    price = prices[coin]
    for tag, run_fn in MODELS:
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
            print(f"{coin} / {tag} / period {i} ({period_start.date()}-{period_end.date()}): "
                  f"done in {time.time() - t0:.1f}s")
        if stopped_early:
            break
    if stopped_early:
        break

total_needed = len(SUBSET_COINS) * len(MODELS) * len(periods)
total_done = sum(
    1 for c in SUBSET_COINS for t, _ in MODELS for i in range(len(periods))
    if (CACHE_DIR / f"{c}_{t}_p{i}.pkl").exists()
)
print(f"\n{total_done}/{total_needed} (coin, model, period) units cached.")
if total_done == total_needed:
    print("All NN/LSTM units done -- ready to stitch and merge.")
