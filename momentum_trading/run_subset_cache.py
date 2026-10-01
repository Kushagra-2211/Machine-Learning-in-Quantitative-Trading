"""
Same checkpointing workaround as run_coin_cache.py, but scoped to a small
SUBSET_COINS list and covering ALL strategies (including NN/LSTM, now that
torch is installed) -- a full 15-coin walk-forward run with NN/LSTM takes
too long for this sandbox's per-command time limit. This subset run exists
purely as a code-correctness sanity check to compare against the user's own
full 15-coin run on their own machine. Not part of the permanent pipeline.
"""

import pickle
import time
from pathlib import Path

import config
import data_utils
import main as m

SUBSET_COINS = ["BTC", "ETH", "XRP", "ADA"]

CACHE_DIR = Path("results/cache_subset_full")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TIME_BUDGET_SECONDS = 36

prices = data_utils.load_wide_prices()
start = time.time()

for coin in SUBSET_COINS:
    cache_path = CACHE_DIR / f"{coin}.pkl"
    if cache_path.exists():
        print(f"{coin}: cached, skipping")
        continue
    if time.time() - start > TIME_BUDGET_SECONDS:
        print(f"Time budget reached, stopping before {coin}. Re-run this script to continue.")
        break
    t0 = time.time()
    results, rmse_diagnostics = m.run_all_strategies_for_coin(prices, coin, verbose=False)
    daily_returns_only = {name: result["daily_returns"] for name, result in results.items()}
    with open(cache_path, "wb") as f:
        pickle.dump({"daily_returns": daily_returns_only, "rmse": rmse_diagnostics}, f)
    print(f"{coin}: done in {time.time() - t0:.1f}s")

done = [c for c in SUBSET_COINS if (CACHE_DIR / f"{c}.pkl").exists()]
print(f"\n{len(done)}/{len(SUBSET_COINS)} subset coins cached: {done}")
