"""
Checkpointed runner: computes run_all_strategies_for_coin() for each coin in
config.COINS and pickles the (daily-returns-only, rmse) results to
results/cache/{coin}.pkl, skipping any coin whose cache file already exists.

This exists purely as a sandbox workaround: background processes don't
survive between separate tool calls here, and a full 15-coin walk-forward
run takes longer than a single tool call's time budget. Running this script
repeatedly (each call picking up where the last left off, via the cache
files) lets the full run complete across several short calls. Not part of
the permanent pipeline -- main.py is still the actual entry point once this
finishes (it will just read a fully-populated cache almost instantly, or you
can ignore this file entirely and run `python3 main.py` directly on a
machine without a per-call time limit).
"""

import pickle
import time
from pathlib import Path

import config
import data_utils
import main as m

m.TORCH_AVAILABLE = False  # sandbox has no PyTorch -- same as main.py's own guard, just forced here

CACHE_DIR = Path("results/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TIME_BUDGET_SECONDS = 38  # stay under the tool call's ~40s ceiling with margin

prices = data_utils.load_wide_prices()
start = time.time()

for coin in config.COINS:
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

done = [c for c in config.COINS if (CACHE_DIR / f"{c}.pkl").exists()]
print(f"\n{len(done)}/{len(config.COINS)} coins cached: {done}")
if len(done) < len(config.COINS):
    print("Run this script again to continue with the remaining coins.")
else:
    print("All coins cached -- ready for final aggregation.")
