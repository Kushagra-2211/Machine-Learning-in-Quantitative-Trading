"""
One-off aggregation step: reads the per-coin cached results/cache/{coin}.pkl
files written by run_coin_cache.py (a sandbox time-limit workaround -- see
that file's docstring) and runs the exact same portfolio/regime/dispersion/
significance/plotting logic main.main() does, just sourcing per-coin results
from the cache instead of recomputing them. Produces the identical
results/*.csv and results/*.png outputs main.py would have written itself.
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
import performance
import regime
import main as m

CACHE_DIR = Path("results/cache")

all_asset_results = {}
all_rmse_diagnostics = {}
for coin in config.COINS:
    with open(CACHE_DIR / f"{coin}.pkl", "rb") as f:
        cached = pickle.load(f)
    all_asset_results[coin] = cached["daily_returns"]
    all_rmse_diagnostics[coin] = cached["rmse"]

strategy_names = list(next(iter(all_asset_results.values())).keys())
print(f"Strategies present: {strategy_names}")

portfolio_returns = {name: [] for name in strategy_names}
for coin, daily_returns_only in all_asset_results.items():
    summary = performance.summarize(daily_returns_only)
    print(f"\n=== {coin} ===")
    print(summary.round(4))
    for strat_name, daily_returns in daily_returns_only.items():
        portfolio_returns[strat_name].append(daily_returns)

print(f"\n=== Equal-weighted portfolio across {len(config.COINS)} coins ===")
portfolio_summary_input = {}
for strat_name, list_of_returns in portfolio_returns.items():
    combined = pd.concat(list_of_returns, axis=1).mean(axis=1)
    portfolio_summary_input[strat_name] = combined
portfolio_summary = performance.summarize(portfolio_summary_input)
print(portfolio_summary.round(4))

print("\n=== Annual breakdown of equal-weighted portfolio performance (regime check) ===")
annual = performance.annual_breakdown(portfolio_summary_input)
print(annual.round(4))

print(f"\n=== Full period (across {len(config.COINS)} coins) ===")
dispersion, significance = m.report_regime_stats("Full period", all_asset_results, portfolio_summary_input)

regime_labels = regime.classify_regime(
    performance.cumulative_return(portfolio_summary_input["Buy & Hold"]),
    window=config.REGIME_WINDOW,
)
counts, runs = regime.regime_summary(regime_labels)
print(f"\n=== Regime classification (Efficiency Ratio, window={config.REGIME_WINDOW}, median split) ===")
print(f"Day counts: {counts.to_dict()}")
print(f"{len(runs)} contiguous regime runs (showing first/last 5):")
for regime_label, start, end in runs[:5] + (runs[-5:] if len(runs) > 5 else []):
    print(f"  {regime_label}: {start.date()} to {end.date()} ({(end - start).days + 1} calendar days)")

regime_dispersion = {}
regime_significance = {}
for regime_value in ("Trending", "Range-bound"):
    filtered_asset_results = performance.filter_asset_results_by_regime(
        all_asset_results, regime_labels, regime_value)
    filtered_portfolio_returns = {
        strat_name: performance.filter_returns_by_regime(daily_returns, regime_labels, regime_value)
        for strat_name, daily_returns in portfolio_summary_input.items()
    }
    regime_dispersion[regime_value], regime_significance[regime_value] = m.report_regime_stats(
        f"Regime: {regime_value}", filtered_asset_results, filtered_portfolio_returns)

print("\n=== Mean prediction RMSE across coins (lower is better) ===")
rmse_df = pd.DataFrame(all_rmse_diagnostics).T
print(rmse_df.mean().round(4).sort_values())

suffix = f"_n{len(config.COINS)}"

plt.figure(figsize=(10, 6))
for strat_name, daily_returns in portfolio_summary_input.items():
    cum = performance.cumulative_return(daily_returns)
    plt.plot(cum.index, cum.values, label=strat_name)
plt.title(f"Crypto momentum -- equal-weighted portfolio across {len(config.COINS)} coins\n"
          f"({config.TRADING_START} to {config.TRADING_END}, walk-forward=on)")
plt.xlabel("Date")
plt.ylabel("Cumulative return (1.0 = starting capital)")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/portfolio_comparison{suffix}.png", dpi=150)
print(f"\nSaved plot: results/portfolio_comparison{suffix}.png")

portfolio_summary.to_csv(f"results/portfolio_summary{suffix}.csv")
dispersion.to_csv(f"results/dispersion_summary{suffix}.csv")
significance.to_csv(f"results/significance{suffix}.csv")
rmse_df.to_csv(f"results/rmse_diagnostics{suffix}.csv")
annual.to_csv(f"results/annual_breakdown{suffix}.csv")
for regime_value in ("Trending", "Range-bound"):
    tag = regime_value.lower().replace(" ", "_").replace("-", "")
    regime_dispersion[regime_value].to_csv(f"results/dispersion_{tag}{suffix}.csv")
    regime_significance[regime_value].to_csv(f"results/significance_{tag}{suffix}.csv")
print("Saved all summary tables to results/*.csv")
