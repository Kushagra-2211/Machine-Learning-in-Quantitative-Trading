"""
Orchestrator: run the classical trend-following strategy, ALL ML strategies
(Random Forest, SVM, Neural Network, LSTM, Gradient Boosting), and a passive
Buy & Hold benchmark on every coin in the finalized 15-coin universe
(config.COINS), and compare.

Run this directly in Spyder (or `python main.py` from a terminal). Neural
Network and LSTM need PyTorch (`pip install torch`); Gradient Boosting needs
xgboost (`pip install xgboost`) -- if either isn't installed, this script
skips that piece and still runs everything else, with a warning printed.

Unlike pairs_trading/main.py, there is NO screening/selection step here --
momentum/trend-following is evaluated per-asset, not as a relationship
between two assets, so every coin in the finalized universe is traded
directly (see DISSERTATION_PLAN.md's "Crypto universe -- FINALIZED" note for
how that 15-coin list was itself arrived at).

PIPELINE OVERVIEW (see run_all_strategies_for_coin() and main() below):
  1. For each of the 15 coins, run every strategy over the same held-out
     trading period (2023-01-01 to 2025-12-31): Classical SMA crossover, ML
     (Random Forest / SVM / Neural Net / LSTM / Gradient Boosting), Buy &
     Hold, and a naive persistence baseline used only for RMSE diagnostics
     (not a tradeable strategy).
  2. Average across all 15 coins into one portfolio per strategy, and
     compare Sharpe ratio / returns / drawdown, plus per-coin dispersion,
     bootstrap confidence intervals, and a significance test against
     Classical.
  3. Regime-conditional analysis (Trending vs Range-bound, via Kaufman's
     Efficiency Ratio on the equal-weighted portfolio's own Buy & Hold
     cumulative return -- see regime.py) -- the natural place to look for
     trend-following's expected edge in genuinely trending conditions and
     its expected weakness in choppy/range-bound conditions (the opposite
     pattern from the pairs-trading strand's mean-reversion strategies).
"""
#%%
import os
import json
import datetime

import matplotlib.pyplot as plt
import pandas as pd

import config
import data_utils
import classical_strategy
import ml_common
import ml_random_forest
import ml_svm
import buy_and_hold
import performance
import walk_forward
import regime
import regime_conditional
import signals

try:
    import ml_neural_net
    import ml_lstm
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch not available ({type(e).__name__}: {e}) -- skipping Neural Network and LSTM "
          f"strategies. Run `pip install torch` and re-run this script to include them.")

try:
    import ml_gradient_boosting
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"xgboost not available ({type(e).__name__}: {e}) -- skipping Gradient Boosting "
          f"strategy. Run `pip install xgboost` and re-run this script to include it.")

#%%
def run_all_strategies_for_coin(prices, coin, verbose=True):
    price = prices[coin]

    def maybe_walk_forward(run_fn):
        """Runs a strategy either as one static formation-period fit, or
        (config.WALK_FORWARD = True, the default) refit periodically on an
        expanding formation window via walk_forward.py. If
        config.INCLUDE_REGIME_FEATURES is set, passes that through to any ML
        run_fn that accepts it (classical/buy-and-hold ignore the kwarg
        entirely, so this only wraps the ML strategies below)."""
        if config.INCLUDE_REGIME_FEATURES:
            bound_run_fn = lambda *a, **kw: run_fn(*a, include_regime_features=True, **kw)
        else:
            bound_run_fn = run_fn
        if config.WALK_FORWARD:
            return walk_forward.run_walk_forward(bound_run_fn, price, verbose=verbose)
        return bound_run_fn(price, config.FORMATION_END, config.TRADING_START, config.TRADING_END)

    # Regime-conditional training (config.REGIME_CONDITIONAL_TRAINING) swaps
    # in regime_conditional.py's two-models-per-period wrappers for
    # RF/SVM/Neural Net/Gradient Boosting -- LSTM is intentionally excluded
    # (see regime_conditional.py's module docstring) and keeps using its
    # normal run_strategy regardless of this flag.
    rc = config.REGIME_CONDITIONAL_TRAINING
    rf_fn = regime_conditional.run_random_forest if rc else ml_random_forest.run_strategy
    svm_fn = regime_conditional.run_svm if rc else ml_svm.run_strategy
    nn_fn = regime_conditional.run_neural_net if rc else ml_neural_net.run_strategy
    gb_fn = regime_conditional.run_gradient_boosting if rc else ml_gradient_boosting.run_strategy

    results = {
        # ----- classical SMA-crossover trend-following strategy (classical_strategy.py) -----
        "Classical": maybe_walk_forward(classical_strategy.run_classical_strategy),
        # ----- ML: Random Forest predicting tomorrow's return (ml_random_forest.py) -----
        "ML: Random Forest": maybe_walk_forward(rf_fn),
        # ----- ML: SVM (Support Vector Regression) predicting tomorrow's return (ml_svm.py) -----
        "ML: SVM": maybe_walk_forward(svm_fn),
        # ----- passive benchmark, no prediction, no refitting involved (buy_and_hold.py) -----
        "Buy & Hold": buy_and_hold.run_buy_and_hold(price, config.TRADING_START, config.TRADING_END),
    }

    if TORCH_AVAILABLE:
        results["ML: Neural Net"] = maybe_walk_forward(nn_fn)
        results["ML: LSTM"] = maybe_walk_forward(ml_lstm.run_strategy)
    if XGBOOST_AVAILABLE:
        results["ML: Gradient Boosting"] = maybe_walk_forward(gb_fn)

    # ----- naive persistence baseline, RMSE diagnostics only (ml_common.py) -----
    naive_baseline = maybe_walk_forward(ml_common.run_naive_baseline)
    rmse_diagnostics = {"Naive Baseline (persistence)":
                         ml_common.prediction_rmse(naive_baseline["predicted_return"], naive_baseline["actual_return"])}
    for name, result in results.items():
        if "predicted_return" in result and "actual_return" in result:
            rmse_diagnostics[name] = ml_common.prediction_rmse(result["predicted_return"], result["actual_return"])

    return results, rmse_diagnostics

#%%
def report_regime_stats(label, asset_results, portfolio_returns_dict):
    """Shared reporting block for "the whole period", "Trending days only",
    and "Range-bound days only" -- runs the exact same dispersion / bootstrap
    CI / Wilcoxon-vs-Classical machinery in all three cases.

    Returns (dispersion_df, significance_df, family_df) so main() can save
    all three to CSV."""
    print(f"\n--- {label}: per-coin Sharpe ratio dispersion ---")
    dispersion = performance.dispersion_summary(asset_results)
    print(dispersion.round(4))

    print(f"\n--- {label}: block-bootstrap 95% CI on Sharpe ratio, and significance vs Classical ---")
    rows = []
    for strat_name, daily_returns in portfolio_returns_dict.items():
        point, lower, upper = performance.block_bootstrap_sharpe_ci(daily_returns)
        row = {"Strategy": strat_name, "Sharpe": point, "CI Lower": lower, "CI Upper": upper,
               "Median Diff vs Classical": None, "Wilcoxon p vs Classical": None}
        line = f"  {strat_name}: Sharpe {point:.3f} [{lower:.3f}, {upper:.3f}]"
        if strat_name != "Classical":
            median_diff, p_value = performance.paired_significance_test(asset_results, strat_name, "Classical")
            row["Median Diff vs Classical"] = median_diff
            row["Wilcoxon p vs Classical"] = p_value
            line += f" -- vs Classical: median per-coin Sharpe diff {median_diff:.3f}, Wilcoxon p={p_value:.3f}"
        print(line)
        rows.append(row)
    significance_df = pd.DataFrame(rows).set_index("Strategy")

    # Benjamini-Hochberg correction across this block's family of ML-vs-
    # Classical Wilcoxon tests (see performance.benjamini_hochberg()).
    pvals = significance_df["Wilcoxon p vs Classical"].astype(float).values
    significance_df["BH Significant (q=0.05)"] = performance.benjamini_hochberg(pvals, q=0.05)

    print(f"\n--- {label}: model-family breakdown (High-capacity vs Simple, see Section 2.4) ---")
    family_df = performance.family_comparison(asset_results)
    print(family_df.round(4))
    p_fam = family_df.attrs.get("family_comparison_p_value", float("nan"))
    print(f"  Mann-Whitney U test, High-capacity vs Simple pooled Sharpe diffs: p={p_fam:.4f}")

    return dispersion, significance_df, family_df

#%%
def main():
    prices = data_utils.load_wide_prices()
    formation, trading = data_utils.split_formation_trading(prices)

    print(f"\n=== Crypto momentum strand: {len(config.COINS)} coins ===")
    print(f"Coins: {config.COINS}")
    print(f"Formation: {config.FORMATION_START} to {config.FORMATION_END} "
          f"({formation.shape[0]} calendar days)")
    print(f"Trading:   {config.TRADING_START} to {config.TRADING_END} "
          f"({trading.shape[0]} calendar days)")

    print(f"\nWalk-forward retraining: {'ON' if config.WALK_FORWARD else 'OFF'}"
          + (f" (refit every {config.WALK_FORWARD_FREQUENCY})" if config.WALK_FORWARD
             else " (single static formation fit)"))
    print(f"ML regime features (EfficiencyRatio + ClassicalSignal): "
          f"{'ON' if config.INCLUDE_REGIME_FEATURES else 'OFF'}")
    print(f"ML target horizon: {config.TARGET_HORIZON} day(s)"
          + (" (default -- tomorrow's return)" if config.TARGET_HORIZON == 1
             else f" (cumulative {config.TARGET_HORIZON}-day forward return, still traded daily)"))
    print(f"ML position sizing: {'confidence-weighted' if config.POSITION_SIZING else 'flat +1/-1 (default)'}")
    print(f"ML regime-conditional training (RF/SVM/NN/GB only, LSTM excluded): "
          f"{'ON' if config.REGIME_CONDITIONAL_TRAINING else 'OFF'}")

    strategy_names = ["Classical", "ML: Random Forest", "ML: SVM", "Buy & Hold"]
    if TORCH_AVAILABLE:
        strategy_names += ["ML: Neural Net", "ML: LSTM"]
    if XGBOOST_AVAILABLE:
        strategy_names += ["ML: Gradient Boosting"]
    portfolio_returns = {name: [] for name in strategy_names}
    # ----- Per-coin TURNOVER, collected alongside returns -- see the matching
    # note in pairs_trading/main.py. Saving the portfolio-level daily gross
    # return AND daily turnover series (below) is enough to reconstruct
    # cost-adjusted returns at any bps level, real-risk-free-rate Sharpes, and
    # regime-interaction tests post-hoc, without another full rerun. -----
    portfolio_turnover = {name: [] for name in strategy_names}

    all_asset_results = {}      # {coin: {strategy_name: daily_returns}}
    all_rmse_diagnostics = {}   # {coin: {strategy_name / "Naive Baseline...": rmse}}
    for coin in config.COINS:
        if coin not in prices.columns:
            print(f"WARNING: {coin} not found in loaded price panel -- skipping.")
            continue
        print(f"\n=== Backtesting coin: {coin} ===")
        results, rmse_diagnostics = run_all_strategies_for_coin(prices, coin)

        daily_returns_only = {name: result["daily_returns"] for name, result in results.items()}
        all_asset_results[coin] = daily_returns_only
        all_rmse_diagnostics[coin] = rmse_diagnostics

        summary = performance.summarize(daily_returns_only)
        print(summary.round(4))
        print("  Prediction RMSE (lower is better -- Naive Baseline = predict tomorrow's "
              "return as today's return):")
        for name, rmse in rmse_diagnostics.items():
            print(f"    {name}: {rmse:.4f}")

        for strat_name, daily_returns in daily_returns_only.items():
            portfolio_returns[strat_name].append(daily_returns)

        # ----- Per-coin turnover for this coin, one series per strategy -----
        # Classical/ML strategies all expose "signal" (see
        # classical_strategy.run_classical_strategy / ml_common.finish_strategy),
        # so signals.position_turnover() works identically for every one of
        # them. Buy & Hold has no signal series (single-asset, never
        # rebalances by construction -- see buy_and_hold.py), so its turnover
        # is built directly from that convention: 1.0 unit of turnover on the
        # first trading day (entering from flat), 0 every day after -- purely
        # for a consistent, reconstructable record, since buy_and_hold.py
        # itself never actually charges this (it never rebalances again).
        for strat_name in strategy_names:
            if strat_name == "Buy & Hold":
                bh_returns = daily_returns_only["Buy & Hold"]
                bh_turnover = pd.Series(0.0, index=bh_returns.index)
                if len(bh_turnover) > 0:
                    bh_turnover.iloc[0] = 1.0
                portfolio_turnover[strat_name].append(bh_turnover)
            else:
                result = results[strat_name]
                turnover = signals.position_turnover(result["signal"])
                portfolio_turnover[strat_name].append(turnover)

    # Equal-weighted portfolio across all 15 coins.
    print(f"\n=== Equal-weighted portfolio across {len(config.COINS)} coins ===")
    portfolio_summary_input = {}
    for strat_name, list_of_returns in portfolio_returns.items():
        combined = pd.concat(list_of_returns, axis=1).mean(axis=1)
        portfolio_summary_input[strat_name] = combined
    portfolio_summary = performance.summarize(portfolio_summary_input)
    print(portfolio_summary.round(4))

    # Equal-weighted portfolio-level turnover, same aggregation as returns.
    portfolio_turnover_input = {}
    for strat_name, list_of_turnover in portfolio_turnover.items():
        combined_turnover = pd.concat(list_of_turnover, axis=1).mean(axis=1)
        portfolio_turnover_input[strat_name] = combined_turnover

    # ----- Annual breakdown: is performance driven by the market regime? -----
    print("\n=== Annual breakdown of equal-weighted portfolio performance (regime check) ===")
    annual = performance.annual_breakdown(portfolio_summary_input)
    print(annual.round(4))

    # ----- Dispersion / bootstrap CI / significance: whole out-of-sample period -----
    print(f"\n=== Full period (across {len(config.COINS)} coins) ===")
    dispersion, significance, family = report_regime_stats("Full period", all_asset_results, portfolio_summary_input)

    # ----- Regime-conditional dispersion / bootstrap CI / significance -----
    # Classified from the equal-weighted portfolio's own Buy & Hold cumulative
    # return (the "crypto market" trendiness proxy) -- same choice
    # pairs_trading/main.py makes with its sector-level proxy. See regime.py's
    # docstring for why trend-following is expected to show the OPPOSITE
    # regime-dependence pattern from pairs trading's mean-reversion strategies.
    regime_labels = regime.classify_regime(
        performance.cumulative_return(portfolio_summary_input["Buy & Hold"]),
        window=config.REGIME_WINDOW,
    )
    counts, runs = regime.regime_summary(regime_labels)
    print(f"\n=== Regime classification (Efficiency Ratio, window={config.REGIME_WINDOW}, median split) ===")
    print(f"Day counts: {counts.to_dict()}")
    print(f"{len(runs)} contiguous regime runs:")
    for regime_label, start, end in runs:
        print(f"  {regime_label}: {start.date()} to {end.date()} ({(end - start).days + 1} calendar days)")

    regime_dispersion = {}
    regime_significance = {}
    regime_family = {}
    for regime_value in ("Trending", "Range-bound"):
        filtered_asset_results = performance.filter_asset_results_by_regime(
            all_asset_results, regime_labels, regime_value)
        filtered_portfolio_returns = {
            strat_name: performance.filter_returns_by_regime(daily_returns, regime_labels, regime_value)
            for strat_name, daily_returns in portfolio_summary_input.items()
        }
        regime_dispersion[regime_value], regime_significance[regime_value], regime_family[regime_value] = (
            report_regime_stats(f"Regime: {regime_value}", filtered_asset_results, filtered_portfolio_returns))

    # ----- RMSE diagnostics averaged across coins -----
    print("\n=== Mean prediction RMSE across coins (lower is better) ===")
    rmse_df = pd.DataFrame(all_rmse_diagnostics).T
    print(rmse_df.mean().round(4).sort_values())

    # ----- Save into final_results/momentum_trading/{run_label}/ -----
    # Same dedicated, self-documented layout as pairs_trading/main.py (a
    # sibling final_results/ folder at the top of the dissertation folder,
    # not code/momentum_trading/results/). Every experiment flag still gets
    # folded into run_label exactly as it did into the old filename suffix
    # (e.g. "n15_possize" stays distinct from the headline "n15" and from
    # "n15_regimefeat"), but each variant now gets its own folder with a
    # run_config.json recording exactly which flags were on, rather than a
    # flat suffix-named CSV sitting alongside every other run's files. The
    # older flat results/*_suffix.csv files from earlier exploratory runs are
    # left untouched in code/momentum_trading/results/ as scratch history.
    run_label = f"n{len(config.COINS)}"
    if config.INCLUDE_REGIME_FEATURES:
        run_label += "_regimefeat"
    if config.TARGET_HORIZON != 1:
        run_label += f"_h{config.TARGET_HORIZON}"
    if config.POSITION_SIZING:
        run_label += "_possize"
    if config.REGIME_CONDITIONAL_TRAINING:
        run_label += "_regimecond"
    # Cost-sensitivity passes get their own suffixed run_label (e.g.
    # "n15_cost5bps") so they NEVER overwrite the no-cost headline results --
    # same convention as pairs_trading/main.py.
    if config.INCLUDE_TRANSACTION_COSTS:
        run_label += f"_cost{int(config.TRANSACTION_COST_BPS)}bps"

    run_dir = os.path.join("..", "..", "final_results", "momentum_trading", run_label)
    os.makedirs(os.path.join(run_dir, "trending"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "rangebound"), exist_ok=True)

    plt.figure(figsize=(10, 6))
    for strat_name, daily_returns in portfolio_summary_input.items():
        cum = performance.cumulative_return(daily_returns)
        plt.plot(cum.index, cum.values, label=strat_name)
    plt.title(f"Crypto momentum -- equal-weighted portfolio across {len(config.COINS)} coins\n"
              f"({config.TRADING_START} to {config.TRADING_END}, "
              f"walk-forward={'on' if config.WALK_FORWARD else 'off'})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return (1.0 = starting capital)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "portfolio_comparison.png"), dpi=150)

    portfolio_summary.to_csv(os.path.join(run_dir, "portfolio_summary.csv"))
    dispersion.to_csv(os.path.join(run_dir, "dispersion_summary.csv"))
    significance.to_csv(os.path.join(run_dir, "significance.csv"))
    family.to_csv(os.path.join(run_dir, "family_comparison.csv"))
    rmse_df.to_csv(os.path.join(run_dir, "rmse_diagnostics.csv"))
    annual.to_csv(os.path.join(run_dir, "annual_breakdown.csv"))

    # ----- Raw portfolio-level daily series (NEW, see pairs_trading/main.py
    # for the full rationale) -- unlocks post-hoc regime-interaction tests,
    # risk-free-rate sensitivity, cost curves, and block-length sensitivity
    # without rerunning the backtest again. -----
    returns_df = pd.DataFrame(portfolio_summary_input)
    returns_df.to_csv(os.path.join(run_dir, "daily_returns_by_strategy.csv"))
    turnover_df = pd.DataFrame(portfolio_turnover_input)
    turnover_df.to_csv(os.path.join(run_dir, "daily_turnover_by_strategy.csv"))
    for regime_value, subfolder in (("Trending", "trending"), ("Range-bound", "rangebound")):
        regime_dispersion[regime_value].to_csv(os.path.join(run_dir, subfolder, "dispersion.csv"))
        regime_significance[regime_value].to_csv(os.path.join(run_dir, subfolder, "significance.csv"))
        regime_family[regime_value].to_csv(os.path.join(run_dir, subfolder, "family_comparison.csv"))

    # Self-documenting run metadata -- exact config values used to produce
    # this folder's results.
    run_config = {
        "run_label": run_label,
        "strand": "momentum_trading",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "universe": {
            "coins": config.COINS,
            "n_coins": len(config.COINS),
        },
        "formation_period": {"start": config.FORMATION_START, "end": config.FORMATION_END},
        "trading_period": {"start": config.TRADING_START, "end": config.TRADING_END},
        "trading_rule": {
            "short_window": config.SHORT_WINDOW,
            "long_window": config.LONG_WINDOW,
        },
        "walk_forward": {"enabled": config.WALK_FORWARD, "frequency": config.WALK_FORWARD_FREQUENCY},
        "regime_classification": {
            "window": config.REGIME_WINDOW,
            "method": "Kaufman Efficiency Ratio, median split",
        },
        "statistical_testing": {
            "block_bootstrap_size": config.BLOCK_SIZE,
            "bootstrap_reps": 2000,
            "bh_fdr_q": 0.05,
        },
        "experiment_flags": {
            "include_regime_features": config.INCLUDE_REGIME_FEATURES,
            "target_horizon": config.TARGET_HORIZON,
            "position_sizing": config.POSITION_SIZING,
            "regime_conditional_training": config.REGIME_CONDITIONAL_TRAINING,
        },
        "models_included": strategy_names,
        "random_seed": config.RANDOM_SEED,
        "trading_days_per_year": config.TRADING_DAYS_PER_YEAR,
        "risk_free_rate": config.RISK_FREE_RATE,
        "transaction_costs": (
            {
                "included": True,
                "cost_bps_per_unit_turnover": config.TRANSACTION_COST_BPS,
                "note": "applied to Classical and every ML strategy via signals.strategy_returns(); "
                        "Buy & Hold untouched (never rebalances after initial entry)",
            }
            if config.INCLUDE_TRANSACTION_COSTS
            else "none modeled in this run"
        ),
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nSaved all results for run '{run_label}' to "
          f"final_results/momentum_trading/{run_label}/ -- portfolio_summary.csv, "
          f"dispersion_summary.csv, significance.csv, family_comparison.csv, "
          f"rmse_diagnostics.csv, annual_breakdown.csv, portfolio_comparison.png, "
          f"run_config.json, plus trending/ and rangebound/ subfolders each with "
          f"their own dispersion.csv, significance.csv, family_comparison.csv")

    return (all_asset_results, portfolio_summary, dispersion, rmse_df, annual,
            regime_dispersion, regime_significance, family, regime_family)


if __name__ == "__main__":
    (all_asset_results, portfolio_summary, dispersion, rmse_df, annual,
     regime_dispersion, regime_significance, family, regime_family) = main()
