"""
Orchestrator: screen pairs within EACH sector in config.SECTORS separately
(preserving the within-sector-only cointegration screening rationale), pool
the top PAIRS_PER_SECTOR pairs from each into one COMBINED multi-sector
universe of config.TOTAL_N_PAIRS pairs, then run the classical,
buy-and-hold, and ALL ML strategies (Random Forest, SVM, Neural Network,
LSTM, Gradient Boosting) on that combined pool, and compare.

Run this directly in Spyder (or `python main.py` from a terminal). Screening
several ~30-80-company sectors takes a few minutes total -- that's normal,
not a hang. Neural Network and LSTM need PyTorch (`pip install torch`);
Gradient Boosting needs xgboost (`pip install xgboost`) -- if either isn't
installed, this script skips that piece and still runs everything else,
with a warning printed.

NOTE ON RUNTIME: with config.WALK_FORWARD = True (the default) every
strategy is refit once per walk-forward sub-period instead of once total,
and the NN/LSTM now also run a small hyperparameter search per fit (see
ml_neural_net.py / ml_lstm.py) -- both changes multiply runtime versus
earlier versions of this script, and runtime now also scales with
config.TOTAL_N_PAIRS (40 pairs takes ~4x as long as 10). If a first run is
too slow, the quickest dials to turn down are config.TOTAL_N_PAIRS,
config.WALK_FORWARD_FREQUENCY, or temporarily setting config.WALK_FORWARD =
False.

PIPELINE OVERVIEW (see run_all_strategies_for_pair() and main() below):
  1. For each sector in config.SECTORS (taking only as many sectors as
     needed to reach config.TOTAL_N_PAIRS at config.PAIRS_PER_SECTOR each):
     screen for cointegrated pairs WITHIN that sector only -> pair_selection.py
     (pairs are selected ONCE on the original static formation period, and
     the same pair identities are used throughout even when walk-forward
     retraining is on -- only each pair's MODEL is refit periodically, not
     which pairs are being traded). Take the top PAIRS_PER_SECTOR from each
     sector and pool them into one combined DataFrame of pairs.
  2. For each pair in the combined pool, run every strategy over the same
     held-out trading period: Classical z-score, ML (Random Forest / SVM /
     Neural Net / LSTM / Gradient Boosting), Buy & Hold, and a naive
     persistence baseline used only for RMSE diagnostics (not a tradeable
     strategy).
  3. Average across all pairs in the combined pool into one portfolio per
     strategy, and compare Sharpe ratio / returns / drawdown, plus per-pair
     dispersion, bootstrap confidence intervals, and a significance test
     against Classical.
"""
#%%
import os
import json
import datetime

import matplotlib.pyplot as plt
import pandas as pd

import config
import data_utils
import pair_selection
import classical_strategy
import ml_common
import ml_random_forest
import ml_svm
import buy_and_hold
import performance
import walk_forward
import regime
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
def run_all_strategies_for_pair(prices, permno1, permno2, verbose=True):
    price1, price2 = prices[permno1], prices[permno2]

    def maybe_walk_forward(run_fn):
        """Runs a strategy either as one static formation-period fit, or
        (config.WALK_FORWARD = True, the default) refit periodically on an
        expanding formation window via walk_forward.py -- see that module's
        docstring for the full rationale."""
        if config.WALK_FORWARD:
            return walk_forward.run_walk_forward(run_fn, price1, price2, verbose=verbose)
        return run_fn(price1, price2, config.FORMATION_END, config.TRADING_START, config.TRADING_END)

    results = {
        # ----- classical z-score strategy (classical_strategy.py) -----
        "Classical": maybe_walk_forward(classical_strategy.run_classical_strategy),
        # ----- ML: Random Forest predicting the spread (ml_random_forest.py) -----
        "ML: Random Forest": maybe_walk_forward(ml_random_forest.run_strategy),
        # ----- ML: SVM (Support Vector Regression) predicting the spread (ml_svm.py) -----
        "ML: SVM": maybe_walk_forward(ml_svm.run_strategy),
        # ----- passive benchmark, no spread/prediction, no refitting involved (buy_and_hold.py) -----
        # (always a single call over the whole trading range regardless of
        # WALK_FORWARD -- see walk_forward.make_buy_and_hold_adapter()'s
        # docstring for why routing this through walk-forward would subtly
        # understate returns at each period boundary)
        "Buy & Hold": buy_and_hold.run_buy_and_hold(price1, price2, config.TRADING_START, config.TRADING_END),
    }

    if TORCH_AVAILABLE:
        # ----- ML: feedforward Neural Network predicting the spread (ml_neural_net.py) -----
        results["ML: Neural Net"] = maybe_walk_forward(ml_neural_net.run_strategy)
        # ----- ML: LSTM predicting the spread from a sequence of past days (ml_lstm.py) -----
        results["ML: LSTM"] = maybe_walk_forward(ml_lstm.run_strategy)
    if XGBOOST_AVAILABLE:
        # ----- ML: Gradient Boosting predicting the spread (ml_gradient_boosting.py) -----
        results["ML: Gradient Boosting"] = maybe_walk_forward(ml_gradient_boosting.run_strategy)

    # ----- naive persistence baseline, RMSE diagnostics only (ml_common.py) -----
    naive_baseline = maybe_walk_forward(ml_common.run_naive_baseline)
    rmse_diagnostics = {"Naive Baseline (persistence)":
                         ml_common.prediction_rmse(naive_baseline["predicted_spread"], naive_baseline["actual_spread"])}
    for name, result in results.items():
        if "predicted_spread" in result and "actual_spread" in result:
            rmse_diagnostics[name] = ml_common.prediction_rmse(result["predicted_spread"], result["actual_spread"])

    return results, rmse_diagnostics

#%%
def report_regime_stats(label, pair_results, portfolio_returns_dict):
    """Shared reporting block for "the whole period", "Trending days only",
    and "Range-bound days only" -- runs the exact same dispersion / bootstrap
    CI / Wilcoxon-vs-Classical machinery in all three cases, just on a
    different (possibly regime-filtered) all_pair_results / pooled-portfolio-
    returns pair. This is what actually tests the regime-dependence claim
    statistically instead of eyeballing the calendar-year breakdown: if
    SVM's edge over Classical is genuinely a mean-reversion-regime effect,
    it should show up as a tighter/more significant result when this function
    is called with pair_results restricted to "Range-bound" days than with
    "Trending" days.

    Returns (dispersion_df, significance_df, family_df) so main() can save
    all three to CSV."""
    print(f"\n--- {label}: per-pair Sharpe ratio dispersion ---")
    dispersion = performance.dispersion_summary(pair_results)
    print(dispersion.round(4))

    print(f"\n--- {label}: block-bootstrap 95% CI on Sharpe ratio, and significance vs Classical ---")
    rows = []
    for strat_name, daily_returns in portfolio_returns_dict.items():
        point, lower, upper = performance.block_bootstrap_sharpe_ci(daily_returns)
        row = {"Strategy": strat_name, "Sharpe": point, "CI Lower": lower, "CI Upper": upper,
               "Median Diff vs Classical": None, "Wilcoxon p vs Classical": None}
        line = f"  {strat_name}: Sharpe {point:.3f} [{lower:.3f}, {upper:.3f}]"
        if strat_name != "Classical":
            median_diff, p_value = performance.paired_significance_test(pair_results, strat_name, "Classical")
            row["Median Diff vs Classical"] = median_diff
            row["Wilcoxon p vs Classical"] = p_value
            line += f" -- vs Classical: median per-pair Sharpe diff {median_diff:.3f}, Wilcoxon p={p_value:.3f}"
        print(line)
        rows.append(row)
    significance_df = pd.DataFrame(rows).set_index("Strategy")

    # Benjamini-Hochberg correction across this block's family of ML-vs-
    # Classical Wilcoxon tests (Classical's own row has no p-value and is
    # left out of the correction -- see performance.benjamini_hochberg()).
    pvals = significance_df["Wilcoxon p vs Classical"].astype(float).values
    significance_df["BH Significant (q=0.05)"] = performance.benjamini_hochberg(pvals, q=0.05)

    print(f"\n--- {label}: model-family breakdown (High-capacity vs Simple, see Section 2.4) ---")
    family_df = performance.family_comparison(pair_results)
    print(family_df.round(4))
    p_fam = family_df.attrs.get("family_comparison_p_value", float("nan"))
    print(f"  Mann-Whitney U test, High-capacity vs Simple pooled Sharpe diffs: p={p_fam:.4f}")

    return dispersion, significance_df, family_df

#%%
def select_combined_pairs(prices, formation):
    """Screen each sector in config.SECTORS INDEPENDENTLY (cointegration
    testing never crosses sector boundaries -- see config.py's rationale),
    take the top config.PAIRS_PER_SECTOR pairs from each, and pool them into
    one combined DataFrame covering config.TOTAL_N_PAIRS pairs in total.

    Returns the combined top_pairs DataFrame (with a "sector" column added
    so every downstream print/plot can still show provenance) and prints a
    clear per-sector breakdown so there's never ambiguity about which
    sectors/how-many-pairs a given run actually covers."""
    assert config.TOTAL_N_PAIRS % config.PAIRS_PER_SECTOR == 0, (
        f"config.TOTAL_N_PAIRS ({config.TOTAL_N_PAIRS}) must be an exact multiple of "
        f"config.PAIRS_PER_SECTOR ({config.PAIRS_PER_SECTOR})."
    )
    n_sectors_needed = config.TOTAL_N_PAIRS // config.PAIRS_PER_SECTOR
    assert n_sectors_needed <= len(config.SECTORS), (
        f"Need {n_sectors_needed} sectors to reach TOTAL_N_PAIRS={config.TOTAL_N_PAIRS} at "
        f"PAIRS_PER_SECTOR={config.PAIRS_PER_SECTOR}, but config.SECTORS only has "
        f"{len(config.SECTORS)} entries."
    )
    active_sectors = config.SECTORS[:n_sectors_needed]
    print(f"\n=== Combined multi-sector pair universe: {config.TOTAL_N_PAIRS} pairs total "
          f"({config.PAIRS_PER_SECTOR} from each of {n_sectors_needed} sectors) ===")
    print(f"Sectors used, in order: {active_sectors}")

    per_sector_pairs = []
    for sector in active_sectors:
        permnos = data_utils.get_sector_permnos(sector)
        print(f"\n--- Screening sector: {sector} ({len(permnos)} companies) ---")
        screen_results = pair_selection.screen_pairs(formation, permnos, sector_label=sector)
        sector_top = pair_selection.top_pairs_with_tickers(
            screen_results, n=config.PAIRS_PER_SECTOR)
        sector_top["sector"] = sector
        print(f"Top {config.PAIRS_PER_SECTOR} pairs from {sector}:")
        print(sector_top[["ticker1", "ticker2", "coint_score", "pvalue", "overlap_days"]])
        per_sector_pairs.append(sector_top)

    combined = pd.concat(per_sector_pairs, ignore_index=True)
    print(f"\n=== Combined pool: {len(combined)} pairs across {n_sectors_needed} sectors "
          f"({combined['sector'].value_counts().to_dict()}) ===")
    return combined


def select_global_top_pairs(prices, formation):
    """GLOBAL_TOP_SELECTION mode: screen EVERY sector in config.SECTORS
    (cointegration testing still never crosses sector boundaries within each
    screen_pairs() call -- only the ranking/selection AFTER screening is
    global), pool every sector's full ranked candidate list together, sort
    by p-value across the WHOLE pool, and take the overall top
    config.TOTAL_N_PAIRS -- "top N pairs of any industry" rather than a
    fixed per-sector quota.

    Because a sector with unusually strong cointegration could in principle
    supply more than PAIRS_PER_SECTOR of the global top N, this prints a
    per-sector breakdown of how many pairs of the FINAL top N actually came
    from each sector, so the industry concentration (or lack of it) is
    reported transparently rather than assumed even-handed."""
    print(f"\n=== Global top-{config.TOTAL_N_PAIRS} pair universe: screening ALL "
          f"{len(config.SECTORS)} sectors, then taking the overall best "
          f"{config.TOTAL_N_PAIRS} pairs by cointegration p-value regardless of sector ===")

    per_sector_results = []
    for sector in config.SECTORS:
        permnos = data_utils.get_sector_permnos(sector)
        print(f"\n--- Screening sector: {sector} ({len(permnos)} companies) ---")
        screen_results = pair_selection.screen_pairs(formation, permnos, sector_label=sector)
        screen_results["sector"] = sector
        per_sector_results.append(screen_results)

    pooled = pd.concat(per_sector_results, ignore_index=True)
    pooled = pooled.sort_values("pvalue").reset_index(drop=True)
    top = pooled.head(config.TOTAL_N_PAIRS).copy()

    sector_map = data_utils.load_sector_map()
    top["ticker1"] = top["permno1"].apply(lambda p: data_utils.ticker_for(p, sector_map))
    top["ticker2"] = top["permno2"].apply(lambda p: data_utils.ticker_for(p, sector_map))

    print(f"\n=== Global top {config.TOTAL_N_PAIRS} pairs pooled from {len(pooled)} candidates "
          f"across {len(config.SECTORS)} sectors ===")
    print(f"Sector breakdown of the final top {config.TOTAL_N_PAIRS} (NOT capped per-sector -- "
          f"this is the actual industry mix the global ranking produced): "
          f"{top['sector'].value_counts().to_dict()}")
    print(top[["sector", "ticker1", "ticker2", "coint_score", "pvalue", "overlap_days"]])
    return top


def main():
    prices = data_utils.load_wide_prices()
    formation, trading = data_utils.split_formation_trading(prices)

    if config.GLOBAL_TOP_SELECTION:
        top_pairs = select_global_top_pairs(prices, formation)
    else:
        top_pairs = select_combined_pairs(prices, formation)
    print("\nFull combined pair list selected for backtesting:")
    print(top_pairs[["sector", "ticker1", "ticker2", "coint_score", "pvalue", "overlap_days"]])

    print(f"\nWalk-forward retraining: {'ON' if config.WALK_FORWARD else 'OFF'}"
          + (f" (refit every {config.WALK_FORWARD_FREQUENCY})" if config.WALK_FORWARD
             else " (single static formation fit, as in earlier runs)"))

    strategy_names = ["Classical", "ML: Random Forest", "ML: SVM", "Buy & Hold"]
    if TORCH_AVAILABLE:
        strategy_names += ["ML: Neural Net", "ML: LSTM"]
    if XGBOOST_AVAILABLE:
        strategy_names += ["ML: Gradient Boosting"]
    portfolio_returns = {name: [] for name in strategy_names}
    # ----- Per-pair TURNOVER, collected alongside returns (see note below the
    # main loop and the run-level save at the bottom of this function) so the
    # portfolio-level daily gross-return AND daily-turnover series can be
    # persisted to disk. Together, those two series are enough to reconstruct
    # cost-adjusted returns at ANY bps level, real-risk-free-rate Sharpes, and
    # regime-interaction tests post-hoc, without rerunning the backtest again
    # for every follow-up robustness question. -----
    portfolio_turnover = {name: [] for name in strategy_names}

    all_pair_results = {}       # {pair_label: {strategy_name: daily_returns}}
    all_rmse_diagnostics = {}   # {pair_label: {strategy_name / "Naive Baseline...": rmse}}
    for _, row in top_pairs.iterrows():
        p1, p2 = row["permno1"], row["permno2"]
        label = f"{row['sector']}:{row['ticker1']}-{row['ticker2']}"
        print(f"\n=== Backtesting pair: {label} ===")
        results, rmse_diagnostics = run_all_strategies_for_pair(prices, p1, p2)

        daily_returns_only = {name: result["daily_returns"] for name, result in results.items()}
        all_pair_results[label] = daily_returns_only
        all_rmse_diagnostics[label] = rmse_diagnostics

        summary = performance.summarize(daily_returns_only)
        print(summary.round(4))
        print("  Prediction RMSE (lower is better -- Naive Baseline = predict tomorrow's "
              "spread as today's spread):")
        for name, rmse in rmse_diagnostics.items():
            print(f"    {name}: {rmse:.4f}")

        for strat_name, daily_returns in daily_returns_only.items():
            portfolio_returns[strat_name].append(daily_returns)

        # ----- Per-pair turnover for this pair, one series per strategy -----
        # Classical/ML strategies all expose position1/position2 (see
        # classical_strategy.run_classical_strategy / ml_common.finish_strategy),
        # so signals.position_turnover() works identically for every one of
        # them. Buy & Hold has no position series (it never rebalances by
        # construction -- see buy_and_hold.py), so its turnover is built
        # directly from that documented convention: 1.0 unit of turnover on
        # the first trading day (entering from flat), 0 every day after.
        for strat_name in strategy_names:
            if strat_name == "Buy & Hold":
                bh_returns = daily_returns_only["Buy & Hold"]
                bh_turnover = pd.Series(0.0, index=bh_returns.index)
                if len(bh_turnover) > 0:
                    bh_turnover.iloc[0] = 1.0
                portfolio_turnover[strat_name].append(bh_turnover)
            else:
                result = results[strat_name]
                turnover = signals.position_turnover(result["position1"], result["position2"])
                portfolio_turnover[strat_name].append(turnover)

    # Equal-weighted "top N pairs" portfolio, in the spirit of Gatev,
    # Goetzmann & Rouwenhorst (2006) -- now pooled across multiple sectors.
    print(f"\n=== Equal-weighted portfolio across {config.TOTAL_N_PAIRS} combined pairs "
          f"({len(top_pairs['sector'].unique())} sectors) ===")
    portfolio_summary_input = {}
    for strat_name, list_of_returns in portfolio_returns.items():
        combined = pd.concat(list_of_returns, axis=1).mean(axis=1)
        portfolio_summary_input[strat_name] = combined
    portfolio_summary = performance.summarize(portfolio_summary_input)
    print(portfolio_summary.round(4))

    # Equal-weighted portfolio-level turnover, same aggregation as returns
    # above -- this is what makes the saved daily_turnover_by_strategy.csv
    # combine correctly with daily_returns_by_strategy.csv post-hoc: since
    # both the per-pair cost drag and the portfolio averaging are linear
    # operations, portfolio_net_return(cost_bps) = portfolio_gross_return -
    # portfolio_avg_turnover * cost_bps / 10000, for ANY cost_bps, not just
    # the one this run happened to be configured with.
    portfolio_turnover_input = {}
    for strat_name, list_of_turnover in portfolio_turnover.items():
        combined_turnover = pd.concat(list_of_turnover, axis=1).mean(axis=1)
        portfolio_turnover_input[strat_name] = combined_turnover

    # ----- Annual breakdown: is performance driven by the market regime (see performance.py)? -----
    print("\n=== Annual breakdown of equal-weighted portfolio performance (regime check) ===")
    annual = performance.annual_breakdown(portfolio_summary_input)
    print(annual.round(4))

    # ----- Dispersion / bootstrap CI / significance: whole out-of-sample period -----
    # This is the conventional, Gatev-et-al.-style "does it work overall" report --
    # it answers a DIFFERENT, more diluted question than the regime blocks below
    # (it averages over ~3 trending years and ~1 range-bound year), so any modest
    # or non-significant result here should be interpreted alongside, not instead
    # of, the regime-conditional tests that follow.
    print(f"\n=== Full period (across the {config.TOTAL_N_PAIRS} combined pairs) ===")
    dispersion, significance, family = report_regime_stats("Full period", all_pair_results, portfolio_summary_input)

    # ----- Regime-conditional dispersion / bootstrap CI / significance -----
    # THIS is the block that actually tests the regime-dependence claim, rather
    # than eyeballing the calendar-year annual_breakdown table above: every
    # trading day is classified as "Trending" or "Range-bound" via a data-driven
    # Efficiency Ratio split (see regime.py), all_pair_results and the pooled
    # portfolio returns are restricted to each regime in turn, and the exact same
    # dispersion/bootstrap/Wilcoxon machinery is re-run within each subset. If
    # SVM's edge over Classical is genuinely a mean-reversion-regime effect, it
    # should be visible here as a tighter or more significant result specifically
    # within "Range-bound" days.
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
        filtered_pair_results = performance.filter_pair_results_by_regime(
            all_pair_results, regime_labels, regime_value)
        filtered_portfolio_returns = {
            strat_name: performance.filter_returns_by_regime(daily_returns, regime_labels, regime_value)
            for strat_name, daily_returns in portfolio_summary_input.items()
        }
        regime_dispersion[regime_value], regime_significance[regime_value], regime_family[regime_value] = (
            report_regime_stats(f"Regime: {regime_value}", filtered_pair_results, filtered_portfolio_returns))

    # ----- RMSE diagnostics averaged across pairs -----
    print("\n=== Mean prediction RMSE across pairs (lower is better) ===")
    rmse_df = pd.DataFrame(all_rmse_diagnostics).T
    print(rmse_df.mean().round(4).sort_values())

    # ----- Save into final_results/pairs_trading/{run_label}/ -----
    # Each run (selection mode x TOTAL_N_PAIRS) gets its own dedicated,
    # self-documented folder under the dissertation's top-level final_results/
    # directory (a sibling of code/ and data/), rather than a flat, suffix-
    # named file dropped into code/pairs_trading/results/. This is the
    # designated home for FINAL results/tables/graphs: distinct run_labels
    # (e.g. "globaltop_n10", "globaltop_n20", "globaltop_n40", "combined_n20")
    # never overwrite each other, each run folder carries a run_config.json
    # recording exactly which settings produced it, and nothing here needs to
    # be regenerated later to recover a forgotten configuration. The older
    # flat results/*_suffix.csv files from earlier exploratory runs are left
    # untouched in code/pairs_trading/results/ as scratch history.
    active_sectors_used = sorted(top_pairs["sector"].unique())
    mode_tag = "globaltop" if config.GLOBAL_TOP_SELECTION else "combined"
    # Cost-sensitivity passes get their own suffixed run_label (e.g.
    # "globaltop_n10_cost5bps") so they NEVER overwrite the original
    # zero-cost n=10/20/40 folders -- the two can be diffed side by side.
    cost_tag = f"_cost{int(config.TRANSACTION_COST_BPS)}bps" if config.INCLUDE_TRANSACTION_COSTS else ""
    run_label = f"{mode_tag}_n{config.TOTAL_N_PAIRS}{cost_tag}"
    title_prefix = f"Global top-{config.TOTAL_N_PAIRS} (any industry)" if config.GLOBAL_TOP_SELECTION \
        else f"Combined {config.TOTAL_N_PAIRS}-pair"

    run_dir = os.path.join("..", "..", "final_results", "pairs_trading", run_label)
    os.makedirs(os.path.join(run_dir, "trending"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "rangebound"), exist_ok=True)

    plt.figure(figsize=(10, 6))
    for strat_name, daily_returns in portfolio_summary_input.items():
        cum = performance.cumulative_return(daily_returns)
        plt.plot(cum.index, cum.values, label=strat_name)
    plt.title(f"{title_prefix} portfolio -- {len(active_sectors_used)} sectors "
              f"({', '.join(active_sectors_used)})\n"
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

    # ----- Raw portfolio-level daily series (NEW) -----
    # Every earlier run only ever saved SUMMARY statistics (Sharpe, terminal
    # return, etc.), never the underlying daily series -- fine until a
    # follow-up robustness question needs to recompute something the summary
    # stats didn't anticipate (a regime-interaction test, a different
    # risk-free rate, a cost-sensitivity curve, a different bootstrap block
    # size). Saving these two frames once, here, means none of those
    # follow-ups ever require rerunning the backtest again.
    returns_df = pd.DataFrame(portfolio_summary_input)
    returns_df.to_csv(os.path.join(run_dir, "daily_returns_by_strategy.csv"))
    turnover_df = pd.DataFrame(portfolio_turnover_input)
    turnover_df.to_csv(os.path.join(run_dir, "daily_turnover_by_strategy.csv"))
    for regime_value, subfolder in (("Trending", "trending"), ("Range-bound", "rangebound")):
        regime_dispersion[regime_value].to_csv(os.path.join(run_dir, subfolder, "dispersion.csv"))
        regime_significance[regime_value].to_csv(os.path.join(run_dir, subfolder, "significance.csv"))
        regime_family[regime_value].to_csv(os.path.join(run_dir, subfolder, "family_comparison.csv"))

    # Self-documenting run metadata -- exact config values used to produce
    # this folder's results, so a run never needs to be repeated just to
    # recall what settings it used.
    run_config = {
        "run_label": run_label,
        "strand": "pairs_trading",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "universe": {
            "selection_mode": "global_top" if config.GLOBAL_TOP_SELECTION else "combined_per_sector",
            "total_n_pairs": config.TOTAL_N_PAIRS,
            "pairs_per_sector": config.PAIRS_PER_SECTOR,
            "sectors_used": active_sectors_used,
            "sectors_configured": config.SECTORS,
        },
        "formation_period": {"start": config.FORMATION_START, "end": config.FORMATION_END},
        "trading_period": {"start": config.TRADING_START, "end": config.TRADING_END},
        "trading_rule": {
            "zscore_window": config.ZSCORE_WINDOW,
            "entry_threshold": config.ENTRY_THRESHOLD,
            "exit_threshold": config.EXIT_THRESHOLD,
            "coint_pvalue_threshold": config.COINT_PVALUE_THRESHOLD,
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
        "models_included": strategy_names,
        "random_seed": config.RANDOM_SEED,
        "trading_days_per_year": config.TRADING_DAYS_PER_YEAR,
        "risk_free_rate": config.RISK_FREE_RATE,
        "transaction_costs": (
            {
                "modeled": True,
                "cost_bps_per_unit_turnover": config.TRANSACTION_COST_BPS,
                "convention": "cost_bps of notional per unit of same-day position "
                              "turnover (entry from flat = 1 unit per leg, a full "
                              "flip = 2 units per leg); Buy & Hold gets a single "
                              "entry-cost charge on day 1 only (see buy_and_hold.py). "
                              "Applied to every strategy via signals.strategy_returns().",
            }
            if config.INCLUDE_TRANSACTION_COSTS
            else "none modeled in this run"
        ),
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nSaved all results for run '{run_label}' to "
          f"final_results/pairs_trading/{run_label}/ -- portfolio_summary.csv, "
          f"dispersion_summary.csv, significance.csv, family_comparison.csv, "
          f"rmse_diagnostics.csv, annual_breakdown.csv, portfolio_comparison.png, "
          f"run_config.json, plus trending/ and rangebound/ subfolders each with "
          f"their own dispersion.csv, significance.csv, family_comparison.csv")

    return (all_pair_results, portfolio_summary, dispersion, rmse_df, annual,
            regime_dispersion, regime_significance, family, regime_family)


if __name__ == "__main__":
    (all_pair_results, portfolio_summary, dispersion, rmse_df, annual,
     regime_dispersion, regime_significance, family, regime_family) = main()
