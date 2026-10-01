"""
Shared performance metrics and comparison table for the momentum strand's
strategies. Adapted from pairs_trading/performance.py -- the metric
functions are unchanged (they're already generic: keyed by an arbitrary
strategy-name/asset-label string, not anything pairs-specific), only the
docstrings and parameter names below are updated from "pair" to "coin"/
"asset" for clarity in this package. See pairs_trading/performance.py for
the fuller original rationale behind each function.

NOTE: config.TRADING_DAYS_PER_YEAR is 365 here (crypto trades every
calendar day), not 252 as in the equities strand -- see config.py.
"""

import numpy as np
import pandas as pd

import config

# Model-capacity grouping used by family_comparison() below -- see
# pairs_trading/performance.py's MODEL_FAMILY for the full rationale tied to
# Section 2.4's structural hypothesis.
MODEL_FAMILY = {
    "ML: Random Forest": "High-capacity (tree/sequence)",
    "ML: Gradient Boosting": "High-capacity (tree/sequence)",
    "ML: LSTM": "High-capacity (tree/sequence)",
    "ML: SVM": "Simple (linear/shallow)",
    "ML: Neural Net": "Simple (linear/shallow)",
}


def cumulative_return(daily_returns):
    return (1 + daily_returns).cumprod()


def terminal_return(daily_returns):
    return cumulative_return(daily_returns).iloc[-1] - 1


def annualized_return(daily_returns, periods_per_year=None):
    periods_per_year = periods_per_year or config.TRADING_DAYS_PER_YEAR
    n = len(daily_returns)
    if n == 0:
        return np.nan
    total_growth = cumulative_return(daily_returns).iloc[-1]
    years = n / periods_per_year
    if years == 0 or total_growth <= 0:
        return np.nan
    return total_growth ** (1 / years) - 1


def annualized_volatility(daily_returns, periods_per_year=None):
    periods_per_year = periods_per_year or config.TRADING_DAYS_PER_YEAR
    return daily_returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(daily_returns, risk_free_rate=None, periods_per_year=None):
    risk_free_rate = risk_free_rate if risk_free_rate is not None else config.RISK_FREE_RATE
    periods_per_year = periods_per_year or config.TRADING_DAYS_PER_YEAR
    excess = daily_returns - risk_free_rate / periods_per_year
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (excess.mean() / vol) * np.sqrt(periods_per_year)


def max_drawdown(daily_returns):
    cum = cumulative_return(daily_returns)
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return drawdown.min()


def summarize(strategy_returns_dict):
    """strategy_returns_dict: {"Classical": daily_returns_series,
    "ML: Random Forest": daily_returns_series, "Buy & Hold": daily_returns_series, ...}"""
    rows = []
    for name, daily_returns in strategy_returns_dict.items():
        rows.append({
            "Strategy": name,
            "Terminal Return": terminal_return(daily_returns),
            "Annualized Return": annualized_return(daily_returns),
            "Annualized Volatility": annualized_volatility(daily_returns),
            "Sharpe Ratio": sharpe_ratio(daily_returns),
            "Max Drawdown": max_drawdown(daily_returns),
        })
    return pd.DataFrame(rows).set_index("Strategy")


# ===================== DISPERSION / SIGNIFICANCE (across coins) =====================
# Same rationale as pairs_trading/performance.py: with only 15 coins, an
# equal-weighted "portfolio" Sharpe can be dominated by one unusually good or
# bad coin rather than reflecting a genuine, repeatable model difference.
# These functions look at the SPREAD of per-coin results, not just the
# pooled mean, and put a rough uncertainty band and a significance check
# around any "model A beat model B" claim.
# =============================================================================================
def per_asset_metric_table(all_asset_results, metric_fn=None):
    """all_asset_results: {coin_label: {strategy_name: daily_returns_series}}
    (this is exactly main.py's all_asset_results). Returns a DataFrame, rows
    = coins, columns = strategies, values = the given metric (default Sharpe
    ratio) computed independently on EACH coin's own daily returns."""
    metric_fn = metric_fn or sharpe_ratio
    rows = {}
    for asset_label, strategy_returns in all_asset_results.items():
        rows[asset_label] = {name: metric_fn(returns) for name, returns in strategy_returns.items()}
    return pd.DataFrame(rows).T


def dispersion_summary(all_asset_results, metric_fn=None, metric_name="Sharpe Ratio"):
    """Mean/std/min/max of a per-coin metric across all coins, per strategy
    -- report alongside the pooled portfolio number in summarize(), not
    instead of it."""
    metric_fn = metric_fn or sharpe_ratio
    table = per_asset_metric_table(all_asset_results, metric_fn)
    return pd.DataFrame({
        f"Mean {metric_name}": table.mean(),
        f"Std {metric_name}": table.std(),
        f"Min {metric_name}": table.min(),
        f"Max {metric_name}": table.max(),
        "N Coins": table.count(),
    })


def bootstrap_sharpe_ci(daily_returns, n_bootstrap=2000, ci=0.95, risk_free_rate=None,
                         periods_per_year=None, random_state=None):
    """i.i.d. bootstrap confidence interval on the Sharpe ratio -- kept for
    reference/comparison; block_bootstrap_sharpe_ci() below is the version
    actually used for reported confidence intervals (see
    pairs_trading/performance.py for the full rationale)."""
    random_state = random_state if random_state is not None else config.RANDOM_SEED
    rng = np.random.default_rng(random_state)
    values = daily_returns.dropna().values
    n = len(values)
    point_estimate = sharpe_ratio(daily_returns, risk_free_rate, periods_per_year)
    if n == 0:
        return point_estimate, np.nan, np.nan

    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_sharpes[i] = sharpe_ratio(pd.Series(sample), risk_free_rate, periods_per_year)

    lower_pct, upper_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lower, upper = np.nanpercentile(boot_sharpes, [lower_pct, upper_pct])
    return point_estimate, lower, upper


def block_bootstrap_sharpe_ci(daily_returns, n_bootstrap=2000, ci=0.95, risk_free_rate=None,
                               periods_per_year=None, block_size=None, random_state=None):
    """Moving-block bootstrap confidence interval on the Sharpe ratio -- see
    pairs_trading/performance.py's version for the full rationale. block_size
    defaults to config.BLOCK_SIZE (20 CALENDAR days here, since crypto trades
    every day). This is the version used for every reported confidence
    interval in this dissertation (see Methodology Section 4.6)."""
    random_state = random_state if random_state is not None else config.RANDOM_SEED
    block_size = block_size or getattr(config, "BLOCK_SIZE", 20)
    rng = np.random.default_rng(random_state)
    values = daily_returns.dropna().values
    n = len(values)
    point_estimate = sharpe_ratio(daily_returns, risk_free_rate, periods_per_year)
    if n == 0:
        return point_estimate, np.nan, np.nan

    block_size = max(1, min(block_size, n))
    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size

    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        if max_start <= 0:
            sample = rng.choice(values, size=n, replace=True)
        else:
            starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
            blocks = [values[s:s + block_size] for s in starts]
            sample = np.concatenate(blocks)[:n]
        boot_sharpes[i] = sharpe_ratio(pd.Series(sample), risk_free_rate, periods_per_year)

    lower_pct, upper_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lower, upper = np.nanpercentile(boot_sharpes, [lower_pct, upper_pct])
    return point_estimate, lower, upper


def benjamini_hochberg(pvalues, q=0.05):
    """Benjamini-Hochberg false-discovery-rate correction across a family of
    p-values tested together -- see pairs_trading/performance.py's version
    for the full rationale. Returns a boolean array, True where that
    hypothesis remains significant after correction."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    reject = np.zeros(n, dtype=bool)
    valid = ~np.isnan(pvalues)
    m = valid.sum()
    if m == 0:
        return reject
    order = np.argsort(np.where(valid, pvalues, np.inf))
    sorted_p = pvalues[order]
    sorted_valid = valid[order]
    ranks = np.arange(1, n + 1)
    thresholds = (ranks / m) * q
    passing = np.where(sorted_valid & (sorted_p <= thresholds))[0]
    if len(passing) > 0:
        k = passing.max()
        reject[order[:k + 1]] = True
    return reject & valid


def family_comparison(all_asset_results, family_map=None, classical_name="Classical"):
    """Model-family breakdown -- see pairs_trading/performance.py's version
    for the full rationale tied to Section 2.4's capacity argument. Pools
    per-coin Sharpe-ratio differences vs Classical across the ML models in
    each capacity family (MODEL_FAMILY), then tests via a two-sided
    Mann-Whitney U test whether the two families' pooled differences differ.
    Call this on a regime-filtered all_asset_results (see
    filter_asset_results_by_regime()) for the regime-conditional version of
    the same question. Returns a DataFrame with the cross-family p-value
    attached as result.attrs['family_comparison_p_value']."""
    from scipy.stats import mannwhitneyu
    family_map = family_map or MODEL_FAMILY
    table = per_asset_metric_table(all_asset_results)

    diffs_by_family = {}
    for strat_name, family in family_map.items():
        if strat_name not in table.columns or classical_name not in table.columns:
            continue
        diffs = (table[strat_name] - table[classical_name]).dropna()
        diffs_by_family.setdefault(family, []).extend(diffs.tolist())

    rows = []
    for family, diffs in diffs_by_family.items():
        arr = np.array(diffs, dtype=float)
        rows.append({
            "Model Family": family,
            "Mean Sharpe Diff vs Classical": arr.mean() if len(arr) else np.nan,
            "Median Sharpe Diff vs Classical": np.median(arr) if len(arr) else np.nan,
            "N Pooled Obs (coins x models in family)": len(arr),
        })
    result = pd.DataFrame(rows).set_index("Model Family") if rows else pd.DataFrame(
        columns=["Mean Sharpe Diff vs Classical", "Median Sharpe Diff vs Classical",
                 "N Pooled Obs (coins x models in family)"])

    families = list(diffs_by_family.keys())
    p_value = np.nan
    if len(families) == 2 and all(len(diffs_by_family[f]) >= 2 for f in families):
        try:
            _, p_value = mannwhitneyu(diffs_by_family[families[0]], diffs_by_family[families[1]],
                                       alternative="two-sided")
        except ValueError:
            p_value = np.nan
    result.attrs["family_comparison_p_value"] = p_value
    return result


# ===================== ANNUAL BREAKDOWN (regime dependence) =====================
def annual_breakdown(strategy_returns_dict, risk_free_rate=None, periods_per_year=None):
    """strategy_returns_dict: {"Classical": daily_returns_series, ...} --
    returns a DataFrame indexed by (Year, Strategy) with Terminal Return and
    Sharpe Ratio computed independently within each calendar year."""
    rows = []
    for name, daily_returns in strategy_returns_dict.items():
        for year, year_returns in daily_returns.groupby(daily_returns.index.year):
            rows.append({
                "Year": year,
                "Strategy": name,
                "Terminal Return": terminal_return(year_returns),
                "Sharpe Ratio": sharpe_ratio(year_returns, risk_free_rate, periods_per_year),
                "N Days": len(year_returns),
            })
    return pd.DataFrame(rows).set_index(["Year", "Strategy"]).sort_index()


# ===================== REGIME-CONDITIONAL FILTERING =====================
# Pairs with regime.classify_regime() -- once every trading day has a
# "Trending"/"Range-bound" label, restricting any daily_returns Series to one
# regime is just a reindex -- every function above then works completely
# unchanged on the filtered series.
# ===========================================================================
def filter_returns_by_regime(daily_returns, regime_labels, regime_value):
    matching_dates = regime_labels[regime_labels == regime_value].index
    return daily_returns.reindex(daily_returns.index.intersection(matching_dates))


def filter_asset_results_by_regime(all_asset_results, regime_labels, regime_value):
    """all_asset_results: {coin_label: {strategy_name: daily_returns}} --
    returns the same nested structure, with every daily_returns Series
    restricted to the given regime."""
    return {
        asset_label: {
            name: filter_returns_by_regime(returns, regime_labels, regime_value)
            for name, returns in strategy_returns.items()
        }
        for asset_label, strategy_returns in all_asset_results.items()
    }


# ===================== REGIME-INTERACTION BOOTSTRAP TESTS (RQ1/RQ3) =====================
# Same rationale and mechanics as pairs_trading/performance.py's identically-
# named functions -- see that module for the full docstrings. Ported here
# unchanged (the block-bootstrap engine is already generic: it only needs
# daily_returns Series and a regime_labels Series, neither of which is
# pairs- or coin-specific), operating on the equal-weighted 15-coin
# portfolio's daily_returns_by_strategy.csv the same way it operates on the
# pairs strand's combined-pairs portfolio. config.BLOCK_SIZE here means 20
# CALENDAR days (crypto trades every day), matching this strand's own
# annualization convention (TRADING_DAYS_PER_YEAR=365) rather than the
# equities strand's 252.
# ==========================================================================================
def bootstrap_two_sided_p_value(boot_values, null_value=0.0):
    """Two-sided bootstrap percentile p-value for a test statistic against
    `null_value` (0.0 for both tests below): p = 2 * min(P(boot >= null),
    P(boot <= null)), capped at 1."""
    boot_values = np.asarray(boot_values, dtype=float)
    boot_values = boot_values[~np.isnan(boot_values)]
    if len(boot_values) == 0:
        return np.nan
    p_ge = np.mean(boot_values >= null_value)
    p_le = np.mean(boot_values <= null_value)
    return min(1.0, 2 * min(p_ge, p_le))


def _joint_block_bootstrap_regime_stat(series_dict, regime_labels, statistic_fn,
                                        n_bootstrap=2000, block_size=None, random_state=None):
    """Shared resampling engine for both regime-interaction tests below --
    see pairs_trading/performance.py's identically-named function for the
    full rationale. Draws contiguous blocks of `block_size` days (default
    config.BLOCK_SIZE), the SAME block start positions applied to every
    series in series_dict AND to regime_labels simultaneously, preserving
    both same-day pairing across series and each resampled day's own true
    regime label."""
    random_state = random_state if random_state is not None else config.RANDOM_SEED
    block_size = block_size or getattr(config, "BLOCK_SIZE", 20)
    rng = np.random.default_rng(random_state)

    common_index = regime_labels.index
    for s in series_dict.values():
        common_index = common_index.intersection(s.index)
    common_index = common_index.sort_values()

    labels_arr = regime_labels.reindex(common_index).values
    values_arr = {name: s.reindex(common_index).values for name, s in series_dict.items()}
    n = len(common_index)
    if n == 0:
        return np.full(n_bootstrap, np.nan)

    block_size = max(1, min(block_size, n))
    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size

    results = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        if max_start <= 0:
            idx = rng.integers(0, n, size=n)
        else:
            starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
            idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        resampled_labels = pd.Series(labels_arr[idx])
        try:
            results[i] = statistic_fn(
                {name: pd.Series(vals[idx]) for name, vals in values_arr.items()},
                resampled_labels,
            )
        except (ValueError, ZeroDivisionError):
            results[i] = np.nan
    return results


def trend_range_sharpe_delta(daily_returns, regime_labels, risk_free_rate=None, periods_per_year=None):
    """Point estimate for the RQ1/Point-6 test: one strategy's own Sharpe
    ratio in Trending days minus its Sharpe ratio in Range-bound days."""
    trending = filter_returns_by_regime(daily_returns, regime_labels, "Trending")
    range_bound = filter_returns_by_regime(daily_returns, regime_labels, "Range-bound")
    return (sharpe_ratio(trending, risk_free_rate, periods_per_year)
            - sharpe_ratio(range_bound, risk_free_rate, periods_per_year))


def trend_range_delta_bootstrap_test(daily_returns, regime_labels, n_bootstrap=2000, ci=0.95,
                                      risk_free_rate=None, periods_per_year=None, block_size=None,
                                      random_state=None):
    """RQ1 / Point 6: is a single strategy's OWN Sharpe ratio itself
    regime-dependent? See pairs_trading/performance.py for the full
    rationale. Returns (point_estimate, ci_lower, ci_upper, p_value)."""
    common = regime_labels.index.intersection(daily_returns.dropna().index)
    labels = regime_labels.reindex(common)
    returns = daily_returns.reindex(common)
    point_estimate = trend_range_sharpe_delta(returns, labels, risk_free_rate, periods_per_year)

    def stat_fn(series_dict, resampled_labels):
        return trend_range_sharpe_delta(series_dict["returns"], resampled_labels,
                                         risk_free_rate, periods_per_year)

    boot = _joint_block_bootstrap_regime_stat({"returns": returns}, labels, stat_fn,
                                               n_bootstrap=n_bootstrap, block_size=block_size,
                                               random_state=random_state)
    lower_pct, upper_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lower, upper = np.nanpercentile(boot, [lower_pct, upper_pct])
    p_value = bootstrap_two_sided_p_value(boot)
    return point_estimate, lower, upper, p_value


def trend_range_delta_table(strategy_returns_dict, regime_labels, strategies=None, n_bootstrap=2000,
                             ci=0.95, risk_free_rate=None, periods_per_year=None, block_size=None,
                             random_state=None, bh_q=0.05):
    """Builds the RQ1/Point-6 summary table -- see pairs_trading/performance.py."""
    strategies = strategies or list(strategy_returns_dict.keys())
    rows = []
    for name in strategies:
        point, lower, upper, p = trend_range_delta_bootstrap_test(
            strategy_returns_dict[name], regime_labels, n_bootstrap=n_bootstrap, ci=ci,
            risk_free_rate=risk_free_rate, periods_per_year=periods_per_year,
            block_size=block_size, random_state=random_state)
        rows.append({
            "Strategy": name,
            "Delta Sharpe (Trend - Range)": point,
            "CI Lower": lower,
            "CI Upper": upper,
            "p-value": p,
        })
    table = pd.DataFrame(rows).set_index("Strategy")
    table["BH Significant (q={:.2f})".format(bh_q)] = benjamini_hochberg(
        table["p-value"].astype(float).values, q=bh_q)
    return table


def ml_classical_regime_interaction(ml_returns, classical_returns, regime_labels,
                                     risk_free_rate=None, periods_per_year=None):
    """Point estimate for the RQ3/Point-5 test: Delta_regime =
    (Sharpe_ML - Sharpe_Classical)_Trending - (Sharpe_ML -
    Sharpe_Classical)_Range-bound."""
    trending_ml = filter_returns_by_regime(ml_returns, regime_labels, "Trending")
    trending_cl = filter_returns_by_regime(classical_returns, regime_labels, "Trending")
    range_ml = filter_returns_by_regime(ml_returns, regime_labels, "Range-bound")
    range_cl = filter_returns_by_regime(classical_returns, regime_labels, "Range-bound")
    trend_gap = (sharpe_ratio(trending_ml, risk_free_rate, periods_per_year)
                 - sharpe_ratio(trending_cl, risk_free_rate, periods_per_year))
    range_gap = (sharpe_ratio(range_ml, risk_free_rate, periods_per_year)
                 - sharpe_ratio(range_cl, risk_free_rate, periods_per_year))
    return trend_gap - range_gap


def ml_classical_regime_interaction_bootstrap_test(ml_returns, classical_returns, regime_labels,
                                                    n_bootstrap=2000, ci=0.95, risk_free_rate=None,
                                                    periods_per_year=None, block_size=None,
                                                    random_state=None):
    """RQ3 / Point 5: block-bootstrap test of the ML-vs-Classical
    INTERACTION with regime -- see pairs_trading/performance.py for the
    full rationale. Returns (point_estimate, ci_lower, ci_upper, p_value)."""
    common = (regime_labels.index
              .intersection(ml_returns.dropna().index)
              .intersection(classical_returns.dropna().index))
    labels = regime_labels.reindex(common)
    ml = ml_returns.reindex(common)
    cl = classical_returns.reindex(common)
    point_estimate = ml_classical_regime_interaction(ml, cl, labels, risk_free_rate, periods_per_year)

    def stat_fn(series_dict, resampled_labels):
        return ml_classical_regime_interaction(series_dict["ml"], series_dict["classical"],
                                                resampled_labels, risk_free_rate, periods_per_year)

    boot = _joint_block_bootstrap_regime_stat({"ml": ml, "classical": cl}, labels, stat_fn,
                                               n_bootstrap=n_bootstrap, block_size=block_size,
                                               random_state=random_state)
    lower_pct, upper_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lower, upper = np.nanpercentile(boot, [lower_pct, upper_pct])
    p_value = bootstrap_two_sided_p_value(boot)
    return point_estimate, lower, upper, p_value


def ml_classical_interaction_table(strategy_returns_dict, regime_labels, classical_name="Classical",
                                    ml_names=None, n_bootstrap=2000, ci=0.95, risk_free_rate=None,
                                    periods_per_year=None, block_size=None, random_state=None, bh_q=0.05):
    """Builds the RQ3/Point-5 summary table -- see pairs_trading/performance.py."""
    if ml_names is None:
        ml_names = [name for name in strategy_returns_dict
                    if name not in (classical_name, "Buy & Hold")]
    rows = []
    for name in ml_names:
        point, lower, upper, p = ml_classical_regime_interaction_bootstrap_test(
            strategy_returns_dict[name], strategy_returns_dict[classical_name], regime_labels,
            n_bootstrap=n_bootstrap, ci=ci, risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year, block_size=block_size, random_state=random_state)
        rows.append({
            "Model": name,
            "Delta_regime (Trend gap - Range gap)": point,
            "CI Lower": lower,
            "CI Upper": upper,
            "p-value": p,
        })
    table = pd.DataFrame(rows).set_index("Model")
    table["BH Significant (q={:.2f})".format(bh_q)] = benjamini_hochberg(
        table["p-value"].astype(float).values, q=bh_q)
    return table


def paired_significance_test(all_asset_results, strategy_a, strategy_b, metric_fn=None):
    """Wilcoxon signed-rank test on the per-coin metric difference between
    two strategies. Answers: "is strategy_a's edge over strategy_b
    consistent across coins, or could it plausibly just be noise?" Returns
    (median_difference, p_value).

    Caveat, same as pairs_trading/performance.py: with only 15 coins this
    test has limited statistical power -- a non-significant p-value here
    means "not enough coins to be sure", not "no difference exists"."""
    from scipy.stats import wilcoxon
    table = per_asset_metric_table(all_asset_results, metric_fn)
    diffs = (table[strategy_a] - table[strategy_b]).dropna()
    if len(diffs) < 2 or (diffs == 0).all():
        return diffs.median() if len(diffs) else np.nan, np.nan
    stat, p_value = wilcoxon(diffs)
    return diffs.median(), p_value
